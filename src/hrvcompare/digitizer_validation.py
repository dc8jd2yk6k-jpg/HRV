"""Validate the screenshot digitizer on charts with known readings.

The real screenshot is used as a template: its HRV line is erased and a new
polyline with known, irregularly spaced readings is drawn in the same colour
and width (anti-aliased by supersampling). Everything else on the screen --
gridlines, labels, icons -- is the app's own rendering. The digitizer is run
on each chart and its readings are matched to the true ones.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation

from .io import clock_to_seconds
from .watch import _line_mask, digitize_hrv_chart

LINE_RGB = (107, 173, 175)
LINE_WIDTH_PX = 3
SUPERSAMPLE = 4


def erase_line(img: np.ndarray, rows: tuple[int, int] = (470, 633), cols: tuple[int, int] = (0, 425), dash_period: int = 22) -> np.ndarray:
    """Remove the chart line, including its faint anti-aliased halo. Inside the
    plot rectangle every row that is not a dashed gridline is reset to the
    black background; on gridline rows the pixels under the dilated line are
    replaced by the dash pattern copied a few periods along the same row."""
    out = img.copy()
    r0, r1 = rows
    c0, c1 = cols
    lum = img.mean(axis=2)
    gridline = (lum[:, 20:420] > 12).mean(axis=1) > 0.3
    mask = np.zeros(img.shape[:2], dtype=bool)
    mask[r0:r1, c0:c1] = _line_mask(img)[r0:r1, c0:c1]
    mask = binary_dilation(mask, iterations=3)
    for y in range(r0, r1):
        if not gridline[y]:
            out[y, c0:c1] = (1, 1, 1)
            continue
        for x in np.flatnonzero(mask[y, c0:c1]) + c0:
            for k in (1, -1, 2, -2, 3, -3):
                xs = x + k * dash_period
                if c0 <= xs < c1 and not mask[y, xs]:
                    out[y, x] = img[y, xs]
                    break
    return out


def random_readings(rng: np.random.Generator, axis_start_s: float, axis_end_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Irregular ~10-min readings with occasional skips; AR(1) values."""
    t = axis_start_s + rng.uniform(10, 25) * 60
    times = []
    while t < axis_end_s - rng.uniform(2, 10) * 60:
        times.append(t)
        step = rng.choice([10, 20, 30], p=[0.85, 0.10, 0.05])
        t += (step + rng.uniform(-0.5, 0.5)) * 60
    n = len(times)
    mean, sd = rng.uniform(30, 65), rng.uniform(8, 18)
    v = np.empty(n)
    v[0] = mean + rng.normal(0, sd)
    for i in range(1, n):
        v[i] = mean + 0.5 * (v[i - 1] - mean) + rng.normal(0, sd * np.sqrt(1 - 0.25))
    return np.asarray(times), np.clip(v, 12, 140)


def render_chart(template: np.ndarray, times_s, values_ms, calib: dict, axis_start_s: float, axis_end_s: float) -> Image.Image:
    h, w, _ = template.shape
    x = calib["x_left"] + (np.asarray(times_s) - axis_start_s) / (axis_end_s - axis_start_s) * (calib["x_right"] - calib["x_left"])
    y = calib["zero_row"] - np.asarray(values_ms) * calib["px_per_ms"]
    overlay = Image.new("RGBA", (w * SUPERSAMPLE, h * SUPERSAMPLE), (0, 0, 0, 0))
    pts = [(float(a) * SUPERSAMPLE, float(b) * SUPERSAMPLE) for a, b in zip(x + 0.5, y + 0.5)]
    ImageDraw.Draw(overlay).line(pts, fill=LINE_RGB + (255,), width=LINE_WIDTH_PX * SUPERSAMPLE, joint="curve")
    overlay = overlay.resize((w, h), Image.LANCZOS)
    base = Image.fromarray(template.astype(np.uint8)).convert("RGBA")
    return Image.alpha_composite(base, overlay).convert("RGB")


def match(true_t, true_v, est_t, est_v, tol_s: float = 300.0) -> dict:
    """Greedy one-to-one matching by time within *tol_s*."""
    pairs = sorted((abs(a - b), i, j) for i, a in enumerate(true_t) for j, b in enumerate(est_t) if abs(a - b) <= tol_s)
    used_i, used_j, m = set(), set(), []
    for _, i, j in pairs:
        if i not in used_i and j not in used_j:
            used_i.add(i)
            used_j.add(j)
            m.append((i, j))
    vi = np.array([est_v[j] - true_v[i] for i, j in m])
    ti = np.array([(est_t[j] - true_t[i]) / 60 for i, j in m])
    return {
        "n_true": len(true_t),
        "n_digitized": len(est_t),
        "recall": len(m) / len(true_t),
        "precision": len(m) / len(est_t) if len(est_t) else np.nan,
        "value_error_median_abs_ms": float(np.median(np.abs(vi))) if len(vi) else np.nan,
        "value_error_p95_abs_ms": float(np.percentile(np.abs(vi), 95)) if len(vi) else np.nan,
        "time_error_median_abs_min": float(np.median(np.abs(ti))) if len(ti) else np.nan,
        "mean_error_ms": float(np.mean(est_v) - np.mean(true_v)),
        "value_errors": vi,
    }


def validate(png: Path, sleep: tuple[str, str] = ("02:29", "09:37"), n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    a0 = clock_to_seconds(sleep[0])
    a1 = clock_to_seconds(sleep[1], ref_s=a0)
    real = digitize_hrv_chart(png, a0, a1)
    template = erase_line(np.asarray(Image.open(png).convert("RGB")).astype(int))
    rng = np.random.default_rng(seed)
    rows, all_err = [], []
    for k in range(n):
        t, v = random_readings(rng, a0, a1)
        chart = render_chart(template, t, v, real.calibration, a0, a1)
        est = digitize_hrv_chart(chart, a0, a1)
        r = match(t, v, est.point_s, est.point_ms)
        all_err.append(r.pop("value_errors"))
        rows.append({"chart": k, **r})
    return pd.DataFrame(rows), np.concatenate(all_err)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m hrvcompare.digitizer_validation")
    p.add_argument("--png", type=Path, default=Path("data/raw/coros_hrv_20260417.png"))
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--out", type=Path, default=Path("paper/generated/digitizer"))
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    df, err = validate(args.png, n=args.n)
    df.to_csv(args.out / "charts.csv", index=False, float_format="%.4g")
    pd.DataFrame({"value_error_ms": err}).to_csv(args.out / "value_errors.csv", index=False, float_format="%.4g")
    print(df.describe().round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
