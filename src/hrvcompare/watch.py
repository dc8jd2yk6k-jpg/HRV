"""Digitize the watch's overnight HRV chart from an app screenshot.

COROS measures overnight HRV as RMSSD over 10-minute intervals during sleep
and reports the night's average. The app only shows a line chart, so the
values are recovered from pixels:

* y: the horizontal gridlines (0/50/100/150 ms) are detected as faint
  full-width rows and fitted linearly, row -> ms;
* x: the chart's baseline spans sleep onset .. wake (the two axis labels);
* the line is traced as the per-column centroid of the line-coloured pixels
  and its vertices -- the watch's actual measurements -- are recovered with a
  free-knot polyline fit. They are *not* evenly spaced: COROS samples roughly
  every 10 min but skips some intervals, and the chart joins what it has.

The calibration is checked against the app's own "Last Night Avg".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.signal import find_peaks

# Fallback calibration measured on the 473x1024 COROS screenshot.
DEFAULT_GRID_ROWS = {150: 408.0, 100: 484.0, 50: 560.0, 0: 635.5}


@dataclass
class WatchHRV:
    point_s: np.ndarray  # wall-clock seconds of each sampled measurement
    point_ms: np.ndarray
    curve_s: np.ndarray  # per-pixel-column trace
    curve_ms: np.ndarray
    axis_start_s: float
    axis_end_s: float
    interval_s: float
    calibration: dict = field(default_factory=dict)
    reported_avg: float | None = None
    normal_range: tuple[float, float] | None = None

    @property
    def digitized_avg(self) -> float:
        return float(np.mean(self.point_ms))

    @property
    def ms_per_px(self) -> float:
        return 1.0 / self.calibration["px_per_ms"]


def _line_mask(img: np.ndarray) -> np.ndarray:
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    return (g > 110) & (b > 110) & (r < 150) & ((g - r) > 40)


def _gridline_rows(img: np.ndarray, y_lo: int, y_hi: int, x_lo: int, x_hi: int) -> list[float]:
    """Centres of faint neutral-grey horizontal lines (dashed or solid)."""
    seg = img[y_lo:y_hi, x_lo:x_hi]
    lum = seg.mean(axis=2)
    neutral = (np.abs(seg[..., 0] - seg[..., 1]) < 8) & (np.abs(seg[..., 1] - seg[..., 2]) < 8)
    frac = ((lum > 12) & (lum < 70) & neutral).mean(axis=1)
    rows = np.flatnonzero(frac > 0.35) + y_lo
    if len(rows) == 0:
        return []
    groups = np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1)
    return [float(g.mean()) for g in groups]


def _fit_grid(rows: list[float], values: tuple[int, ...]) -> dict | None:
    """Pick the run of len(values) equally spaced rows and fit row -> ms."""
    k = len(values)
    best = None
    for i in range(len(rows) - k + 1):
        cand = np.array(rows[i : i + k])
        gaps = np.diff(cand)
        if gaps.min() <= 0 or (gaps.max() - gaps.min()) > 3:
            continue
        if best is None or gaps.mean() > np.diff(best).mean():
            best = cand
    if best is None:
        return None
    vals = np.array(sorted(values, reverse=True), dtype=float)  # top row = largest value
    slope, intercept = np.polyfit(vals, best, 1)  # row = intercept + slope * ms
    return {"rows": best.tolist(), "values": vals.tolist(), "px_per_ms": -slope, "zero_row": intercept, "source": "detected"}


def _hat_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Design matrix of a continuous piecewise-linear function with the given knots."""
    a = np.zeros((len(x), len(knots)))
    k = np.clip(np.searchsorted(knots, x, side="right") - 1, 0, len(knots) - 2)
    t = (x - knots[k]) / (knots[k + 1] - knots[k])
    a[np.arange(len(x)), k] = 1 - t
    a[np.arange(len(x)), k + 1] = t
    return a


def _polyline_fit(x: np.ndarray, y: np.ndarray, knots: np.ndarray) -> tuple[np.ndarray, float]:
    a = _hat_basis(x, knots)
    vals, *_ = np.linalg.lstsq(a, y, rcond=None)
    return vals, float(np.sum((a @ vals - y) ** 2))


def _fit_vertices(
    x: np.ndarray, y: np.ndarray, w: int = 4, min_angle: float = 0.10, max_err_px: float = 1.5, min_gap: float = 3.0
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Recover the polyline's vertices (the watch's actual measurements).

    1. seed knots where the local slope changes (left/right line fits);
    2. knot values by least squares on a piecewise-linear basis, so a corner's
       value comes from the straight segments either side (the centroid of a
       thick anti-aliased line rounds peaks off);
    3. nudge each knot to the position that minimises the squared error;
    4. add a knot wherever the fit still misses the drawn line.
    """
    n = len(x)
    ang = np.zeros(n)
    for i in range(w, n - w):
        sl = np.polyfit(x[i - w : i + 1], y[i - w : i + 1], 1)[0]
        sr = np.polyfit(x[i : i + w + 1], y[i : i + w + 1], 1)[0]
        ang[i] = abs(np.arctan(sr) - np.arctan(sl))
    peaks, _ = find_peaks(ang, height=min_angle, distance=5)
    knots = np.unique(np.concatenate([[x[0]], x[peaks], [x[-1]]])).astype(float)

    for _ in range(6):
        for k in range(1, len(knots) - 1):
            cands = knots[k] + np.arange(-3, 3.01, 0.5)
            cands = cands[(cands > knots[k - 1] + min_gap) & (cands < knots[k + 1] - min_gap)]
            if len(cands) == 0:
                continue
            best = min(cands, key=lambda c: _polyline_fit(x, y, np.r_[knots[:k], c, knots[k + 1 :]])[1])
            knots[k] = best
        vals, _ = _polyline_fit(x, y, knots)
        err = np.abs(_hat_basis(x, knots) @ vals - y)
        worst = int(np.argmax(err))
        if err[worst] <= max_err_px or np.min(np.abs(knots - x[worst])) < min_gap:
            break
        knots = np.sort(np.r_[knots, x[worst]])

    vals, sse = _polyline_fit(x, y, knots)
    err = np.abs(_hat_basis(x, knots) @ vals - y)
    return knots, vals, {"rmse_px": float(np.sqrt(sse / n)), "max_err_px": float(err.max())}


def digitize_hrv_chart(
    png: str | Path | Image.Image,
    axis_start_s: float,
    axis_end_s: float,
    interval_min: float = 10.0,
    grid_values: tuple[int, ...] = (0, 50, 100, 150),
    reported_avg: float | None = None,
    normal_range: tuple[float, float] | None = None,
) -> WatchHRV:
    image = png if isinstance(png, Image.Image) else Image.open(png)
    img = np.asarray(image.convert("RGB")).astype(int)
    h, w, _ = img.shape
    ys, xs = np.nonzero(_line_mask(img))
    if len(ys) == 0:
        raise ValueError("no chart line found in screenshot")
    # Other UI elements share the line colour (status text, the card icon);
    # the chart line is the vertical band holding the most such pixels.
    band_rows = np.unique(ys)
    bands = np.split(band_rows, np.flatnonzero(np.diff(band_rows) > 20) + 1)
    band = max(bands, key=lambda b: np.isin(ys, b).sum())
    keep = (ys >= band[0]) & (ys <= band[-1])
    ys, xs = ys[keep], xs[keep]
    y_top, y_bot = ys.min(), ys.max()

    rows = _gridline_rows(img, max(0, y_top - 300), min(h, y_bot + 80), int(w * 0.05), int(w * 0.85))
    calib = _fit_grid(rows, grid_values)
    if calib is None:
        vals = np.array(sorted(DEFAULT_GRID_ROWS), dtype=float)
        rws = np.array([DEFAULT_GRID_ROWS[v] for v in sorted(DEFAULT_GRID_ROWS)])
        slope, intercept = np.polyfit(vals, rws, 1)
        calib = {"rows": rws.tolist(), "values": vals.tolist(), "px_per_ms": -slope, "zero_row": intercept, "source": "fallback"}

    # Time axis = horizontal extent of the baseline (lowest gridline).
    base = int(round(max(calib["rows"])))
    lit = np.flatnonzero(img[base, :, :].mean(axis=1) > 12)
    x_left, x_right = float(lit.min()), float(lit.max())
    calib.update({"x_left": x_left, "x_right": x_right})

    def x_to_s(x):
        return axis_start_s + (np.asarray(x, dtype=float) - x_left) / (x_right - x_left) * (axis_end_s - axis_start_s)

    def row_to_ms(y):
        return (calib["zero_row"] - np.asarray(y, dtype=float)) / calib["px_per_ms"]

    cols = np.arange(xs.min(), xs.max() + 1)
    centroid = np.array([ys[xs == c].mean() if np.any(xs == c) else np.nan for c in cols])
    good = ~np.isnan(centroid)
    cols, centroid = cols[good], centroid[good]
    curve_s, curve_ms = x_to_s(cols), row_to_ms(centroid)

    knots, knot_rows, fit = _fit_vertices(cols.astype(float), centroid)
    point_s, point_ms = x_to_s(knots), row_to_ms(knot_rows)
    gaps = np.diff(point_s) / 60
    calib.update(fit)
    calib["n_points"] = int(len(point_s))
    calib["spacing_median_min"] = float(np.median(gaps)) if len(gaps) else np.nan
    calib["n_gaps_over_1_5x"] = int((gaps > 1.5 * interval_min).sum())

    interval_s = interval_min * 60
    return WatchHRV(
        point_s=point_s,
        point_ms=point_ms,
        curve_s=curve_s,
        curve_ms=curve_ms,
        axis_start_s=axis_start_s,
        axis_end_s=axis_end_s,
        interval_s=interval_s,
        calibration=calib,
        reported_avg=reported_avg,
        normal_range=normal_range,
    )
