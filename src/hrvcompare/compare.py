"""Strap-vs-watch agreement: matched windows, statistics, sensitivity grid."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats

from .artifacts import CleanResult
from .io import StrapRecording
from .metrics import GATES, Gate, window_metrics
from .timeline import HYPOTHESES, wall_end_times
from .watch import WatchHRV

CONVENTIONS = ("ending", "centred")


def watch_windows(point_s: np.ndarray, length_s: float, convention: str = "ending") -> tuple[np.ndarray, np.ndarray]:
    """Strap window for each watch point: the *length_s* before the point
    ("ending") or centred on it ("centred")."""
    if convention == "ending":
        return point_s - length_s, point_s
    if convention == "centred":
        return point_s - length_s / 2, point_s + length_s / 2
    raise ValueError(f"unknown convention {convention!r}")


def matched_table(
    clean: CleanResult,
    wall_end_s: np.ndarray,
    watch: WatchHRV,
    length_s: float = 600.0,
    convention: str = "ending",
    gate: Gate = GATES["default"],
) -> pd.DataFrame:
    starts, ends = watch_windows(watch.point_s, length_s, convention)
    df = window_metrics(clean.beats, wall_end_s, starts, ends, gate)
    df.insert(0, "point_s", watch.point_s)
    df.insert(1, "watch_rmssd", watch.point_ms)
    return df


def agreement(strap: np.ndarray, watch: np.ndarray) -> dict:
    """Agreement of watch against strap (reference). Bland-Altman is reported
    both in ms and on ln scale (back-transformed to a watch/strap ratio), since
    RMSSD is log-normally distributed and its error scales with its level."""
    s = np.asarray(strap, dtype=float)
    w = np.asarray(watch, dtype=float)
    keep = np.isfinite(s) & np.isfinite(w) & (s > 0) & (w > 0)
    s, w = s[keep], w[keep]
    n = len(s)
    out = {"n": n, "strap_mean": np.nan, "watch_mean": np.nan}
    if n == 0:
        return out
    d = w - s
    dl = np.log(w) - np.log(s)
    out.update(
        {
            "strap_mean": float(s.mean()),
            "watch_mean": float(w.mean()),
            "bias_ms": float(d.mean()),
            "mae_ms": float(np.abs(d).mean()),
            "mape_pct": float(np.mean(np.abs(d) / s) * 100),
            "ratio": float(np.exp(dl.mean())),
        }
    )
    if n >= 3:
        sd, sdl = d.std(ddof=1), dl.std(ddof=1)
        out.update(
            {
                "loa_low_ms": float(d.mean() - 1.96 * sd),
                "loa_high_ms": float(d.mean() + 1.96 * sd),
                "ratio_loa_low": float(np.exp(dl.mean() - 1.96 * sdl)),
                "ratio_loa_high": float(np.exp(dl.mean() + 1.96 * sdl)),
                "ccc": lin_ccc(s, w),
                "ccc_ln": lin_ccc(np.log(s), np.log(w)),
            }
        )
        r = stats.pearsonr(np.log(s), np.log(w))
        out["pearson_r_ln"] = float(r.statistic)
        out["pearson_p"] = float(r.pvalue)
        if n > 3:
            ci = r.confidence_interval(0.95)
            out["pearson_ci_low"], out["pearson_ci_high"] = float(ci.low), float(ci.high)
        out["spearman_rho"] = float(stats.spearmanr(s, w).statistic)
    return out


def lin_ccc(x: np.ndarray, y: np.ndarray) -> float:
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = ((x - mx) * (y - my)).mean()
    return float(2 * cov / (vx + vy + (mx - my) ** 2))


def epoch_agreement(matched: pd.DataFrame) -> dict:
    v = matched[matched["valid"]]
    return agreement(v["rmssd"].to_numpy(), v["watch_rmssd"].to_numpy())


def block_agreement(matched: pd.DataFrame, points_per_block: int = 3, min_valid: int = 2) -> tuple[dict, pd.DataFrame]:
    """Agreement after averaging consecutive watch points into blocks (e.g. 3 x
    10 min = 30 min), using only the points whose strap window is valid."""
    m = matched.copy()
    m["block"] = np.arange(len(m)) // points_per_block
    v = m[m["valid"]]
    g = v.groupby("block").agg(
        point_s=("point_s", "mean"), strap=("rmssd", "mean"), watch=("watch_rmssd", "mean"), n=("rmssd", "size")
    )
    g = g[g["n"] >= min_valid]
    return agreement(g["strap"].to_numpy(), g["watch"].to_numpy()), g.reset_index()


def night_level(matched: pd.DataFrame, watch: WatchHRV) -> dict:
    """Primary endpoint: the night's average HRV, which needs no fine
    alignment. The coverage-matched version compares the two devices over the
    same measurement points only."""
    v = matched[matched["valid"]]
    out = {
        "watch_reported_avg": watch.reported_avg,
        "watch_digitized_avg_all": watch.digitized_avg,
        "n_points": int(len(matched)),
        "n_valid": int(len(v)),
        "strap_avg_valid": float(v["rmssd"].mean()) if len(v) else np.nan,
        "watch_avg_matched": float(v["watch_rmssd"].mean()) if len(v) else np.nan,
    }
    out["matched_diff_ms"] = out["watch_avg_matched"] - out["strap_avg_valid"]
    out["matched_ratio"] = out["watch_avg_matched"] / out["strap_avg_valid"]
    if watch.reported_avg is not None:
        out["reported_vs_strap_diff_ms"] = watch.reported_avg - out["strap_avg_valid"]
    return out


def sensitivity_grid(
    rec: StrapRecording,
    clean: CleanResult,
    watch: WatchHRV,
    hypotheses=tuple(HYPOTHESES),
    gates=tuple(GATES),
    conventions=CONVENTIONS,
    lengths_min=(10, 5),
) -> pd.DataFrame:
    walls = {h: wall_end_times(rec, clean, h) for h in hypotheses}
    rows = []
    for h, g, c, L in itertools.product(hypotheses, gates, conventions, lengths_min):
        m = matched_table(clean, walls[h], watch, L * 60, c, GATES[g])
        a = epoch_agreement(m)
        rows.append({"hypothesis": h, "gate": g, "convention": c, "window_min": L, **night_level(m, watch), **a})
    return pd.DataFrame(rows)


def lag_scan(
    rec: StrapRecording,
    clean: CleanResult,
    watch: WatchHRV,
    lags_min=range(-20, 21, 2),
    hypothesis: str = "H1",
    gate: Gate = GATES["default"],
) -> pd.DataFrame:
    """Diagnostic only: agreement when the strap timeline is shifted. With few
    valid windows this surface is noisy, so no offset is fitted from it."""
    rows = []
    for lag in lags_min:
        wall = wall_end_times(rec, clean, hypothesis, lag_s=lag * 60)
        m = matched_table(clean, wall, watch, 600, "ending", gate)
        a = epoch_agreement(m)
        rows.append({"lag_min": lag, "n": a["n"], "pearson_r_ln": a.get("pearson_r_ln", np.nan), "mae_ms": a.get("mae_ms", np.nan)})
    return pd.DataFrame(rows)


def window_length_scan(
    clean: CleanResult,
    wall_end_s: np.ndarray,
    watch: WatchHRV,
    lengths_min=(1, 2, 3, 5, 10),
    conventions=CONVENTIONS,
    gate: Gate = GATES["default"],
) -> pd.DataFrame:
    """Agreement when the strap window behind each watch reading is shorter.
    If the watch's reading were a short sample, short strap windows would agree
    better. The minimum pair count scales with the window length."""
    rows = []
    for L, c in itertools.product(lengths_min, conventions):
        g = Gate(gate.name, gate.min_coverage, gate.max_corrected, min_pairs=max(10, int(L * 60 / 1.25 * 0.5)))
        m = matched_table(clean, wall_end_s, watch, L * 60, c, g)
        a = epoch_agreement(m)
        v = m[m["valid"]]
        rows.append(
            {
                "window_min": L,
                "convention": c,
                "n": a["n"],
                "matched_diff_ms": night_level(m, watch)["matched_diff_ms"],
                "pearson_r_ln": a.get("pearson_r_ln", np.nan),
                "ccc": a.get("ccc", np.nan),
                "mae_ms": a.get("mae_ms", np.nan),
                "sd_strap": float(v["rmssd"].std()) if len(v) > 1 else np.nan,
                "sd_watch": float(v["watch_rmssd"].std()) if len(v) > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def watch_noise(matched: pd.DataFrame, n_readings: int) -> dict:
    """Extra scatter of the watch over the strap across matched windows, read
    as per-reading measurement noise, and what it implies for the nightly mean
    (noise / sqrt(readings))."""
    v = matched[matched["valid"]]
    if len(v) < 3:
        return {"excess_sd_ms": np.nan, "nightly_mean_se_ms": np.nan}
    var_w, var_s = v["watch_rmssd"].var(), v["rmssd"].var()
    excess = float(np.sqrt(max(var_w - var_s, 0.0)))
    return {
        "sd_watch_ms": float(np.sqrt(var_w)),
        "sd_strap_ms": float(np.sqrt(var_s)),
        "excess_sd_ms": excess,
        "nightly_mean_se_ms": excess / np.sqrt(n_readings),
    }
