"""HRV metrics on a cleaned interval table, with per-window quality gating.

RMSSD and pNN50 use only successive differences between two *adjacent valid*
intervals, so a masked stretch never produces a difference that spans a gap.
Frequency-domain power uses the Lomb-Scargle periodogram on the unevenly
sampled valid NN series, so masked gaps don't need to be interpolated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import lombscargle

from .artifacts import CORRECTED, MASKED

LF_BAND = (0.04, 0.15)
HF_BAND = (0.15, 0.40)


@dataclass(frozen=True)
class Gate:
    name: str
    min_coverage: float  # valid-NN time / window length
    max_corrected: float  # corrected beats / valid beats
    min_pairs: int = 30


GATES = {
    "strict": Gate("strict", 0.90, 0.02),
    "default": Gate("default", 0.80, 0.05),
    "lenient": Gate("lenient", 0.60, 0.10),
}


def time_domain(rr_ms: np.ndarray, valid: np.ndarray) -> dict:
    rr = np.asarray(rr_ms, dtype=float)
    v = np.asarray(valid, dtype=bool)
    nn = rr[v]
    pair = v[1:] & v[:-1]
    d = np.diff(rr)[pair]
    out = {
        "n_nn": int(v.sum()),
        "n_pairs": int(pair.sum()),
        "mean_nn": float(nn.mean()) if len(nn) else np.nan,
        "sdnn": float(nn.std(ddof=1)) if len(nn) > 1 else np.nan,
        "rmssd": float(np.sqrt(np.mean(d**2))) if len(d) else np.nan,
        "pnn50": float(np.mean(np.abs(d) > 50) * 100) if len(d) else np.nan,
    }
    out["hr"] = 60000.0 / out["mean_nn"] if len(nn) else np.nan
    return out


def lomb_scargle_bands(t_s: np.ndarray, nn_ms: np.ndarray, fmax: float = 0.5) -> dict:
    """LF/HF power (ms^2) from the Lomb-Scargle PSD of an unevenly sampled NN
    series. PSD scaling: P * 2T/N, so a sinusoid of amplitude A integrates to
    A^2/2 and white noise of variance s^2 to s^2 over [0, Nyquist]."""
    t = np.asarray(t_s, dtype=float)
    y = np.asarray(nn_ms, dtype=float)
    nan = {"lf": np.nan, "hf": np.nan, "lf_hf": np.nan, "lf_nu": np.nan, "hf_nu": np.nan}
    if len(t) < 30:
        return nan
    span = t[-1] - t[0]
    if span < 1 / LF_BAND[0]:
        return nan
    y = y - np.polyval(np.polyfit(t - t[0], y, 1), t - t[0])
    f = np.arange(1 / span, fmax, 1 / (4 * span))
    p = lombscargle(t, y, 2 * np.pi * f)
    psd = p * 2 * span / len(t)

    def band(lo, hi):
        m = (f >= lo) & (f < hi)
        return float(np.trapezoid(psd[m], f[m])) if m.sum() > 1 else np.nan

    lf, hf = band(*LF_BAND), band(*HF_BAND)
    return {"lf": lf, "hf": hf, "lf_hf": lf / hf if hf else np.nan, "lf_nu": 100 * lf / (lf + hf), "hf_nu": 100 * hf / (lf + hf)}


def window_metrics(
    beats: pd.DataFrame,
    wall_end_s: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    gate: Gate,
    frequency: bool = False,
) -> pd.DataFrame:
    """Metrics for each window [start, end). Intervals are assigned by their
    end time on the wall clock."""
    rr = beats["rr_ms"].to_numpy()
    status = beats["status"].to_numpy()
    valid = status != MASKED
    corrected = status == CORRECTED
    lo = np.searchsorted(wall_end_s, starts, side="left")
    hi = np.searchsorted(wall_end_s, ends, side="left")
    rows = []
    for s, e, a, b in zip(starts, ends, lo, hi):
        v = valid[a:b]
        row = {"start_s": s, "end_s": e}
        row.update(time_domain(rr[a:b], v))
        length = e - s
        row["data_time_s"] = rr[a:b].sum() / 1000
        row["nn_time_s"] = rr[a:b][v].sum() / 1000
        row["coverage"] = row["nn_time_s"] / length
        row["corrected_frac"] = corrected[a:b].sum() / max(v.sum(), 1)
        row["valid"] = bool(
            row["coverage"] >= gate.min_coverage
            and row["corrected_frac"] <= gate.max_corrected
            and row["n_pairs"] >= gate.min_pairs
        )
        if frequency:
            row.update(lomb_scargle_bands(wall_end_s[a:b][v], rr[a:b][v]))
        rows.append(row)
    df = pd.DataFrame(rows)
    df["gate"] = gate.name
    return df


def regular_windows(t0: float, t1: float, length_s: float) -> tuple[np.ndarray, np.ndarray]:
    starts = np.arange(t0, t1 - length_s + 1e-9, length_s)
    return starts, starts + length_s


def resting_hr(
    beats: pd.DataFrame,
    wall_end_s: np.ndarray,
    t0: float,
    t1: float,
    min_coverage: float = 0.8,
) -> tuple[dict, pd.DataFrame]:
    """Several strap RHR definitions over [t0, t1) from per-minute heart rate.
    Each minute's HR is 60 x valid beats / valid NN time; minutes with less than
    *min_coverage* valid time are left out."""
    starts, ends = regular_windows(t0, t1, 60.0)
    rr = beats["rr_ms"].to_numpy()
    valid = (beats["status"] != MASKED).to_numpy()
    lo = np.searchsorted(wall_end_s, starts)
    hi = np.searchsorted(wall_end_s, ends)
    nn_time = np.array([rr[a:b][valid[a:b]].sum() / 1000 for a, b in zip(lo, hi)])
    n_nn = np.array([valid[a:b].sum() for a, b in zip(lo, hi)])
    ok = nn_time >= min_coverage * 60
    hr = np.where(ok, 60 * n_nn / np.where(nn_time > 0, nn_time, np.nan), np.nan)
    per_min = pd.DataFrame({"start_s": starts, "hr": hr, "coverage": nn_time / 60})
    s = pd.Series(hr)
    roll5 = s.rolling(5, min_periods=4).mean()
    roll30 = s.rolling(30, min_periods=24).mean()
    per_min["hr_5min"] = roll5.to_numpy()
    per_min["hr_30min"] = roll30.to_numpy()

    def at_min(series):
        if series.notna().any():
            i = int(series.idxmin())
            return float(series[i]), float(starts[i] + 60)  # end of the minute closing the window
        return np.nan, np.nan

    low5, low5_t = at_min(roll5)
    low30, low30_t = at_min(roll30)
    out = {
        "lowest_5min_mean_hr": low5,
        "lowest_5min_at_s": low5_t,
        "lowest_30min_mean_hr": low30,
        "lowest_30min_at_s": low30_t,
        "mean_sleep_hr": float(60 * n_nn[ok].sum() / nn_time[ok].sum()) if ok.any() else np.nan,
        "p5_1min_hr": float(np.nanpercentile(hr, 5)) if ok.any() else np.nan,
        "lowest_1min_hr": float(np.nanmin(hr)) if ok.any() else np.nan,
        "minutes_valid": int(ok.sum()),
        "minutes_total": int(len(ok)),
    }
    return out, per_min
