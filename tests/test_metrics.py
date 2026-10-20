import numpy as np
import pandas as pd
import pytest

from hrvcompare.metrics import GATES, lomb_scargle_bands, resting_hr, time_domain, window_metrics


def test_rmssd_skips_pairs_touching_masked_beats():
    rr = np.array([1000, 1010, 1500, 990, 1000, 1020], dtype=float)
    valid = np.array([True, True, False, True, True, True])
    out = time_domain(rr, valid)
    d = np.array([10, 10, 20])  # 1000->1010, 990->1000, 1000->1020
    assert out["n_pairs"] == 3
    assert out["rmssd"] == pytest.approx(np.sqrt(np.mean(d**2)))
    assert out["mean_nn"] == pytest.approx(np.mean([1000, 1010, 990, 1000, 1020]))


def test_lomb_scargle_power_calibration():
    rng = np.random.default_rng(0)
    rr = 1000 + 30 * np.sin(2 * np.pi * 0.25 * np.arange(600)) + rng.normal(0, 1, 600)
    t = np.cumsum(rr) / 1000
    rr = 1000 + 30 * np.sin(2 * np.pi * 0.25 * t) + rng.normal(0, 1, 600)
    keep = rng.random(600) > 0.1  # uneven sampling with gaps
    bands = lomb_scargle_bands(t[keep], rr[keep])
    assert bands["hf"] == pytest.approx(30**2 / 2, rel=0.15)
    assert bands["lf"] < 0.05 * bands["hf"]


def _beats(rr, status=None):
    rr = np.asarray(rr, dtype=float)
    return pd.DataFrame({"rr_ms": rr, "status": status if status is not None else ["ok"] * len(rr), "t_end_s": np.cumsum(rr) / 1000})


def test_window_gating_on_coverage():
    rr = np.full(1200, 1000.0)
    status = np.array(["ok"] * 1200, dtype=object)
    status[600:900] = "masked"
    b = _beats(rr, status)
    wall = b["t_end_s"].to_numpy()
    starts = np.array([0.0, 600.0, 900.0])
    w = window_metrics(b, wall, starts, starts + 300, GATES["default"])
    assert w["valid"].tolist() == [True, False, True]
    assert w.loc[1, "coverage"] == pytest.approx(0.0, abs=0.01)


def test_resting_hr_definitions():
    rr = np.concatenate([np.full(600, 1000.0), np.full(600, 1200.0), np.full(600, 1000.0)])
    b = _beats(rr)
    wall = b["t_end_s"].to_numpy()
    out, per_min = resting_hr(b, wall, 0, wall[-1])
    assert out["lowest_5min_mean_hr"] == pytest.approx(50, abs=0.5)
    assert out["lowest_30min_mean_hr"] < 60
    assert out["mean_sleep_hr"] == pytest.approx(60 * 1800 / (rr.sum() / 1000), rel=1e-3)
