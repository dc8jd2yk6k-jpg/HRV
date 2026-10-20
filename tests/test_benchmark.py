import numpy as np
import pytest

from hrvcompare.benchmark import (
    _rmssd_in_windows,
    _windows,
    corrupt,
    estimate,
    malik_flags,
    mitdb_ectopy_benchmark,
    noise_library,
    parametric_episode,
)
from hrvcompare.synthetic import synthetic_rr


def test_malik_flags_rejects_jumps_and_resets_after_a_step_change():
    rr = np.array([1000, 1010, 1500, 990, 1000] + [700] * 10, dtype=float)
    f = malik_flags(rr, reset_after=5)
    assert f[2] and not f[3]
    # A genuine step to 700 ms is rejected 5 times, then accepted as the new reference.
    assert f[5:10].all() and not f[10:].any()


@pytest.mark.parametrize("noise", ["empirical", "parametric"])
def test_corruption_preserves_elapsed_time(noise):
    base = synthetic_rr(n=900, seed=1)
    lib = [parametric_episode(d, np.random.default_rng(0)) for d in (5, 12, 30)]
    bad = corrupt(base, np.random.default_rng(3), contamination=0.2, iso_rate=0.01, noise=noise, library=lib)
    assert bad.sum() == pytest.approx(base.sum(), abs=1e-6)
    assert len(bad) != len(base)


def test_estimate_returns_one_value_per_window():
    base = synthetic_rr(n=1300, seed=2)
    starts = _windows(base.sum() / 1000)
    for method in ("none", "malik_delete", "malik_adjacent", "malik_interp", "proposed"):
        assert estimate(method, base, starts).shape == starts.shape


def test_methods_leave_clean_data_almost_unchanged():
    base = synthetic_rr(n=1300, seed=4)
    starts = _windows(base.sum() / 1000)
    truth = _rmssd_in_windows(np.cumsum(base) / 1000, base, np.ones(len(base) - 1, bool), starts)
    for method in ("malik_adjacent", "proposed"):
        est = estimate(method, base, starts)
        assert np.nanmax(np.abs(est / truth - 1)) < 0.05


def test_proposed_beats_naive_deletion_under_noise():
    base = synthetic_rr(n=1300, seed=5)
    lib = [parametric_episode(d, np.random.default_rng(1)) for d in (6, 10, 20)]
    bad = corrupt(base, np.random.default_rng(7), contamination=0.1, iso_rate=0.01, noise="parametric", library=lib)
    starts = _windows(base.sum() / 1000)
    truth = _rmssd_in_windows(np.cumsum(base) / 1000, base, np.ones(len(base) - 1, bool), starts)
    err = {m: np.nanmedian(np.abs(estimate(m, bad, starts) / truth - 1)) for m in ("malik_delete", "proposed_lenient")}
    assert err["proposed_lenient"] < err["malik_delete"]


def test_noise_library_finds_episodes_in_the_recording(recording):
    lib = noise_library(recording.rr_ms)
    assert len(lib) > 100
    assert all(len(e) >= 5 for e in lib)


def test_mitdb_record_100_ectopy(tmp_path):
    from pathlib import Path

    local = Path(__file__).resolve().parents[1] / "data" / "external" / "mitdb_sample"
    res = mitdb_ectopy_benchmark(["100"], local_dir=local).iloc[0]
    assert res["ectopic_beats"] == 34
    assert res["sensitivity"] == 1.0
    assert abs(res["rmssd_proposed"] / res["rmssd_reference"] - 1) < 0.05
