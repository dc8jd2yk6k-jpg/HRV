import numpy as np
import pytest

from hrvcompare.artifacts import CORRECTED, MASKED, OK, clean_rr, detect_artifacts
from hrvcompare.metrics import time_domain
from hrvcompare.synthetic import inject_artifacts, synthetic_rr


def touched(clean, n):
    t = np.zeros(n, dtype=bool)
    b = clean.beats
    for a, z, st in zip(b["raw_first"], b["raw_last"], b["status"]):
        if st != OK:
            t[a : z + 1] = True
    return t


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_detects_and_corrects_isolated_artifacts(seed):
    rr = synthetic_rr(seed=seed)
    truth_rmssd = time_domain(rr, np.ones(len(rr), bool))["rmssd"]
    bad, truth = inject_artifacts(rr, rate=0.02, seed=seed + 10)
    clean = clean_rr(bad)
    hit = touched(clean, len(bad))
    sens = np.mean([hit[a : z + 1].any() for _, a, z in truth])
    assert sens >= 0.9
    rmssd = time_domain(clean.beats["rr_ms"].to_numpy(), clean.valid)["rmssd"]
    assert abs(rmssd / truth_rmssd - 1) < 0.05
    # Without cleaning RMSSD is wildly inflated.
    assert time_domain(bad, np.ones(len(bad), bool))["rmssd"] > 2 * truth_rmssd


def test_corrections_preserve_elapsed_time():
    bad, _ = inject_artifacts(synthetic_rr(seed=3), rate=0.03, bursts=3, seed=4)
    clean = clean_rr(bad)
    assert clean.beats["rr_ms"].sum() == pytest.approx(bad.sum(), abs=1e-6)
    assert clean.beats["t_end_s"].iloc[-1] == pytest.approx(bad.sum() / 1000)


def test_ectopic_beat_is_restored_to_normal_rhythm():
    rr = np.full(400, 1000.0) + np.random.default_rng(0).normal(0, 5, 400)
    rr[200], rr[201] = 650.0, 1350.0  # premature beat + compensatory pause
    clean = clean_rr(rr)
    b = clean.beats
    fixed = b[(b["raw_first"] >= 200) & (b["raw_last"] <= 201)]
    assert (fixed["status"] == CORRECTED).all()
    assert np.allclose(fixed["rr_ms"], 1000, atol=15)


def test_missed_and_extra_beats_are_split_and_merged():
    rr = np.full(400, 1000.0) + np.random.default_rng(1).normal(0, 5, 400)
    rr = np.concatenate([rr[:100], [2000.0], rr[100:200], [400.0, 600.0], rr[200:]])
    clean = clean_rr(rr)
    b = clean.beats
    missed = b[b["raw_first"] == 100]
    assert len(missed) == 2 and (missed["kind"] == "missed").all()
    extra = b[(b["raw_first"] == 201) & (b["raw_last"] == 202)]
    assert len(extra) == 1 and extra["rr_ms"].iloc[0] == pytest.approx(1000)


def test_noise_bursts_are_masked_not_interpolated():
    rr = synthetic_rr(seed=5)
    bad, truth = inject_artifacts(rr, rate=0.0, bursts=4, seed=6)
    clean = clean_rr(bad)
    for kind, a, z in truth:
        assert kind == "burst"
        assert clean.raw_masked[a : z + 1].mean() > 0.9
    # Masked beats are excluded from the NN series entirely.
    assert (clean.beats.loc[~clean.valid, "status"] == MASKED).all()


def test_clean_series_is_mostly_untouched():
    rr = synthetic_rr(seed=7)
    clean = clean_rr(rr)
    s = clean.summary()
    assert s["masked_time_frac"] < 0.03
    assert s["corrected_frac_of_valid"] < 0.03


def test_agrees_with_neurokit_kubios():
    nk = pytest.importorskip("neurokit2")
    bad, _ = inject_artifacts(synthetic_rr(seed=8), rate=0.02, seed=9)
    peaks = np.concatenate([[0], np.cumsum(bad)]).round().astype(int)
    info, _ = nk.signal_fixpeaks(peaks, sampling_rate=1000, method="kubios", iterative=False)
    # NeuroKit index i is the interval ending at peak i, i.e. our interval i-1.
    theirs = {i - 1 for k in ("ectopic", "missed", "extra", "longshort") for i in info[k]}
    ours = np.flatnonzero(detect_artifacts(bad).labels != OK)
    near = [any(abs(i - j) <= 1 for j in ours) for i in theirs]
    assert np.mean(near) >= 0.9


def test_real_recording_cleaning(recording, cleaned):
    s = cleaned.summary()
    assert s["total_time_s"] == pytest.approx(recording.rr_total_s)
    # Heavily contaminated recording: a large share is masked, few beats corrected.
    assert 0.2 < s["masked_time_frac"] < 0.5
    assert s["corrected_frac_of_valid"] < 0.05
    nn = cleaned.beats.loc[cleaned.valid, "rr_ms"]
    assert nn.between(300, 2000).all()
