import numpy as np
import pytest

from hrvcompare.timeline import HYPOTHESES, wall_end_times


@pytest.mark.parametrize("hyp", list(HYPOTHESES))
def test_wall_clock_is_monotonic_and_starts_at_session_start(recording, cleaned, hyp):
    w = wall_end_times(recording, cleaned, hyp)
    assert np.all(np.diff(w) > 0)
    assert w[0] - recording.start_s == pytest.approx(cleaned.beats["rr_ms"].iloc[0] / 1000, abs=2)


def test_hypothesis_end_points(recording, cleaned):
    session_end = recording.start_s + recording.duration_s
    assert wall_end_times(recording, cleaned, "H1")[-1] == pytest.approx(recording.start_s + recording.rr_total_s)
    assert wall_end_times(recording, cleaned, "H2")[-1] == pytest.approx(session_end)
    assert wall_end_times(recording, cleaned, "H3")[-1] == pytest.approx(session_end)


def test_lag_shifts_everything(recording, cleaned):
    a = wall_end_times(recording, cleaned, "H1")
    b = wall_end_times(recording, cleaned, "H1", lag_s=120)
    assert np.allclose(b - a, 120)
