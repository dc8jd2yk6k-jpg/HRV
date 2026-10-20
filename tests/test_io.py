import numpy as np

from hrvcompare.io import clock_to_seconds, seconds_to_clock


def test_reads_metadata_and_beats(recording):
    assert recording.n == 23816
    assert recording.start.strftime("%Y-%m-%d %H:%M:%S") == "2026-04-17 02:19:50"
    assert recording.duration_s == 7 * 3600 + 21 * 60 + 3
    assert recording.metadata["device name"] == "BRT30969"
    assert recording.rr_ms[:3].tolist() == [918, 940, 938]


def test_device_time_is_cumulative_rr(recording):
    # The strap has no independent clock: Time == start + cumsum(RRI).
    assert np.abs(recording.clock_residual_ms()).max() < 1e-3


def test_clock_deficit(recording):
    # Header duration exceeds the summed intervals by ~42 min.
    assert abs(recording.clock_deficit_s / 60 - 42.2) < 0.1


def test_clock_parsing():
    assert clock_to_seconds("02:29") == 2 * 3600 + 29 * 60
    assert clock_to_seconds("09:37:30") == 9 * 3600 + 37 * 60 + 30
    assert clock_to_seconds("00:30", ref_s=clock_to_seconds("23:00")) == 86400 + 1800
    assert seconds_to_clock(8390.9) == "02:19:50"
