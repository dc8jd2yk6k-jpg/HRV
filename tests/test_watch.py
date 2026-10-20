import numpy as np
import pytest

from hrvcompare.io import clock_to_seconds
from hrvcompare.watch import digitize_hrv_chart


@pytest.fixture(scope="module")
def watch(watch_png):
    return digitize_hrv_chart(watch_png, clock_to_seconds("02:29"), clock_to_seconds("09:37"), reported_avg=48)


def test_gridlines_detected(watch):
    c = watch.calibration
    assert c["source"] == "detected"
    assert c["px_per_ms"] == pytest.approx(1.52, abs=0.02)
    assert c["zero_row"] == pytest.approx(635.5, abs=1.0)


def test_vertices_reproduce_the_drawn_line(watch):
    c = watch.calibration
    assert 36 <= c["n_points"] <= 46
    assert c["rmse_px"] < 1.0
    # COROS measures about every 10 minutes.
    assert 9.0 < c["spacing_median_min"] < 11.0


def test_digitized_mean_matches_app_average(watch):
    assert abs(watch.digitized_avg - watch.reported_avg) < 1.5


def test_values_within_chart_range(watch):
    assert np.all((watch.point_ms > 5) & (watch.point_ms < 150))
