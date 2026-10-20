import numpy as np

from hrvcompare.digitizer_validation import validate


def test_digitizer_recovers_known_readings(watch_png):
    df, err = validate(watch_png, n=5, seed=1)
    assert df["recall"].mean() > 0.9
    assert np.median(np.abs(err)) < 0.5  # ms
    assert df["mean_error_ms"].abs().max() < 3.0
