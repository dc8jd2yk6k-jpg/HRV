import numpy as np
import pytest

from hrvcompare.compare import agreement, lin_ccc


def test_perfect_agreement():
    x = np.array([30.0, 45, 60, 80, 55])
    a = agreement(x, x)
    assert a["bias_ms"] == 0 and a["ratio"] == pytest.approx(1)
    assert a["ccc"] == pytest.approx(1) and a["pearson_r_ln"] == pytest.approx(1)


def test_proportional_bias_shows_in_ratio():
    x = np.array([30.0, 45, 60, 80, 55])
    a = agreement(x, 0.8 * x)
    assert a["ratio"] == pytest.approx(0.8)
    assert a["ratio_loa_low"] == pytest.approx(0.8) and a["ratio_loa_high"] == pytest.approx(0.8)
    assert a["ccc"] < 1


def test_lin_ccc_penalises_offset():
    x = np.arange(10.0)
    assert lin_ccc(x, x + 5) < lin_ccc(x, x + 1) < 1


def test_drops_missing_pairs():
    a = agreement(np.array([30, np.nan, 50.0, 60]), np.array([32, 40, np.nan, 58.0]))
    assert a["n"] == 2
