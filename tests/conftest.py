from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


@pytest.fixture(scope="session")
def strap_csv():
    return RAW / "strap_rri_20260417.csv"


@pytest.fixture(scope="session")
def watch_png():
    return RAW / "coros_hrv_20260417.png"


@pytest.fixture(scope="session")
def recording(strap_csv):
    from hrvcompare.io import read_strap_csv

    return read_strap_csv(strap_csv)


@pytest.fixture(scope="session")
def cleaned(recording):
    from hrvcompare.artifacts import clean_rr

    return clean_rr(recording.rr_ms)
