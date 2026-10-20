import json
import shutil

import pytest

from hrvcompare.cli import main


@pytest.fixture(scope="module")
def outdir(tmp_path_factory, strap_csv, watch_png):
    out = tmp_path_factory.mktemp("report")
    rc = main([
        "run", "--rri", str(strap_csv), "--watch-png", str(watch_png), "--sleep", "02:29", "09:37",
        "--watch-avg", "48", "--watch-range", "56", "68", "--watch-rhr", "50", "--out", str(out),
    ])
    assert rc == 0
    return out


def test_report_and_artifacts_written(outdir):
    report = (outdir / "report.md").read_text()
    assert "## Key results" in report and "Watch RHR" in report
    for fig in ("01_tachogram", "03_rmssd_overnight", "04_agreement", "05_sensitivity", "06_heart_rate"):
        assert (outdir / "figures" / f"{fig}.png").stat().st_size > 10_000
    assert (outdir / "tables" / "matched_windows.csv").exists()
    tex = (outdir / "report.tex").read_text()
    assert r"\begin{document}" in tex and "figures/03_rmssd_overnight.pdf" in tex
    if shutil.which("pdflatex"):
        assert (outdir / "report.pdf").read_bytes()[:4] == b"%PDF"


def test_summary_numbers_are_plausible(outdir):
    s = json.loads((outdir / "summary.json").read_text())
    assert abs(s["watch"]["digitized_avg"] - 48) < 1.5
    night = s["night_level"]
    assert night["n_valid"] >= 5
    assert 30 < night["strap_avg_valid"] < 100
    assert 40 < s["resting_hr"]["lowest_5min_mean_hr"] < 60
    assert s["resting_hr"]["lowest_5min_mean_hr"] <= s["resting_hr"]["mean_sleep_hr"]
