# hrvcompare — chest strap vs smartwatch overnight HRV & resting HR

Cleans a chest strap's raw R-R interval (RRI) export, recovers the watch's overnight HRV readings from an app
screenshot, aligns the two, and reports how well they agree. Built for one night (2026-04-17) from a chest strap
plus a COROS watch, and reusable for further nights.

**Results for the first night: [`reports/report.pdf`](reports/report.pdf)** (LaTeX-typeset; the same report is also in
[`reports/report.md`](reports/report.md))

| | |
|---|---|
| Night-level HRV, same windows | watch − strap **+1.7 ms** with the strap clock as recorded (H1), +1.6 ms (H2), −8.0 ms (H3): within 8 ms (14 %) however the clock is aligned |
| Single 10-min readings | watch/strap ×1.01, 95 % limits ×0.64–×1.60: individual watch readings are noisy |
| Strap resting HR | lowest 5-min mean 45.2 bpm · lowest 30-min mean 46.6 bpm · mean sleeping HR 51.2 bpm |
| Strap data quality | 37 % of the recording masked as contact noise; 3 % of usable beats corrected |

## Quick start

```bash
pip install -e .    # Python >= 3.10; use .[dev] for pytest + NeuroKit2 cross-checks
hrvcompare run \
  --rri data/raw/strap_rri_20260417.csv \
  --watch-png data/raw/coros_hrv_20260417.png \
  --sleep 02:29 09:37 \
  --watch-avg 48 \
  --watch-range 56 68 \
  --out reports
```

* `--sleep`: the sleep onset and wake times printed under the watch chart.
* `--watch-avg`: the app's "Last Night Avg", used to check the digitization.
* `--watch-range`: the app's "Normal Range" (optional).
* `--watch-rhr <bpm>`: the watch's resting HR (optional). It adds the RHR comparison.

The run writes `reports/report.pdf` (typeset with LaTeX from `reports/report.tex`), the same report as `reports/report.md`,
`reports/figures/*.png|pdf`, `reports/tables/*.csv` (including the full cleaned RR series) and `reports/summary.json`, which holds
the headline numbers for combining nights. The PDF needs a TeX Live installation (on Debian/Ubuntu:
`apt-get install texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended lmodern latexmk`). Without one, or
with `--no-pdf`, the run still writes `report.tex`. `python -m hrvcompare run …` works too. The command-line options also let
you change the clock hypothesis used for figures (`--hypothesis H1|H2|H3`), the window
quality gate (`--gate strict|default|lenient`), the strap window behind each watch reading (`--window-min`,
`--convention ending|centred`) and the physiological RR range (`--rr-min/--rr-max`, default 400–2000 ms for sleep).

## Method in brief

**1 · Clean the RRI without breaking the series** (`artifacts.py`). Deleting outliers makes successive differences
span beats that were never adjacent, which inflates RMSSD, and it shifts every later beat time. Instead:

* beats are classified with **Lipponen & Tarvainen (2019)** (the Kubios automatic method) as ectopic / missed / extra /
  long-short. A fixed 30 % deviation-from-median safety net, plus a 400–2000 ms range check, stops dense noise from
  inflating the adaptive thresholds;
* **noise bursts** (≥3 flagged beats in any 10) are **masked**, not interpolated. They keep their duration on the time axis;
* **isolated artifacts are corrected by type with time-preserving operations**: merge extra beats, split missed beats,
  move a misplaced R-peak to the midpoint of its two intervals. Total elapsed time is unchanged to the millisecond;
* RMSSD uses only successive differences between two **usable adjacent** beats. LF/HF use **Lomb–Scargle** on the
  unevenly spaced usable beats, so nothing is interpolated across gaps;
* a window counts only if ≥80 % of its time is usable NN and ≤5 % of its beats were corrected.

Validated on synthetic RR with known artifacts: 100 % of injected ectopic, missed and extra beats found, bursts fully masked,
and RMSSD within ~2 % of truth (vs ~4× inflated if uncleaned).

**2 · Read the watch chart** (`watch.py`). The gridlines calibrate ms per pixel, and the axis labels give clock time. A free-knot
polyline fit recovers the actual readings (the chart's corners), which COROS takes about every 10 minutes but
irregularly. The mean of the digitized readings is 47.98 ms, against the app's 48 ms.

**3 · Align the devices** (`timeline.py`). The strap has no clock of its own: its timestamps are `start + cumsum(RR)`,
and the session header is 42.2 min longer than the summed intervals. Three hypotheses for where that time went are
carried through every result: H1, the strap clock as recorded (the stream ended early); H2, the gap is spread uniformly; H3, the
gap sits inside the noise episodes. Most of the evidence points to time lost during the night: the file never holds an interval
longer than 3.03 s, so longer contact losses vanish, and the final noise burst coincides with waking only if the clock is
stretched. One night can't pin down how much was lost, though. The figures use H1, because it needs no assumption, and the
headline numbers are given under all three. No offset is fitted, because fitting one on HRV and then scoring HRV agreement
would be circular.

**4 · Compare** (`compare.py`). The main endpoint is the **night-level average over coverage-matched windows**,
which depends far less on alignment than window-by-window matching. Epoch-level results use Bland–Altman on the ratio (log) scale, Lin's CCC and Pearson/Spearman correlation.
Sensitivity is reported across alignment × gate × window convention × window length, along with a lag scan and a window-length scan.
Resting HR is reported under several standard definitions, since COROS doesn't publish its own.

## Layout

```
data/raw/                 input files (strap CSV, watch screenshot)
src/hrvcompare/
  io.py                   parse the strap export (metadata + beat table)
  artifacts.py            detection, burst masking, time-preserving correction
  metrics.py              time/frequency-domain HRV, window gating, resting HR
  timeline.py             clock-deficit hypotheses -> wall-clock beat times
  watch.py                screenshot digitizer (gridline calibration, vertex fit)
  compare.py              matched windows, agreement statistics, sensitivity
  synthetic.py            synthetic RR + artifact injection for validation
  doc.py                  document model rendered to Markdown and LaTeX, PDF build
  pipeline.py, report.py, plots.py, cli.py
tests/                    pytest suite (synthetic ground truth, NeuroKit2 cross-check, end-to-end)
reports/                  generated report (PDF, LaTeX source, Markdown), figures, tables
```

## Tests

```bash
pip install -e .[dev]
pytest
```
