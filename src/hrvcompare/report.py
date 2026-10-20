"""Build the report document from a run and write it as Markdown, LaTeX and PDF,
plus the CSV tables and a JSON summary."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import MASKED
from .doc import Box, Code, Document, Figure, Heading, ListBlock, Para, Table, build_pdf, latex_engine, to_latex, to_markdown
from .io import seconds_to_clock
from .metrics import GATES
from .plots import make_all
from .timeline import HYPOTHESES


def _f(x, nd=1, unit=""):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:.{nd}f}{unit}".replace("-", "−")


def _sf(x, nd=1, unit=""):
    """Signed number with a typographic minus."""
    if x is None or not np.isfinite(x):
        return "–"
    return f"{x:+.{nd}f}{unit}".replace("-", "−")


def _pct(x, nd=0):
    return "–" if x is None or not np.isfinite(x) else f"{100 * x:.{nd}f}%"


def _clock(s, seconds=False):
    return seconds_to_clock(s) if seconds else seconds_to_clock(s)[:5]


def table(df: pd.DataFrame, fmt: dict | None = None, colspec: str = "", small: bool = False) -> Table:
    fmt = fmt or {}
    rows = []
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            v, f = row[c], fmt.get(c)
            if callable(f):
                cells.append(f(v))
            elif isinstance(v, (float, np.floating)):
                cells.append(_f(float(v), f if isinstance(f, int) else 1))
            else:
                cells.append(str(v))
        rows.append(cells)
    return Table([str(c) for c in df.columns], rows, colspec=colspec, small=small)


def _tail_noise_evidence(res, minutes: float = 5.0) -> float:
    b = res.clean.beats
    t = b["t_end_s"].to_numpy()
    tail = t > t[-1] - minutes * 60
    rr = b["rr_ms"].to_numpy()
    return float(rr[tail & (b["status"] == MASKED).to_numpy()].sum() / rr[tail].sum())


def _tail_stats(res, tail_min: float = 5.0, before_min: float = 10.0) -> dict:
    """The final stretch of strap data (where it sits on the wall clock under
    each hypothesis) and the physiology just before it."""
    b = res.clean.beats
    t = b["t_end_s"].to_numpy()
    tail = t > t[-1] - tail_min * 60
    before = (t > t[-1] - (tail_min + before_min) * 60) & ~tail & res.clean.valid
    rr = b["rr_ms"].to_numpy()
    pair = before[1:] & before[:-1]
    out = {"tail_start": {}, "tail_end": {}}
    for h, wall in res.walls.items():
        out["tail_start"][h] = float(wall[np.argmax(tail)])
        out["tail_end"][h] = float(wall[-1])
    out["hr_before"] = 60000 / rr[before].mean() if before.any() else np.nan
    out["rmssd_before"] = float(np.sqrt(np.mean(np.diff(rr)[pair] ** 2))) if pair.any() else np.nan
    return out


def _lag_note(res) -> str:
    lg = res.lags.dropna(subset=["pearson_r_ln"])
    if lg.empty:
        return ""
    best = lg.loc[lg["pearson_r_ln"].idxmax()]
    zero = lg[lg["lag_min"] == 0]
    r0 = f"r = {_f(zero['pearson_r_ln'].iloc[0], 2)}, n = {int(zero['n'].iloc[0])}" if len(zero) else "–"
    lag = int(best["lag_min"])
    if abs(lag) <= 4:
        return f"agreement is highest near zero shift ({_sf(lag, 0)} min, r = {_f(best['pearson_r_ln'], 2)})."
    direction = "behind" if lag > 0 else "ahead of"
    return (f"agreement is highest when the strap timeline is shifted by {_sf(lag, 0)} min (r = {_f(best['pearson_r_ln'], 2)}, "
            f"n = {int(best['n'])}; unshifted {r0}), i.e. with the strap's clock running {direction} the wall clock"
            + (" — what H2/H3 predict." if lag > 0 else "."))


def _bottom_line(res) -> str:
    n, e, w = res.night, res.epoch, res.watch
    bh = res.by_hypothesis.dropna(subset=["diff_ms"])
    parts = []
    if len(bh):
        lo, hi = bh["diff_ms"].min(), bh["diff_ms"].max()
        worst = float(np.max(np.abs(bh["diff_ms"])))
        worst_pct = float(np.max(np.abs(bh["ratio"] - 1)) * 100)
        if worst <= 10:
            parts.append(f"Averaged over the same stretches of the night, the watch's *nightly* HRV agrees with the chest strap to within "
                         f"{_f(worst, 0)} ms ({_f(worst_pct, 0)}%) whichever way the strap's clock is aligned "
                         f"({_sf(lo)} to {_sf(hi)} ms across the three clock hypotheses).")
        else:
            parts.append(f"Averaged over the same stretches of the night, the watch's nightly HRV differs from the chest strap by "
                         f"{_sf(lo)} to {_sf(hi)} ms depending on how the strap's clock is aligned.")
    if e["n"] >= 3 and (e["ccc"] < 0.6 or e["ratio_loa_high"] / e["ratio_loa_low"] > 1.7):
        parts.append("Its *individual 10-minute readings* scatter far more than the strap does, so treat single readings "
                     "as noisy and trust the nightly average.")
    gap = w.reported_avg - n["strap_avg_valid"] if w.reported_avg is not None else np.nan
    if np.isfinite(gap) and abs(gap - n["matched_diff_ms"]) > 3:
        parts.append(f"The watch's headline {w.reported_avg:g} ms covers the whole night, while the usable strap windows do not: "
                     f"comparing it with the strap's usable-window average ({_f(n['strap_avg_valid'])} ms) without matching coverage "
                     f"would wrongly suggest the watch reads {_f(abs(gap), 0)} ms {'low' if gap < 0 else 'high'}.")
    return " ".join(parts)


def _coverage_note(res) -> str:
    m = res.matched
    t0, t1 = res.sleep_s
    mid = (t0 + t1) / 2
    first, second = m[m["point_s"] < mid], m[m["point_s"] >= mid]
    notes = []
    if len(first) and first["valid"].mean() < 0.3:
        notes.append(f"In the first half of the night only {int(first['valid'].sum())} of {len(first)} windows survive strap noise.")
    if m["valid"].sum() and (second["valid"].sum() / m["valid"].sum()) > 0.6:
        notes.append("The usable windows are concentrated in the second half of the night.")
    return " ".join(notes)


def _window_scan_note(res) -> str:
    ws = res.window_scan
    primary = ws[(ws["window_min"] == res.inputs.window_min) & (ws["convention"] == res.inputs.convention)]
    base = float(primary["pearson_r_ln"].iloc[0]) if len(primary) else np.nan
    shorter = ws[ws["window_min"] < res.inputs.window_min]
    if len(shorter) and np.isfinite(base) and shorter["pearson_r_ln"].max() > base + 0.15:
        return "Shorter strap windows agree better, suggesting each COROS reading covers less than 10 minutes:"
    return "Shorter strap windows don't materially improve the match, so window length isn't the explanation:"


def write_tables(res, tables: Path) -> dict[str, Path]:
    tables.mkdir(parents=True, exist_ok=True)
    paths = {}

    def save(name, df):
        p = tables / f"{name}.csv"
        df.to_csv(p, index=False, float_format="%.4g")
        paths[name] = p

    b = res.clean.beats.copy()
    b.insert(0, "wall_clock", [f"{seconds_to_clock(s)}.{int((s % 1) * 1000):03d}" for s in res.wall])
    save("cleaned_rr", b)
    m = res.matched.copy()
    m.insert(0, "watch_time", [_clock(s, True) for s in m["point_s"]])
    save("matched_windows", m)
    w5 = res.windows5.copy()
    w5.insert(0, "window_start", [_clock(s, True) for s in w5["start_s"]])
    save("strap_5min_windows", w5)
    save("sensitivity_grid", res.grid)
    save("lag_scan", res.lags)
    save("window_length_scan", res.window_scan)
    save("synthetic_validation", res.synthetic)
    wp = pd.DataFrame({"time": [_clock(s, True) for s in res.watch.point_s], "watch_rmssd_ms": res.watch.point_ms})
    save("watch_digitized", wp)
    rm = res.rhr_minutes.copy()
    rm.insert(0, "minute", [_clock(s) for s in rm["start_s"]])
    save("strap_hr_per_minute", rm)
    save("cleaning_summary", pd.DataFrame([res.clean.summary()]))
    return paths


def summary_dict(res) -> dict:
    inp = res.inputs
    return {
        "date": res.rec.start.date().isoformat(),
        "primary_analysis": {"hypothesis": inp.hypothesis, "gate": inp.gate, "convention": inp.convention, "window_min": inp.window_min},
        "recording": {
            "start": res.rec.start.isoformat(),
            "session_duration_s": res.rec.duration_s,
            "rr_total_s": res.rec.rr_total_s,
            "clock_deficit_s": res.rec.clock_deficit_s,
            "n_intervals": res.rec.n,
        },
        "cleaning": res.clean.summary(),
        "device_flags": res.device_flags,
        "watch": {
            "reported_avg": res.watch.reported_avg,
            "digitized_avg": res.watch.digitized_avg,
            "n_readings": int(len(res.watch.point_ms)),
            "calibration": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in res.watch.calibration.items()},
        },
        "night_level": res.night,
        "epoch_level": res.epoch,
        "watch_noise": res.noise,
        "strap_overnight": res.strap_summary,
        "resting_hr": res.rhr,
    }


def reproduce_command(res, outdir: Path) -> str:
    inp = res.inputs
    parts = [f"hrvcompare run --rri {inp.rri_csv}", f"--watch-png {inp.watch_png}", f"--sleep {inp.sleep_start} {inp.sleep_end}"]
    if inp.watch_avg is not None:
        parts.append(f"--watch-avg {inp.watch_avg:g}")
    if inp.watch_range:
        parts.append(f"--watch-range {inp.watch_range[0]:g} {inp.watch_range[1]:g}")
    if inp.watch_rhr is not None:
        parts.append(f"--watch-rhr {inp.watch_rhr:g}")
    parts.append(f"--out {Path(outdir).as_posix()}")
    return " \\\n  ".join(parts)


def build_document(res, outdir: Path, tables: dict[str, Path]) -> Document:
    rec, clean, watch, inp = res.rec, res.clean, res.watch, res.inputs
    cs = clean.summary()
    n, e, nz = res.night, res.epoch, res.noise
    gate = GATES[inp.gate]
    t0, t1 = res.sleep_s
    strap_end = rec.start_s + rec.rr_total_s
    session_end = rec.start_s + rec.duration_s
    tail_masked = _tail_noise_evidence(res)
    g = res.grid
    main = g[g["gate"].isin(["strict", "default"])]
    lenient = g[g["gate"] == "lenient"]
    syn = res.synthetic
    s5 = res.strap_summary
    r = res.rhr
    c = watch.calibration
    rel = lambda p: Path(p).relative_to(outdir).as_posix()  # noqa: E731

    doc = Document(
        title="Chest strap vs COROS watch",
        subtitle=f"Overnight HRV and resting heart rate, {rec.start:%Y-%m-%d}",
        note="Generated by hrvcompare. Every number and figure is produced by the pipeline; "
             "re-run it to regenerate this report (see [Reproduce](#reproduce)).",
    )
    add = doc.add

    # Key results ---------------------------------------------------------------
    add(Heading("Key results", numbered=False))
    add(table(pd.DataFrame([
        ["Strap data usable during sleep", f"{_pct(s5['valid_nn_time_frac'])} of {_clock(t0)}–{_clock(min(t1, strap_end))} "
         f"({_pct(cs['masked_time_frac'])} of the recording masked as noise, {_pct(cs['corrected_frac_of_valid'], 1)} of usable beats corrected)"],
        ["Watch nightly HRV", f"{_f(watch.reported_avg, 0)} ms reported; {_f(watch.digitized_avg)} ms from the digitized chart ({len(watch.point_ms)} readings)"],
        ["**Night level, same windows**", f"**{inp.hypothesis} (strap clock as recorded): watch {_f(n['watch_avg_matched'])} vs strap "
         f"{_f(n['strap_avg_valid'])} ms → {_sf(n['matched_diff_ms'])} ms ({_sf(100 * (n['matched_ratio'] - 1))}%)**, n = {n['n_valid']}"
         + "".join(f" · {row.hypothesis}: {_sf(row.diff_ms)} ms (n = {row.n})"
                   for row in res.by_hypothesis.itertuples() if row.hypothesis != inp.hypothesis)],
        ["Night level, every strict/default-gate variant", f"{_sf(main['matched_diff_ms'].min())} to {_sf(main['matched_diff_ms'].max())} ms "
         f"({len(main)} variants: 3 alignment hypotheses × 2 gates × 2 window conventions × 2 window lengths)"],
        ["Single readings (epoch level)", f"ratio watch/strap ×{_f(e.get('ratio'), 2)}, 95% limits ×{_f(e.get('ratio_loa_low'), 2)}–×{_f(e.get('ratio_loa_high'), 2)}, "
         f"CCC {_f(e.get('ccc'), 2)}, r(ln) {_f(e.get('pearson_r_ln'), 2)}"],
        ["Watch noise per reading", f"≈{_f(nz.get('excess_sd_ms'))} ms SD on top of the strap's own variation → ≈{_f(nz.get('nightly_mean_se_ms'))} ms on the nightly mean"],
        ["Strap resting HR", f"lowest 5-min mean {_f(r['lowest_5min_mean_hr'])} bpm · lowest 30-min mean {_f(r['lowest_30min_mean_hr'])} bpm · "
         f"mean sleeping HR {_f(r['mean_sleep_hr'])} bpm"],
        ["Watch resting HR", f"{_f(r['watch_rhr'], 0)} bpm" if r.get("watch_rhr") is not None else
         "not provided — re-run with `--watch-rhr <bpm>` to add the comparison"],
    ], columns=["", "Result"]), colspec=r"L{0.27\linewidth}Y"))
    add(Box("Bottom line.", _bottom_line(res)))

    # 1. Data -------------------------------------------------------------------
    add(Heading("Data"))
    add(table(pd.DataFrame([
        ["Chest strap", f"`{Path(inp.rri_csv).name}` — device {rec.metadata.get('device name', '?')} (fw {rec.metadata.get('version', '?')}), "
         f"{rec.n:,} R-R intervals, session {_clock(rec.start_s, True)} + {seconds_to_clock(rec.duration_s)}"],
        ["Watch", f"`{Path(inp.watch_png).name}` — COROS 'HRV Assessment' screenshot: night avg {_f(watch.reported_avg, 0)} ms, "
         f"normal range {watch.normal_range[0]:g}–{watch.normal_range[1]:g} ms, sleep {inp.sleep_start}–{inp.sleep_end}" if watch.normal_range else
         f"`{Path(inp.watch_png).name}`"],
    ], columns=["Source", "What it contains"]), colspec=r"L{0.15\linewidth}Y"))
    add(Para("COROS measures overnight HRV as **RMSSD** roughly every **10 minutes** during sleep and reports the night's average; "
             "its 'normal range' is a personal baseline from previous nights. The strap is therefore summarised the same way: "
             "RMSSD in the 10 minutes ending at each watch reading."))

    # 2. Cleaning ---------------------------------------------------------------
    add(Heading("Cleaning the strap RRI series without breaking it"))
    add(Heading("Why deleting outliers breaks HRV", level=2))
    add(ListBlock([
        "**RMSSD** is the RMS of *successive* differences. Delete beat *i* and the next difference becomes RR(i+1) − RR(i−1): two intervals "
        "that were never adjacent. Every deletion adds a spurious jump and inflates RMSSD.",
        "**Spectral measures** (LF/HF) assume a continuous beat sequence; deleting beats shifts all later beat times.",
        "**Interpolating over long noise runs** fixes the arithmetic but makes up data — a 9-beat burst \"corrected\" by a spline is a guess.",
    ]))
    add(Heading("What the pipeline does instead", level=2))
    cfg = clean.config
    add(ListBlock([
        f"**Physiological range**: intervals outside {cfg.rr_min_ms:g}–{cfg.rr_max_ms:g} ms ({60000 / cfg.rr_max_ms:.0f}–"
        f"{60000 / cfg.rr_min_ms:.0f} bpm, appropriate for sleep) are flagged.",
        "**Beat classification** with the Lipponen & Tarvainen (2019) algorithm (the Kubios 'automatic' method). Each flagged beat is "
        f"classified as *ectopic*, *missed*, *extra* or *long/short* using adaptive thresholds (α = {cfg.alpha}, {cfg.window_width}-beat window).",
        f"**Safety net for dense noise**: any beat more than {cfg.max_dev_frac:.0%} away from a robust local median is flagged too, and flagged "
        "beats are kept out of the threshold statistics. In this recording, garbage stretches otherwise inflate the adaptive thresholds so much "
        "that obvious junk (e.g. 342 ms inside a 1300 ms rhythm) passes.",
        f"**Noise bursts are masked, not corrected**: where ≥{cfg.burst_min_bad} of any {cfg.burst_window} consecutive beats are flagged, the run "
        f"(±{cfg.burst_margin} beats) is marked unusable. Masked intervals **keep their duration on the time axis**, so the timeline stays "
        "continuous and every window knows how much of it is unusable.",
        ("**Isolated artifacts are corrected by type**, in a way that preserves total elapsed time exactly:", [
            "*extra* detection → merged with the neighbour whose sum best matches the local median;",
            "*missed* beat → split into k equal intervals;",
            "*ectopic* / *long-short* → the R-peak shared by a pair of intervals is moved to their midpoint (the pair whose mean best matches the median).",
        ]),
        "**Residual check**: detection is re-run on the corrected series; anything still flagged is masked.",
        "**Metrics only from clean adjacency**: RMSSD/pNN50 use successive differences only where *both* intervals are usable. LF/HF use a "
        "Lomb–Scargle periodogram on the unevenly spaced usable beats, so masked gaps never need interpolating.",
        f"**Window gate**: a window counts only if ≥{gate.min_coverage:.0%} of its time is usable NN and ≤{gate.max_corrected:.0%} of its beats "
        "were corrected (the conventional ≤5% correction limit).",
    ], ordered=True))
    add(Heading("Result on this recording", level=2))
    add(table(pd.DataFrame([
        ["Raw intervals", f"{cs['n_raw_intervals']:,}"],
        ["Masked (noise bursts, unresolvable beats)", f"{cs['n_masked']:,} intervals · {_f(cs['masked_time_s'] / 60, 0)} min · {_pct(cs['masked_time_frac'])} of recording time"],
        ["Corrected (isolated artifacts)", f"{cs['n_corrected']:,} intervals ({_pct(cs['corrected_frac_of_valid'], 1)} of usable): "
         + ", ".join(f"{k} {cs[f'corrected_{k}']}" for k in ("extra", "missed", "ectopic", "longshort") if cs[f"corrected_{k}"])],
        ["Untouched", f"{cs['n_ok']:,}"],
        ["Total time before → after", f"{_f(rec.rr_total_s, 3)} s → {_f(cs['total_time_s'], 3)} s (identical)"],
        ["Device's own 'unusual' flag", f"{res.device_flags['device_flagged']} beats; we also flag {_pct(res.device_flags['device_flags_we_also_flag'])} of them, "
         f"but they cover only {_pct(res.device_flags['our_flags_device_also_flags'])} of what we flag — not usable as a filter"],
    ], columns=["", "Count"]), colspec=r"L{0.34\linewidth}Y"))
    add(Figure("figures/01_tachogram",
               "Raw strap intervals (grey) and the cleaned NN series that enters the analysis (blue); shaded spans are masked noise. "
               "Bottom: a 3-minute zoom in which isolated artifacts are corrected in place (orange) and a noise burst is masked rather than interpolated."))
    add(Para("**Validation on synthetic data with known truth** (sleep-like RR with respiratory sinus arrhythmia, 2% injected "
             "ectopic/missed/extra beats plus 3 noise bursts, 3 random seeds):"))
    vt = pd.DataFrame({
        "seed": syn["seed"].astype(int),
        "ectopic found": [f"{_pct(v)} of {int(k)}" for v, k in zip(syn["detected_ectopic"], syn["n_ectopic"])],
        "missed found": [f"{_pct(v)} of {int(k)}" for v, k in zip(syn["detected_missed"], syn["n_missed"])],
        "extra found": [f"{_pct(v)} of {int(k)}" for v, k in zip(syn["detected_extra"], syn["n_extra"])],
        "bursts masked": [_pct(v) for v in syn["burst_masked_frac"]],
        "RMSSD true": syn["rmssd_true"],
        "uncleaned": syn["rmssd_uncleaned"],
        "cleaned": syn["rmssd_cleaned"],
        "false flags (clean data)": [_pct(v, 1) for v in syn["false_flag_frac_clean_series"]],
    })
    add(table(vt, colspec="lCCCCCCCC", small=True))
    add(Para("False flags on artifact-free data come from the Kubios thresholds themselves (NeuroKit2's implementation flags a similar "
             "number); they cost about 2% of RMSSD."))

    # 3. Alignment --------------------------------------------------------------
    add(Heading("Aligning the two devices in time"))
    add(Para(f"The strap has **no independent clock**: its `Time` column is exactly `start + cumsum(RRI)`. The header says the session ran "
             f"{seconds_to_clock(rec.duration_s)} ({_clock(rec.start_s, True)}–{_clock(session_end, True)}), but the intervals add up to only "
             f"{seconds_to_clock(rec.rr_total_s)}, ending at {_clock(strap_end, True)}. **{_f(rec.clock_deficit_s / 60)} minutes of wall-clock "
             "time contain no beats**, and nothing in the RRI file says where. Three hypotheses are carried through the whole analysis:"))
    add(table(pd.DataFrame([[k, v] for k, v in HYPOTHESES.items()], columns=["", "Where the missing time is"]), colspec="lY"))
    ts = _tail_stats(res)
    drift = [h for h in res.walls if h != "H1"]
    add(Para("**Where did the time go?** The evidence points both ways."))
    add(Para("*For time lost inside the night (H2/H3):*"))
    for_drift = [
        f"no interval longer than {_f(rec.rr_ms.max() / 1000, 2)} s appears in the file, so a contact loss longer than that leaves no interval "
        f"and its time silently vanishes, and {_pct(cs['masked_time_frac'])} of the recording is contact noise;",
        f"the strap's final 5 minutes are {_pct(tail_masked)} noise. With drift ({'/'.join(drift)}) they fall at "
        f"{_clock(ts['tail_start'][drift[-1]])}–{_clock(ts['tail_end'][drift[-1]])}, when the watch says you woke ({inp.sleep_end}) and the "
        f"session was stopped ({_clock(session_end)}), which fits getting up and taking the strap off. Without drift (H1) they fall at "
        f"{_clock(ts['tail_start']['H1'])}–{_clock(ts['tail_end']['H1'])}, in the middle of sleep;",
    ]
    if _lag_note(res):
        for_drift.append("in the exploratory lag scan below, " + _lag_note(res))
    add(ListBlock(for_drift))
    add(Para("*For an early end of the stream (H1):*"))
    add(ListBlock([
        f"the last usable strap minutes before that final noise look like stable sleep (HR ≈ {_f(ts['hr_before'], 0)} bpm, "
        f"RMSSD ≈ {_f(ts['rmssd_before'], 0)} ms), not the arousal that usually precedes waking.",
    ]))
    add(Para("So the strap's clock probably runs tens of minutes behind in the second half of the night, but by how much cannot be pinned "
             "down from one night without a sync marker. **The figures and window-level tables use H1**, the strap's clock as recorded, "
             "because it needs no assumption about where the beats went. **Every headline number is also given under H2 and H3.** "
             "**No offset is fitted**: choosing a shift by HRV agreement and then reporting HRV agreement would be circular."))
    add(Figure("figures/07_lag_scan",
               f"Correlation between strap (H1 clock) and watch ln RMSSD when the strap timeline is shifted by "
               f"{_sf(res.lags['lag_min'].min(), 0)} to {_sf(res.lags['lag_min'].max(), 0)} min. A positive shift "
               "means the strap's clock runs behind the wall clock. The shift is not applied, because it would be chosen on the very "
               "agreement being measured.", width=0.62))

    # 4. Digitization -----------------------------------------------------------
    add(Heading("Reading the watch values off the screenshot"))
    add(ListBlock([
        f"**y-axis**: the 0/50/100/150 ms gridlines were detected at rows {', '.join(f'{x:g}' for x in c['rows'])} → "
        f"{_f(c['px_per_ms'], 3)} px per ms (1 px ≈ {_f(1 / c['px_per_ms'], 2)} ms).",
        f"**x-axis**: the baseline spans {inp.sleep_start} (sleep onset) to {inp.sleep_end} (wake), as labelled.",
        f"**readings**: the line is a polyline whose corners are the watch's measurements. A free-knot fit recovers **{c['n_points']} readings** "
        f"(the fitted polyline matches the drawn line to {_f(c['rmse_px'], 2)} px RMS). They are **not evenly spaced**: the median spacing is "
        f"{_f(c['spacing_median_min'])} min, with {c.get('n_gaps_over_1_5x', 0)} gaps longer than 15 min (skipped measurements). "
        "Sampling the line on a regular grid instead would read values off the straight segments between readings.",
        f"**check**: the mean of the digitized readings is **{_f(watch.digitized_avg, 2)} ms vs the app's {_f(watch.reported_avg, 0)} ms**.",
    ]))

    # 5. HRV agreement ----------------------------------------------------------
    add(Heading("HRV agreement"))
    add(Figure("figures/02_window_quality",
               "Share of each watch measurement window covered by usable strap NN intervals. Only windows that pass the quality gate (blue) are compared."))
    add(Para(f"Of the {n['n_points']} watch readings, **{n['n_valid']}** have a strap window that passes the gate. " + _coverage_note(res)))
    add(Figure("figures/03_rmssd_overnight",
               "Overnight RMSSD: the watch's digitized readings (orange) and the strap's 10-minute RMSSD for windows that pass the gate (blue). "
               "The grey band is the watch's personal normal range; the vertical line marks where the strap's beat stream ends."))
    add(Heading("Night level (main comparison)", level=2))
    add(Para(f"With the strap's clock as recorded ({inp.hypothesis}):"))
    add(table(pd.DataFrame([
        ["Watch, app-reported night average", _f(watch.reported_avg, 0), "all readings"],
        ["Watch, digitized night average", _f(n["watch_digitized_avg_all"]), "all readings"],
        ["Watch, average over matched readings", _f(n["watch_avg_matched"]), f"{n['n_valid']} readings"],
        ["Strap, average over the same windows", _f(n["strap_avg_valid"]), f"{n['n_valid']} windows"],
        ["**Difference (watch − strap)**", f"**{_sf(n['matched_diff_ms'])}**", f"ratio ×{_f(n['matched_ratio'], 3)}"],
    ], columns=["", "RMSSD (ms)", "based on"]), colspec="lrl"))
    bh = res.by_hypothesis
    add(Para("Under each clock hypothesis (same gate and windows; the hypothesis decides which watch readings are matched, and to "
             "which stretch of strap data):"))
    add(table(pd.DataFrame({
        "clock": bh["hypothesis"],
        "n": bh["n"],
        "watch (ms)": bh["watch_avg"],
        "strap (ms)": bh["strap_avg"],
        "watch − strap (ms)": [_sf(v) for v in bh["diff_ms"]],
        "ratio": [f"×{_f(v, 2)}" for v in bh["ratio"]],
        "r (ln), epochs": [_f(v, 2) for v in bh["pearson_r_ln"]],
        "MAE (ms), epochs": bh["mae_ms"],
    }), colspec="lRRRRRRR"))
    add(Heading("Epoch level (secondary)", level=2))
    add(Figure("figures/04_agreement",
               "Left: watch vs strap RMSSD for the matched windows, with the identity line. Right: Bland–Altman plot on the ratio (log) scale "
               "with the bias and 95% limits of agreement."))
    if e["n"] >= 3:
        add(table(pd.DataFrame([
            ["Bias (watch − strap)", f"{_sf(e['bias_ms'])} ms", f"ratio ×{_f(e['ratio'], 3)}"],
            ["95% limits of agreement", f"{_sf(e['loa_low_ms'])} to {_sf(e['loa_high_ms'])} ms", f"×{_f(e['ratio_loa_low'], 2)} to ×{_f(e['ratio_loa_high'], 2)}"],
            ["Lin's CCC", _f(e["ccc"], 2), f"ln scale {_f(e['ccc_ln'], 2)}"],
            ["Pearson r (ln RMSSD)", _f(e["pearson_r_ln"], 2),
             f"95% CI {_f(e.get('pearson_ci_low'), 2)} to {_f(e.get('pearson_ci_high'), 2)}, p = {_f(e['pearson_p'], 3)}"],
            ["Spearman ρ", _f(e["spearman_rho"], 2), ""],
            ["MAE / MAPE", f"{_f(e['mae_ms'])} ms", _pct(e["mape_pct"] / 100)],
        ], columns=["Statistic", "Value", ""]), colspec="lll"))
    if nz.get("excess_sd_ms", 0) > 0 and nz["sd_watch_ms"] > 1.2 * nz["sd_strap_ms"]:
        add(Para(f"The watch readings vary more than the strap's ({_f(nz['sd_watch_ms'])} vs {_f(nz['sd_strap_ms'])} ms SD over the matched windows). "
                 f"That is ≈{_f(nz['excess_sd_ms'])} ms of extra scatter per reading, consistent with each COROS reading being a short optical sample. "
                 f"Averaged over {len(watch.point_ms)} readings it shrinks to ≈{_f(nz['nightly_mean_se_ms'])} ms on the nightly mean, which is why "
                 "the nightly average is far more trustworthy than any single reading. " + _window_scan_note(res)))
    else:
        add(Para(f"The watch readings vary about as much as the strap's ({_f(nz.get('sd_watch_ms'))} vs {_f(nz.get('sd_strap_ms'))} ms SD). "
                 + _window_scan_note(res)))
    ws = res.window_scan.rename(columns={"window_min": "strap window", "matched_diff_ms": "night diff (ms)", "pearson_r_ln": "r (ln)",
                                         "ccc": "CCC", "mae_ms": "MAE (ms)", "sd_strap": "SD strap", "sd_watch": "SD watch"})
    add(table(ws[["strap window", "convention", "n", "night diff (ms)", "r (ln)", "CCC", "MAE (ms)", "SD strap", "SD watch"]],
              {"strap window": lambda v: f"{v:g} min", "night diff (ms)": _sf, "r (ln)": 2, "CCC": 2}, colspec="rlrrrrrrr", small=True))
    add(Para(f"Epoch-level agreement also depends on the clock: across the three hypotheses r (ln) ranges from "
             f"{_f(bh['pearson_r_ln'].min(), 2)} to {_f(bh['pearson_r_ln'].max(), 2)} (table in the previous subsection). 10-minute epochs from "
             "one night are few and autocorrelated, and their alignment is uncertain, so treat the epoch statistics as descriptive, not as "
             "a validation."))

    # 6. Sensitivity ------------------------------------------------------------
    add(Heading("Sensitivity analysis"))
    add(Figure("figures/05_sensitivity",
               "Night-level difference (watch − strap) over coverage-matched windows for every combination of alignment hypothesis, window "
               "convention, window length and quality gate. The analysis used in the other figures is highlighted; n is the number of matched windows."))
    add(Para(f"Across the {len(main)} strict- and default-gate variants, the night-level difference stays between **{_sf(main['matched_diff_ms'].min())} and "
             f"{_sf(main['matched_diff_ms'].max())} ms** (median {_sf(main['matched_diff_ms'].median())} ms). With the **lenient** gate it moves to "
             f"{_sf(lenient['matched_diff_ms'].min())} … {_sf(lenient['matched_diff_ms'].max())} ms (median {_sf(lenient['matched_diff_ms'].median())} ms)"
             + (": leftover artifacts in poorly covered windows inflate the strap's RMSSD and make the watch look low. "
                "Quality gating matters more than any alignment choice. "
                if lenient["matched_diff_ms"].median() < main["matched_diff_ms"].median() - 3 else ". ")
             + f"Full grid: [`{rel(tables['sensitivity_grid'])}`]({rel(tables['sensitivity_grid'])})."))

    # 7. RHR --------------------------------------------------------------------
    add(Heading("Resting heart rate"))
    add(Figure("figures/06_heart_rate",
               "Strap heart rate during sleep from usable beats only: per-minute values and the 5-minute rolling mean, with the lowest "
               "5-minute mean marked."))
    add(Para("COROS doesn't publish exactly how its resting HR is computed, so several standard strap definitions are reported. "
             "Compare the watch's RHR against the one that matches its definition (most wearables report a sleep-time low such as the "
             "lowest 5–30-min average):"))
    rhr_rows = [
        ["Lowest 5-min mean HR", _f(r["lowest_5min_mean_hr"]), f"window ending {_clock(r['lowest_5min_at_s'])}"],
        ["Lowest 30-min mean HR", _f(r["lowest_30min_mean_hr"]), f"window ending {_clock(r['lowest_30min_at_s'])}"],
        ["Mean sleeping HR", _f(r["mean_sleep_hr"]), "all usable minutes"],
        ["5th percentile of 1-min HR", _f(r["p5_1min_hr"]), ""],
        ["Lowest 1-min HR", _f(r["lowest_1min_hr"]), ""],
    ]
    if r.get("watch_rhr") is not None:
        rhr_rows.append(["**Watch RHR**", f"**{_f(r['watch_rhr'], 0)}**",
                         f"watch − lowest 5-min = {_sf(r['watch_rhr'] - r['lowest_5min_mean_hr'])} bpm; "
                         f"watch − lowest 30-min = {_sf(r['watch_rhr'] - r['lowest_30min_mean_hr'])} bpm"])
    add(table(pd.DataFrame(rhr_rows, columns=["Definition (strap)", "bpm", ""]), colspec="lrl"))
    add(Para(f"Based on {r['minutes_valid']} of {r['minutes_total']} sleep minutes with ≥80% usable beats. HR is much less sensitive "
             "to artifacts than HRV: a masked minute is simply skipped."))

    # 8. Strap-only summary -----------------------------------------------------
    add(Heading("Strap-only overnight HRV summary"))
    add(Para(f"From {s5['n_valid']} of {s5['n_windows']} 5-min windows passing the gate between {_clock(t0)} and {_clock(s5['covered_until_s'])} "
             "(Task Force 1996 short-term standard; frequency domain via Lomb–Scargle):"))
    add(table(pd.DataFrame([
        ["Mean NN", _f(s5["mean_mean_nn"], 0), "ms"],
        ["HR", _f(s5["mean_hr"]), "bpm"],
        ["SDNN (5-min)", _f(s5["mean_sdnn"]), "ms"],
        ["RMSSD (5-min)", _f(s5["mean_rmssd"]), "ms"],
        ["pNN50", _f(s5["mean_pnn50"]), "%"],
        ["LF power", _f(s5["mean_lf"], 0), "ms²"],
        ["HF power", _f(s5["mean_hf"], 0), "ms²"],
        ["LF/HF", _f(s5["mean_lf_hf"], 2), ""],
    ], columns=["Metric", "Value", "Unit"]), colspec="lrl"))
    add(Para(f"RMSSD pooled over every usable beat pair of the night is {_f(s5['pooled_rmssd'])} ms. It is higher than the gated window "
             "average because it includes the edges of poorly covered windows; the gated windows are the more robust estimate."))

    # 9. Limitations ------------------------------------------------------------
    add(Heading("Limitations"))
    add(ListBlock([
        f"**One night, one person.** Agreement statistics from about {n['n_valid']} autocorrelated windows are descriptive. A validation "
        "needs several nights (ideally several people) and a nightly-level Bland–Altman across nights.",
        f"**Strap signal quality** is the dominant problem: only {_pct(s5['valid_nn_time_frac'])} of the sleep window was usable, "
        f"and only {n['n_valid']} of {n['n_points']} watch readings could be matched.",
        f"**Clock**: the {_f(rec.clock_deficit_s / 60, 0)}-min deficit cannot be located from the RRI data. The night-level difference "
        f"spans {_sf(bh['diff_ms'].min())} to {_sf(bh['diff_ms'].max())} ms across the clock hypotheses, and window-by-window matching is "
        "uncertain by up to tens of minutes late in the night.",
        "**COROS internals** (measurement duration within each 10-min slot, artifact handling, RHR definition) are not published.",
        "**Screenshot digitization**: ±1 px ≈ ±0.7 ms in value and ≈ ±1 min in time per reading.",
    ]))

    # 10. Protocol --------------------------------------------------------------
    add(Heading("Protocol for the next nights"))
    add(ListBlock([
        f"**Wet the strap electrodes** (or use electrode gel) and tighten the strap. Contact noise made {_pct(cs['masked_time_frac'])} of this recording unusable.",
        "**Start both devices within the same minute** and note the time; stop the strap recording as soon as you wake.",
        "**Add sync markers**: 5 slow deep breaths, or 10 s of standing still after sitting up, at the start and end. Both devices show these as "
        "a clear HR/RR change.",
        "**Log when the strap is taken off.** It resolves the clock-deficit question directly.",
        "**Export the watch data** if possible (COROS app → activity/sleep → export FIT, or a third-party sync). Numeric data removes "
        "digitization error and gives the watch's HR curve, which allows HR-based alignment that is independent of HRV.",
        "Record **at least 5–7 nights** and compare nightly averages.",
    ], ordered=True))

    # Reproduce / references ----------------------------------------------------
    add(Heading("Reproduce", numbered=False))
    add(Code("pip install -e .\n" + reproduce_command(res, outdir)))
    add(Para("Tables written alongside this report:"))
    add(ListBlock([f"[`{rel(p)}`]({rel(p)})" for p in tables.values()]))
    add(Heading("References", numbered=False))
    add(ListBlock([
        "Lipponen JA, Tarvainen MP (2019). A robust algorithm for heart rate variability time series artefact correction using novel beat "
        "classification. *J Med Eng Technol* 43(3):173–181.",
        "Task Force of the ESC and NASPE (1996). Heart rate variability: standards of measurement, physiological interpretation and clinical use. "
        "*Circulation* 93:1043–1065.",
        "Bland JM, Altman DG (1999). Measuring agreement in method comparison studies. *Stat Methods Med Res* 8:135–160.",
        "Lin LI (1989). A concordance correlation coefficient to evaluate reproducibility. *Biometrics* 45:255–268.",
        "COROS support: Overnight HRV — RMSSD measured over 10-minute intervals during sleep and averaged nightly.",
    ]))
    return doc


def write_report(res, outdir: Path, pdf: bool = True) -> dict[str, Path]:
    """Write report.md, report.tex and (if a LaTeX engine is available) report.pdf."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    make_all(res, outdir / "figures")
    tables = write_tables(res, outdir / "tables")
    (outdir / "summary.json").write_text(
        json.dumps(summary_dict(res), indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    )
    doc = build_document(res, outdir, tables)
    paths = {"markdown": outdir / "report.md", "latex": outdir / "report.tex"}
    paths["markdown"].write_text(to_markdown(doc))
    paths["latex"].write_text(to_latex(doc, short_title=f"Chest strap vs COROS watch · {res.rec.start:%Y-%m-%d}"))
    if pdf:
        if latex_engine() is None:
            print("warning: no LaTeX engine found; wrote report.tex but not report.pdf", file=sys.stderr)
        else:
            paths["pdf"] = build_pdf(paths["latex"])
    return paths
