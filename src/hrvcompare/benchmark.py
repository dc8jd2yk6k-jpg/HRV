"""Benchmark artifact-handling methods against ground truth.

Ground-truth NN series are corrupted with isolated artifacts and contact-noise
bursts at controlled levels. Every corruption replaces a stretch of beats by
artifacts of exactly the same total duration, so wall-clock windows of the
corrupted series line up with those of the clean series. Each method estimates
RMSSD in 5-min windows of the corrupted series; the error is the difference to
RMSSD of the clean series in the same window.

Methods compared:

none            RMSSD of the corrupted series as recorded
malik_delete    Malik 20 % filter, flagged beats deleted, differences taken
                across the gaps (the "just delete the outliers" approach)
malik_adjacent  same filter, differences only between originally adjacent
                accepted beats
malik_interp    same filter, flagged beats replaced by linear interpolation
kubios_nk2      Lipponen & Tarvainen (2019) correction as implemented in
                NeuroKit2 (``signal_fixpeaks(method="kubios", iterative=True)``)
proposed        this package: classification + burst masking + time-preserving
                correction + adjacency-aware RMSSD + window gate (default)
proposed_lenient  the same with the lenient gate (>=60 % usable time), to show
                the trade-off between how many windows are kept and accuracy

External validation on PhysioNet's MIT-BIH Arrhythmia Database (real ectopy)
is implemented in ``mitdb_ectopy_benchmark``; ``physionet_rr`` also reads other
beat-annotated databases, such as NSR2DB, for future ground-truth sets. Both
need network access to physionet.org.
"""

from __future__ import annotations

import argparse
import itertools
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import ArtifactConfig, CORRECTED, MASKED, OK, clean_rr
from .metrics import GATES, Gate
from .synthetic import synthetic_rr

METHODS = ("none", "malik_delete", "malik_adjacent", "malik_interp", "kubios_nk2", "proposed", "proposed_lenient")
METHOD_LABELS = {
    "none": "No cleaning",
    "malik_delete": "20% filter, delete",
    "malik_adjacent": "20% filter, delete, adjacent pairs only",
    "malik_interp": "20% filter, interpolate",
    "kubios_nk2": "Kubios-type correction (NeuroKit2)",
    "proposed": "Proposed",
    "proposed_lenient": "Proposed, lenient gate",
}
WINDOW_S = 300.0


# ----------------------------------------------------------------------------
# Comparator methods


def malik_flags(rr: np.ndarray, thr: float = 0.20, lo: float = 400.0, hi: float = 2000.0, reset_after: int = 5) -> np.ndarray:
    """Flag intervals out of range or differing by more than *thr* from the last
    accepted interval. After *reset_after* consecutive rejections the reference
    is reset to the current beat, so a genuine step change cannot lock the
    filter out for the rest of the recording."""
    rr = np.asarray(rr, dtype=float)
    flags = np.zeros(len(rr), dtype=bool)
    ref, streak = None, 0
    for i, x in enumerate(rr):
        if not lo <= x <= hi:
            flags[i] = True
            continue
        if ref is None or abs(x - ref) <= thr * ref:
            ref, streak = x, 0
            continue
        flags[i] = True
        streak += 1
        if streak >= reset_after:
            ref, streak = x, 0
    return flags


def _windows(total_s: float) -> np.ndarray:
    return np.arange(0.0, total_s - WINDOW_S + 1e-9, WINDOW_S)


def _rmssd_in_windows(t_end: np.ndarray, values: np.ndarray, pair_ok: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """RMSSD of successive differences ``values[k+1]-values[k]`` allowed by
    *pair_ok[k]*, for pairs whose second interval ends inside each window."""
    d = np.diff(values)
    t2 = t_end[1:]
    out = np.full(len(starts), np.nan)
    for w, a in enumerate(starts):
        m = pair_ok & (t2 >= a) & (t2 < a + WINDOW_S)
        if m.sum() >= 10:
            out[w] = np.sqrt(np.mean(d[m] ** 2))
    return out


def estimate(method: str, rr: np.ndarray, starts: np.ndarray, cfg: ArtifactConfig | None = None, gate: Gate = GATES["default"]) -> np.ndarray:
    """RMSSD per window [start, start+300 s) of the corrupted series *rr*."""
    rr = np.asarray(rr, dtype=float)
    t_end = np.cumsum(rr) / 1000.0
    n = len(rr)
    if method == "none":
        return _rmssd_in_windows(t_end, rr, np.ones(n - 1, bool), starts)
    if method.startswith("malik"):
        f = malik_flags(rr)
        if method == "malik_delete":
            keep = ~f
            return _rmssd_in_windows(t_end[keep], rr[keep], np.ones(keep.sum() - 1, bool), starts)
        if method == "malik_adjacent":
            return _rmssd_in_windows(t_end, rr, ~f[1:] & ~f[:-1], starts)
        if method == "malik_interp":
            idx = np.arange(n)
            filled = rr.copy()
            if (~f).sum() >= 2:
                filled[f] = np.interp(idx[f], idx[~f], rr[~f])
            return _rmssd_in_windows(t_end, filled, np.ones(n - 1, bool), starts)
    if method == "kubios_nk2":
        import neurokit2 as nk

        peaks = np.concatenate([[0], np.round(np.cumsum(rr))]).astype(int)
        try:
            _, fixed = nk.signal_fixpeaks(peaks, sampling_rate=1000, method="kubios", iterative=True)
        except Exception:  # NeuroKit2 can fail on pathological input
            return np.full(len(starts), np.nan)
        fixed = np.unique(np.asarray(fixed, dtype=float))
        rr2 = np.diff(fixed)
        return _rmssd_in_windows(fixed[1:] / 1000.0, rr2, np.ones(len(rr2) - 1, bool), starts)
    if method in ("proposed", "proposed_lenient"):
        if method == "proposed_lenient":
            gate = GATES["lenient"]
        res = clean_rr(rr, cfg)
        b = res.beats
        valid = (b["status"] != MASKED).to_numpy()
        corrected = (b["status"] == CORRECTED).to_numpy()
        t = b["t_end_s"].to_numpy()
        v = b["rr_ms"].to_numpy()
        est = _rmssd_in_windows(t, v, valid[1:] & valid[:-1], starts)
        for w, a in enumerate(starts):  # window gate
            m = (t >= a) & (t < a + WINDOW_S)
            cover = v[m & valid].sum() / 1000.0 / WINDOW_S
            corr = corrected[m].sum() / max((m & valid).sum(), 1)
            if cover < gate.min_coverage or corr > gate.max_corrected:
                est[w] = np.nan
        return est
    raise ValueError(f"unknown method {method!r}")


# ----------------------------------------------------------------------------
# Ground truth and corruption


def real_clean_segments(rec_rr: np.ndarray, min_minutes: float = 5.0, malik_thr: float = 0.20) -> list[np.ndarray]:
    """Stretches of a real recording that are clean by two independent
    criteria: untouched by ``clean_rr`` and no successive change above
    *malik_thr*. Returns the RR arrays."""
    res = clean_rr(rec_rr)
    b = res.beats
    rr = b["rr_ms"].to_numpy()
    good = ((b["status"] == OK) & (b["raw_first"] == b["raw_last"])).to_numpy()
    jump = np.abs(np.diff(rr)) / rr[:-1] > malik_thr
    segs, i, n = [], 0, len(rr)
    while i < n:
        if not good[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and good[j + 1] and not jump[j]:
            j += 1
        if rr[i : j + 1].sum() / 60000 >= min_minutes:
            segs.append(rr[i : j + 1].copy())
        i = j + 1
    return segs


def synthetic_segments(n: int = 20, minutes: float = 10.0, seed: int = 100) -> list[np.ndarray]:
    """Synthetic sleep-like NN series spanning a range of heart rates and HRV."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        mean = rng.uniform(850, 1300)
        beats = int(minutes * 60000 / mean) + 1
        out.append(
            synthetic_rr(
                n=beats,
                mean_ms=mean,
                rsa_ms=rng.uniform(10, 60),
                resp_hz=rng.uniform(0.2, 0.3),
                lf_ms=rng.uniform(5, 30),
                noise_ms=rng.uniform(5, 20),
                seed=seed + k,
            )
        )
    return out


def noise_library(rec_rr: np.ndarray, dev: float = 0.30, merge_gap: int = 5, pad: int = 2, min_len: int = 5) -> list[np.ndarray]:
    """Contact-noise episodes cut from a real recording: runs where raw
    intervals deviate more than *dev* from a 91-beat rolling median (or leave
    400-2000 ms), merged across short gaps and padded. This segmentation is
    deliberately simpler than, and independent of, ``clean_rr``."""
    rr = np.asarray(rec_rr, dtype=float)
    med = pd.Series(rr).rolling(91, center=True, min_periods=10).median().bfill().ffill().to_numpy()
    bad = (np.abs(rr / med - 1) > dev) | (rr < 400) | (rr > 2000)
    idx = np.flatnonzero(bad)
    if len(idx) == 0:
        return []
    runs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > merge_gap:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))
    eps = []
    for a, b in runs:
        a, b = max(a - pad, 0), min(b + pad + 1, len(rr))
        if b - a >= min_len:
            eps.append(rr[a:b].copy())
    return eps


def parametric_episode(duration_s: float, rng: np.random.Generator) -> np.ndarray:
    """Random spurious detections (250-1500 ms) filling *duration_s*."""
    out, total = [], 0.0
    while total < duration_s * 1000:
        x = rng.uniform(250, 1500)
        out.append(x)
        total += x
    return np.asarray(out)


def corrupt(
    base: np.ndarray,
    rng: np.random.Generator,
    contamination: float,
    iso_rate: float,
    noise: str,
    library: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Time-preserving corruption of *base*: noise episodes covering about
    *contamination* of the time, then isolated ectopic / missed / extra beats
    at *iso_rate* per beat elsewhere."""
    base = np.asarray(base, dtype=float)
    n = len(base)
    total = base.sum()
    cum = np.concatenate([[0.0], np.cumsum(base)])
    taken = np.zeros(n, dtype=bool)
    episodes: dict[int, tuple[int, np.ndarray]] = {}
    budget = contamination * total
    tries = 0
    while budget > 0 and tries < 200:
        tries += 1
        if noise == "empirical":
            ep = library[rng.integers(len(library))]
        else:
            durs = np.array([e.sum() for e in library]) / 1000 if library else np.array([20.0])
            ep = parametric_episode(float(rng.choice(durs)), rng)
        dur = min(ep.sum(), budget + 2000.0)
        s = int(rng.integers(5, max(n - 10, 6)))
        e = int(np.searchsorted(cum, cum[s] + dur)) if cum[-1] > cum[s] + dur else n
        e = max(e, s + 1)
        lo, hi = max(s - 12, 0), min(e + 12, n)
        if e >= n - 2 or taken[lo:hi].any():
            continue
        seg_t = cum[e] - cum[s]
        ep = ep[: max(np.searchsorted(np.cumsum(ep), seg_t) + 1, 1)]
        ep = ep * (seg_t / ep.sum())
        episodes[s] = (e, ep)
        taken[s:e] = True
        budget -= seg_t
    iso: dict[int, str] = {}
    candidates = [i for i in range(15, n - 15, 12) if not taken[max(i - 12, 0) : i + 13].any()]
    k = min(int(round(iso_rate * n)), len(candidates))
    for i in rng.choice(candidates, size=k, replace=False) if k else []:
        iso[int(i)] = str(rng.choice(["ectopic", "missed", "extra"]))
    out, i = [], 0
    while i < n:
        if i in episodes:
            e, ep = episodes[i]
            out.extend(ep.tolist())
            i = e
        elif i in iso and i + 1 < n:
            kind = iso[i]
            if kind == "ectopic":
                s = 0.65 * base[i]
                out += [s, base[i] - s + base[i + 1]]
                i += 2
            elif kind == "missed":
                out.append(base[i] + base[i + 1])
                i += 2
            else:
                a = rng.uniform(0.3, 0.6) * base[i]
                out += [a, base[i] - a]
                i += 1
        else:
            out.append(base[i])
            i += 1
    return np.asarray(out)


# ----------------------------------------------------------------------------
# Experiment


def _seed(*parts) -> int:
    """Deterministic seed (Python's hash() is randomised per process)."""
    return zlib.crc32("|".join(map(str, parts)).encode())


def run_noise_benchmark(
    bases: dict[str, list[np.ndarray]],
    library: list[np.ndarray],
    contaminations=(0.0, 0.05, 0.10, 0.20, 0.30),
    iso_rate: float = 0.01,
    noises=("empirical", "parametric"),
    seeds: int | dict[str, int] = 5,
    methods=METHODS,
) -> pd.DataFrame:
    rows = []
    for (set_name, segs), noise, c in itertools.product(bases.items(), noises, contaminations):
        n_seeds = seeds[set_name] if isinstance(seeds, dict) else seeds
        for k, base in enumerate(segs):
            starts = _windows(base.sum() / 1000.0)
            if len(starts) == 0:
                continue
            truth = _rmssd_in_windows(np.cumsum(base) / 1000.0, base, np.ones(len(base) - 1, bool), starts)
            for seed in range(n_seeds):
                rng = np.random.default_rng(_seed(set_name, noise, c, k, seed))
                bad = corrupt(base, rng, c, iso_rate, noise, library)
                for method in methods:
                    est = estimate(method, bad, starts)
                    for w in range(len(starts)):
                        rows.append(
                            {
                                "base": set_name,
                                "noise": noise,
                                "contamination": c,
                                "segment": k,
                                "seed": seed,
                                "window": w,
                                "method": method,
                                "truth": truth[w],
                                "estimate": est[w],
                            }
                        )
    df = pd.DataFrame(rows)
    df["error_pct"] = 100 * (df["estimate"] - df["truth"]) / df["truth"]
    return df


def run_clean_baseline(bases: dict[str, list[np.ndarray]], methods=METHODS) -> pd.DataFrame:
    """Uncorrupted input: how much does each method change RMSSD on clean data?"""
    rows = []
    for set_name, segs in bases.items():
        for k, base in enumerate(segs):
            starts = _windows(base.sum() / 1000.0)
            truth = _rmssd_in_windows(np.cumsum(base) / 1000.0, base, np.ones(len(base) - 1, bool), starts)
            for method in methods:
                est = estimate(method, base, starts)
                for w in range(len(starts)):
                    rows.append({"base": set_name, "segment": k, "window": w, "method": method, "truth": truth[w], "estimate": est[w]})
    df = pd.DataFrame(rows)
    df["error_pct"] = 100 * (df["estimate"] - df["truth"]) / df["truth"]
    return df


def summarize(df: pd.DataFrame, by=("base", "noise", "contamination", "method")) -> pd.DataFrame:
    def agg(g):
        e = g["error_pct"].dropna()
        return pd.Series(
            {
                "windows": len(g),
                "yield": len(e) / len(g) if len(g) else np.nan,
                "median_abs_error_pct": float(np.median(np.abs(e))) if len(e) else np.nan,
                "p90_abs_error_pct": float(np.percentile(np.abs(e), 90)) if len(e) else np.nan,
                "median_error_pct": float(np.median(e)) if len(e) else np.nan,
            }
        )

    return df.groupby(list(by), sort=False).apply(agg, include_groups=False).reset_index()


# ----------------------------------------------------------------------------
# PhysioNet (needs network access to physionet.org)

NORMAL = set("NLRej")
ECTOPIC = set("AaJSVEF")
MITDB_PACED = {"102", "104", "107", "217"}
MITDB_RECORDS = [
    "100", "101", "103", "105", "106", "108", "109", "111", "112", "113", "114", "115", "116", "117", "118", "119",
    "121", "122", "123", "124", "200", "201", "202", "203", "205", "207", "208", "209", "210", "212", "213", "214",
    "215", "219", "220", "221", "222", "223", "228", "230", "231", "232", "233", "234",
]


def physionet_rr(record: str, db: str, extension: str, local_dir: str | Path | None = None) -> dict:
    """RR intervals (ms) from a PhysioNet beat-annotation file, with the
    symbols of the beats that end and start each interval and the set of
    rhythm labels present (e.g. '(AFIB')."""
    import wfdb

    if local_dir is not None:
        ann = wfdb.rdann(str(Path(local_dir) / record), extension)
        hdr = wfdb.rdheader(str(Path(local_dir) / record))
    else:
        ann = wfdb.rdann(record, extension, pn_dir=db)
        hdr = wfdb.rdheader(record, pn_dir=db)
    symbols_all = np.asarray(ann.symbol)
    beats = np.array([s in NORMAL | ECTOPIC | set("/fQ") for s in symbols_all])
    samples = np.asarray(ann.sample)[beats]
    symbols = symbols_all[beats]
    rhythms = {a.strip("\x00").strip() for a, s in zip(ann.aux_note or [], symbols_all) if s == "+" and a}
    return {
        "rr": np.diff(samples) / hdr.fs * 1000.0,
        "end_symbol": symbols[1:],
        "start_symbol": symbols[:-1],
        "rhythms": rhythms,
    }


def mitdb_ectopy_benchmark(records=MITDB_RECORDS, local_dir: str | Path | None = None) -> pd.DataFrame:
    """Detection of expert-annotated ectopic beats and RMSSD error against the
    NN-only reference (successive differences of normal-to-normal intervals
    that are adjacent and bounded by normal beats)."""
    rows = []
    for rec in records:
        if rec in MITDB_PACED:
            continue
        d = physionet_rr(rec, "mitdb", "atr", local_dir)
        if any(r.startswith(("(AFIB", "(AFL")) for r in d["rhythms"]):
            continue  # HRV is not meaningful in atrial fibrillation / flutter
        rr, end_sym, start_sym = d["rr"], d["end_symbol"], d["start_symbol"]
        nn = np.array([e in NORMAL and s in NORMAL for e, s in zip(end_sym, start_sym)])
        ref_pairs = nn[1:] & nn[:-1]
        ref = float(np.sqrt(np.mean(np.diff(rr)[ref_pairs] ** 2)))
        ectopic_beats = np.flatnonzero([e in ECTOPIC for e in end_sym])  # interval ending at the ectopic beat
        res = clean_rr(rr)
        flagged_raw = (res.raw_labels != OK) | res.raw_masked
        hit = [flagged_raw[i] or (i + 1 < len(rr) and flagged_raw[i + 1]) for i in ectopic_beats]
        far = np.ones(len(rr), bool)
        for i in ectopic_beats:
            far[max(i - 2, 0) : i + 3] = False
        b = res.beats
        valid = (b["status"] != MASKED).to_numpy()
        v = b["rr_ms"].to_numpy()
        est = float(np.sqrt(np.mean(np.diff(v)[valid[1:] & valid[:-1]] ** 2)))
        rows.append(
            {
                "record": rec,
                "beats": len(rr) + 1,
                "ectopic_beats": len(ectopic_beats),
                "sensitivity": float(np.mean(hit)) if hit else np.nan,
                "false_flag_rate_normal": float(flagged_raw[far & nn].mean()),
                "rmssd_reference": ref,
                "rmssd_proposed": est,
                "rmssd_uncleaned": float(np.sqrt(np.mean(np.diff(rr) ** 2))),
                "masked_frac": float(res.raw_masked.mean()),
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Command line


def main(argv: list[str] | None = None) -> int:
    from .io import read_strap_csv

    p = argparse.ArgumentParser(prog="python -m hrvcompare.benchmark", description=__doc__.split("\n\n")[0])
    p.add_argument("--rri", type=Path, default=Path("data/raw/strap_rri_20260417.csv"), help="recording for real ground truth and the noise library")
    p.add_argument("--out", type=Path, default=Path("paper/generated/benchmark"))
    p.add_argument("--physionet", action="store_true", help="also run the MIT-BIH ectopy benchmark (needs physionet.org)")
    p.add_argument("--mitdb-dir", type=Path, help="local directory with MIT-BIH .atr/.hea files instead of downloading")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    rec = read_strap_csv(args.rri)
    bases = {"real": real_clean_segments(rec.rr_ms), "synthetic": synthetic_segments()}
    library = noise_library(rec.rr_ms)
    df = run_noise_benchmark(bases, library, seeds={"real": 5, "synthetic": 3})
    df.to_csv(args.out / "noise_windows.csv", index=False, float_format="%.4g")
    summarize(df).to_csv(args.out / "noise_summary.csv", index=False, float_format="%.4g")
    base = run_clean_baseline(bases)
    base.to_csv(args.out / "clean_windows.csv", index=False, float_format="%.4g")
    summarize(base, by=("base", "method")).to_csv(args.out / "clean_summary.csv", index=False, float_format="%.4g")
    if args.physionet or args.mitdb_dir:
        records = MITDB_RECORDS if args.physionet else sorted(p.stem for p in args.mitdb_dir.glob("*.atr"))
        mitdb_ectopy_benchmark(records, local_dir=args.mitdb_dir).to_csv(args.out / "mitdb_ectopy.csv", index=False, float_format="%.4g")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
