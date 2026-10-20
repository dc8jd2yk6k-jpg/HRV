"""Synthetic sleep-like RR series with known artifacts, for validating the
cleaning algorithm against ground truth."""

from __future__ import annotations

import numpy as np


def synthetic_rr(
    n: int = 3000,
    mean_ms: float = 1000.0,
    rsa_ms: float = 35.0,
    resp_hz: float = 0.25,
    lf_ms: float = 20.0,
    noise_ms: float = 12.0,
    seed: int = 0,
) -> np.ndarray:
    """Clean NN intervals: respiratory sinus arrhythmia + a 0.1 Hz LF wave +
    white noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) * mean_ms / 1000.0
    phase = rng.uniform(0, 2 * np.pi)
    return (
        mean_ms
        + rsa_ms * np.sin(2 * np.pi * resp_hz * t)
        + lf_ms * np.sin(2 * np.pi * 0.1 * t + phase)
        + rng.normal(0, noise_ms, n)
    )


def inject_artifacts(
    rr: np.ndarray,
    rate: float = 0.02,
    kinds: tuple[str, ...] = ("ectopic", "missed", "extra"),
    bursts: int = 0,
    burst_len: tuple[int, int] = (8, 15),
    seed: int = 1,
) -> tuple[np.ndarray, list[tuple[str, int, int]]]:
    """Corrupt a clean series. Returns (rr_bad, truth) where truth lists
    (kind, first, last) index ranges in rr_bad. All corruptions preserve the
    total elapsed time, as a real detector does:

    * ectopic: premature beat -> short coupling interval + compensatory pause
    * missed:  an R-peak not detected -> two intervals merged
    * extra:   a spurious detection -> one interval split in two
    * burst:   a stretch replaced by random detections (contact noise)
    """
    rng = np.random.default_rng(seed)
    n = len(rr)
    events = {}
    # Space events at least 12 beats apart so they stay "isolated".
    candidates = np.arange(20, n - 20, 12)
    pick = rng.choice(candidates, size=min(int(rate * n), len(candidates)), replace=False)
    for j in pick:
        events[int(j)] = str(rng.choice(kinds))
    if bursts:
        free = [c for c in np.arange(40, n - 60, 60) if all(abs(c - e) > 25 for e in events)]
        for j in rng.choice(free, size=min(bursts, len(free)), replace=False):
            events[int(j)] = "burst"

    out: list[float] = []
    truth: list[tuple[str, int, int]] = []
    j = 0
    while j < n:
        kind = events.get(j)
        if kind is None:
            out.append(rr[j])
            j += 1
        elif kind == "ectopic":
            s = 0.65 * rr[j]
            truth.append(("ectopic", len(out), len(out) + 1))
            out += [s, rr[j] - s + rr[j + 1]]
            j += 2
        elif kind == "missed":
            truth.append(("missed", len(out), len(out)))
            out.append(rr[j] + rr[j + 1])
            j += 2
        elif kind == "extra":
            a = rng.uniform(0.3, 0.6) * rr[j]
            truth.append(("extra", len(out), len(out) + 1))
            out += [a, rr[j] - a]
            j += 1
        else:  # burst
            k = int(rng.integers(burst_len[0], burst_len[1] + 1))
            total = rr[j : j + k].sum()
            pieces = []
            while total > 0:
                p = min(rng.uniform(200, 1400), total)
                pieces.append(p)
                total -= p
            truth.append(("burst", len(out), len(out) + len(pieces) - 1))
            out += pieces
            j += k
    return np.asarray(out), truth


def validation_table(seeds=(0, 1, 2), rate: float = 0.02, bursts: int = 3):
    """Run the cleaner on corrupted synthetic series with known truth."""
    import pandas as pd

    from .artifacts import OK, clean_rr
    from .metrics import time_domain

    rows = []
    for seed in seeds:
        rr = synthetic_rr(seed=seed)
        bad, truth = inject_artifacts(rr, rate=rate, bursts=bursts, seed=seed + 10)
        clean = clean_rr(bad)
        b = clean.beats
        touched = np.zeros(len(bad), dtype=bool)
        for a, z, st in zip(b["raw_first"], b["raw_last"], b["status"]):
            if st != OK:
                touched[a : z + 1] = True
        row = {"seed": seed}
        for kind in ("ectopic", "missed", "extra", "burst"):
            hits = [touched[a : z + 1].any() for k, a, z in truth if k == kind]
            row[f"n_{kind}"] = len(hits)
            row[f"detected_{kind}"] = float(np.mean(hits)) if hits else np.nan
        row["burst_masked_frac"] = float(np.mean([clean.raw_masked[a : z + 1].mean() for k, a, z in truth if k == "burst"]))
        row["rmssd_true"] = time_domain(rr, np.ones(len(rr), bool))["rmssd"]
        row["rmssd_uncleaned"] = time_domain(bad, np.ones(len(bad), bool))["rmssd"]
        row["rmssd_cleaned"] = time_domain(b["rr_ms"].to_numpy(), clean.valid)["rmssd"]
        clean0 = clean_rr(rr)
        row["false_flag_frac_clean_series"] = 1 - clean0.summary()["n_ok"] / len(clean0.beats)
        rows.append(row)
    return pd.DataFrame(rows)
