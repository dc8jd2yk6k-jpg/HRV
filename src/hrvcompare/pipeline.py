"""Run every step once and collect the results the report and figures use."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import ArtifactConfig, CleanResult, clean_rr
from .compare import (
    block_agreement,
    epoch_agreement,
    lag_scan,
    matched_table,
    night_level,
    sensitivity_grid,
    watch_noise,
    window_length_scan,
)
from .io import StrapRecording, clock_to_seconds, read_strap_csv
from .metrics import GATES, regular_windows, resting_hr, time_domain, window_metrics
from .timeline import HYPOTHESES, wall_end_times
from .synthetic import validation_table
from .watch import WatchHRV, digitize_hrv_chart


@dataclass
class Inputs:
    rri_csv: Path
    watch_png: Path
    sleep_start: str  # "HH:MM" as shown on the watch chart
    sleep_end: str
    watch_avg: float | None = None
    watch_range: tuple[float, float] | None = None
    watch_rhr: float | None = None
    watch_interval_min: float = 10.0
    hypothesis: str = "H1"
    gate: str = "default"
    convention: str = "ending"
    window_min: float = 10.0
    artifact_config: ArtifactConfig = field(default_factory=ArtifactConfig)


@dataclass
class Results:
    inputs: Inputs
    rec: StrapRecording
    clean: CleanResult
    watch: WatchHRV
    walls: dict[str, np.ndarray]
    sleep_s: tuple[float, float]
    matched: pd.DataFrame
    night: dict
    epoch: dict
    blocks: dict
    blocks_table: pd.DataFrame
    grid: pd.DataFrame
    lags: pd.DataFrame
    window_scan: pd.DataFrame
    noise: dict
    synthetic: pd.DataFrame
    rhr: dict
    rhr_minutes: pd.DataFrame
    windows5: pd.DataFrame
    strap_summary: dict
    device_flags: dict

    @property
    def wall(self) -> np.ndarray:
        return self.walls[self.inputs.hypothesis]


def _device_flag_overlap(rec: StrapRecording, clean: CleanResult) -> dict:
    dev = rec.unusual.astype(bool)
    ours = (clean.raw_labels != "ok") | clean.raw_masked
    return {
        "device_flagged": int(dev.sum()),
        "ours_flagged_or_masked": int(ours.sum()),
        "device_flags_we_also_flag": float(ours[dev].mean()) if dev.any() else np.nan,
        "our_flags_device_also_flags": float(dev[ours].mean()) if ours.any() else np.nan,
    }


def run(inp: Inputs) -> Results:
    rec = read_strap_csv(inp.rri_csv)
    clean = clean_rr(rec.rr_ms, inp.artifact_config)
    t0 = clock_to_seconds(inp.sleep_start, ref_s=rec.start_s - 3600)
    t1 = clock_to_seconds(inp.sleep_end, ref_s=t0)
    watch = digitize_hrv_chart(
        inp.watch_png,
        t0,
        t1,
        interval_min=inp.watch_interval_min,
        reported_avg=inp.watch_avg,
        normal_range=inp.watch_range,
    )
    walls = {h: wall_end_times(rec, clean, h) for h in HYPOTHESES}
    wall = walls[inp.hypothesis]
    gate = GATES[inp.gate]

    matched = matched_table(clean, wall, watch, inp.window_min * 60, inp.convention, gate)
    blocks, blocks_table = block_agreement(matched)

    # Strap-only overnight summary over the sleep window it covers.
    s_end = min(t1, float(wall[-1]))
    starts, ends = regular_windows(t0, s_end, 300.0)
    windows5 = window_metrics(clean.beats, wall, starts, ends, gate, frequency=True)
    v5 = windows5[windows5["valid"]]
    in_sleep = (wall >= t0) & (wall < s_end)
    pooled = time_domain(clean.beats["rr_ms"].to_numpy()[in_sleep], clean.valid[in_sleep])
    strap_summary = {
        "window": "5 min",
        "n_windows": int(len(windows5)),
        "n_valid": int(len(v5)),
        **{f"mean_{k}": float(v5[k].mean()) for k in ("mean_nn", "sdnn", "rmssd", "pnn50", "hr", "lf", "hf", "lf_hf")},
        "pooled_rmssd": pooled["rmssd"],
        "pooled_sdnn": pooled["sdnn"],
        "valid_nn_time_frac": float(
            clean.beats["rr_ms"].to_numpy()[in_sleep & clean.valid].sum() / 1000 / (s_end - t0)
        ),
        "covered_until_s": s_end,
    }
    rhr, rhr_minutes = resting_hr(clean.beats, wall, t0, s_end)
    if inp.watch_rhr is not None:
        rhr["watch_rhr"] = inp.watch_rhr

    return Results(
        inputs=inp,
        rec=rec,
        clean=clean,
        watch=watch,
        walls=walls,
        sleep_s=(t0, t1),
        matched=matched,
        night=night_level(matched, watch),
        epoch=epoch_agreement(matched),
        blocks=blocks,
        blocks_table=blocks_table,
        grid=sensitivity_grid(rec, clean, watch),
        lags=lag_scan(rec, clean, watch, hypothesis=inp.hypothesis, gate=gate),
        window_scan=window_length_scan(clean, wall, watch, gate=gate),
        noise=watch_noise(matched, len(watch.point_ms)),
        synthetic=validation_table(),
        rhr=rhr,
        rhr_minutes=rhr_minutes,
        windows5=windows5,
        strap_summary=strap_summary,
        device_flags=_device_flag_overlap(rec, clean),
    )
