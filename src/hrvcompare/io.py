"""Read the chest-strap RRI export.

The export is a Polar-Flow-style CSV: a few header/value line pairs with
session and device metadata, then a beat table ``Time,RRI,HR unusual-rate,
unusual`` with one row per R-R interval.  ``Time`` is ``HH:MM:SS mmm`` and is
nothing more than ``start + cumsum(RRI)`` -- the strap has no independent
clock, which matters for alignment (see :mod:`hrvcompare.timeline`).
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

BEAT_TABLE_HEADER = "Time,RRI"


@dataclass
class StrapRecording:
    rr_ms: np.ndarray  # raw R-R intervals in ms, never modified
    device_time_s: np.ndarray  # device "Time" column, seconds after start-date midnight
    unusual: np.ndarray  # device's own per-beat "unusual" flag (0/1)
    unusual_rate: np.ndarray  # device's running "HR unusual-rate"
    start: datetime  # session start (time of the R-peak preceding the first interval)
    duration_s: float  # session duration from the header
    metadata: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.rr_ms)

    @property
    def start_s(self) -> float:
        """Session start as seconds after midnight of the start date."""
        midnight = self.start.replace(hour=0, minute=0, second=0, microsecond=0)
        return (self.start - midnight).total_seconds()

    @property
    def rr_end_s(self) -> np.ndarray:
        """End of each interval in seconds since session start (strap clock)."""
        return np.cumsum(self.rr_ms) / 1000.0

    @property
    def rr_total_s(self) -> float:
        return float(self.rr_ms.sum() / 1000.0)

    @property
    def clock_deficit_s(self) -> float:
        """Header duration minus the summed RR intervals (time with no beats)."""
        return self.duration_s - self.rr_total_s

    def clock_residual_ms(self) -> np.ndarray:
        """Device Time minus (start + cumsum RR); ~0 if Time is derived from RR."""
        return (self.device_time_s - self.start_s - self.rr_end_s) * 1000.0


def _parse_hms(text: str) -> float:
    """``HH:MM:SS`` or ``HH:MM`` -> seconds."""
    parts = [int(p) for p in text.strip().split(":")]
    if len(parts) == 2:
        parts.append(0)
    h, m, s = parts
    return h * 3600 + m * 60 + s


def _parse_device_time(text: str) -> float:
    hms, _, ms = text.strip().partition(" ")
    return _parse_hms(hms) + (int(ms) / 1000.0 if ms else 0.0)


def _strip_empty(row: list[str]) -> list[str]:
    while row and row[-1] == "":
        row = row[:-1]
    return row


def read_strap_csv(path: str | Path) -> StrapRecording:
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    try:
        table_at = next(i for i, ln in enumerate(lines) if ln.startswith(BEAT_TABLE_HEADER))
    except StopIteration:
        raise ValueError(f"{path}: no '{BEAT_TABLE_HEADER}' beat table found") from None

    # Metadata comes as consecutive header/value line pairs.
    metadata: dict[str, str] = {}
    head = list(csv.reader(lines[:table_at]))
    for keys, values in zip(head[0::2], head[1::2]):
        for k, v in zip(_strip_empty(keys), values):
            if k:
                metadata[k.strip()] = v.strip()

    beats = pd.read_csv(io.StringIO("\n".join(lines[table_at:])))
    beats.columns = [c.strip() for c in beats.columns]
    rr = beats["RRI"].to_numpy(dtype=float)

    t = beats["Time"].map(_parse_device_time).to_numpy(dtype=float)
    # Unwrap midnight roll-overs.
    t = t + 86400.0 * np.concatenate([[0], np.cumsum(np.diff(t) < -43200)])

    start = datetime.strptime(f"{metadata['Date']} {metadata['Start time']}", "%Y-%m-%d %H:%M:%S")
    duration_s = _parse_hms(metadata["Duration"])

    return StrapRecording(
        rr_ms=rr,
        device_time_s=t,
        unusual=beats.get("unusual", pd.Series(np.zeros(len(rr)))).to_numpy(dtype=int),
        unusual_rate=beats.get("HR unusual-rate", pd.Series(np.zeros(len(rr)))).to_numpy(dtype=float),
        start=start,
        duration_s=float(duration_s),
        metadata=metadata,
    )


def seconds_to_clock(s: float) -> str:
    s = float(s) % 86400
    return f"{int(s // 3600):02d}:{int(s % 3600 // 60):02d}:{int(s % 60):02d}"


def clock_to_seconds(text: str, ref_s: float | None = None) -> float:
    """Parse ``HH:MM[:SS]`` to seconds after midnight; if *ref_s* is given and the
    time is more than 12 h before it, it is taken to be on the next day."""
    s = float(_parse_hms(text))
    if ref_s is not None and s < ref_s - 43200:
        s += 86400
    return s


def wall_datetime(rec: StrapRecording, wall_s: np.ndarray | float):
    midnight = rec.start.replace(hour=0, minute=0, second=0, microsecond=0)
    if np.ndim(wall_s) == 0:
        return midnight + timedelta(seconds=float(wall_s))
    return pd.to_datetime(midnight) + pd.to_timedelta(np.asarray(wall_s), unit="s")
