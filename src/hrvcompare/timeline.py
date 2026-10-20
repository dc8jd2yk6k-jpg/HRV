"""Map the strap's beat clock onto wall-clock time.

The strap's timestamps are ``start + cumsum(RR)``. Whenever the strap stops
emitting intervals (contact loss, radio dropout) that time silently vanishes,
so the beat clock falls behind the wall clock. Here the header duration
(7:21:03) exceeds the summed intervals by ~42 min. Where that time went cannot
be recovered from the RR data alone, so each hypothesis is carried through the
analysis:

H1  the strap's own clock taken as recorded: the RR stream simply ends early
    and the app session ran on empty until stopped. Needs no assumption about
    where beats went, so the figures use it for display.
H2  the missing time is spread uniformly through the night (linear stretch).
H3  the missing time is hidden inside the masked artifact episodes, in
    proportion to their duration (contact loss is where beats go missing).
"""

from __future__ import annotations

import numpy as np

from .artifacts import CleanResult, MASKED
from .io import StrapRecording

HYPOTHESES = {
    "H1": "strap clock taken as recorded; the RR stream ended early and the session ran on empty",
    "H2": "missing time spread uniformly (linear stretch)",
    "H3": "missing time inside masked artifact episodes",
}


def wall_end_times(rec: StrapRecording, clean: CleanResult, hypothesis: str = "H1", lag_s: float = 0.0) -> np.ndarray:
    """Wall-clock end time (seconds after start-date midnight) of every
    interval in ``clean.beats``."""
    t = clean.beats["t_end_s"].to_numpy()
    total = t[-1]
    deficit = max(rec.clock_deficit_s, 0.0)
    if hypothesis == "H1" or deficit == 0:
        wall = t
    elif hypothesis == "H2":
        wall = t * (total + deficit) / total
    elif hypothesis == "H3":
        rr = clean.beats["rr_ms"].to_numpy()
        masked_cum = np.cumsum(np.where(clean.beats["status"].to_numpy() == MASKED, rr, 0.0)) / 1000
        wall = t + deficit * masked_cum / masked_cum[-1] if masked_cum[-1] > 0 else t
    else:
        raise ValueError(f"unknown hypothesis {hypothesis!r}; expected one of {list(HYPOTHESES)}")
    return rec.start_s + wall + lag_s


def coverage_span(rec: StrapRecording, clean: CleanResult, hypothesis: str) -> tuple[float, float]:
    w = wall_end_times(rec, clean, hypothesis)
    return rec.start_s, float(w[-1])
