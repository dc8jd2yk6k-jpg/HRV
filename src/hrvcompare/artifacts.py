"""Beat-level artifact detection and correction for R-R interval series.

Detection follows Lipponen & Tarvainen (2019), "A robust algorithm for heart
rate variability time series artefact correction using novel beat
classification", J Med Eng Technol 43(3):173-181 -- the Kubios "automatic"
method, also ported in NeuroKit2's ``signal_fixpeaks(method="kubios")``.
Three changes make it usable on a noisy consumer strap:

* thresholds are computed NaN-aware, excluding beats that are out of the
  physiological range or already masked, so noise bursts don't inflate them;
* a fixed safety net flags any beat deviating more than 30 % from a robust
  local median ("deviant"). In stretches where most beats are noise the
  adaptive Kubios thresholds grow so wide that obvious garbage passes;
* dense clusters of flagged beats ("noise bursts") are *masked*, not
  corrected. Correcting a 9-beat run of garbage would make up data.

Isolated artifacts are corrected in the beat-time domain, so the total
elapsed time is preserved exactly:

* extra beat  -> merge with the neighbour whose sum is closest to the median
* missed beat -> split into k equal intervals
* ectopic / long-short -> move the shared R-peak to the midpoint of the pair
  of intervals whose mean is closest to the local median

Nothing is ever deleted. Masked intervals keep their raw duration on the time
axis, so downstream windows know exactly how much of their time is unusable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

OK = "ok"
CORRECTED = "corrected"
MASKED = "masked"

DETECTION_LABELS = ("ectopic", "missed", "extra", "longshort", "out_of_range", "deviant")
PRIORITY = ("extra", "missed", "ectopic", "longshort", "out_of_range", "deviant")


@dataclass
class ArtifactConfig:
    # Physiological range for a resting / sleep recording (150-30 bpm). Lower
    # rr_min_ms for exercise data.
    rr_min_ms: float = 400.0
    rr_max_ms: float = 2000.0
    # Safety net for dense noise, where in-range garbage inflates the adaptive
    # thresholds: beats deviating more than this from the local median are
    # always flagged and kept out of the threshold statistics.
    max_dev_frac: float = 0.30
    alpha: float = 5.2
    c1: float = 0.13
    c2: float = 0.17
    window_width: int = 91
    medfilt_order: int = 11
    burst_window: int = 10  # beats
    burst_min_bad: int = 3  # flagged beats within burst_window that make a burst
    burst_margin: int = 2  # beats masked either side of a burst
    max_iter: int = 4


@dataclass
class Detection:
    labels: np.ndarray  # per interval: "ok" or one of DETECTION_LABELS
    medrr: np.ndarray
    th2: np.ndarray
    drrs: np.ndarray
    mrrs: np.ndarray


@dataclass
class CleanResult:
    beats: pd.DataFrame  # t_end_s, rr_ms, status, kind, raw_first, raw_last
    raw_labels: np.ndarray  # detection label per raw interval
    raw_masked: np.ndarray  # raw interval ends up masked
    config: ArtifactConfig = field(default_factory=ArtifactConfig)

    @property
    def valid(self) -> np.ndarray:
        return (self.beats["status"] != MASKED).to_numpy()

    def summary(self) -> dict:
        b = self.beats
        rr = b["rr_ms"].to_numpy()
        status = b["status"].to_numpy()
        kinds = b["kind"].to_numpy()
        out = {
            "n_raw_intervals": int(len(self.raw_labels)),
            "n_clean_intervals": int(len(b)),
            "total_time_s": float(rr.sum() / 1000),
            "n_ok": int((status == OK).sum()),
            "n_corrected": int((status == CORRECTED).sum()),
            "n_masked": int((status == MASKED).sum()),
            "masked_time_s": float(rr[status == MASKED].sum() / 1000),
            "masked_time_frac": float(rr[status == MASKED].sum() / rr.sum()),
            "corrected_frac_of_valid": float((status == CORRECTED).sum() / max((status != MASKED).sum(), 1)),
        }
        for lab in DETECTION_LABELS:
            out[f"raw_flagged_{lab}"] = int((self.raw_labels == lab).sum())
        for k in DETECTION_LABELS:
            out[f"corrected_{k}"] = int(((status == CORRECTED) & (kinds == k)).sum())
        for k in np.unique(kinds[status == MASKED]):
            out[f"masked_{k}"] = int(((status == MASKED) & (kinds == k)).sum())
        return out


def _rolling_qd(x: np.ndarray, width: int) -> np.ndarray:
    """NaN-aware centred rolling quartile deviation."""
    r = pd.Series(x).rolling(width, center=True, min_periods=max(5, width // 10))
    qd = ((r.quantile(0.75) - r.quantile(0.25)) / 2).to_numpy()
    if np.all(np.isnan(qd)):
        return np.full(len(x), np.nanmedian(np.abs(x)) if np.any(~np.isnan(x)) else np.nan)
    return pd.Series(qd).ffill().bfill().to_numpy()


def _neighbour_reduce(drrs: np.ndarray, offsets: tuple[int, int], pick_max: np.ndarray) -> np.ndarray:
    pad = 2
    p = np.pad(drrs, pad, mode="reflect")
    a = p[pad + offsets[0] : pad + offsets[0] + len(drrs)]
    b = p[pad + offsets[1] : pad + offsets[1] + len(drrs)]
    return np.where(pick_max, np.maximum(a, b), np.minimum(a, b))


def detect_artifacts(rr_ms: np.ndarray, cfg: ArtifactConfig | None = None, exclude: np.ndarray | None = None) -> Detection:
    """Classify every interval as ok / ectopic / missed / extra / longshort /
    out_of_range / deviant. *exclude* marks intervals left out of the threshold and
    median statistics (e.g. already-masked noise)."""
    cfg = cfg or ArtifactConfig()
    rr = np.asarray(rr_ms, dtype=float)
    n = len(rr)
    oor = (rr < cfg.rr_min_ms) | (rr > cfg.rr_max_ms)
    skip = oor | np.isnan(rr)
    if exclude is not None:
        skip |= exclude

    def local_median(skip_mask, order):
        m = pd.Series(np.where(skip_mask, np.nan, rr)).rolling(order, center=True, min_periods=1).median()
        return m.interpolate(limit_direction="both").to_numpy()

    with np.errstate(invalid="ignore"):
        # A wide median is robust inside noise bursts; it defines "deviant".
        deviant = np.abs(rr / local_median(skip, 4 * cfg.medfilt_order + 1) - 1) > cfg.max_dev_frac
    skip |= deviant
    rr_stats = np.where(skip, np.nan, rr)

    drr = np.diff(rr, prepend=rr[0])
    th1 = cfg.alpha * _rolling_qd(np.abs(np.diff(rr_stats, prepend=np.nan)), cfg.window_width)
    drrs = drr / th1

    medrr = local_median(skip, cfg.medfilt_order)
    mrr = rr - medrr
    mrr = np.where(mrr < 0, 2 * mrr, mrr)
    th2 = cfg.alpha * _rolling_qd(np.abs(np.where(skip, np.nan, mrr)), cfg.window_width)
    mrrs = mrr / th2

    s12 = _neighbour_reduce(drrs, (-1, 1), pick_max=drrs > 0)
    s22 = _neighbour_reduce(drrs, (1, 2), pick_max=drrs < 0)

    labels = np.full(n, OK, dtype=object)

    def set_label(j: int, lab: str) -> None:
        if labels[j] == OK:
            labels[j] = lab

    c1, c2 = cfg.c1, cfg.c2
    with np.errstate(invalid="ignore"):
        for i in range(n - 2):
            d = drrs[i]
            if not (abs(d) > 1 or abs(mrrs[i]) > 3):
                continue
            if (d > 1 and s12[i] < -c1 * d - c2) or (d < -1 and s12[i] > -c1 * d + c2):
                set_label(i, "ectopic")
                continue
            candidates = [i]
            if abs(drrs[i + 1]) < abs(drrs[i + 2]):
                candidates.append(i + 1)
            for j in candidates:
                long_ = drrs[j] > 1 and s22[j] < -1
                short = drrs[j] < -1 and s22[j] > 1
                if not (long_ or short or abs(mrrs[j]) > 3):
                    continue
                k = max(int(round(rr[j] / medrr[j])), 2)
                missed = abs(rr[j] / k - medrr[j]) < th2[j]
                extra = j + 1 < n and abs(rr[j] + rr[j + 1] - medrr[j]) < th2[j]
                if short and extra:
                    set_label(j, "extra")
                elif long_ and missed:
                    set_label(j, "missed")
                else:
                    set_label(j, "longshort")

    labels[(labels == OK) & oor] = "out_of_range"
    labels[(labels == OK) & deviant] = "deviant"
    return Detection(labels=labels, medrr=medrr, th2=th2, drrs=drrs, mrrs=mrrs)


def burst_mask(labels: np.ndarray, cfg: ArtifactConfig | None = None) -> np.ndarray:
    """Mask runs where flagged beats are too dense to correct individually."""
    cfg = cfg or ArtifactConfig()
    bad = labels != OK
    n = len(bad)
    w = min(cfg.burst_window, n)
    mask = np.zeros(n, dtype=bool)
    cs = np.concatenate([[0], np.cumsum(bad)])
    for s in np.flatnonzero(cs[w:] - cs[:-w] >= cfg.burst_min_bad):
        idx = np.flatnonzero(bad[s : s + w])
        mask[s + idx[0] : s + idx[-1] + 1] = True
    if cfg.burst_margin and mask.any():
        k = np.ones(2 * cfg.burst_margin + 1)
        mask = np.convolve(mask.astype(float), k, mode="same") > 0
    return mask


def _plan_corrections(rr: np.ndarray, det: Detection, mask: np.ndarray) -> tuple[list[tuple], np.ndarray]:
    """Decide one correction per isolated flagged interval.

    Returns (ops, extra_mask). Each op is (kind, label, a, b, k): "merge" a+b,
    "split" a into k, "equalize" a and b. Anything that cannot be corrected
    confidently is added to extra_mask instead.
    """
    n = len(rr)
    labels = det.labels
    extra_mask = np.zeros(n, dtype=bool)
    covered = np.zeros(n, dtype=bool)  # already part of a planned correction
    ops: list[tuple] = []

    def usable(j: int) -> bool:
        return 0 <= j < n and not (mask[j] or covered[j] or extra_mask[j])

    # Most specific explanations first, so e.g. both halves of a split beat are
    # explained by one merge instead of two competing corrections.
    flagged = np.flatnonzero((labels != OK) & ~mask)
    order = sorted(flagged, key=lambda j: (PRIORITY.index(labels[j]), j))
    for j in order:
        if covered[j]:
            continue
        lab = labels[j]
        med, th = det.medrr[j], det.th2[j]
        op = None
        if lab in ("extra", "out_of_range", "deviant") and rr[j] < med:
            pairs = [(a, a + 1) for a in (j - 1, j) if usable(a) and usable(a + 1)]
            pairs = [p for p in pairs if abs(rr[p[0]] + rr[p[1]] - med) < th]
            if pairs:
                a, b = min(pairs, key=lambda p: abs(rr[p[0]] + rr[p[1]] - med))
                op = ("merge", "extra" if lab == "extra" else lab, a, b, 1)
        if op is None and lab in ("missed", "out_of_range", "deviant") and rr[j] > med:
            k = max(int(round(rr[j] / med)), 2)
            if abs(rr[j] / k - med) < th:
                op = ("split", "missed" if lab == "missed" else lab, j, j, k)
        if op is None and lab in ("ectopic", "longshort", "extra", "missed"):
            pairs = [(a, a + 1) for a in (j - 1, j) if usable(a) and usable(a + 1)]
            if pairs:
                a, b = min(pairs, key=lambda p: abs((rr[p[0]] + rr[p[1]]) / 2 - med))
                op = ("equalize", lab, a, b, 2)
        if op is None:
            extra_mask[j] = True
            continue
        ops.append(op)
        covered[op[2] : op[3] + 1] = True
    return ops, extra_mask


def _emit(rr: np.ndarray, labels: np.ndarray, burst: np.ndarray, extra_mask: np.ndarray, ops: list[tuple]) -> pd.DataFrame:
    n = len(rr)
    op_at = {op[2]: op for op in ops}
    out_rr, out_status, out_kind, out_first, out_last = [], [], [], [], []

    def emit(v, status, kind, a, b):
        out_rr.append(v)
        out_status.append(status)
        out_kind.append(kind)
        out_first.append(a)
        out_last.append(b)

    j = 0
    while j < n:
        if burst[j] or extra_mask[j]:
            kind = "burst" if burst[j] else (labels[j] if labels[j] != OK else "conflict")
            emit(rr[j], MASKED, kind, j, j)
            j += 1
            continue
        op = op_at.get(j)
        if op is None:
            emit(rr[j], OK, "", j, j)
            j += 1
            continue
        kind, lab, a, b, k = op
        if kind == "merge":
            emit(rr[a] + rr[b], CORRECTED, lab, a, b)
        elif kind == "split":
            for _ in range(k):
                emit(rr[a] / k, CORRECTED, lab, a, a)
        else:
            m = (rr[a] + rr[b]) / 2
            emit(m, CORRECTED, lab, a, a)
            emit(m, CORRECTED, lab, b, b)
        j = b + 1

    rr_out = np.asarray(out_rr, dtype=float)
    return pd.DataFrame(
        {
            "t_end_s": np.cumsum(rr_out) / 1000.0,
            "rr_ms": rr_out,
            "status": out_status,
            "kind": out_kind,
            "raw_first": out_first,
            "raw_last": out_last,
        }
    )


def clean_rr(rr_ms: np.ndarray, cfg: ArtifactConfig | None = None) -> CleanResult:
    """Detect, mask and correct artifacts. Returns the corrected interval table;
    the sum of ``rr_ms`` equals the raw sum exactly."""
    cfg = cfg or ArtifactConfig()
    rr = np.asarray(rr_ms, dtype=float)

    # Detect, mask bursts, re-detect with bursts excluded from the statistics
    # until the mask stops growing.
    mask = np.zeros(len(rr), dtype=bool)
    for _ in range(cfg.max_iter):
        det = detect_artifacts(rr, cfg, exclude=mask)
        new_mask = mask | burst_mask(det.labels, cfg)
        if new_mask.sum() == mask.sum():
            break
        mask = new_mask

    ops, extra_mask = _plan_corrections(rr, det, mask)
    beats = _emit(rr, det.labels, mask, extra_mask, ops)

    # Residual check: anything still flagged after correction is masked.
    rr_c = beats["rr_ms"].to_numpy()
    masked = (beats["status"] == MASKED).to_numpy()
    resid = detect_artifacts(np.where(masked, np.nan, rr_c), cfg, exclude=masked).labels
    resid_idx = (resid != OK) & ~masked & ~np.isnan(rr_c)
    if resid_idx.any():
        beats.loc[resid_idx, "status"] = MASKED
        beats.loc[resid_idx, "kind"] = "residual"

    raw_masked = np.zeros(len(rr), dtype=bool)
    m = beats["status"] == MASKED
    for a, b in zip(beats.loc[m, "raw_first"], beats.loc[m, "raw_last"]):
        raw_masked[a : b + 1] = True
    return CleanResult(beats=beats, raw_labels=det.labels, raw_masked=raw_masked, config=cfg)
