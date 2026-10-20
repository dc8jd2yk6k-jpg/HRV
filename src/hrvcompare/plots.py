"""Report figures (static PNG, light surface).

Colour roles follow one rule: blue = chest strap, orange = watch, greys =
context (masked data, gates, reference lines). Text always uses ink tones.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, MultipleLocator  # noqa: E402

from .artifacts import CORRECTED, MASKED  # noqa: E402
from .io import seconds_to_clock  # noqa: E402
from .metrics import GATES  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
WASH = "#f0efec"
STRAP = "#2a78d6"
STRAP_LIGHT = "#86b6ef"
WATCH = "#eb6834"
DPI = 150


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK2,
            "ytick.labelcolor": INK2,
            "ytick.left": False,
            "text.color": INK,
            "font.size": 9,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
        }
    )


def _hours(s):
    return np.asarray(s, dtype=float) / 3600.0


_clock = FuncFormatter(lambda h, _: seconds_to_clock(h * 3600)[:5])


def _time_axis(ax, major_h: float = 1.0) -> None:
    ax.xaxis.set_major_formatter(_clock)
    ax.xaxis.set_major_locator(MultipleLocator(major_h))
    ax.grid(axis="x", visible=False)


def _dots(ax, x, y, color, size=26, label=None, zorder=3):
    ax.scatter(x, y, s=size, color=color, edgecolors=SURFACE, linewidths=1.2, zorder=zorder, label=label)


def _masked_spans(beats, wall, merge_gap_s: float = 10.0) -> list[tuple[float, float]]:
    masked = (beats["status"] == MASKED).to_numpy()
    starts = wall - beats["rr_ms"].to_numpy() / 1000
    spans: list[list[float]] = []
    edges = np.diff(np.concatenate([[0], masked.astype(int), [0]]))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1):
        s, e = starts[a], wall[b]
        if spans and s - spans[-1][1] < merge_gap_s:
            spans[-1][1] = e
        else:
            spans.append([s, e])
    return [(s, e) for s, e in spans]


def _save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _pick_zoom(res, length_s: float = 180.0) -> tuple[float, float]:
    beats, wall = res.clean.beats, res.wall
    rr = beats["rr_ms"].to_numpy() / 1000
    masked = (beats["status"] == MASKED).to_numpy()
    corrected = (beats["status"] == CORRECTED).to_numpy()
    t0, t1 = res.sleep_s[0], min(res.sleep_s[1], wall[-1])
    best, best_score = (t0, t0 + length_s), -1.0
    for s in np.arange(t0, t1 - length_s, length_s / 2):
        m = (wall >= s) & (wall < s + length_s)
        if not m.any():
            continue
        mfrac = rr[m & masked].sum() / length_s
        score = corrected[m].sum() + (5 if 0.15 <= mfrac <= 0.5 else 0)
        if score > best_score:
            best, best_score = (s, s + length_s), score
    return best


def fig_tachogram(res, path: Path) -> Path:
    _style()
    beats, wall = res.clean.beats, res.wall
    raw_wall = res.rec.start_s + res.rec.rr_end_s
    fig, (ax, az) = plt.subplots(2, 1, figsize=(10, 6.2), gridspec_kw={"height_ratios": [1.6, 1], "hspace": 0.45})

    for s, e in _masked_spans(beats, wall):
        ax.axvspan(_hours(s), _hours(e), color=WASH, lw=0, zorder=0)
    ax.scatter(_hours(raw_wall), res.rec.rr_ms, s=1.0, color=MUTED, alpha=0.35, lw=0, rasterized=True)
    v = res.clean.valid
    ax.scatter(_hours(wall[v]), beats["rr_ms"].to_numpy()[v], s=1.0, color=STRAP, lw=0, rasterized=True)
    ax.set_ylim(150, 2300)
    ax.set_ylabel("R-R interval (ms)")
    _time_axis(ax)
    ax.set_title("Overnight R-R intervals: raw strap output vs cleaned NN series")
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=MUTED, ms=4, label="raw intervals"),
        plt.Line2D([], [], marker="o", ls="", color=STRAP, ms=4, label="cleaned NN (used)"),
        Patch(color=WASH, label="masked noise"),
    ]
    ax.legend(handles=handles, loc="upper left", ncol=3, bbox_to_anchor=(0, 1.0))

    z0, z1 = _pick_zoom(res)
    ax.axvspan(_hours(z0), _hours(z1), color=INK2, alpha=0.12, lw=0)
    rm = (raw_wall >= z0) & (raw_wall < z1)
    cm = (wall >= z0) & (wall < z1)
    for s, e in _masked_spans(beats.loc[cm], wall[cm], merge_gap_s=0):
        az.axvspan(s - z0, e - z0, color=WASH, lw=0, zorder=0)
    az.plot(raw_wall[rm] - z0, res.rec.rr_ms[rm], color=MUTED, lw=0.8, marker="o", ms=2.5, label="raw intervals")
    nn = np.where(v, beats["rr_ms"].to_numpy(), np.nan)
    az.plot(wall[cm] - z0, nn[cm], color=STRAP, lw=1.4, label="cleaned NN")
    corr = cm & (beats["status"] == CORRECTED).to_numpy()
    _dots(az, wall[corr] - z0, beats["rr_ms"].to_numpy()[corr], WATCH, size=30, label="corrected beat")
    _dots(az, wall[cm & v & ~corr] - z0, nn[cm & v & ~corr], STRAP, size=12)
    az.set_xlim(0, z1 - z0)
    az.set_xlabel(f"seconds after {seconds_to_clock(z0)}")
    az.set_ylabel("R-R interval (ms)")
    az.set_title(f"Zoom {seconds_to_clock(z0)[:5]}–{seconds_to_clock(z1)[:5]}: isolated artifacts corrected, noise bursts masked")
    handles = az.get_legend_handles_labels()[0] + [Patch(color=WASH, label="masked noise")]
    az.legend(handles=handles, loc="upper left", ncol=4, bbox_to_anchor=(0, 1.0))
    az.grid(axis="x", visible=False)
    return _save(fig, path)


def fig_quality(res, path: Path) -> Path:
    _style()
    m = res.matched
    gate = GATES[res.inputs.gate]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    x = _hours(m["point_s"])
    width = 0.7 * res.watch.interval_s / 3600
    colors = np.where(m["valid"], STRAP, AXIS)
    ax.bar(x, m["coverage"] * 100, width=width, color=colors, align="center")
    ax.axhline(gate.min_coverage * 100, color=INK2, lw=0.8)
    ax.text(x.min() - width, gate.min_coverage * 100 + 3, f"gate: ≥{gate.min_coverage:.0%} usable time and ≤{gate.max_corrected:.0%} corrected beats",
            color=INK2, fontsize=8.5, ha="left")
    end = _hours(res.wall[-1])
    ax.axvline(end, color=MUTED, lw=0.8)
    ax.text(end + 0.03, 102, "strap stream ends", color=INK2, fontsize=8.5, va="bottom")
    ax.set_ylim(0, 112)
    ax.set_ylabel("usable NN time (%)")
    _time_axis(ax)
    ax.set_title(f"Usable strap data in each {res.inputs.window_min:g}-min watch measurement window "
                 f"({int(m['valid'].sum())} of {len(m)} pass)")
    ax.legend(handles=[Patch(color=STRAP, label="passes gate"), Patch(color=AXIS, label="fails gate")],
              loc="upper left", bbox_to_anchor=(0, -0.12), ncol=2)
    return _save(fig, path)


def fig_rmssd_overnight(res, path: Path) -> Path:
    _style()
    m, w = res.matched, res.watch
    fig, ax = plt.subplots(figsize=(10, 4.2))
    if w.normal_range:
        ax.axhspan(*w.normal_range, color=WASH, lw=0, zorder=0)
        ax.text(_hours(w.point_s[0]), w.normal_range[1] + 1.5, "watch 'normal range' (personal baseline)",
                color=INK2, fontsize=8.5)
    ax.plot(_hours(w.point_s), w.point_ms, color=WATCH, lw=1.0, alpha=0.6, zorder=2)
    _dots(ax, _hours(w.point_s), w.point_ms, WATCH, size=22, label="watch (COROS, digitized)")
    v = m[m["valid"]]
    _dots(ax, _hours(v["point_s"]), v["rmssd"], STRAP, size=46, label="chest strap (windows that pass the gate)", zorder=4)
    end = _hours(res.wall[-1])
    ax.axvline(end, color=MUTED, lw=0.8)
    ax.text(end + 0.03, 4, "strap stream ends", color=INK2, fontsize=8.5)
    if w.reported_avg is not None:
        ax.axhline(w.reported_avg, color=WATCH, lw=0.8, alpha=0.8)
        ax.text(_hours(w.axis_end_s) + 0.03, w.reported_avg, f"watch night avg {w.reported_avg:g}", color=INK2,
                fontsize=8.5, va="center")
    ax.set_ylim(0, max(100, np.nanmax(v["rmssd"]) + 10 if len(v) else 100))
    ax.set_xlim(_hours(res.sleep_s[0]) - 0.1, _hours(res.sleep_s[1]) + 0.1)
    ax.set_ylabel("RMSSD (ms)")
    _time_axis(ax)
    ax.set_title("Overnight HRV (RMSSD, 10-min windows): chest strap vs watch")
    ax.legend(loc="upper left", ncol=2)
    return _save(fig, path)


def fig_agreement(res, path: Path) -> Path:
    _style()
    v = res.matched[res.matched["valid"]]
    s, w = v["rmssd"].to_numpy(), v["watch_rmssd"].to_numpy()
    a = res.epoch
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 4.3), gridspec_kw={"wspace": 0.3})

    lim = (min(s.min(), w.min()) * 0.8, max(s.max(), w.max()) * 1.15) if len(s) else (0, 100)
    ax.plot(lim, lim, color=MUTED, lw=0.8)
    ax.text(lim[1], lim[1], "  identity", color=INK2, fontsize=8.5, va="center", ha="left")
    _dots(ax, s, w, STRAP, size=46)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("chest strap RMSSD (ms)")
    ax.set_ylabel("watch RMSSD (ms)")
    if a["n"] >= 3:
        ax.text(0.03, 0.97, f"n = {a['n']} windows\nCCC = {a['ccc']:.2f}\nr (ln) = {a['pearson_r_ln']:.2f}",
                transform=ax.transAxes, va="top", color=INK2, fontsize=8.5)
    ax.set_title("Watch vs strap, matched windows")

    if a["n"] >= 3:
        mean = (s + w) / 2
        ratio = w / s
        bx.set_yscale("log")
        for val, lab, col in [
            (a["ratio"], f"bias ×{a['ratio']:.2f}", INK2),
            (a["ratio_loa_high"], f"+1.96 SD ×{a['ratio_loa_high']:.2f}", MUTED),
            (a["ratio_loa_low"], f"−1.96 SD ×{a['ratio_loa_low']:.2f}", MUTED),
        ]:
            bx.axhline(val, color=col, lw=0.9)
            bx.text(1.01, val, lab, transform=bx.get_yaxis_transform(), color=INK2, fontsize=8.5, va="center")
        bx.axhline(1.0, color=AXIS, lw=0.8)
        _dots(bx, mean, ratio, STRAP, size=46)
        lo = min(ratio.min(), a["ratio_loa_low"]) / 1.1
        hi = max(ratio.max(), a["ratio_loa_high"]) * 1.1
        bx.set_ylim(lo, hi)
        ticks = [t for t in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0) if lo <= t <= hi]
        bx.set_yticks(ticks)
        bx.set_yticklabels([f"{t:g}" for t in ticks])
        bx.minorticks_off()
    bx.set_xlabel("mean of strap and watch (ms)")
    bx.set_ylabel("watch / strap ratio (log scale)")
    bx.set_title("Bland–Altman (ratio)")
    return _save(fig, path)


def fig_sensitivity(res, path: Path) -> Path:
    _style()
    g = res.grid.copy()
    g["label"] = g["hypothesis"] + " · " + g["convention"] + " · " + g["window_min"].astype(int).astype(str) + " min"
    labels = list(dict.fromkeys(g["label"]))
    gates = list(dict.fromkeys(g["gate"]))
    inp = res.inputs
    primary = f"{inp.hypothesis} · {inp.convention} · {int(inp.window_min)} min"
    fig, axes = plt.subplots(1, len(gates), figsize=(10, 4.6), sharey=True, gridspec_kw={"wspace": 0.12})
    y = np.arange(len(labels))[::-1]
    lim = np.nanmax(np.abs(g["matched_diff_ms"])) * 1.15
    for ax, gate in zip(axes, gates):
        sub = g[g["gate"] == gate].set_index("label").loc[labels]
        is_primary = (sub.index == primary) & (gate == inp.gate)
        ax.axvline(0, color=AXIS, lw=0.9)
        _dots(ax, sub["matched_diff_ms"][~is_primary], y[~is_primary], MUTED, size=26)
        _dots(ax, sub["matched_diff_ms"][is_primary], y[is_primary], STRAP, size=46, zorder=4)
        for yy, n in zip(y, sub["n_valid"]):
            ax.text(1.0, yy, f" n={n}", transform=ax.get_yaxis_transform(), fontsize=7.5, color=MUTED, va="center")
        ax.set_xlim(-lim, lim)
        g_ = GATES[gate]
        ax.set_title(f"{gate} gate\n≥{g_.min_coverage:.0%} usable, ≤{g_.max_corrected:.0%} corrected", fontsize=9.5)
        ax.set_xlabel("watch − strap (ms)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    fig.suptitle("Night-level difference (coverage-matched) under every analysis choice", x=0.125, ha="left",
                 fontsize=11, fontweight="bold", y=1.04)
    return _save(fig, path)


def fig_heart_rate(res, path: Path) -> Path:
    _style()
    pm, r = res.rhr_minutes, res.rhr
    fig, ax = plt.subplots(figsize=(10, 3.6))
    x = _hours(pm["start_s"] + 30)
    ax.plot(x, pm["hr"], color=STRAP_LIGHT, lw=1.0, label="1-min HR (valid minutes)")
    ax.plot(x, pm["hr_5min"], color=STRAP, lw=1.6, label="5-min rolling mean")
    if np.isfinite(r["lowest_5min_mean_hr"]):
        tx = _hours(r["lowest_5min_at_s"] - 150)
        _dots(ax, [tx], [r["lowest_5min_mean_hr"]], STRAP, size=50, zorder=4)
        ax.annotate(f"lowest 5-min mean {r['lowest_5min_mean_hr']:.1f} bpm", (tx, r["lowest_5min_mean_hr"]),
                    xytext=(0, -16), textcoords="offset points", ha="center", color=INK2, fontsize=8.5)
    ax.axhline(r["mean_sleep_hr"], color=MUTED, lw=0.8)
    ax.text(x.min(), r["mean_sleep_hr"] + 0.6, f"mean sleeping HR {r['mean_sleep_hr']:.1f} bpm", color=INK2, fontsize=8.5)
    if r.get("watch_rhr") is not None:
        ax.axhline(r["watch_rhr"], color=WATCH, lw=1.0)
        ax.text(x.max(), r["watch_rhr"] + 0.6, f"watch RHR {r['watch_rhr']:g} bpm", color=INK2, fontsize=8.5, ha="right")
    lo = np.nanmin(pm["hr"]) if pm["hr"].notna().any() else 40
    hi = np.nanmax(pm["hr"]) if pm["hr"].notna().any() else 80
    ax.set_ylim(lo - 5, hi + 4)
    ax.set_ylabel("heart rate (bpm)")
    _time_axis(ax)
    ax.set_title("Chest-strap heart rate during sleep (masked noise excluded)")
    ax.legend(loc="upper right", ncol=2)
    return _save(fig, path)


def fig_lag_scan(res, path: Path) -> Path:
    _style()
    lg = res.lags
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.axvline(0, color=AXIS, lw=0.9)
    ax.plot(lg["lag_min"], lg["pearson_r_ln"], color=STRAP, lw=1.4)
    _dots(ax, lg["lag_min"], lg["pearson_r_ln"], STRAP, size=22)
    ax.set_ylim(-0.1, 1.0)
    ax.set_xlabel("strap timeline shifted by (min)")
    ax.set_ylabel("Pearson r, ln RMSSD")
    ax.set_title("Lag scan: no sharp optimum, so no offset is fitted")
    ax.grid(axis="x", visible=False)
    return _save(fig, path)


def make_all(res, outdir: Path) -> dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    return {
        "tachogram": fig_tachogram(res, outdir / "01_tachogram.png"),
        "quality": fig_quality(res, outdir / "02_window_quality.png"),
        "rmssd": fig_rmssd_overnight(res, outdir / "03_rmssd_overnight.png"),
        "agreement": fig_agreement(res, outdir / "04_agreement.png"),
        "sensitivity": fig_sensitivity(res, outdir / "05_sensitivity.png"),
        "heart_rate": fig_heart_rate(res, outdir / "06_heart_rate.png"),
        "lag": fig_lag_scan(res, outdir / "07_lag_scan.png"),
    }
