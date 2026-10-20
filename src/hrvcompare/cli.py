"""Command line: ``hrvcompare run ...`` (or ``python -m hrvcompare run ...``)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .artifacts import ArtifactConfig
from .compare import CONVENTIONS
from .metrics import GATES
from .pipeline import Inputs, run
from .report import write_report
from .timeline import HYPOTHESES


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hrvcompare", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="clean the strap RRI, digitize the watch chart, compare, write the report")
    r.add_argument("--rri", required=True, type=Path, help="chest-strap RRI export (CSV)")
    r.add_argument("--watch-png", required=True, type=Path, help="screenshot of the watch's overnight HRV chart")
    r.add_argument("--sleep", required=True, nargs=2, metavar=("ONSET", "WAKE"),
                   help="sleep onset and wake times printed under the watch chart, HH:MM")
    r.add_argument("--watch-avg", type=float, help="watch's reported nightly HRV average (ms), used as a digitization check")
    r.add_argument("--watch-range", type=float, nargs=2, metavar=("LOW", "HIGH"), help="watch's 'normal range' (ms)")
    r.add_argument("--watch-rhr", type=float, help="watch's resting heart rate for the night (bpm)")
    r.add_argument("--watch-interval", type=float, default=10.0, help="watch HRV measurement interval in minutes (default 10)")
    r.add_argument("--hypothesis", choices=list(HYPOTHESES), default="H1", help="primary clock-alignment hypothesis")
    r.add_argument("--gate", choices=list(GATES), default="default", help="primary window quality gate")
    r.add_argument("--convention", choices=CONVENTIONS, default="ending", help="strap window relative to each watch reading")
    r.add_argument("--window-min", type=float, default=10.0, help="strap window length behind each watch reading (min)")
    r.add_argument("--rr-min", type=float, default=ArtifactConfig.rr_min_ms, help="shortest plausible RR interval in ms (default for sleep)")
    r.add_argument("--rr-max", type=float, default=ArtifactConfig.rr_max_ms, help="longest plausible RR interval in ms")
    r.add_argument("--out", type=Path, default=Path("reports"), help="output directory")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inputs = Inputs(
        rri_csv=args.rri,
        watch_png=args.watch_png,
        sleep_start=args.sleep[0],
        sleep_end=args.sleep[1],
        watch_avg=args.watch_avg,
        watch_range=tuple(args.watch_range) if args.watch_range else None,
        watch_rhr=args.watch_rhr,
        watch_interval_min=args.watch_interval,
        hypothesis=args.hypothesis,
        gate=args.gate,
        convention=args.convention,
        window_min=args.window_min,
        artifact_config=ArtifactConfig(rr_min_ms=args.rr_min, rr_max_ms=args.rr_max),
    )
    res = run(inputs)
    report = write_report(res, args.out)
    n = res.night
    w = res.watch
    print(f"report: {report}")
    print(f"strap: {res.clean.summary()['masked_time_frac']:.0%} of recording masked; "
          f"{n['n_valid']}/{n['n_points']} watch readings matched")
    print(f"watch digitized avg {w.digitized_avg:.1f} ms (reported {w.reported_avg}); "
          f"night level watch - strap = {n['matched_diff_ms']:+.1f} ms")
    if w.reported_avg is not None and abs(w.digitized_avg - w.reported_avg) > 1.5:
        print("warning: digitized average differs from the reported one by >1.5 ms; check the chart calibration", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
