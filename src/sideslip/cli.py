"""Command line interface.

Examples
--------
    sideslip evaluate
    sideslip evaluate --telemetry data/telemetry_bahrain_2024_lap20.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .evaluate import load_telemetry, run_scenario, scenarios


def _evaluate(args: argparse.Namespace) -> int:
    telemetry = load_telemetry(args.telemetry)
    runs = [run_scenario(telemetry, scenario) for scenario in scenarios()]

    print(f"Telemetry: {args.telemetry}")
    print(f"Samples: {len(telemetry)}, duration {telemetry['t_s'].max():.1f}s")
    print()
    print(f"{'scenario':22s} {'EKF RMSE':>10s} {'steady':>10s} {'integrate':>10s} {'NEES':>8s}")
    for run in runs:
        metrics = run.metrics()
        print(
            f"{run.scenario.name:22s} {metrics['ekf']['rmse_deg']:10.3f} "
            f"{metrics['steady_state']['rmse_deg']:10.3f} "
            f"{metrics['integration']['rmse_deg']:10.3f} "
            f"{metrics['nees']['mean']:8.2f}"
        )

    if args.output:
        payload = {run.scenario.name: run.metrics() for run in runs}
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWritten {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sideslip", description="Sideslip angle estimation")
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate = sub.add_parser("evaluate", help="compare estimators across scenarios")
    evaluate.add_argument("--telemetry", default="data/telemetry_bahrain_2024_lap20.csv")
    evaluate.add_argument("--output", default=None)
    evaluate.set_defaults(func=_evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
