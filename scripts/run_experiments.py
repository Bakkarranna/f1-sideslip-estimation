"""Run every scenario, write the metrics and draw the report figures.

Usage:
    python scripts/run_experiments.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sideslip import plots  # noqa: E402
from sideslip.evaluate import (  # noqa: E402
    load_telemetry,
    noise_sensitivity,
    run_scenario,
    scenarios,
    sensitivity,
)

FIGURES = ROOT / "reports" / "figures"
ARTIFACTS = ROOT / "artifacts"


def main() -> None:
    telemetry = load_telemetry(ROOT / "data" / "telemetry_bahrain_2024_lap20.csv")
    runs = [run_scenario(telemetry, scenario) for scenario in scenarios()]
    nominal = runs[0]

    summary = {
        "lap": {
            "samples": int(len(telemetry)),
            "duration_s": round(float(telemetry["t_s"].max()), 2),
            "speed_min_mps": round(float(telemetry["speed_mps"].min()), 2),
            "speed_max_mps": round(float(telemetry["speed_mps"].max()), 2),
            "peak_steer_deg": round(float(telemetry["steer_rad"].abs().max() * 57.2958), 2),
        },
        "scenarios": {},
        "sensitivity": [],
    }
    for run in runs:
        summary["scenarios"][run.scenario.name] = {
            "description": run.scenario.description,
            **run.metrics(),
        }

    sensitivity_rows = []
    sensitivity_rows += sensitivity(telemetry, "rear_stiffness_scale", [0.7, 0.85, 1.0, 1.15, 1.3])
    sensitivity_rows += sensitivity(telemetry, "mass_scale", [0.95, 1.0, 1.05])
    sensitivity_rows += noise_sensitivity(telemetry, [0.5, 1.0, 2.0])
    summary["sensitivity"] = sensitivity_rows

    ARTIFACTS.mkdir(exist_ok=True)
    with (ARTIFACTS / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    plots.plot_overview(nominal, FIGURES / "lap_overview.png")
    plots.plot_estimate(nominal, FIGURES / "sideslip_estimate.png")
    plots.plot_scenarios(runs, FIGURES / "scenario_errors.png")
    plots.plot_sensitivity(sensitivity_rows, FIGURES / "sensitivity.png")
    plots.plot_nees(nominal, FIGURES / "nees.png")

    print("Scenario results, sideslip RMSE in degrees")
    header = f"{'scenario':22s} {'EKF':>7s} {'steady':>7s} {'integ':>7s} {'NEES':>7s}"
    print(header)
    print("-" * len(header))
    for run in runs:
        metrics = run.metrics()
        print(
            f"{run.scenario.name:22s} {metrics['ekf']['rmse_deg']:7.3f} "
            f"{metrics['steady_state']['rmse_deg']:7.3f} "
            f"{metrics['integration']['rmse_deg']:7.3f} "
            f"{metrics['nees']['mean']:7.2f}"
        )

    print("\nSensitivity, EKF RMSE in degrees")
    for row in sensitivity_rows:
        print(f"  {row['parameter']:24s} {row['value']:5.2f}  {row['rmse_deg']:.3f}")

    print(f"\nWritten {ARTIFACTS / 'metrics.json'}")
    print(f"Figures in {FIGURES}")


if __name__ == "__main__":
    main()
