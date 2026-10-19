"""Scenario definitions and evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .estimators import IntegrationEstimator, SteadyStateEstimator, run_filter
from .model import VehicleParams, sideslip
from .simulation import SensorConfig, simulate_lap

TRUTH_PARAMS = VehicleParams()
WARMUP_SECONDS = 1.0


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    filter_params: VehicleParams = TRUTH_PARAMS
    sensors: SensorConfig = SensorConfig()
    steering_gain: float = 1.0
    seed: int = 3


def scenarios() -> list[Scenario]:
    return [
        Scenario(
            name="nominal",
            description="Filter uses the same parameters as the reference car. Only tyre relaxation is unmodelled.",
        ),
        Scenario(
            name="rear_stiffness_low",
            description="Rear cornering stiffness 26 percent below the reference car.",
            filter_params=TRUTH_PARAMS.with_changes(b_rear=13.0),
        ),
        Scenario(
            name="cg_forward",
            description="Filter assumes the centre of gravity is 10 cm further forward.",
            filter_params=TRUTH_PARAMS.with_changes(a=1.80, b=1.80),
        ),
        Scenario(
            name="mass_high",
            description="Filter assumes a 40 kg heavier car, roughly the fuel load at the start of a race.",
            filter_params=TRUTH_PARAMS.with_changes(mass=838.0),
        ),
        Scenario(
            name="noise_high",
            description="Sensor noise doubled, with a gyro bias of 0.004 rad/s and an accelerometer bias of 0.15 m/s2.",
            sensors=SensorConfig(
                wheel_speed_std=0.20,
                yaw_rate_std=0.012,
                ay_std=0.20,
                steer_std=0.003,
                ax_std=0.30,
                yaw_rate_bias=0.004,
                ay_bias=0.15,
            ),
        ),
        Scenario(
            name="limit_handling",
            description="Steering input scaled by 1.12 to push the car closer to the grip limit.",
            steering_gain=1.12,
        ),
    ]


def load_telemetry(path: str | Path = "data/telemetry_bahrain_2024_lap20.csv") -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"t_s", "speed_mps", "steer_rad", "ax_mps2"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Telemetry is missing columns: {sorted(missing)}")
    if frame["t_s"].isna().any():
        raise ValueError("Telemetry contains missing timestamps")
    return frame


def error_metrics(estimate: np.ndarray, truth: np.ndarray, start_index: int = 0) -> dict[str, float]:
    error = np.degrees(estimate[start_index:] - truth[start_index:])
    return {
        "rmse_deg": float(np.sqrt(np.mean(error**2))),
        "mae_deg": float(np.mean(np.abs(error))),
        "p95_deg": float(np.percentile(np.abs(error), 95)),
        "max_deg": float(np.max(np.abs(error))),
        "bias_deg": float(np.mean(error)),
    }


@dataclass
class ScenarioRun:
    scenario: Scenario
    result: object
    truth_beta: np.ndarray
    ekf_beta: np.ndarray
    ekf_std: np.ndarray
    steady_beta: np.ndarray
    integration_beta: np.ndarray
    nees: np.ndarray

    def metrics(self) -> dict[str, dict[str, float]]:
        dt = float(np.median(np.diff(self.result.time)))
        start = int(WARMUP_SECONDS / dt)
        return {
            "ekf": error_metrics(self.ekf_beta, self.truth_beta, start),
            "steady_state": error_metrics(self.steady_beta, self.truth_beta, start),
            "integration": error_metrics(self.integration_beta, self.truth_beta, start),
            "nees": {
                "mean": float(np.mean(self.nees[start:])),
                "within_95_percent": float(
                    np.mean((self.nees[start:] > 0.352) & (self.nees[start:] < 7.814))
                ),
            },
        }


def run_scenario(telemetry: pd.DataFrame, scenario: Scenario) -> ScenarioRun:
    frame = telemetry.copy()
    frame["steer_rad"] = frame["steer_rad"] * scenario.steering_gain
    result = simulate_lap(
        frame,
        TRUTH_PARAMS,
        sensors=scenario.sensors,
        seed=scenario.seed,
        substeps=8,
    )
    dt = float(np.median(np.diff(frame["t_s"])))
    trace = run_filter(result, result.truth, scenario.filter_params, dt)
    truth_beta = np.array([sideslip(state) for state in result.truth])
    steady = SteadyStateEstimator(scenario.filter_params).run(
        result.steer_input, result.measured_ay
    )
    integration = IntegrationEstimator(dt).run(
        result.measured_vx, result.measured_r, result.measured_ay
    )
    return ScenarioRun(
        scenario=scenario,
        result=result,
        truth_beta=truth_beta,
        ekf_beta=trace.sideslip_rad,
        ekf_std=trace.beta_std_rad,
        steady_beta=steady,
        integration_beta=integration,
        nees=trace.nees,
    )


def sensitivity(
    telemetry: pd.DataFrame,
    parameter: str,
    values: list[float],
) -> list[dict[str, float]]:
    """Sweep one filter parameter and report sideslip RMSE."""
    rows = []
    for value in values:
        if parameter == "rear_stiffness_scale":
            base = TRUTH_PARAMS
            params = base.with_changes(b_rear=base.b_rear * value)
        elif parameter == "mass_scale":
            params = TRUTH_PARAMS.with_changes(mass=TRUTH_PARAMS.mass * value)
        elif parameter == "front_stiffness_scale":
            base = TRUTH_PARAMS
            params = base.with_changes(b_front=base.b_front * value)
        else:
            raise ValueError(f"Unknown parameter: {parameter}")
        scenario = Scenario(name=f"{parameter}_{value:.2f}", description="", filter_params=params)
        run = run_scenario(telemetry, scenario)
        dt = float(np.median(np.diff(run.result.time)))
        start = int(WARMUP_SECONDS / dt)
        rows.append({"parameter": parameter, "value": value, **error_metrics(run.ekf_beta, run.truth_beta, start)})
    return rows


def noise_sensitivity(telemetry: pd.DataFrame, scales: list[float]) -> list[dict[str, float]]:
    rows = []
    for scale in scales:
        sensors = SensorConfig(
            wheel_speed_std=0.10 * scale,
            yaw_rate_std=0.006 * scale,
            ay_std=0.10 * scale,
            steer_std=0.0015 * scale,
            ax_std=0.15 * scale,
        )
        scenario = Scenario(name=f"noise_{scale:.1f}", description="", sensors=sensors)
        run = run_scenario(telemetry, scenario)
        dt = float(np.median(np.diff(run.result.time)))
        start = int(WARMUP_SECONDS / dt)
        rows.append(
            {"parameter": "noise_scale", "value": scale, **error_metrics(run.ekf_beta, run.truth_beta, start)}
        )
    return rows
