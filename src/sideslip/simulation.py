"""Reference simulation and sensor model.

The reference car uses the same single track equations as the filter but with
two differences that matter: tyre forces are filtered by a first order
relaxation model and the physical parameters can be set independently. That
gives the filter something real to be wrong about.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model import VehicleParams, axle_forces, lateral_acceleration


@dataclass(frozen=True)
class SensorConfig:
    wheel_speed_std: float = 0.10
    yaw_rate_std: float = 0.006
    ay_std: float = 0.10
    steer_std: float = 0.0015
    ax_std: float = 0.15
    wheel_speed_bias: float = 0.0
    yaw_rate_bias: float = 0.0
    ay_bias: float = 0.0


@dataclass
class SimulationResult:
    time: np.ndarray
    truth: np.ndarray
    measured_vx: np.ndarray
    measured_r: np.ndarray
    measured_ay: np.ndarray
    steer_input: np.ndarray
    ax_input: np.ndarray
    delta: np.ndarray
    speed_reference: np.ndarray


def _relaxed_derivatives(
    state: np.ndarray,
    delta: float,
    ax: float,
    params: VehicleParams,
    forces: np.ndarray,
) -> np.ndarray:
    vx, vy, r = state[:3]
    force_f, force_r = forces
    cos_delta = np.cos(delta)
    vx_dot = ax + vy * r
    vy_dot = (force_f * cos_delta + force_r) / params.mass - vx * r
    r_dot = (params.a * force_f * cos_delta - params.b * force_r) / params.inertia
    return np.array([vx_dot, vy_dot, r_dot])


def simulate_lap(
    telemetry: pd.DataFrame,
    params: VehicleParams,
    sensors: SensorConfig | None = None,
    seed: int = 3,
    substeps: int = 4,
    relaxation: bool = True,
) -> SimulationResult:
    """Drive the reference model with a real speed and steering trace.

    The telemetry is expected at 50 Hz with columns ``speed_mps``,
    ``steer_rad`` and ``ax_mps2``.
    """
    sensors = sensors or SensorConfig()
    rng = np.random.default_rng(seed)
    dt = float(np.median(np.diff(telemetry["t_s"])))
    steps = len(telemetry)
    sub_dt = dt / substeps

    speed = telemetry["speed_mps"].to_numpy(float)
    delta_command = telemetry["steer_rad"].to_numpy(float)
    ax_command = telemetry["ax_mps2"].to_numpy(float)

    state = np.array([speed[0], 0.0, 0.0])
    force_f, force_r, _, _ = axle_forces(state[0], state[1], state[2], delta_command[0], params)
    forces = np.array([force_f, force_r])

    truth = np.zeros((steps, 3))
    ay_true = np.zeros(steps)
    truth[0] = state

    for index in range(1, steps):
        delta = delta_command[index - 1]
        ax = ax_command[index - 1]
        for _ in range(substeps):
            if relaxation:
                steady_f, steady_r, _, _ = axle_forces(
                    state[0], state[1], state[2], delta, params
                )
                target = np.array([steady_f, steady_r])
                tau = np.array([params.relaxation_front, params.relaxation_rear])
                rate = (abs(state[0]) / np.maximum(tau, 0.05)) * (target - forces)
                forces = forces + sub_dt * rate
            else:
                force_f, force_r, _, _ = axle_forces(
                    state[0], state[1], state[2], delta, params
                )
                forces = np.array([force_f, force_r])

            k1 = _relaxed_derivatives(state, delta, ax, params, forces)
            k2 = _relaxed_derivatives(state + 0.5 * sub_dt * k1, delta, ax, params, forces)
            k3 = _relaxed_derivatives(state + 0.5 * sub_dt * k2, delta, ax, params, forces)
            k4 = _relaxed_derivatives(state + sub_dt * k3, delta, ax, params, forces)
            state = state + sub_dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

        truth[index] = state
        ay_true[index] = lateral_acceleration(state, delta, params)

    measured_vx = truth[:, 0] + rng.normal(0, sensors.wheel_speed_std, steps) + sensors.wheel_speed_bias
    measured_r = truth[:, 2] + rng.normal(0, sensors.yaw_rate_std, steps) + sensors.yaw_rate_bias
    measured_ay = ay_true + rng.normal(0, sensors.ay_std, steps) + sensors.ay_bias
    steer_input = delta_command + rng.normal(0, sensors.steer_std, steps)
    ax_input = ax_command + rng.normal(0, sensors.ax_std, steps)

    return SimulationResult(
        time=telemetry["t_s"].to_numpy(float),
        truth=truth,
        measured_vx=measured_vx,
        measured_r=measured_r,
        measured_ay=measured_ay,
        steer_input=steer_input,
        ax_input=ax_input,
        delta=delta_command,
        speed_reference=speed,
    )
