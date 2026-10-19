"""Baseline estimators used to contextualise the Kalman filter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import VehicleParams, sideslip, steady_state_sideslip


class SteadyStateEstimator:
    """Textbook linear bicycle model, fed with measured lateral acceleration."""

    def __init__(self, params: VehicleParams) -> None:
        self.params = params

    def run(self, delta: np.ndarray, ay: np.ndarray) -> np.ndarray:
        return np.array(
            [steady_state_sideslip(float(d), float(a), self.params) for d, a in zip(delta, ay, strict=False)]
        )


class IntegrationEstimator:
    """Integrate lateral acceleration to get lateral speed.

    This is included because it is a common first attempt. It drifts whenever
    the accelerometer has a bias or the car is on a gradient.
    """

    def __init__(self, dt: float) -> None:
        self.dt = dt

    def run(self, vx: np.ndarray, yaw_rate: np.ndarray, ay: np.ndarray) -> np.ndarray:
        vy = 0.0
        output = np.zeros(len(vx))
        for index in range(1, len(vx)):
            vy += self.dt * (ay[index] - vx[index] * yaw_rate[index])
            output[index] = np.arctan2(vy, max(abs(vx[index]), 1.0))
        return output


@dataclass
class FilterTrace:
    states: np.ndarray
    nees: np.ndarray
    beta_std_rad: np.ndarray

    @property
    def sideslip_rad(self) -> np.ndarray:
        return np.array([sideslip(state) for state in self.states])


def run_filter(result, truth: np.ndarray, params: VehicleParams, dt: float, **kwargs) -> FilterTrace:
    """Run the EKF over a simulation result and return the diagnostics."""
    from .ekf import ExtendedKalmanFilter

    initial = np.array([result.measured_vx[0], 0.0, 0.0])
    ekf = ExtendedKalmanFilter(params, dt=dt, initial_state=initial, **kwargs)
    states = np.zeros((len(result.time), 3))
    nees = np.zeros(len(result.time))
    beta_std = np.zeros(len(result.time))
    for index in range(len(result.time)):
        ekf.step(
            float(result.steer_input[index]),
            float(result.ax_input[index]),
            float(result.measured_vx[index]),
            float(result.measured_r[index]),
            float(result.measured_ay[index]),
        )
        states[index] = ekf.x
        nees[index] = ekf.nees(truth[index])
        vy, vx = ekf.x[1], max(abs(ekf.x[0]), 1.0)
        gradient = np.array([-vy / (vx**2 + vy**2), vx / (vx**2 + vy**2), 0.0])
        beta_std[index] = float(np.sqrt(gradient @ ekf.p @ gradient.T))
    return FilterTrace(states=states, nees=nees, beta_std_rad=beta_std)
