"""Extended Kalman Filter for sideslip angle estimation.

The filter estimates longitudinal speed, lateral speed and yaw rate from three
measurements: wheel speed, yaw rate and lateral acceleration. Steering angle
and longitudinal acceleration are treated as known inputs.
"""

from __future__ import annotations

import numpy as np

from .model import (
    VehicleParams,
    derivatives,
    jacobian,
    lateral_acceleration,
    lateral_jacobian,
    sideslip,
)


class ExtendedKalmanFilter:
    """EKF for the three degree of freedom vehicle model.

    Parameters
    ----------
    params:
        Model parameters the filter believes in. These are deliberately kept
        separate from the parameters of the reference simulation so that model
        mismatch can be studied.
    dt:
        Sample time in seconds.
    q:
        Diagonal of the process noise covariance, in (state units)^2 per second.
    r:
        Diagonal of the effective measurement noise covariance. The values are
        deliberately larger than the sensor noise alone because they also
        absorb the part of the model error that shows up at measurement
        frequency, such as tyre relaxation and suspension compliance. Tuning
        them on the nominal lap brings the normalised error close to three.
    """

    def __init__(
        self,
        params: VehicleParams,
        dt: float,
        q: tuple[float, float, float] = (0.08, 0.45, 0.05),
        r: tuple[float, float, float] = (0.08, 0.00032, 0.08),
        initial_state: np.ndarray | None = None,
        initial_covariance: np.ndarray | None = None,
    ) -> None:
        self.params = params
        self.dt = dt
        self.q = np.diag(q)
        self.r = np.diag(r)
        self.x = np.zeros(3) if initial_state is None else np.array(initial_state, dtype=float)
        self.p = (
            np.diag([0.5**2, 0.5**2, 0.1**2])
            if initial_covariance is None
            else np.array(initial_covariance, dtype=float)
        )
        self.history: list[np.ndarray] = []
        self.covariance_history: list[np.ndarray] = []
        self.delta = 0.0

    @property
    def sideslip(self) -> float:
        return sideslip(self.x)

    def predict(self, delta: float, ax: float) -> np.ndarray:
        """Propagate the state and covariance one step."""
        self.x = self.x + self.dt * derivatives(self.x, delta, ax, self.params)
        transition = np.eye(3) + self.dt * jacobian(self.x, delta, ax, self.params)
        self.p = transition @ self.p @ transition.T + self.dt * self.q
        return self.x.copy()

    def update(self, vx: float, yaw_rate: float, ay: float) -> np.ndarray:
        """Fuse one set of sensor readings."""
        measurement = np.array([vx, yaw_rate, ay])
        predicted = np.array(
            [self.x[0], self.x[2], lateral_acceleration(self.x, self.delta, self.params)]
        )
        gradient = lateral_jacobian(self.x, self.delta, self.params)
        observation = np.zeros((3, 3))
        observation[0, 0] = 1.0  # wheel speed measures vx
        observation[1, 2] = 1.0  # gyro measures yaw rate
        observation[2, :] = gradient

        innovation = measurement - predicted
        covariance = observation @ self.p @ observation.T + self.r
        gain = self.p @ observation.T @ np.linalg.inv(covariance)
        self.x = self.x + gain @ innovation

        identity = np.eye(3)
        correction = identity - gain @ observation
        self.p = correction @ self.p @ correction.T + gain @ self.r @ gain.T
        self.p = 0.5 * (self.p + self.p.T)
        self.history.append(self.x.copy())
        self.covariance_history.append(self.p.copy())
        return self.x.copy()

    def step(self, delta: float, ax: float, vx: float, yaw_rate: float, ay: float) -> np.ndarray:
        """Predict and update in one call. ``delta`` is stored for the update."""
        self.delta = delta
        self.predict(delta, ax)
        return self.update(vx, yaw_rate, ay)

    def nees(self, truth: np.ndarray) -> float:
        """Normalised estimation error squared for the current state."""
        error = self.x - truth
        return float(error @ np.linalg.inv(self.p) @ error)
