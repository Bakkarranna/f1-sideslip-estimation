"""Single track vehicle model with a Pacejka style tyre curve.

The model is written in the ISO 8855 body frame: x forward, y to the left and
yaw rate positive when turning left. Steering angle and slip angles use the
same sign convention, so a left turn has positive steering, positive slip
angles at the front axle and positive yaw rate.

Downforce is included because a Formula 1 car generates most of its grip from
it. Without it the model would be far away from the lateral accelerations that
appear in real telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRAVITY = 9.81
AIR_DENSITY = 1.225
MIN_SPEED = 1.0


@dataclass(frozen=True)
class VehicleParams:
    """Physical parameters of the car.

    Defaults describe a 2024 specification car with a full tank removed, which
    is roughly the state of the car in the middle of a race.
    """

    mass: float = 798.0
    inertia: float = 1200.0
    a: float = 1.90
    b: float = 1.70
    mu: float = 1.70
    b_front: float = 16.5
    b_rear: float = 17.5
    c_front: float = 1.40
    c_rear: float = 1.45
    e: float = 0.30
    cla: float = 3.0
    aero_front_share: float = 0.45
    relaxation_front: float = 0.55
    relaxation_rear: float = 0.65

    @property
    def wheelbase(self) -> float:
        return self.a + self.b

    def static_loads(self) -> tuple[float, float]:
        weight = self.mass * GRAVITY
        front = weight * self.b / self.wheelbase
        rear = weight * self.a / self.wheelbase
        return front, rear

    def axle_loads(self, speed: float) -> tuple[float, float]:
        """Vertical load per axle including downforce."""
        front, rear = self.static_loads()
        downforce = 0.5 * AIR_DENSITY * self.cla * speed**2
        return front + downforce * self.aero_front_share, rear + downforce * (
            1.0 - self.aero_front_share
        )

    def with_changes(self, **changes: float) -> VehicleParams:
        values = {**self.__dict__, **changes}
        return VehicleParams(**values)


def tyre_force(alpha: np.ndarray | float, load: float, b: float, c: float, e: float, mu: float) -> np.ndarray | float:
    """Pacejka magic formula with zero camber and zero longitudinal slip."""
    alpha = np.asarray(alpha, dtype=float) if isinstance(alpha, np.ndarray) else alpha
    d = mu * load
    shape = b * alpha - e * (b * alpha - np.arctan(b * alpha))
    return d * np.sin(c * np.arctan(shape))


def tyre_gradient(alpha: float, load: float, b: float, c: float, e: float, mu: float) -> float:
    """dFy/dalpha for the same tyre curve."""
    d = mu * load
    shape = b * alpha - e * (b * alpha - np.arctan(b * alpha))
    dshape = b - e * (b - b / (1.0 + (b * alpha) ** 2))
    return d * np.cos(c * np.arctan(shape)) * c / (1.0 + shape**2) * dshape


def slip_angles(vx: float, vy: float, r: float, delta: float, params: VehicleParams) -> tuple[float, float]:
    speed = max(abs(vx), MIN_SPEED)
    front = delta - np.arctan((vy + params.a * r) / speed)
    rear = -np.arctan((vy - params.b * r) / speed)
    return float(front), float(rear)


def axle_forces(
    vx: float, vy: float, r: float, delta: float, params: VehicleParams
) -> tuple[float, float, float, float]:
    """Return front and rear lateral force and the two slip angles."""
    alpha_f, alpha_r = slip_angles(vx, vy, r, delta, params)
    load_f, load_r = params.axle_loads(abs(vx))
    force_f = tyre_force(alpha_f, load_f, params.b_front, params.c_front, params.e, params.mu)
    force_r = tyre_force(alpha_r, load_r, params.b_rear, params.c_rear, params.e, params.mu)
    return float(force_f), float(force_r), alpha_f, alpha_r


def derivatives(
    state: np.ndarray, delta: float, ax: float, params: VehicleParams
) -> np.ndarray:
    """State derivative for x = [vx, vy, r].

    ``ax`` is the longitudinal specific force at the contact patch, which is
    what a longitudinal accelerometer measures.
    """
    vx, vy, r = state
    force_f, force_r, _, _ = axle_forces(vx, vy, r, delta, params)
    cos_delta = np.cos(delta)
    vx_dot = ax + vy * r
    vy_dot = (force_f * cos_delta + force_r) / params.mass - vx * r
    r_dot = (params.a * force_f * cos_delta - params.b * force_r) / params.inertia
    return np.array([vx_dot, vy_dot, r_dot])


def lateral_acceleration(
    state: np.ndarray, delta: float, params: VehicleParams
) -> float:
    """Specific lateral force measured by a body fixed accelerometer."""
    vx, vy, r = state
    force_f, force_r, _, _ = axle_forces(vx, vy, r, delta, params)
    return float((force_f * np.cos(delta) + force_r) / params.mass)


def jacobian(state: np.ndarray, delta: float, ax: float, params: VehicleParams) -> np.ndarray:
    """Analytic Jacobian of :func:`derivatives` with respect to the state."""
    vx, vy, r = state
    speed = max(abs(vx), MIN_SPEED)
    load_f, load_r = params.axle_loads(abs(vx))
    alpha_f, alpha_r = slip_angles(vx, vy, r, delta, params)
    cos_delta = np.cos(delta)

    front_gradient = tyre_gradient(alpha_f, load_f, params.b_front, params.c_front, params.e, params.mu)
    rear_gradient = tyre_gradient(alpha_r, load_r, params.b_rear, params.c_rear, params.e, params.mu)

    # dFy/dLoad is the sine term of the magic formula.
    shape_f = params.b_front * alpha_f - params.e * (
        params.b_front * alpha_f - np.arctan(params.b_front * alpha_f)
    )
    shape_r = params.b_rear * alpha_r - params.e * (
        params.b_rear * alpha_r - np.arctan(params.b_rear * alpha_r)
    )
    dload_f = params.mu * np.sin(params.c_front * np.arctan(shape_f))
    dload_r = params.mu * np.sin(params.c_rear * np.arctan(shape_r))

    # How axle load changes with speed.
    dload_dv = AIR_DENSITY * params.cla * abs(vx)
    dload_f_dv = dload_dv * params.aero_front_share
    dload_r_dv = dload_dv * (1.0 - params.aero_front_share)

    q_f = vy + params.a * r
    q_r = vy - params.b * r
    denom_f = speed**2 + q_f**2
    denom_r = speed**2 + q_r**2

    dalpha_f_dvy = -speed / denom_f
    dalpha_f_dr = -params.a * speed / denom_f
    dalpha_f_dvx = q_f / denom_f
    dalpha_r_dvy = -speed / denom_r
    dalpha_r_dr = params.b * speed / denom_r
    dalpha_r_dvx = q_r / denom_r

    dforce_f_dvy = front_gradient * dalpha_f_dvy
    dforce_f_dr = front_gradient * dalpha_f_dr
    dforce_f_dvx = front_gradient * dalpha_f_dvx + dload_f * dload_f_dv
    dforce_r_dvy = rear_gradient * dalpha_r_dvy
    dforce_r_dr = rear_gradient * dalpha_r_dr
    dforce_r_dvx = rear_gradient * dalpha_r_dvx + dload_r * dload_r_dv

    jac = np.zeros((3, 3))
    jac[0, 0] = 0.0
    jac[0, 1] = r
    jac[0, 2] = vy
    # Speed enters the lateral rows twice: through the yaw coupling term and
    # through the downforce that loads the tyres.
    jac[1, 0] = (dforce_f_dvx * cos_delta + dforce_r_dvx) / params.mass - r
    jac[1, 1] = (dforce_f_dvy * cos_delta + dforce_r_dvy) / params.mass
    jac[1, 2] = (dforce_f_dr * cos_delta + dforce_r_dr) / params.mass - vx
    jac[2, 0] = (params.a * dforce_f_dvx * cos_delta - params.b * dforce_r_dvx) / params.inertia
    jac[2, 1] = (params.a * dforce_f_dvy * cos_delta - params.b * dforce_r_dvy) / params.inertia
    jac[2, 2] = (params.a * dforce_f_dr * cos_delta - params.b * dforce_r_dr) / params.inertia
    return jac


def lateral_jacobian(state: np.ndarray, delta: float, params: VehicleParams) -> np.ndarray:
    """Gradient of the measured lateral acceleration with respect to the state."""
    vx, vy, r = state
    speed = max(abs(vx), MIN_SPEED)
    load_f, load_r = params.axle_loads(abs(vx))
    alpha_f, alpha_r = slip_angles(vx, vy, r, delta, params)
    cos_delta = np.cos(delta)

    front_gradient = tyre_gradient(alpha_f, load_f, params.b_front, params.c_front, params.e, params.mu)
    rear_gradient = tyre_gradient(alpha_r, load_r, params.b_rear, params.c_rear, params.e, params.mu)
    shape_f = params.b_front * alpha_f - params.e * (
        params.b_front * alpha_f - np.arctan(params.b_front * alpha_f)
    )
    shape_r = params.b_rear * alpha_r - params.e * (
        params.b_rear * alpha_r - np.arctan(params.b_rear * alpha_r)
    )
    dload_f = params.mu * np.sin(params.c_front * np.arctan(shape_f))
    dload_r = params.mu * np.sin(params.c_rear * np.arctan(shape_r))
    dload_dv = AIR_DENSITY * params.cla * abs(vx)

    q_f = vy + params.a * r
    q_r = vy - params.b * r
    denom_f = speed**2 + q_f**2
    denom_r = speed**2 + q_r**2

    dforce_f_dvy = front_gradient * (-speed / denom_f)
    dforce_f_dr = front_gradient * (-params.a * speed / denom_f)
    dforce_f_dvx = (
        front_gradient * (q_f / denom_f) + dload_f * dload_dv * params.aero_front_share
    )
    dforce_r_dvy = rear_gradient * (-speed / denom_r)
    dforce_r_dr = rear_gradient * (params.b * speed / denom_r)
    dforce_r_dvx = (
        rear_gradient * (q_r / denom_r)
        + dload_r * dload_dv * (1.0 - params.aero_front_share)
    )

    return np.array(
        [
            (dforce_f_dvx * cos_delta + dforce_r_dvx) / params.mass,
            (dforce_f_dvy * cos_delta + dforce_r_dvy) / params.mass,
            (dforce_f_dr * cos_delta + dforce_r_dr) / params.mass,
        ]
    )


def sideslip(state: np.ndarray) -> float:
    """Sideslip angle in radians, positive when the car points left of travel."""
    return float(np.arctan2(state[1], max(abs(state[0]), MIN_SPEED)))


def steady_state_sideslip(
    delta: float, ay: float, params: VehicleParams, rear_stiffness: float | None = None
) -> float:
    """Classic steady state estimate from the linear bicycle model.

    This is the baseline estimator. It assumes small angles, no load transfer
    and a linear tyre, so it is expected to degrade near the limit.
    """
    if rear_stiffness is None:
        load_r = params.axle_loads(0.0)[1]
        rear_stiffness = tyre_gradient(0.0, load_r, params.b_rear, params.c_rear, params.e, params.mu)
    return float(
        params.b / params.wheelbase * delta
        - params.mass * params.a / (params.wheelbase * rear_stiffness) * ay
    )


def integrate_rk4(
    state: np.ndarray, delta: float, ax: float, params: VehicleParams, dt: float
) -> np.ndarray:
    """One Runge-Kutta step, used by the reference simulator."""
    k1 = derivatives(state, delta, ax, params)
    k2 = derivatives(state + 0.5 * dt * k1, delta, ax, params)
    k3 = derivatives(state + 0.5 * dt * k2, delta, ax, params)
    k4 = derivatives(state + dt * k3, delta, ax, params)
    return state + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
