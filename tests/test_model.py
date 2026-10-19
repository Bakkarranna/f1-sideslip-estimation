from __future__ import annotations

import numpy as np
import pytest

from sideslip.model import (
    VehicleParams,
    derivatives,
    jacobian,
    lateral_acceleration,
    lateral_jacobian,
    sideslip,
    slip_angles,
    steady_state_sideslip,
    tyre_force,
    tyre_gradient,
)


def test_tyre_force_is_symmetric_and_zero_at_zero_slip():
    params = VehicleParams()
    load = 4000.0
    assert tyre_force(0.0, load, params.b_front, params.c_front, params.e, params.mu) == 0.0
    left = tyre_force(0.1, load, params.b_front, params.c_front, params.e, params.mu)
    right = tyre_force(-0.1, load, params.b_front, params.c_front, params.e, params.mu)
    assert left == pytest.approx(-right)


def test_tyre_force_peaks_near_friction_limit():
    params = VehicleParams()
    load = 4000.0
    peak = max(
        tyre_force(float(alpha), load, params.b_front, params.c_front, params.e, params.mu)
        for alpha in np.linspace(0.0, 0.4, 200)
    )
    assert peak == pytest.approx(params.mu * load, rel=0.02)


def test_tyre_gradient_matches_finite_difference():
    params = VehicleParams()
    alpha = 0.045
    load = 3800.0
    step = 1e-6
    numeric = (
        tyre_force(alpha + step, load, params.b_front, params.c_front, params.e, params.mu)
        - tyre_force(alpha - step, load, params.b_front, params.c_front, params.e, params.mu)
    ) / (2 * step)
    assert tyre_gradient(alpha, load, params.b_front, params.c_front, params.e, params.mu) == pytest.approx(
        numeric, rel=1e-5
    )


@pytest.mark.parametrize("state", [[40.0, 0.3, 0.15], [20.0, -0.2, -0.4], [75.0, 0.05, 0.05]])
def test_state_jacobian_matches_finite_difference(state):
    params = VehicleParams()
    delta, ax = 0.05, 1.0
    numeric = np.zeros((3, 3))
    step = 1e-6
    for index in range(3):
        bump = np.zeros(3)
        bump[index] = step
        numeric[:, index] = (
            derivatives(np.array(state) + bump, delta, ax, params)
            - derivatives(np.array(state) - bump, delta, ax, params)
        ) / (2 * step)
    analytic = jacobian(np.array(state), delta, ax, params)
    assert np.allclose(analytic, numeric, rtol=2e-4, atol=2e-5)


def test_lateral_jacobian_matches_finite_difference():
    params = VehicleParams()
    state = np.array([50.0, 0.25, 0.2])
    delta = 0.04
    numeric = np.zeros(3)
    step = 1e-6
    for index in range(3):
        bump = np.zeros(3)
        bump[index] = step
        numeric[index] = (
            lateral_acceleration(state + bump, delta, params)
            - lateral_acceleration(state - bump, delta, params)
        ) / (2 * step)
    assert np.allclose(lateral_jacobian(state, delta, params), numeric, rtol=2e-4, atol=2e-5)


def test_slip_angles_follow_the_sign_convention():
    params = VehicleParams()
    front, rear = slip_angles(50.0, 0.0, 0.0, 0.05, params)
    assert front == pytest.approx(0.05)
    assert rear == pytest.approx(0.0)


def test_steady_state_matches_a_slow_constant_radius_turn():
    params = VehicleParams()
    delta = 0.02
    speed = 30.0
    state = np.array([speed, 0.0, 0.0])
    for _ in range(4000):
        rate = derivatives(state, delta, 0.0, params)
        state = state + 0.001 * rate
    ay = lateral_acceleration(state, delta, params)
    estimate = steady_state_sideslip(delta, ay, params)
    assert abs(estimate - sideslip(state)) < 0.003


def test_sideslip_is_zero_when_not_sliding():
    assert sideslip(np.array([30.0, 0.0, 0.2])) == pytest.approx(0.0)
