from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sideslip.ekf import ExtendedKalmanFilter
from sideslip.model import lateral_acceleration
from sideslip.evaluate import Scenario, load_telemetry, run_scenario
from sideslip.model import VehicleParams

DATA = Path(__file__).resolve().parents[1] / "data" / "telemetry_bahrain_2024_lap20.csv"


def test_covariance_stays_symmetric_and_positive():
    ekf = ExtendedKalmanFilter(VehicleParams(), dt=0.02)
    for index in range(200):
        ekf.step(0.01 * np.sin(index / 20), 0.5, 50.0, 0.05, 1.5)
        assert np.allclose(ekf.p, ekf.p.T)
        assert np.all(np.linalg.eigvals(ekf.p) > 0)


def test_update_moves_the_state_towards_the_measurements():
    params = VehicleParams()
    ekf = ExtendedKalmanFilter(params, dt=0.02, initial_state=np.array([40.0, 0.0, 0.0]))
    ekf.delta = 0.02
    measured_yaw, measured_ay = 0.4, 8.0
    before = abs(lateral_acceleration(ekf.x, ekf.delta, params) - measured_ay)
    ekf.update(vx=40.0, yaw_rate=measured_yaw, ay=measured_ay)
    after = abs(lateral_acceleration(ekf.x, ekf.delta, params) - measured_ay)
    assert after < before
    assert ekf.x[2] == pytest.approx(measured_yaw, abs=0.05)


def test_filter_beats_baselines_on_the_reference_lap():
    telemetry = load_telemetry(DATA)
    run = run_scenario(telemetry, Scenario(name="nominal", description=""))
    metrics = run.metrics()
    assert metrics["ekf"]["rmse_deg"] < metrics["steady_state"]["rmse_deg"]
    assert metrics["ekf"]["rmse_deg"] < 0.5
