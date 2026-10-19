"""Download one clean lap of 2024 Bahrain telemetry and derive model inputs.

OpenF1 reports speed at roughly 3.7 Hz and position in decimetres. The script
slices one lap, resamples it to a uniform 50 Hz grid, smooths the path and
derives curvature and steering angle from the path geometry. Steering is not
broadcast by Formula 1, so it has to be reconstructed.

Usage:
    python scripts/fetch_telemetry.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.interpolate import splev, splprep
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "telemetry_bahrain_2024_lap20.csv"
CACHE = ROOT / ".cache"

SESSION_KEY = 9472  # 2024 Bahrain Grand Prix
DRIVER = 1
LAP = 20
WHEELBASE = 3.6
RATE_HZ = 50.0


def fetch(endpoint: str, **params) -> list[dict]:
    CACHE.mkdir(exist_ok=True)
    key = endpoint + "_" + "_".join(f"{name}-{value}" for name, value in sorted(params.items()))
    path = CACHE / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    response = requests.get(
        f"https://api.openf1.org/v1/{endpoint}",
        params=params,
        timeout=90,
        headers={"User-Agent": "f1-sideslip-estimation/1.1"},
    )
    response.raise_for_status()
    payload = response.json()
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def main() -> None:
    laps = pd.DataFrame(fetch("laps", session_key=SESSION_KEY, driver_number=DRIVER))
    laps["date_start"] = pd.to_datetime(laps["date_start"], utc=True, format="ISO8601")
    this_lap = laps[laps["lap_number"] == LAP].iloc[0]
    next_lap = laps[laps["lap_number"] == LAP + 1].iloc[0]
    start, end = this_lap["date_start"], next_lap["date_start"]

    car = pd.DataFrame(fetch("car_data", session_key=SESSION_KEY, driver_number=DRIVER))
    loc = pd.DataFrame(fetch("location", session_key=SESSION_KEY, driver_number=DRIVER))
    for frame in (car, loc):
        frame["date"] = pd.to_datetime(frame["date"], utc=True, format="ISO8601")

    car = car[(car["date"] >= start) & (car["date"] < end)].sort_values("date")
    loc = loc[(loc["date"] >= start) & (loc["date"] < end)].sort_values("date")
    if len(car) < 100 or len(loc) < 100:
        raise RuntimeError("Not enough samples in the selected lap")

    duration = (end - start).total_seconds()
    grid = np.arange(0.0, duration, 1.0 / RATE_HZ)

    def resample(frame: pd.DataFrame, column: str) -> np.ndarray:
        seconds = (frame["date"] - start).dt.total_seconds().to_numpy()
        return np.interp(grid, seconds, frame[column].to_numpy(dtype=float))

    speed = resample(car, "speed") / 3.6
    x = resample(loc, "x") / 10.0
    y = resample(loc, "y") / 10.0

    # Position from the timing feed is quantised, so fit a smoothing spline
    # through the lap before differentiating it.
    x = savgol_filter(x, 31, 3)
    y = savgol_filter(y, 31, 3)
    distance = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    keep = np.r_[True, np.diff(distance) > 0.05]
    tck, _ = splprep([x[keep], y[keep]], u=distance[keep], s=len(x) * 0.05, k=3)
    dx, dy = splev(distance, tck, der=1)
    ddx, ddy = splev(distance, tck, der=2)
    curvature = (dx * ddy - dy * ddx) / np.maximum((dx**2 + dy**2) ** 1.5, 1e-9)
    steering = np.arctan(WHEELBASE * curvature)
    steering = savgol_filter(steering, 21, 2)
    ax = savgol_filter(np.gradient(speed, 1.0 / RATE_HZ), 31, 2)

    telemetry = pd.DataFrame(
        {
            "t_s": grid,
            "speed_mps": speed,
            "x_m": x,
            "y_m": y,
            "curvature_1pm": curvature,
            "steer_rad": steering,
            "ax_mps2": ax,
        }
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    telemetry.to_csv(OUTPUT, index=False)
    print(f"Lap duration {duration:.2f}s, {len(telemetry)} samples")
    print(f"Speed {speed.min():.1f} to {speed.max():.1f} m/s")
    print(f"Steering {np.degrees(np.abs(steering)).max():.1f} deg peak")
    print(f"Written {OUTPUT}")


if __name__ == "__main__":
    sys.exit(main())
