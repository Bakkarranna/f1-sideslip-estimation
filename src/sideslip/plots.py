"""Figures for the estimation report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .evaluate import WARMUP_SECONDS, ScenarioRun

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 160,
        "font.size": 9,
        "axes.titlesize": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
    }
)


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_overview(run: ScenarioRun, path: str | Path) -> Path:
    result = run.result
    fig, axes = plt.subplots(3, 1, figsize=(9, 5.6), sharex=True)
    axes[0].plot(result.time, result.speed_reference, color="#264653", lw=1.4)
    axes[0].set_ylabel("Speed (m/s)")
    axes[0].set_title("2024 Bahrain, lap 20: reference lap used for the study")
    axes[1].plot(result.time, np.degrees(result.delta), color="#2a9d8f", lw=1.4)
    axes[1].set_ylabel("Steering (deg)")
    axes[2].plot(result.time, np.degrees(run.truth_beta), color="#e76f51", lw=1.4)
    axes[2].set_ylabel("Sideslip (deg)")
    axes[2].set_xlabel("Time (s)")
    return _save(fig, path)


def plot_estimate(run: ScenarioRun, path: str | Path) -> Path:
    result = run.result
    time = result.time
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    truth = np.degrees(run.truth_beta)
    ekf = np.degrees(run.ekf_beta)
    band = 2 * np.degrees(run.ekf_std)

    axes[0].fill_between(time, ekf - band, ekf + band, color="#8ecae6", alpha=0.35, label="EKF 95 percent band")
    axes[0].plot(time, truth, color="black", lw=1.5, label="reference")
    axes[0].plot(time, ekf, color="#1d3557", lw=1.1, label="EKF")
    axes[0].plot(time, np.degrees(run.steady_beta), color="#e63946", lw=0.9, ls="--", label="steady state")
    axes[0].plot(time, np.degrees(run.integration_beta), color="#f4a261", lw=0.9, ls=":", label="integration")
    axes[0].set_ylabel("Sideslip (deg)")
    axes[0].legend(frameon=False, ncol=5, fontsize=8, loc="upper right")
    axes[0].set_title("Sideslip estimate on the nominal scenario")

    axes[1].plot(time, ekf - truth, color="#1d3557", lw=1.0, label="EKF")
    axes[1].plot(time, np.degrees(run.steady_beta) - truth, color="#e63946", lw=0.8, alpha=0.8)
    axes[1].plot(time, np.degrees(run.integration_beta) - truth, color="#f4a261", lw=0.8, alpha=0.8)
    axes[1].axhline(0, color="black", lw=0.7)
    axes[1].set_ylabel("Error (deg)")
    axes[1].set_xlabel("Time (s)")
    return _save(fig, path)


def plot_scenarios(runs: list[ScenarioRun], path: str | Path) -> Path:
    names = [run.scenario.name for run in runs]
    ekf = [run.metrics()["ekf"]["rmse_deg"] for run in runs]
    steady = [run.metrics()["steady_state"]["rmse_deg"] for run in runs]
    integration = [run.metrics()["integration"]["rmse_deg"] for run in runs]
    positions = np.arange(len(names))
    width = 0.26

    fig, axis = plt.subplots(figsize=(9, 3.6))
    axis.bar(positions - width, ekf, width, label="EKF", color="#1d3557")
    axis.bar(positions, steady, width, label="Steady state", color="#e63946")
    axis.bar(positions + width, integration, width, label="Integration", color="#f4a261")
    axis.set_xticks(positions)
    axis.set_xticklabels([name.replace("_", " ") for name in names], rotation=18, ha="right")
    axis.set_ylabel("Sideslip RMSE (deg)")
    axis.set_title("Error by scenario, lower is better")
    axis.legend(frameon=False, ncol=3)
    return _save(fig, path)


def plot_sensitivity(rows: list[dict], path: str | Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    stiffness = [row for row in rows if row["parameter"] == "rear_stiffness_scale"]
    if stiffness:
        axes[0].plot([row["value"] for row in stiffness], [row["rmse_deg"] for row in stiffness], marker="o", color="#1d3557")
        axes[0].set_xlabel("Rear stiffness relative to reference")
        axes[0].set_ylabel("Sideslip RMSE (deg)")
        axes[0].set_title("Rear stiffness mismatch")
    noise = [row for row in rows if row["parameter"] == "noise_scale"]
    if noise:
        axes[1].plot([row["value"] for row in noise], [row["rmse_deg"] for row in noise], marker="o", color="#2a9d8f")
        axes[1].set_xlabel("Sensor noise multiplier")
        axes[1].set_ylabel("Sideslip RMSE (deg)")
        axes[1].set_title("Sensor noise")
    return _save(fig, path)


def plot_nees(run: ScenarioRun, path: str | Path) -> Path:
    dt = float(np.median(np.diff(run.result.time)))
    start = int(WARMUP_SECONDS / dt)
    values = run.nees[start:]
    fig, axis = plt.subplots(figsize=(4.8, 3.2))
    axis.hist(np.clip(values, 0, 20), bins=40, color="#457b9d", alpha=0.85)
    axis.axvline(np.mean(values), color="#e63946", lw=1.2, label=f"mean {np.mean(values):.2f}")
    axis.axvline(3.0, color="black", lw=0.9, ls="--", label="expected 3")
    axis.set_xlabel("NEES per sample")
    axis.set_ylabel("Count")
    axis.set_title("Filter consistency, nominal scenario")
    axis.legend(frameon=False, fontsize=8)
    return _save(fig, path)
