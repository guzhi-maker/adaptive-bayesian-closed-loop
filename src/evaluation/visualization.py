"""Visualization utilities for the closed-loop framework."""

from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.core.closed_loop import StepRecord

matplotlib.use("Agg")  # Non-interactive backend


def plot_cost_over_time(
    records: list[StepRecord],
    ax: Optional[plt.Axes] = None,
    title: str = "Decision Cost Over Time",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot per-step decision cost over time."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    steps = [r.step for r in records]
    costs = [r.decision.expected_cost for r in records]

    ax.plot(steps, costs, alpha=0.5, linewidth=0.5, label="Per-step cost")
    # Smooth
    if len(costs) > 100:
        kernel = np.ones(100) / 100
        smoothed = np.convolve(costs, kernel, mode="valid")
        ax.plot(
            steps[len(steps) - len(smoothed):],
            smoothed,
            color="red",
            linewidth=2,
            label="Smoothed (window=100)",
        )

    ax.set_xlabel("Step")
    ax.set_ylabel("Decision Cost")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_state_trajectory(
    records: list[StepRecord],
    ax: Optional[plt.Axes] = None,
    title: str = "State Estimate Over Time",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot tracker state estimate over time."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    steps = [r.step for r in records]
    states = np.array([r.state_estimate for r in records])

    for i in range(states.shape[1]):
        ax.plot(steps, states[:, i], label=f"State dim {i}")

    ax.fill_between(
        steps,
        states[:, 0] - np.sqrt([r.state_uncertainty for r in records]),
        states[:, 0] + np.sqrt([r.state_uncertainty for r in records]),
        alpha=0.2,
        label="Uncertainty (±1σ)",
    )

    ax.set_xlabel("Step")
    ax.set_ylabel("State")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_calibration_reliability(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    ax: Optional[plt.Axes] = None,
    title: str = "Reliability Diagram",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot calibration reliability diagram."""
    from src.evaluation.metrics import compute_ece

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    ece, mce, bin_accs, bin_confs, bin_counts = compute_ece(
        probs, labels, n_bins=n_bins, adaptive=False
    )

    # Only plot non-empty bins
    valid = bin_counts > 0
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.bar(
        bin_confs[valid] - 0.5 / n_bins,
        bin_accs[valid],
        width=1.0 / n_bins,
        alpha=0.6,
        color="steelblue",
        edgecolor="navy",
        label="Actual",
    )
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{title}\nECE={ece:.6f}, MCE={mce:.6f}")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_decision_breakdown(
    records: list[StepRecord],
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot pie chart of decision type breakdown."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    breakdown = {}
    for r in records:
        action = r.decision.action.value
        breakdown[action] = breakdown.get(action, 0) + 1

    labels = list(breakdown.keys())
    sizes = list(breakdown.values())
    colors = ["green", "red", "orange"]

    ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors[: len(labels)],
        startangle=90,
    )
    ax.set_title(f"Decision Breakdown (Total: {len(records)})")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_summary_dashboard(
    records: list[StepRecord],
    probs: np.ndarray,
    labels: np.ndarray,
    save_dir: str = "results/figures",
    prefix: str = "",
) -> dict:
    """Generate summary dashboard with all key plots.

    Args:
        records: Step records from closed-loop run.
        probs: Calibrated probabilities.
        labels: True labels.
        save_dir: Directory to save figures.
        prefix: Filename prefix.

    Returns:
        Dict mapping plot names to file paths.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    fig1 = plot_cost_over_time(
        records, save_path=str(save_dir / f"{prefix}cost_over_time.pdf")
    )
    plt.close(fig1)
    paths["cost_over_time"] = str(save_dir / f"{prefix}cost_over_time.pdf")

    fig2 = plot_state_trajectory(
        records, save_path=str(save_dir / f"{prefix}state_trajectory.pdf")
    )
    plt.close(fig2)
    paths["state_trajectory"] = str(save_dir / f"{prefix}state_trajectory.pdf")

    fig3 = plot_calibration_reliability(
        probs, labels, save_path=str(save_dir / f"{prefix}reliability.pdf")
    )
    plt.close(fig3)
    paths["reliability"] = str(save_dir / f"{prefix}reliability.pdf")

    fig4 = plot_decision_breakdown(
        records, save_path=str(save_dir / f"{prefix}decision_breakdown.pdf")
    )
    plt.close(fig4)
    paths["decision_breakdown"] = str(save_dir / f"{prefix}decision_breakdown.pdf")

    return paths
