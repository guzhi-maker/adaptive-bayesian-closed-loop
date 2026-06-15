"""Evaluation metrics for the closed-loop framework.

Extends standard calibration metrics (ECE, MCE, NLL, Brier) with
decision-focused metrics for the non-stationary cost-sensitive setting:

1. Average Decision Cost: Mean cost of decisions made
2. Cumulative Cost Ratio: Total cost relative to optimal oracle
3. Cost-Weighted ECE: ECE weighted by decision costs
4. Drift Adaptation Delay: Steps to recover after a regime change
5. Regret: Cumulative cost difference from optimal decisions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.special import expit


@dataclass
class DecisionMetrics:
    """Container for decision-focused evaluation metrics."""

    # Standard calibration
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    nll: float  # Negative Log-Likelihood
    brier: float  # Brier Score

    # Decision cost
    average_cost: float  # Mean decision cost
    cumulative_cost: float  # Total decision cost
    oracle_cost: float  # Total cost of optimal oracle
    cost_ratio: float  # cumulative_cost / oracle_cost

    # Cost-weighted calibration
    cost_weighted_ece: float  # ECE weighted by decision costs

    # Decision breakdown
    accept_rate: float  # Fraction of ACCEPT decisions
    reject_rate: float  # Fraction of REJECT decisions
    error_rate: float  # Fraction of incorrect decisions

    # Drift adaptation
    drift_adapt_delay: float = -1.0  # Steps to recover after drift

    # Auxiliary
    ece_std: float = 0.0
    mce_std: float = 0.0

    def __str__(self) -> str:
        return (
            f"Cost={self.average_cost:.4f} | "
            f"CostRatio={self.cost_ratio:.4f} | "
            f"ECE={self.ece:.6f} | "
            f"NLL={self.nll:.6f} | "
            f"Accept={self.accept_rate:.2%} / Reject={self.reject_rate:.2%}"
        )

    def as_dict(self) -> dict:
        return {
            "ece": self.ece,
            "ece_std": self.ece_std,
            "mce": self.mce,
            "mce_std": self.mce_std,
            "nll": self.nll,
            "brier": self.brier,
            "average_cost": self.average_cost,
            "cumulative_cost": self.cumulative_cost,
            "oracle_cost": self.oracle_cost,
            "cost_ratio": self.cost_ratio,
            "cost_weighted_ece": self.cost_weighted_ece,
            "accept_rate": self.accept_rate,
            "reject_rate": self.reject_rate,
            "error_rate": self.error_rate,
            "drift_adapt_delay": self.drift_adapt_delay,
        }


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    adaptive: bool = False,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Expected Calibration Error (ECE)."""
    assert len(probs) == len(labels)
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)

    # Use confidence = max(probs, 1 - probs) for binning
    conf = np.maximum(probs, 1.0 - probs)

    if adaptive:
        order = np.argsort(conf)
        sorted_conf = conf[order]
        sorted_probs = probs[order]
        sorted_labels = labels[order]
        bin_edges = np.linspace(0, len(sorted_probs), n_bins + 1).astype(int)
        bin_edges[-1] = len(sorted_probs)
    else:
        bin_edges = np.linspace(0, 1, n_bins + 1)

    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        if adaptive:
            start, end = bin_edges[i], bin_edges[i + 1]
            mask = slice(start, end)
        else:
            mask = (conf >= bin_edges[i]) & (conf < bin_edges[i + 1])
            if i == n_bins - 1:
                mask |= (conf == 1.0)

        bin_counts[i] = np.sum(mask)
        if bin_counts[i] > 0:
            bin_labels = labels[mask]
            bin_probs = probs[mask]
            bin_conf = conf[mask]
            bin_preds = (bin_probs >= 0.5).astype(float)
            bin_accuracies[i] = np.mean(bin_preds == bin_labels)
            bin_confidences[i] = np.mean(bin_conf)

    total = max(bin_counts.sum(), 1e-10)
    ece = 0.0
    mce = 0.0
    for i in range(n_bins):
        if bin_counts[i] > 0:
            weight = bin_counts[i] / total
            diff = abs(bin_accuracies[i] - bin_confidences[i])
            ece += weight * diff
            mce = max(mce, diff)

    return ece, mce, bin_accuracies, bin_confidences, bin_counts


def compute_nll(probs: np.ndarray, labels: np.ndarray) -> float:
    """Compute Negative Log-Likelihood."""
    eps = 1e-12
    probs = np.clip(np.asarray(probs, dtype=np.float64), eps, 1.0 - eps)
    return float(
        -np.mean(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs))
    )


def compute_brier(probs: np.ndarray, labels: np.ndarray) -> float:
    """Compute Brier Score."""
    return float(np.mean((np.asarray(probs) - np.asarray(labels)) ** 2))


def compute_cost_weighted_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    costs_no_action: np.ndarray,
    costs_action: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute ECE weighted by decision costs.

    Bins with higher decision costs get more weight, so calibration
    errors in high-stakes regions are penalized more.
    """
    conf = np.maximum(probs, 1.0 - probs)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(conf, bin_edges, right=False) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Decision cost for each sample: min(c_na * P(y=1), c_a * P(y=0))
    decision_costs = np.minimum(
        costs_no_action * probs, costs_action * (1.0 - probs)
    )

    total_cost = 0.0
    weighted_ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        n_bin = mask.sum()
        if n_bin == 0:
            continue

        bin_labels = labels[mask]
        bin_probs = probs[mask]
        bin_costs = decision_costs[mask]
        bin_preds = (bin_probs >= 0.5).astype(float)
        bin_conf = conf[mask]

        acc = np.mean(bin_preds == bin_labels)
        avg_conf = np.mean(bin_conf)
        cost_weight = np.mean(bin_costs)

        total_cost += np.sum(bin_costs)
        weighted_ece += cost_weight * abs(acc - avg_conf)

    return weighted_ece / max(total_cost, 1e-10)


def compute_decision_costs(
    probs: np.ndarray,
    labels: np.ndarray,
    costs_no_action: np.ndarray,
    costs_action: np.ndarray,
    decisions: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Compute decision costs relative to optimal oracle.

    Args:
        probs: Predicted probabilities P(y=1).
        labels: True labels in {0, 1}.
        costs_no_action: Cost of false negative.
        costs_action: Cost of false positive.

    Returns:
        (cumulative_cost, oracle_cost, cost_ratio).
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    c_na = np.asarray(costs_no_action, dtype=float)
    c_a = np.asarray(costs_action, dtype=float)

    # Use passed decisions, or default to 0.5 threshold
    if decisions is None:
        decisions = (probs >= 0.5).astype(float)
    else:
        decisions = np.asarray(decisions, dtype=float)

    # Actual cost
    fn_cost = c_na * (decisions == 0) * labels
    fp_cost = c_a * (decisions == 1) * (1.0 - labels)
    actual_cost = np.sum(fn_cost + fp_cost)

    # Oracle cost: optimal cost-aware decisions using the same threshold policy
    # but with perfect knowledge of the cost ratio at each step
    oracle_threshold = c_a.astype(float) / (c_na.astype(float) + c_a.astype(float) + 1e-12)
    oracle_decisions = (probs >= oracle_threshold).astype(float)
    oracle_cost = np.sum(
        c_na * (oracle_decisions == 0) * labels
        + c_a * (oracle_decisions == 1) * (1.0 - labels)
    )

    cost_ratio = actual_cost / max(oracle_cost, 1e-6)
    return float(actual_cost), float(oracle_cost), float(cost_ratio)


def compute_drift_adapt_delay(
    costs: np.ndarray,
    window: int = 500,
    threshold_factor: float = 1.5,
) -> float:
    """Estimate how many steps the system takes to recover after drift.

    Computes the number of steps after a cost spike until costs return
    to near-baseline levels.

    Args:
        costs: Per-step decision costs, shape (n_steps,).
        window: Smoothing window.
        threshold_factor: Multiple of baseline to define recovery.

    Returns:
        Estimated adaptation delay in steps, or -1 if not computable.
    """
    if len(costs) < window * 2:
        return -1.0

    # Smooth
    kernel = np.ones(window) / window
    smoothed = np.convolve(costs, kernel, mode="valid")

    # Baseline: first window
    baseline = np.median(smoothed[:window])
    threshold = baseline * threshold_factor

    # Find the first spike
    above = np.where(smoothed > threshold)[0]
    if len(above) == 0:
        return 0.0

    spike_start = above[0]
    if spike_start + window >= len(smoothed):
        return -1.0

    # Find recovery: first point after spike where cost returns below threshold
    post_spike = smoothed[spike_start:]
    recovered = np.where(post_spike <= threshold)[0]
    if len(recovered) == 0:
        return float(len(post_spike))

    return float(recovered[0])


def compute_all_decision_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    costs_no_action: np.ndarray,
    costs_action: np.ndarray,
    decisions: np.ndarray | None = None,
    n_bins: int = 15,
    adaptive: bool = False,
    n_bootstrap: int = 200,
) -> DecisionMetrics:
    """Compute all decision-focused evaluation metrics.

    Args:
        probs: Predicted probabilities, shape (n_samples,).
        labels: True labels in {0, 1}, shape (n_samples,).
        costs_no_action: False negative costs, shape (n_samples,).
        costs_action: False positive costs, shape (n_samples,).
        decisions: Decision outcomes (1=REJECT, 0=ACCEPT). Auto-computed if None.
        n_bins: Number of bins for ECE.
        adaptive: Use adaptive binning.
        n_bootstrap: Bootstrap resamples for error bars.

    Returns:
        DecisionMetrics with all evaluation results.
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    c_na = np.asarray(costs_no_action, dtype=float)
    c_a = np.asarray(costs_action, dtype=float)

    if decisions is None:
        decisions = (probs >= 0.5).astype(float)

    # Standard calibration
    ece, mce, _, _, _ = compute_ece(probs, labels, n_bins, adaptive)
    nll = compute_nll(probs, labels)
    brier = compute_brier(probs, labels)

    # Bootstrap for ECE/MCE
    rng = np.random.RandomState(42)
    n = len(probs)
    ece_boot, mce_boot = [], []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        e_b, m_b, _, _, _ = compute_ece(probs[idx], labels[idx], n_bins, adaptive)
        ece_boot.append(e_b)
        mce_boot.append(m_b)
    ece_std = float(np.std(ece_boot))
    mce_std = float(np.std(mce_boot))

    # Decision costs
    cum_cost, oracle_cost, cost_ratio = compute_decision_costs(
        probs, labels, c_na, c_a, decisions
    )
    avg_cost = cum_cost / n

    # Cost-weighted ECE
    cw_ece = compute_cost_weighted_ece(probs, labels, c_na, c_a, n_bins)

    # Decision breakdown
    accept_rate = float(np.mean(decisions == 0))
    reject_rate = float(np.mean(decisions == 1))
    error_rate = float(np.mean(
        (decisions == 1) != labels
    ))

    # Drift adaptation delay
    per_step_costs = np.where(
        decisions == 1,
        c_a * (1.0 - labels),
        c_na * labels,
    )
    adapt_delay = compute_drift_adapt_delay(per_step_costs)

    return DecisionMetrics(
        ece=ece,
        mce=mce,
        nll=nll,
        brier=brier,
        average_cost=avg_cost,
        cumulative_cost=cum_cost,
        oracle_cost=oracle_cost,
        cost_ratio=cost_ratio,
        cost_weighted_ece=cw_ece,
        accept_rate=accept_rate,
        reject_rate=reject_rate,
        error_rate=error_rate,
        drift_adapt_delay=adapt_delay,
        ece_std=ece_std,
        mce_std=mce_std,
    )

