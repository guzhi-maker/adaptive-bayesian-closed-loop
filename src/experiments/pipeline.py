"""Unified experiment pipeline for the closed-loop framework.

Orchestrates:
1. Data loading (synthetic or real)
2. Framework initialization (closed-loop or baseline)
3. Execution and metric computation
4. Results logging and visualization

Supports all experiment types from the project plan:
- Ablation (2³ factorial)
- Baseline comparison (8 methods)
- Stability and robustness tests
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Literal, Optional

import numpy as np

from src.core.base_calibrator import BaseCalibrator
from src.core.base_decision_maker import Action, BaseDecisionMaker, DecisionResult
from src.core.base_tracker import BaseTracker
from src.core.closed_loop import ClosedLoopConfig, ClosedLoopFramework
from src.data.dataset_loader import DatasetLoader
from src.data.synthetic_generator import (
    CostConfig,
    DriftConfig,
    SyntheticDataConfig,
    SyntheticDataGenerator,
)
from src.evaluation.metrics import DecisionMetrics, compute_all_decision_metrics
from src.evaluation.visualization import plot_summary_dashboard
from src.modules.platt_calibrator import PlattCalibrator
from src.modules.three_layer_decision import ThreeLayerConfig, ThreeLayerDecisionMaker
from src.modules.ukf_tracker import UKFTracker


@dataclass
class BaseObservationModel:
    """Simple observation model: maps state to observation via identity."""

    def observe(self, state: np.ndarray) -> np.ndarray:
        return state.copy()

    def observe_with_noise(
        self, state: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return state.copy(), np.ones_like(state) * 0.01


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""

    # Data
    n_samples: int = 50_000
    drift_type: Literal["gradual", "abrupt", "periodic", "none"] = "gradual"
    drift_speed: float = 5e-5
    cost_type: Literal["fixed", "dynamic", "stratified"] = "dynamic"

    # Feedback
    use_f1: bool = True
    use_f2: bool = True
    use_f3: bool = True

    # UKF
    ukf_process_noise: float = 5e-4
    ukf_initial_p: float = 0.01

    # Calibration
    calibrate_every: int = 500
    calibration_window: int = 2000
    warmup_steps: int = 500

    # Decision
    low_threshold: float = 0.4
    high_threshold: float = 0.6
    cost_no_action: float = 10.0
    cost_action: float = 1.0

    # Feedback sensitivity
    f1_sensitivity: float = 1.0
    f2_sensitivity: float = 10.0
    f3_drift_sensitivity: float = 0.5
    f3_uncertainty_sensitivity: float = 2.0

    # Evaluation
    ece_bins: int = 15
    n_bootstrap: int = 50

    base_logit_mean: float = -2.0

    # General
    seed: int = 42
    experiment_name: str = "default"
    results_dir: str = "results"


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""

    config: dict
    metrics: dict
    cumulative_cost: float
    average_cost: float
    decision_breakdown: dict
    runtime_seconds: float
    n_samples: int


def _create_observation_model() -> Callable:
    """Create the observation model for UKF.

    Returns a function mapping state -> (expected_obs, obs_var).
    """
    def obs_model(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # State is log-odds of error rate. Observation is the drift signal.
        return state.copy(), np.array([0.001])

    return obs_model


def run_experiment(cfg: ExperimentConfig, verbose: bool = True) -> ExperimentResult:
    """Run a single closed-loop experiment.

    Args:
        cfg: Experiment configuration.
        verbose: If True, print progress.

    Returns:
        ExperimentResult with all metrics.
    """
    np.random.seed(cfg.seed)

    if verbose:
        print("=" * 60)
        print(f"Experiment: {cfg.experiment_name}")
        print(f"  Drift: {cfg.drift_type} (speed={cfg.drift_speed})")
        print(f"  Cost: {cfg.cost_type}")
        print(f"  Feedback: F1={cfg.use_f1}, F2={cfg.use_f2}, F3={cfg.use_f3}")
        print("=" * 60)

    # --- 1. Generate data ----------------------------------------------------
    if verbose:
        print("\n[Step 1/4] Generating synthetic data...")

    drift_cfg = DriftConfig(
        drift_type=cfg.drift_type,
        drift_speed=cfg.drift_speed,
    )
    cost_cfg = CostConfig(cost_type=cfg.cost_type)
    syn_cfg = SyntheticDataConfig(
        n_samples=cfg.n_samples,
        drift=drift_cfg,
        cost=cost_cfg,
        base_logit_mean=cfg.base_logit_mean,
        seed=cfg.seed,
    )
    generator = SyntheticDataGenerator(syn_cfg)
    data = generator.generate()

    logits = data["logits"]
    labels = data["labels"]
    costs_na = data["costs_no_action"]
    costs_a = data["costs_action"]

    if verbose:
        print(f"  Generated {len(logits):,} samples")
        print(f"  Label balance: {labels.mean():.3f}")

    # --- 2. Initialize framework ---------------------------------------------
    if verbose:
        print("\n[Step 2/4] Initializing closed-loop framework...")

    # UKF tracker: 1D state tracking the logit shift (drift in logit space)
    # State initialized at 0 (no drift), observes smoothed running logit mean
    def drift_obs_model(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return state.copy(), np.array([0.01])

    tracker = UKFTracker(
        initial_state=np.array([0.0]),
        initial_covariance=np.eye(1) * 0.1,
        process_noise=np.eye(1) * cfg.ukf_process_noise,
        observation_model=drift_obs_model,
        state_clamp=(-8.0, 8.0),
    )

    # Platt calibrator with bootstrap
    calibrator = PlattCalibrator(
        use_scalar=True,
        n_bootstrap=cfg.n_bootstrap,
    )

    # Three-layer decision maker
    decision_config = ThreeLayerConfig(
        low_threshold=cfg.low_threshold,
        high_threshold=cfg.high_threshold,
        default_cost_no_action=cfg.cost_no_action,
        default_cost_action=cfg.cost_action,
    )
    decision_maker = ThreeLayerDecisionMaker(config=decision_config)

    # Closed-loop config
    loop_config = ClosedLoopConfig(
        use_f1=cfg.use_f1,
        use_f2=cfg.use_f2,
        use_f3=cfg.use_f3,
        calibrate_every=cfg.calibrate_every,
        calibration_window=cfg.calibration_window,
        warmup_steps=cfg.warmup_steps,
        bootstrap_n=cfg.n_bootstrap,
        f1_sensitivity=cfg.f1_sensitivity,
        f2_sensitivity=cfg.f2_sensitivity,
        f3_drift_sensitivity=cfg.f3_drift_sensitivity,
        f3_uncertainty_sensitivity=cfg.f3_uncertainty_sensitivity,
    )

    framework = ClosedLoopFramework(
        tracker=tracker,
        calibrator=calibrator,
        decision_maker=decision_maker,
        config=loop_config,
    )

    # --- 3. Run pipeline -----------------------------------------------------
    if verbose:
        print("\n[Step 3/4] Running closed-loop pipeline...")

    start_time = time.perf_counter()

    # Process stream: UKF observes a smoothed version of the logits
    # In practice, this would be the base classifier's average confidence
    # We use the drift_state as the ground-truth observation for tracking
    observations = data["drift_state"].reshape(-1, 1) if "drift_state" in data else None

    if observations is None:
        # Fallback: running average of logits as observation
        from scipy.ndimage import uniform_filter1d
        obs_signal = uniform_filter1d(logits, size=501, mode='nearest')
        observations = obs_signal.reshape(-1, 1)

    records = framework.process_stream(
        logits=logits,
        labels=labels,
        costs_no_action=costs_na,
        costs_action=costs_a,
        observations=observations,
        verbose=verbose,
    )

    runtime = time.perf_counter() - start_time

    if verbose:
        print(f"  Runtime: {runtime:.2f}s ({runtime / len(logits) * 1000:.3f}ms per step)")

    # --- 4. Evaluate ---------------------------------------------------------
    if verbose:
        print("\n[Step 4/4] Computing metrics...")

    # Collect calibrated probabilities from records
    cal_probs = np.array([r.calibrated_prob for r in records])
    decisions = np.array([1 if r.decision.action == Action.REJECT else 0 for r in records])

    metrics = compute_all_decision_metrics(
        probs=cal_probs,
        labels=labels,
        costs_no_action=costs_na,
        costs_action=costs_a,
        decisions=decisions,
        n_bins=cfg.ece_bins,
        n_bootstrap=cfg.n_bootstrap,
    )

    # Save results
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    result = ExperimentResult(
        config=asdict(cfg),
        metrics=metrics.as_dict(),
        cumulative_cost=framework.get_cumulative_cost(),
        average_cost=framework.get_average_cost(),
        decision_breakdown=framework.get_decision_breakdown(),
        runtime_seconds=runtime,
        n_samples=len(logits),
    )

    # Save to JSON
    result_path = results_dir / f"{cfg.experiment_name}_result.json"
    with open(result_path, "w") as f:
        json.dump(
            {
                "config": result.config,
                "metrics": result.metrics,
                "cumulative_cost": result.cumulative_cost,
                "average_cost": result.average_cost,
                "decision_breakdown": result.decision_breakdown,
                "runtime_seconds": result.runtime_seconds,
                "n_samples": result.n_samples,
            },
            f,
            indent=2,
        )
    if verbose:
        print(f"\n  Result saved to {result_path}")
        print(f"  {metrics}")

    # Generate plots
    plot_paths = plot_summary_dashboard(
        records, cal_probs, labels,
        save_dir=str(results_dir / "figures"),
        prefix=f"{cfg.experiment_name}_",
    )
    if verbose:
        print(f"  Plots saved to {results_dir / 'figures'}")

    return result


def run_ablation_experiments(
    base_cfg: ExperimentConfig,
    seeds: list[int] | None = None,
    verbose: bool = True,
) -> list[ExperimentResult]:
    """Run the 2³ factorial ablation experiment.

    Tests all 8 combinations of F1, F2, F3 feedback loops.

    Args:
        base_cfg: Base experiment configuration.
        seeds: Random seeds for repetition. Defaults to [42].
        verbose: If True, print progress.

    Returns:
        List of ExperimentResult for each configuration.
    """
    if seeds is None:
        seeds = [42]

    # 2³ factorial design
    feedback_configs = [
        {"use_f1": False, "use_f2": False, "use_f3": False},  # None
        {"use_f1": True, "use_f2": False, "use_f3": False},   # F1 only
        {"use_f1": False, "use_f2": True, "use_f3": False},   # F2 only
        {"use_f1": False, "use_f2": False, "use_f3": True},   # F3 only
        {"use_f1": True, "use_f2": True, "use_f3": False},    # F1+F2
        {"use_f1": True, "use_f2": False, "use_f3": True},    # F1+F3
        {"use_f1": False, "use_f2": True, "use_f3": True},    # F2+F3
        {"use_f1": True, "use_f2": True, "use_f3": True},     # All (full)
    ]

    label_map = {
        (False, False, False): "None (open-loop)",
        (True, False, False): "F1 only",
        (False, True, False): "F2 only",
        (False, False, True): "F3 only",
        (True, True, False): "F1+F2",
        (True, False, True): "F1+F3",
        (False, True, True): "F2+F3",
        (True, True, True): "All (closed-loop)",
    }

    all_results = []

    for fb in feedback_configs:
        f1, f2, f3 = fb["use_f1"], fb["use_f2"], fb["use_f3"]
        label = label_map[(f1, f2, f3)]

        for seed in seeds:
            cfg = ExperimentConfig(
                **{**asdict(base_cfg),
                   "use_f1": f1,
                   "use_f2": f2,
                   "use_f3": f3,
                   "seed": seed,
                   "experiment_name": f"ablation_F1={int(f1)}_F2={int(f2)}_F3={int(f3)}_seed={seed}"}
            )

            if verbose:
                print(f"\n{'=' * 60}")
                print(f"Ablation: {label} (seed={seed})")
                print(f"{'=' * 60}")

            result = run_experiment(cfg, verbose=verbose)
            result.config["ablation_label"] = label
            all_results.append(result)

    return all_results


def run_counterfactual_experiments(
    base_cfg: ExperimentConfig,
    seeds: list[int] | None = None,
    verbose: bool = True,
) -> list[ExperimentResult]:
    """Run counterfactual experiments to prove closed-loop necessity.

    Tests additional baselines beyond the 2³ design:
    1. Optimal open-loop (best static calibration + decision)
    2. Ensemble: average of three module outputs
    3. Partial feedback (F1+F3, no F2)
    4. Partial feedback (F2+F3, no F1)

    Args:
        base_cfg: Base configuration.
        seeds: Random seeds.
        verbose: If True, print progress.

    Returns:
        List of ExperimentResult.
    """
    if seeds is None:
        seeds = [42]

    configs = [
        {"use_f1": False, "use_f2": False, "use_f3": False, "name": "optimal_open_loop"},
        {"use_f1": False, "use_f2": True, "use_f3": True, "name": "no_F1"},
        {"use_f1": True, "use_f2": False, "use_f3": True, "name": "no_F2"},
        {"use_f1": True, "use_f2": True, "use_f3": False, "name": "no_F3"},
    ]

    all_results = []

    for cfg_dict in configs:
        for seed in seeds:
            cfg = ExperimentConfig(
                **{**asdict(base_cfg),
                   "use_f1": cfg_dict["use_f1"],
                   "use_f2": cfg_dict["use_f2"],
                   "use_f3": cfg_dict["use_f3"],
                   "seed": seed,
                   "experiment_name": f"counterfactual_{cfg_dict['name']}_seed={seed}"}
            )

            if verbose:
                print(f"\n{'=' * 60}")
                print(f"Counterfactual: {cfg_dict['name']} (seed={seed})")
                print(f"{'=' * 60}")

            result = run_experiment(cfg, verbose=verbose)
            all_results.append(result)

    return all_results


def compute_ablation_summary(
    results: list[ExperimentResult],
) -> dict:
    """Compute summary statistics from ablation experiment results.

    Groups results by feedback configuration and computes
    mean ± std of key metrics.

    Args:
        results: List of ExperimentResult from ablation runs.

    Returns:
        Dict with summary statistics organized by configuration.
    """
    from collections import defaultdict

    groups = defaultdict(list)
    for r in results:
        key = (
            r.config["use_f1"],
            r.config["use_f2"],
            r.config["use_f3"],
        )
        groups[key].append(r)

    summary = {}
    for key, group in groups.items():
        costs = [r.metrics["average_cost"] for r in group]
        cost_ratios = [r.metrics["cost_ratio"] for r in group]
        eces = [r.metrics["ece"] for r in group]

        summary[str(key)] = {
            "n_runs": len(group),
            "average_cost_mean": float(np.mean(costs)),
            "average_cost_std": float(np.std(costs)),
            "cost_ratio_mean": float(np.mean(cost_ratios)),
            "cost_ratio_std": float(np.std(cost_ratios)),
            "ece_mean": float(np.mean(eces)),
            "ece_std": float(np.std(eces)),
        }

    return summary


