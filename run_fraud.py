"""Fraud detection experiment: closed-loop vs baselines on realistic fraud data."""
import sys
sys.path.insert(0, ".")

import time
import json
import numpy as np

from src.data.fraud_generator import FraudDataConfig, FraudDataGenerator
from src.core.closed_loop import ClosedLoopConfig, ClosedLoopFramework
from src.modules.ukf_tracker import UKFTracker
from src.modules.platt_calibrator import PlattCalibrator
from src.modules.three_layer_decision import ThreeLayerConfig, ThreeLayerDecisionMaker
from src.evaluation.metrics import compute_all_decision_metrics
from src.baselines.methods import (
    get_baseline, run_baseline_on_stream, BASELINE_REGISTRY
)

# =========================================================================
# Configuration
# =========================================================================
N_WEEKS = 10
SAMPLES_PER_WEEK = 5000  # 260K total for faster runtime
SEEDS = [42, 43]

all_results = {}

for seed in SEEDS:
    print(f"\n{'=' * 60}")
    print(f"Fraud Experiment | Seed = {seed}")
    print(f"{'=' * 60}")

    # Generate fraud data
    cfg = FraudDataConfig(
        n_weeks=N_WEEKS,
        samples_per_week=SAMPLES_PER_WEEK,
        seed=seed,
    )
    gen = FraudDataGenerator(cfg)
    data = gen.generate()
    n = len(data["logits"])

    print(f"  Samples: {n:,}")
    print(f"  Fraud rate: {data['labels'].mean():.4f}")
    print(f"  Avg amount: ${data['amounts'].mean():.2f}")
    print(f"  Weekly fraud rates: [{data['fraud_rate'][0]:.4f} -> {data['fraud_rate'][-1]:.4f}]")

    # ---- Baselines ----
    for name in BASELINE_REGISTRY:
        t0 = time.perf_counter()
        bl = get_baseline(name)
        result = run_baseline_on_stream(
            bl, data["logits"], data["labels"],
            data["costs_no_action"], data["costs_action"],
            warmup=2000,
        )
        elapsed = time.perf_counter() - t0

        metrics = compute_all_decision_metrics(
            probs=result["probs"],
            labels=data["labels"],
            costs_no_action=data["costs_no_action"],
            costs_action=data["costs_action"],
            decisions=result["decisions"],
            n_bootstrap=20,
        )
        key = f"{name}_seed={seed}"
        all_results.setdefault(name, []).append({
            "seed": seed,
            "cost": metrics.average_cost,
            "cumulative_cost": metrics.cumulative_cost,
            "ece": metrics.ece,
            "accept_rate": metrics.accept_rate,
            "runtime": elapsed,
        })
        print(f"  {bl.name:25s} | cost=${metrics.average_cost:.4f} | "
              f"ece={metrics.ece:.6f} | accept={metrics.accept_rate:.2%} | "
              f"{elapsed:.1f}s")

    # ---- Closed-loop framework ----
    t0 = time.perf_counter()

    # UKF tracker
    def obs_model(s):
        return s.copy(), np.array([0.01])

    tracker = UKFTracker(
        initial_state=np.array([0.0]),
        initial_covariance=np.eye(1) * 0.1,
        process_noise=np.eye(1) * 1e-4,
        observation_model=obs_model,
    )

    # Platt calibrator
    calibrator = PlattCalibrator(use_scalar=True, n_bootstrap=30)

    # Decision maker
    decision_config = ThreeLayerConfig(
        low_threshold=0.4, high_threshold=0.6,
        default_cost_no_action=150.0, default_cost_action=15.0,
    )
    decision_maker = ThreeLayerDecisionMaker(config=decision_config)

    # Closed-loop config
    loop_config = ClosedLoopConfig(
        use_f1=True, use_f2=True, use_f3=True,
        calibrate_every=500,
        calibration_window=2000,
        warmup_steps=1000,
        bootstrap_n=30,
    )

    framework = ClosedLoopFramework(
        tracker=tracker,
        calibrator=calibrator,
        decision_maker=decision_maker,
        config=loop_config,
    )

    # Use logit running mean as UKF observation (no cheating with ground truth)
    from scipy.ndimage import uniform_filter1d
    obs_signal = uniform_filter1d(data["logits"], size=501, mode="nearest")

    records = framework.process_stream(
        logits=data["logits"],
        labels=data["labels"],
        costs_no_action=data["costs_no_action"],
        costs_action=data["costs_action"],
        observations=obs_signal.reshape(-1, 1),
        verbose=False,
    )
    elapsed = time.perf_counter() - t0

    cal_probs = np.array([r.calibrated_prob for r in records])
    decisions = np.array([1 if r.decision.action.value == "reject" else 0 for r in records])

    metrics = compute_all_decision_metrics(
        probs=cal_probs,
        labels=data["labels"],
        costs_no_action=data["costs_no_action"],
        costs_action=data["costs_action"],
        decisions=decisions,
        n_bootstrap=20,
    )

    all_results.setdefault("closed_loop", []).append({
        "seed": seed,
        "cost": metrics.average_cost,
        "cumulative_cost": metrics.cumulative_cost,
        "ece": metrics.ece,
        "accept_rate": metrics.accept_rate,
        "runtime": elapsed,
    })
    print(f"  {'Closed-Loop Framework':25s} | cost=${metrics.average_cost:.4f} | "
          f"ece={metrics.ece:.6f} | accept={metrics.accept_rate:.2%} | "
          f"{elapsed:.1f}s")

# =========================================================================
# Summary
# =========================================================================
print("\n" + "=" * 60)
print("FRAUD EXPERIMENT SUMMARY (mean across seeds)")
print("=" * 60)
print(f"{'Method':28s} {'Avg Cost':>10s} {'ECE':>12s} {'Accept':>8s}")
print("-" * 60)

results_sorted = sorted(all_results.items(), key=lambda x: np.mean([r["cost"] for r in x[1]]))
for name, runs in results_sorted:
    costs = [r["cost"] for r in runs]
    eces = [r["ece"] for r in runs]
    accept = [r["accept_rate"] for r in runs]
    print(f"{name[:28]:28s} ${np.mean(costs):>7.2f} {np.mean(eces):>11.6f} {np.mean(accept):>7.2%}")

best_name = results_sorted[0][0]
best_cost = np.mean([r["cost"] for r in all_results[best_name]])
second_cost = np.mean([r["cost"] for r in all_results[results_sorted[1][0]]]) if len(results_sorted) > 1 else best_cost
improvement = (second_cost - best_cost) / second_cost * 100
print(f"\nBest: {best_name} (${best_cost:.2f})")
print(f"Improvement over #2: {improvement:.1f}%")

with open("results/fraud_comparison.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print("Results saved to results/fraud_comparison.json")
