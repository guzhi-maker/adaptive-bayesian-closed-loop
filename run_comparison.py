"""Comparison experiment: closed-loop vs all baselines."""
import sys
sys.path.insert(0, ".")

import json
import time
import numpy as np
# Plain formatting

from src.experiments.pipeline import ExperimentConfig, run_experiment
from src.baselines.methods import (
    get_baseline, run_baseline_on_stream, BASELINE_REGISTRY
)
from src.evaluation.metrics import compute_all_decision_metrics

# =========================================================================
# Configuration
# =========================================================================
N_SAMPLES = 10000
DRIFT_SPEED = 6e-4  # Fast drift: 5% -> 95% positives
SEEDS = [42, 43, 44]

all_results = {}

for seed in SEEDS:
    print(f"\n{'=' * 60}")
    print(f"Seed = {seed}")
    print(f"{'=' * 60}")

    # Generate shared data
    cfg = ExperimentConfig(
        n_samples=N_SAMPLES,
        drift_type="gradual",
        drift_speed=DRIFT_SPEED,
        cost_type="fixed",
        cost_no_action=10.0,
        cost_action=1.0,
        base_logit_mean=-3.0,
        seed=seed,
    )

    # Quick data generation (reuse the generator logic)
    from src.data.synthetic_generator import (
        CostConfig, DriftConfig, SyntheticDataConfig, SyntheticDataGenerator
    )
    syn_cfg = SyntheticDataConfig(
        n_samples=N_SAMPLES,
        drift=DriftConfig(drift_type="gradual", drift_speed=DRIFT_SPEED),
        cost=CostConfig(cost_type="fixed", base_fn=10.0, base_fp=1.0),
        base_logit_mean=-3.0,
        seed=seed,
    )
    data = SyntheticDataGenerator(syn_cfg).generate()
    logits = data["logits"]
    labels = data["labels"]
    costs_na = data["costs_no_action"]
    costs_a = data["costs_action"]

    # ---- Baselines ----
    for name in BASELINE_REGISTRY:
        t0 = time.perf_counter()
        bl = get_baseline(name)
        result = run_baseline_on_stream(bl, logits, labels, costs_na, costs_a, warmup=500)
        elapsed = time.perf_counter() - t0

        metrics = compute_all_decision_metrics(
            probs=result["probs"],
            labels=labels,
            costs_no_action=costs_na,
            costs_action=costs_a,
            decisions=result["decisions"],
            n_bootstrap=20,
        )
        all_results.setdefault(name, []).append({
            "seed": seed,
            "cost": metrics.average_cost,
            "cumulative_cost": metrics.cumulative_cost,
            "oracle_cost": metrics.oracle_cost,
            "ece": metrics.ece,
            "cost_ratio": metrics.cost_ratio,
            "accept_rate": metrics.accept_rate,
            "runtime": elapsed,
        })
        print(f"  {bl.name:25s} | cost={metrics.average_cost:.4f} | "
              f"ece={metrics.ece:.6f} | accept={metrics.accept_rate:.2%} | "
              f"{elapsed:.2f}s")

    # ---- Closed-loop framework ----
    t0 = time.perf_counter()
    cl_result = run_experiment(cfg, verbose=False)
    elapsed = time.perf_counter() - t0

    all_results.setdefault("closed_loop", []).append({
        "seed": seed,
        "cost": cl_result.average_cost,
        "cumulative_cost": cl_result.cumulative_cost,
        "oracle_cost": cl_result.metrics["oracle_cost"],
        "ece": cl_result.metrics["ece"],
        "cost_ratio": cl_result.metrics["cost_ratio"],
        "accept_rate": cl_result.metrics["accept_rate"],
        "runtime": cl_result.runtime_seconds,
    })
    print(f"  {'Closed-Loop Framework':25s} | cost={cl_result.average_cost:.4f} | "
          f"ece={cl_result.metrics['ece']:.6f} | "
          f"accept={cl_result.metrics['accept_rate']:.2%} | "
          f"{cl_result.runtime_seconds:.2f}s")

# =========================================================================
# Summary across seeds
# =========================================================================
print("\n" + "=" * 60)
print("COMPARISON SUMMARY (mean +/- std across seeds)")
print("=" * 60)

rows = []
for name, runs in sorted(all_results.items()):
    costs = [r["cost"] for r in runs]
    eces = [r["ece"] for r in runs]
    accept = [r["accept_rate"] for r in runs]
    runtimes = [r["runtime"] for r in runs]
    display_name = runs[0].get("display_name", name)

    rows.append([
        name[:28],
        f"{np.mean(costs):.4f} +/- {np.std(costs):.4f}",
        f"{np.mean(eces):.6f}",
        f"{np.mean(accept):.2%}",
        f"{np.mean(runtimes):.2f}s",
    ])

headers = ["Method", "Avg Cost", "ECE", "Accept", "Runtime"]
print(f"{'Method':28s} {'Avg Cost':>20s} {'ECE':>12s} {'Accept':>8s} {'Runtime':>8s}")
print("-" * 80)
for r in rows:
    print(f"{r[0]:28s} {r[1]:>20s} {r[2]:>12s} {r[3]:>8s} {r[4]:>8s}")

# Find best
best = min(all_results.items(), key=lambda x: np.mean([r["cost"] for r in x[1]]))
print(f"\nBest method: {best[0]} (cost={np.mean([r['cost'] for r in best[1]]):.4f})")

# Save
with open("results/comparison_results.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print("\nResults saved to results/comparison_results.json")
