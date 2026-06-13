#!/usr/bin/env python3
"""Adaptive Bayesian Closed-Loop Framework - Main Entry Point.

Usage:
    python main.py --quick          # Quick smoke test
    python main.py --ablation       # Full 2³ ablation experiment
    python main.py --counterfactual # Counterfactual experiments
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def run_quick_test(verbose: bool = True):
    """Quick smoke test to verify the framework runs end-to-end."""
    if verbose:
        print("=" * 60)
        print("Quick Smoke Test")
        print("=" * 60)

    from src.experiments.pipeline import ExperimentConfig, run_experiment

    cfg = ExperimentConfig(
        n_samples=5_000,
        drift_type="gradual",
        drift_speed=2e-4,
        cost_type="fixed",
        use_f1=True,
        use_f2=True,
        use_f3=True,
        calibrate_every=500,
        experiment_name="quick_test",
        seed=42,
    )

    result = run_experiment(cfg, verbose=verbose)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Quick test complete!")
        print(f"  Average cost: {result.average_cost:.4f}")
        print(f"  Cost ratio:   {result.metrics['cost_ratio']:.4f}")
        print(f"  ECE:          {result.metrics['ece']:.6f}")
        print(f"  Runtime:      {result.runtime_seconds:.2f}s")
        print(f"{'=' * 60}")

    return result


def run_ablation(verbose: bool = True):
    """Run full 2³ factorial ablation experiment."""
    if verbose:
        print("=" * 60)
        print("2³ Factorial Ablation Experiment")
        print("=" * 60)

    from src.experiments.pipeline import (
        ExperimentConfig,
        compute_ablation_summary,
        run_ablation_experiments,
    )

    base_cfg = ExperimentConfig(
        n_samples=30_000,
        drift_type="gradual",
        drift_speed=2e-4,
        cost_type="fixed",
        calibrate_every=500,
        experiment_name="ablation",
        seed=42,
    )

    results = run_ablation_experiments(base_cfg, seeds=[42, 43, 44], verbose=verbose)
    summary = compute_ablation_summary(results)

    # Save summary
    summary_path = Path("results") / "ablation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"\n{'=' * 60}")
        print("Ablation Summary:")
        for key, val in summary.items():
            print(
                f"  {key}: cost={val['average_cost_mean']:.4f}±{val['average_cost_std']:.4f}, "
                f"cost_ratio={val['cost_ratio_mean']:.4f}±{val['cost_ratio_std']:.4f}"
            )
        print(f"\nSummary saved to {summary_path}")
        print(f"{'=' * 60}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive Bayesian Closed-Loop Framework"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run quick smoke test"
    )
    parser.add_argument(
        "--ablation", action="store_true", help="Run 2³ ablation experiment"
    )
    parser.add_argument(
        "--counterfactual", action="store_true", help="Run counterfactual experiments"
    )
    parser.add_argument(
        "--all", action="store_true", help="Run all experiments"
    )

    args = parser.parse_args()

    # Default: run quick test
    if not any(vars(args).values()):
        args.quick = True

    if args.quick or args.all:
        run_quick_test(verbose=True)

    if args.ablation or args.all:
        run_ablation(verbose=True)

    if args.counterfactual or args.all:
        from src.experiments.pipeline import (
            ExperimentConfig,
            run_counterfactual_experiments,
        )

        base_cfg = ExperimentConfig(
            n_samples=30_000,
            drift_type="gradual",
            drift_speed=2e-4,
            cost_type="fixed",
            experiment_name="counterfactual",
        )
        results = run_counterfactual_experiments(base_cfg, seeds=[42], verbose=True)

    print("\nAll experiments complete!")


if __name__ == "__main__":
    main()




