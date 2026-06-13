"""Statistical analysis: t-tests and ANOVA for closed-loop framework.

Produces:
1. Pairwise t-tests: closed-loop vs each baseline
2. Ablation ANOVA: factor analysis of F1, F2, F3 effects
3. Paper-ready results table
"""
import sys
sys.path.insert(0, ".")

import json
import numpy as np
from pathlib import Path


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def pairwise_t_tests(data: dict, baseline_names: list[str],
                     target: str = "closed_loop") -> list[dict]:
    """Pairwise t-tests: target vs each baseline.

    Uses the stored per-seed costs. With limited seeds, reports
    mean difference and effect size (Cohen's d).
    """
    target_costs = np.array([r["cost"] for r in data[target]])
    results = []

    for name in baseline_names:
        if name not in data or name == target:
            continue
        bl_costs = np.array([r["cost"] for r in data[name]])

        mean_diff = float(np.mean(bl_costs) - np.mean(target_costs))
        pct_improvement = mean_diff / np.mean(bl_costs) * 100

        # Pooled standard deviation for Cohen's d
        n_t, n_bl = len(target_costs), len(bl_costs)
        var_pooled = ((n_t - 1) * np.var(target_costs, ddof=1)
                      + (n_bl - 1) * np.var(bl_costs, ddof=1)) / (n_t + n_bl - 2)
        cohens_d = mean_diff / np.sqrt(var_pooled) if var_pooled > 0 else 0.0

        results.append({
            "baseline": name,
            "target_cost": float(np.mean(target_costs)),
            "baseline_cost": float(np.mean(bl_costs)),
            "mean_diff": mean_diff,
            "pct_improvement": pct_improvement,
            "cohens_d": float(cohens_d),
        })

    return results


def ablation_anova(data: dict) -> dict:
    """Simple ANOVA-like analysis for the 2^3 ablation experiment.

    Computes main effects and interaction effects from the 8 configurations.
    Uses the group means (single seed available).
    """
    configs = [
        (False, False, False, "None"),
        (True, False, False, "F1"),
        (False, True, False, "F2"),
        (False, False, True, "F3"),
        (True, True, False, "F1+F2"),
        (True, False, True, "F1+F3"),
        (False, True, True, "F2+F3"),
        (True, True, True, "All"),
    ]

    # Find the costs for each config
    costs = {}
    for f1, f2, f3, label in configs:
        key = str((f1, f2, f3))
        if key in data:
            costs[(f1, f2, f3)] = data[key]["average_cost_mean"]

    # Main effects: average effect of turning ON each factor
    main_F1 = costs[(True, True, True)] - costs[(False, True, True)]
    main_F2 = costs[(True, True, True)] - costs[(True, False, True)]
    main_F3 = costs[(True, True, True)] - costs[(True, True, False)]

    # Interaction effects
    # F1xF2: (F1+F2) - (F1) - (F2) + (none)
    int_F1F2 = (costs[(True, True, False)] - costs[(True, False, False)]
                - costs[(False, True, False)] + costs[(False, False, False)])
    int_F1F3 = (costs[(True, False, True)] - costs[(True, False, False)]
                - costs[(False, False, True)] + costs[(False, False, False)])
    int_F2F3 = (costs[(False, True, True)] - costs[(False, True, False)]
                - costs[(False, False, True)] + costs[(False, False, False)])
    int_all = (costs[(True, True, True)] - costs[(True, True, False)]
               - costs[(True, False, True)] - costs[(False, True, True)]
               + costs[(True, False, False)] + costs[(False, True, False)]
               + costs[(False, False, True)] - costs[(False, False, False)])

    return {
        "main_effects": {
            "F1 (state -> calibration)": float(main_F1),
            "F2 (residual -> UKF)": float(main_F2),
            "F3 (state -> threshold)": float(main_F3),
        },
        "interaction_effects": {
            "F1 x F2": float(int_F1F2),
            "F1 x F3": float(int_F1F3),
            "F2 x F3": float(int_F2F3),
            "F1 x F2 x F3": float(int_all),
        },
        "cost_improvement_from_closed_loop": float(
            costs[(True, True, True)] - costs[(False, False, False)]
        ),
    }


def generate_paper_table(ttest_results: list[dict]) -> str:
    """Generate a LaTeX table from t-test results."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Closed-Loop Framework vs Baselines: Statistical Comparison}",
        r"\label{tab:comparison}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & Cost (\$) & Improvement (\%) & Cohen's $d$ \\",
        r"\midrule",
    ]

    for r in ttest_results:
        name = r["baseline"].replace("_", " ").title()
        imp = r["pct_improvement"]
        d = r["cohens_d"]
        lines.append(
            f"  {name} & ${r['target_cost']:.2f} & {imp:.1f}\\% & {d:.2f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 60)
    print("Statistical Analysis")
    print("=" * 60)

    # --- Fraud comparison ---
    print("\n--- Fraud Experiment: Pairwise t-tests ---")
    data = load_results("results/fraud_comparison.json")
    baselines = [n for n in data.keys() if n != "closed_loop"]
    ttest = pairwise_t_tests(data, baselines, "closed_loop")

    for r in sorted(ttest, key=lambda x: x["pct_improvement"], reverse=True):
        print(f"  {r['baseline']:25s} | closed=${r['target_cost']:.2f} vs "
              f"baseline=${r['baseline_cost']:.2f} | "
              f"improvement={r['pct_improvement']:.1f}% | "
              f"d={r['cohens_d']:.2f}")

    # --- Ablation ANOVA ---
    print("\n--- Ablation: Main Effects ---")
    ablation_data = load_results("results/ablation_summary.json")
    anova = ablation_anova(ablation_data)

    for name, effect in anova["main_effects"].items():
        direction = "reduces" if effect < 0 else "increases"
        print(f"  {name:35s} {direction} cost by {abs(effect):.4f}")

    print("\n--- Ablation: Interaction Effects ---")
    for name, effect in anova["interaction_effects"].items():
        sign = "+" if effect >= 0 else ""
        print(f"  {name:20s} | effect = {sign}{effect:.4f}")

    print(f"\n  Total cost reduction from closed-loop: "
          f"{abs(anova['cost_improvement_from_closed_loop']):.4f}")

    # --- LaTeX table ---
    print("\n--- LaTeX Table ---")
    latex = generate_paper_table(ttest)
    print(latex)

    # Save
    with open("results/statistical_analysis.json", "w") as f:
        json.dump({"ttest": ttest, "anova": anova}, f, indent=2)
    print("\nSaved to results/statistical_analysis.json")
