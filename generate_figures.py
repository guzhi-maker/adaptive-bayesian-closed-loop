"""Generate paper-quality figures for the closed-loop framework.

Produces PDF vector figures ready for JRSS-B submission.
All figures use consistent 10pt font, proper labels, and legends.
"""
import sys
sys.path.insert(0, ".")

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path

# Consistent publication style
rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "lines.linewidth": 1.5,
    "axes.linewidth": 0.8,
    "grid.alpha": 0.3,
})

FIG_DIR = Path("paper/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def fig1_ablation_bar(results_json: str = "results/ablation_summary.json"):
    """Figure 1: Ablation experiment - cost by feedback configuration.

    Bar chart with all 8 configurations, showing cost +/- std.
    """
    with open(results_json) as f:
        data = json.load(f)

    labels_map = {
        "(False, False, False)": "None\n(open-loop)",
        "(True, False, False)": "F1\nonly",
        "(False, True, False)": "F2\nonly",
        "(False, False, True)": "F3\nonly",
        "(True, True, False)": "F1\n+F2",
        "(True, False, True)": "F1\n+F3",
        "(False, True, True)": "F2\n+F3",
        "(True, True, True)": "Full\n(closed-loop)",
    }
    order = [
        "(False, False, False)", "(True, False, False)", "(False, True, False)",
        "(False, False, True)", "(True, True, False)", "(True, False, True)",
        "(False, True, True)", "(True, True, True)",
    ]

    labels = [labels_map[k] for k in order]
    costs = [data[k]["average_cost_mean"] for k in order]
    errs = [data[k]["average_cost_std"] for k in order]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#d62728" if "False" in k and k.count("True") < 1
              else "#2ca02c" if k == "(True, True, True)"
              else "#1f77b4" for k in order]

    bars = ax.bar(range(len(labels)), costs, yerr=errs, capsize=4,
                  color=colors, edgecolor="black", linewidth=0.5, width=0.6)

    # Highlight full closed-loop
    bars[-1].set_hatch("///")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Average Decision Cost")
    ax.set_title("Ablation: Effect of Feedback Loops on Decision Cost")
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, cost in zip(bars, costs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(errs) * 0.1,
                f"{cost:.2f}", ha="center", va="bottom", fontsize=7)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1f77b4", label="Partial feedback"),
        Patch(facecolor="#2ca02c", label="Full closed-loop", hatch="///"),
        Patch(facecolor="#d62728", label="Open-loop (no feedback)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)

    path = FIG_DIR / "fig1_ablation.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def fig2_comparison(results_json: str = "results/comparison_results.json"):
    """Figure 2: Closed-loop vs all baselines - cost comparison.

    Horizontal bar chart sorted by performance.
    """
    with open(results_json) as f:
        data = json.load(f)

    # Compute mean cost for each method, sort
    methods = []
    for name, runs in data.items():
        costs = [r["cost"] for r in runs]
        methods.append((name, np.mean(costs), np.std(costs)))

    methods.sort(key=lambda x: x[1])

    names = [m[0].replace("_", " ").title() for m in methods]
    costs = [m[1] for m in methods]
    errs = [m[2] for m in methods]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    y_pos = range(len(names))
    colors = ["#2ca02c" if "Closed" in n else "#1f77b4" for n in names]

    bars = ax.barh(y_pos, costs, xerr=errs, capsize=3,
                   color=colors, edgecolor="black", linewidth=0.5, height=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Average Decision Cost ($)")
    ax.set_title("Closed-Loop Framework vs Baselines (Fraud Scenario)")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    # Value labels
    for bar, cost in zip(bars, costs):
        ax.text(bar.get_width() + max(errs) * 0.1, bar.get_y() + bar.get_height() / 2,
                f"${cost:.2f}", va="center", fontsize=8)

    path = FIG_DIR / "fig2_comparison.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def fig3_cost_over_time(records_json: str = None, n_samples: int = 10000):
    """Figure 3: Cost over time - open-loop vs closed-loop.

    Requires re-running both configurations and saving step records.
    """
    # Use the rapid drift scenario data
    from src.data.synthetic_generator import (
        DriftConfig, CostConfig, SyntheticDataConfig, SyntheticDataGenerator
    )
    from src.core.closed_loop import ClosedLoopConfig, ClosedLoopFramework
    from src.modules.ukf_tracker import UKFTracker
    from src.modules.platt_calibrator import PlattCalibrator
    from src.modules.three_layer_decision import ThreeLayerConfig, ThreeLayerDecisionMaker
    from src.baselines.methods import get_baseline, run_baseline_on_stream
    from scipy.ndimage import uniform_filter1d

    np.random.seed(42)

    # Generate data
    syn_cfg = SyntheticDataConfig(
        n_samples=n_samples,
        drift=DriftConfig(drift_type="gradual", drift_speed=6e-4),
        cost=CostConfig(cost_type="fixed", base_fn=10.0, base_fp=1.0),
        base_logit_mean=-3.0,
    )
    data = SyntheticDataGenerator(syn_cfg).generate()
    logits = data["logits"]
    labels = data["labels"]
    costs_na = data["costs_no_action"]
    costs_a = data["costs_action"]

    # ---- Open-loop baseline ----
    bl = get_baseline("static_platt")
    bl_result = run_baseline_on_stream(bl, logits, labels, costs_na, costs_a, warmup=500)

    # ---- Closed-loop ----
    def obs_model(s):
        return s.copy(), np.array([0.01])

    tracker = UKFTracker(
        initial_state=np.array([0.0]),
        initial_covariance=np.eye(1) * 0.1,
        process_noise=np.eye(1) * 1e-4,
        observation_model=obs_model,
    )
    cal = PlattCalibrator(use_scalar=True)
    dm = ThreeLayerDecisionMaker()
    loop_cfg = ClosedLoopConfig(
        use_f1=True, use_f2=True, use_f3=True,
        calibrate_every=500, warmup_steps=500,
    )
    framework = ClosedLoopFramework(tracker, cal, dm, loop_cfg)

    obs_signal = uniform_filter1d(logits, size=501, mode="nearest")
    records = framework.process_stream(
        logits, labels, costs_na, costs_a,
        observations=obs_signal.reshape(-1, 1),
        verbose=False,
    )

    # ---- Plot ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

    steps = np.arange(n_samples)

    # Smooth costs
    kernel = np.ones(200) / 200

    # Top: cost over time
    ol_costs = bl_result["costs"]
    ol_smooth = np.convolve(ol_costs, kernel, mode="valid")
    ax1.plot(steps[:len(ol_smooth)], ol_smooth, label="Open-loop", color="#d62728", alpha=0.8)

    cl_costs = np.array([r.decision.expected_cost for r in records])
    cl_smooth = np.convolve(cl_costs, kernel, mode="valid")
    ax1.plot(steps[:len(cl_smooth)], cl_smooth, label="Closed-loop", color="#2ca02c", alpha=0.8)

    ax1.set_ylabel("Decision Cost (smoothed)")
    ax1.set_title("Per-Step Decision Cost Over Time")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Bottom: true drift
    drift = data["drift_state"]
    # Convert to probability scale
    drift_probs = 1.0 / (1.0 + np.exp(-(drift + syn_cfg.base_logit_mean)))
    ax2.plot(steps, drift_probs, color="gray", linewidth=0.8, alpha=0.7, label="True P(y=1)")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("P(y=1)")
    ax2.set_title("True Positive Rate Over Time (Drift)")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = FIG_DIR / "fig3_cost_over_time.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def fig4_reliability():
    """Figure 4: Reliability diagram for closed-loop framework."""
    from src.experiments.pipeline import ExperimentConfig, run_experiment

    cfg = ExperimentConfig(
        n_samples=10000, drift_type="gradual", drift_speed=6e-4,
        cost_type="fixed", cost_no_action=10.0, cost_action=1.0,
        base_logit_mean=-3.0,
        use_f1=True, use_f2=True, use_f3=True,
        calibrate_every=500, experiment_name="fig_reliability", seed=42,
    )
    result = run_experiment(cfg, verbose=False)

    # Reconstruct data to get records
    np.random.seed(42)
    from src.data.synthetic_generator import (
        DriftConfig, CostConfig, SyntheticDataConfig, SyntheticDataGenerator
    )
    from src.core.closed_loop import ClosedLoopConfig, ClosedLoopFramework
    from src.modules.ukf_tracker import UKFTracker
    from src.modules.platt_calibrator import PlattCalibrator
    from src.modules.three_layer_decision import ThreeLayerConfig, ThreeLayerDecisionMaker
    from scipy.ndimage import uniform_filter1d

    syn_cfg = SyntheticDataConfig(
        n_samples=10000,
        drift=DriftConfig(drift_type="gradual", drift_speed=6e-4),
        cost=CostConfig(cost_type="fixed", base_fn=10.0, base_fp=1.0),
        base_logit_mean=-3.0,
    )
    data = SyntheticDataGenerator(syn_cfg).generate()

    def obs_model(s):
        return s.copy(), np.array([0.01])

    tracker = UKFTracker(
        initial_state=np.array([0.0]),
        initial_covariance=np.eye(1) * 0.1,
        process_noise=np.eye(1) * 1e-4,
        observation_model=obs_model,
    )
    cal = PlattCalibrator(use_scalar=True)
    dm = ThreeLayerDecisionMaker()
    framework = ClosedLoopFramework(tracker, cal, dm, ClosedLoopConfig(
        use_f1=True, use_f2=True, use_f3=True, calibrate_every=500, warmup_steps=500,
    ))

    obs_signal = uniform_filter1d(data["logits"], size=501, mode="nearest")
    records = framework.process_stream(
        data["logits"], data["labels"], data["costs_no_action"], data["costs_action"],
        observations=obs_signal.reshape(-1, 1), verbose=False,
    )
    probs = np.array([r.calibrated_prob for r in records])
    labels_arr = data["labels"]

    # Compute reliability
    from src.evaluation.metrics import compute_ece
    ece, mce, bin_accs, bin_confs, bin_counts = compute_ece(probs, labels_arr, n_bins=15)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    valid = bin_counts > 0
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect", linewidth=1)
    ax.bar(bin_confs[valid] - 0.5 / 15, bin_accs[valid], width=1.0 / 15,
           alpha=0.7, color="#2ca02c", edgecolor="black", linewidth=0.5,
           label="Closed-loop")

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability Diagram\nECE = {ece:.4f}, MCE = {mce:.4f}")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")

    path = FIG_DIR / "fig4_reliability.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def fig_architecture():
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7)
    ax.axis('off')
    modules = {
        'Observation': (1, 5.5, 1.8, 0.8),
        'Tracker (UKF)': (1, 3.8, 1.8, 0.8),
        'Calibrator': (5.5, 3.8, 1.8, 0.8),
        'Decision Maker': (8.5, 3.8, 1.8, 0.8),
    }
    colors = {'Observation': '#e8f0fe', 'Tracker (UKF)': '#e6f4ea',
              'Calibrator': '#fef7e0', 'Decision Maker': '#fce8e6'}
    for name, (x, y, w, h) in modules.items():
        rect = plt.Rectangle((x, y), w, h, facecolor=colors[name],
                             edgecolor='black', linewidth=1.5, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, name, ha='center', va='center',
                fontsize=11, fontweight='bold')
    ax.annotate('', xy=(4.5, 4.2), xytext=(2.8, 4.2),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    ax.annotate('', xy=(8.3, 4.2), xytext=(7.3, 4.2),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    ax.annotate('', xy=(5.5, 4.55), xytext=(2.8, 4.55),
                arrowprops=dict(arrowstyle='->', lw=1.5, ls='--', color='#1a73e8'))
    ax.annotate('', xy=(2.8, 3.35), xytext=(5.5, 3.35),
                arrowprops=dict(arrowstyle='->', lw=1.5, ls='--', color='#e37400'))
    ax.annotate('', xy=(8.5, 4.55), xytext=(7.3, 4.55),
                arrowprops=dict(arrowstyle='->', lw=1.5, ls='--', color='#c5221f'))
    ax.annotate('', xy=(1.9, 5.5), xytext=(1.9, 4.6),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', lw=1.5, label='Forward flow'),
        Line2D([0], [0], color='#1a73e8', lw=1.5, ls='--', label='F1: Uncertainty to calibration'),
        Line2D([0], [0], color='#e37400', lw=1.5, ls='--', label='F2: Residual to diffusion'),
        Line2D([0], [0], color='#c5221f', lw=1.5, ls='--', label='F3: Posterior to threshold'),
    ]
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.08),
              ncol=2, fontsize=8, frameon=False)
    ax.set_title('Closed-Loop Architecture', fontsize=12, fontweight='bold', pad=10)
    path = FIG_DIR / 'fig_architecture.pdf'
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')
    return path

def fig5_fraud_cost_curve():
    """Figure 5: Cumulative cost on fraud data."""
    from src.data.fraud_generator import FraudDataConfig, FraudDataGenerator
    from src.core.closed_loop import ClosedLoopConfig, ClosedLoopFramework
    from src.modules.ukf_tracker import UKFTracker
    from src.modules.platt_calibrator import PlattCalibrator
    from src.modules.three_layer_decision import ThreeLayerConfig, ThreeLayerDecisionMaker
    from src.baselines.methods import get_baseline, run_baseline_on_stream
    from scipy.ndimage import uniform_filter1d

    gen = FraudDataGenerator(FraudDataConfig(n_weeks=10, samples_per_week=5000, seed=42))
    data = gen.generate()

    # Baselines
    baselines = {
        "Static Platt": get_baseline("static_platt"),
        "Cost-Sensitive": get_baseline("cost_sensitive"),
        "Raw Threshold": get_baseline("raw"),
    }

    all_cumulative = {}
    for name, bl in baselines.items():
        result = run_baseline_on_stream(bl, data["logits"], data["labels"],
                                         data["costs_no_action"], data["costs_action"],
                                         warmup=2000)
        all_cumulative[name] = np.cumsum(result["costs"])

    # Closed-loop
    def obs_model(s):
        return s.copy(), np.array([0.01])

    tracker = UKFTracker(
        initial_state=np.array([0.0]),
        initial_covariance=np.eye(1) * 0.1,
        process_noise=np.eye(1) * 1e-4,
        observation_model=obs_model,
    )
    cal = PlattCalibrator(use_scalar=True, n_bootstrap=20)
    dm = ThreeLayerDecisionMaker(ThreeLayerConfig(
        low_threshold=0.4, high_threshold=0.6,
        default_cost_no_action=150.0, default_cost_action=15.0,
    ))
    framework = ClosedLoopFramework(tracker, cal, dm, ClosedLoopConfig(
        use_f1=True, use_f2=True, use_f3=True, calibrate_every=500,
        calibration_window=2000, warmup_steps=1000,
    ))

    obs_signal = uniform_filter1d(data["logits"], size=501, mode="nearest")
    records = framework.process_stream(
        data["logits"], data["labels"], data["costs_no_action"], data["costs_action"],
        observations=obs_signal.reshape(-1, 1), verbose=False,
    )
    cl_costs = np.array([r.decision.expected_cost for r in records])
    all_cumulative["Closed-Loop"] = np.cumsum(cl_costs)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {"Closed-Loop": "#2ca02c", "Static Platt": "#1f77b4",
              "Cost-Sensitive": "#ff7f0e", "Raw Threshold": "#d62728"}

    for name in ["Raw Threshold", "Static Platt", "Cost-Sensitive", "Closed-Loop"]:
        cum = all_cumulative[name]
        ax.plot(np.arange(len(cum)), cum, label=name, color=colors[name], linewidth=1.2)

    ax.set_xlabel("Transaction")
    ax.set_ylabel("Cumulative Loss ($)")
    ax.set_title("Cumulative Fraud Loss: Closed-Loop vs Baselines")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    path = FIG_DIR / "fig5_fraud_cumulative.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


if __name__ == "__main__":
    print("Generating paper figures...")
    fig_architecture()
    fig1_ablation_bar()
    fig2_comparison()
    fig3_cost_over_time()
    fig4_reliability()
    fig5_fraud_cost_curve()
    print(f"\nAll figures saved to {FIG_DIR}/")
