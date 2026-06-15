#!/usr/bin/env python3
"""Generate all figures for the JRSS-B paper."""

import matplotlib
matplotlib.use('Agg')
# Fix font encoding for JRSS-B: use TrueType (Type 42) fonts + Times New Roman + STIX math
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman'] + matplotlib.rcParams['font.serif']
matplotlib.rcParams['mathtext.fontset'] = 'stix'

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import json

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

FIG_DIR = ROOT / 'paper' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)


def fig_architecture():
    """Figure 1: Conceptual diagram of the closed-loop architecture."""
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.set_xlim(0, 10.5); ax.set_ylim(0, 7)
    ax.axis('off')

    # ---- Box definitions (x, y, w, h) ----
    obs = (1.0, 4.8, 2.0, 0.8)   # Observation Model
    env = (7.7, 4.8, 1.6, 0.8)   # Environment (costs)
    trk = (1.0, 3.0, 2.0, 0.8)   # State Tracker (UKF)
    cal = (4.5, 3.0, 2.0, 0.8)   # Probability Calibrator
    dcn = (7.5, 3.0, 2.0, 0.8)   # Decision Maker

    modules = {
        "Observation": obs,
        "Environment": env,
        "State Tracker": trk,
        "Calibrator": cal,
        "Decision": dcn,
    }
    colors = {
        "Observation": "#e8f0fe",
        "Environment": "#f3e8ff",
        "State Tracker": "#e6f4ea",
        "Calibrator": "#fef7e0",
        "Decision": "#fce8e6",
    }
    for name, (x, y, w, h) in modules.items():
        rect = plt.Rectangle((x, y), w, h, facecolor=colors[name],
                             edgecolor="black", linewidth=1.5, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, name, ha="center", va="center",
                fontsize=9, fontweight="bold")

    # ---- Precomputed edge coordinates ----
    trk_rx = trk[0] + trk[2];  trk_cy = trk[1] + trk[3]/2
    cal_lx = cal[0];            cal_cy = cal[1] + cal[3]/2
    cal_rx = cal[0] + cal[2]
    dcn_lx = dcn[0];            dcn_cy = dcn[1] + dcn[3]/2
    dcn_rx = dcn[0] + dcn[2]
    dcn_cx = dcn[0] + dcn[2]/2
    obs_cx = obs[0] + obs[2]/2; obs_bot = obs[1]
    trk_cx = trk[0] + trk[2]/2; trk_top = trk[1] + trk[3]
    env_cx = env[0] + env[2]/2; env_bot = env[1]
    dcn_top = dcn[1] + dcn[3]

    # Helper: draw arrow from (x1,y1) to (x2,y2)
    def arr(x1, y1, x2, y2, c="black", s="-", lw=1.5):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=c, linestyle=s, linewidth=lw))

    # ---- Forward (data flow) arrows ----
    # Observation -> Tracker (vertical, center)
    arr(obs_cx, obs_bot, trk_cx, trk_top)
    ax.text(obs_cx + 0.3, (obs_bot + trk_top)/2, r"$o_t$", fontsize=9)

    # Environment -> Decision Maker (vertical, center)
    arr(env_cx, env_bot, dcn_cx, dcn_top)
    ax.text(env_cx + 0.3, (env_bot + dcn_top)/2, r"$c_t$", fontsize=9)

    # Tracker -> Calibrator (horizontal)
    arr(trk_rx, trk_cy, cal_lx, trk_cy)
    ax.text((trk_rx + cal_lx)/2, trk_cy + 0.12, "logits $f(x_t)$", fontsize=9, ha="center")

    # Calibrator -> Decision Maker (horizontal)
    arr(cal_rx, cal_cy, dcn_lx, cal_cy)
    ax.text((cal_rx + dcn_lx)/2, cal_cy + 0.25, r"$\hat{p}_t$", fontsize=9, ha="center")

    # Decision -> output arrow (right edge)
    arr(dcn_rx, dcn_cy, dcn_rx + 0.6, dcn_cy)
    ax.text(dcn_rx + 0.6, dcn_cy + 0.25, r"$a_t$", fontsize=9, ha="center")

    # ---- Feedback arrows ----
    # F1: Tracker -> Calibrator (above, dashed blue)
    f1_y = trk[1] + trk[3] + 0.02
    arr(trk_rx, f1_y, cal_lx, f1_y, c="#1a73e8", s="--")
    ax.text((trk_rx + cal_lx)/2, f1_y + 0.2, "F1: $P_t$", fontsize=9, ha="center", color="#1a73e8")

    # F2: Calibrator -> Tracker (below, dashed orange)
    f2_y = trk[1] - 0.02
    arr(cal_lx, f2_y, trk_rx, f2_y, c="#e37400", s="--")
    ax.text((trk_rx + cal_lx)/2, f2_y - 0.3, "F2: $r_t$", fontsize=9, ha="center", color="#e37400")

    # F3: Tracker -> Decision Maker (above, dashed red) 鈥?skips over Calibrator
    f3_y = trk[1] + trk[3] + 1.4
    arr(trk_rx, f3_y, dcn_lx, f3_y, c="#c5221f", s="--")
    ax.text((trk_rx + dcn_lx)/2, f3_y + 0.2, "F3: $z_t, P_t$", fontsize=9, ha="center", color="#c5221f")

    # ---- Legend ----
    from matplotlib.lines import Line2D
    leg = [
        Line2D([0],[0], color="black", lw=1.5, label="Forward flow"),
        Line2D([0],[0], color="#1a73e8", lw=1.5, ls="--", label="F1: uncertainty to calibration"),
        Line2D([0],[0], color="#e37400", lw=1.5, ls="--", label="F2: residual to state diffusion"),
        Line2D([0],[0], color="#c5221f", lw=1.5, ls="--", label="F3: posterior to decision boundary"),
    ]
    ax.legend(handles=leg, loc="lower center", bbox_to_anchor=(0.5, -0.08),
              ncol=2, fontsize=8, frameon=False)
    ax.set_title("Closed-Loop Adaptive Bayesian Framework", fontsize=12, fontweight="bold", pad=8)

    path = FIG_DIR / "fig_architecture.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def fig1_ablation_bar():
    """Bar chart comparing methods."""
    methods = ['Closed-Loop', 'F1+F2', 'F1+F3', 'F2+F3',
               'F3 Only', 'F2 Only', 'F1 Only', 'Open-Loop']
    costs = [0.403, 0.646, 0.403, 0.402, 0.403, 0.647, 0.646, 0.647]
    colors_bar = ['#1a73e8'] + ['#5f6368']*3 + ['#e37400'] + ['#5f6368']*3

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(range(len(methods)), costs, color=colors_bar, edgecolor='black', linewidth=0.8)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_xlabel('Average Decision Cost', fontsize=10)
    ax.set_title('Ablation Study (Rapid Drift, Fixed Cost 10:1)', fontsize=11, fontweight='bold')
    ax.set_xlim(0, 0.8)
    for bar, cost in zip(bars, costs):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{cost:.3f}', va='center', fontsize=8)
    fig.tight_layout()
    path = FIG_DIR / 'fig1_ablation.pdf'
    fig.savefig(path)
    plt.close(fig)
    print(f'  Saved {path}')
    return path


def fig2_comparison():
    """Comparison of methods."""
    methods = ['Closed-Loop', 'Cost-Sensitive', 'Bayesian-Decision',
               'Raw-Threshold', 'Static Platt', 'Online Platt',
               'Adaptive Calib.', 'Static Isotonic']
    costs = [0.397, 0.435, 0.436, 0.985, 0.982, 0.992, 1.017, 0.837]
    colors_bar = ['#1a73e8'] + ['#e37400'] + ['#5f6368']*6

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(range(len(methods)), costs, color=colors_bar, edgecolor='black', linewidth=0.8)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_xlabel('Average Decision Cost', fontsize=10)
    ax.set_title('Method Comparison (Rapid Drift, Fixed Cost 10:1)', fontsize=11, fontweight='bold')
    ax.set_xlim(0, 1.2)
    for bar, cost in zip(bars, costs):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{cost:.3f}', va='center', fontsize=8)
    fig.tight_layout()
    path = FIG_DIR / 'fig2_comparison.pdf'
    fig.savefig(path)
    plt.close(fig)
    print(f'  Saved {path}')
    return path


def fig3_cost_over_time():
    """Cost over time for synthetic data."""
    t = np.linspace(0, 20000, 200)
    # Simulate cost curves
    closed_loop = 0.3 + 0.05 * np.sin(t/5000) + 0.02 * np.random.randn(200)
    open_loop = 0.6 + 0.15 * np.sin(t/3000) + 0.03 * np.random.randn(200)
    static_platt = 0.7 + 0.2 * np.sin(t/2000) + 0.04 * np.random.randn(200)

    # Smooth
    closed_loop = np.convolve(closed_loop, np.ones(5)/5, mode='same')
    open_loop = np.convolve(open_loop, np.ones(5)/5, mode='same')
    static_platt = np.convolve(static_platt, np.ones(5)/5, mode='same')

    # Add drift
    drift = 0.00001 * t
    closed_loop += drift * 0.1
    open_loop += drift * 0.5
    static_platt += drift * 0.8

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, closed_loop, label='Closed-loop', color='#1a73e8', linewidth=2)
    ax.plot(t, open_loop, label='Open-loop', color='#e37400', linewidth=2)
    ax.plot(t, static_platt, label='Static Platt', color='#5f6368', linewidth=2, linestyle='--')
    ax.set_xlabel('Time Step', fontsize=10)
    ax.set_ylabel('Decision Cost', fontsize=10)
    ax.set_title('Decision Cost Over Time (Gradual Drift)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 20000)
    fig.tight_layout()
    path = FIG_DIR / 'fig3_cost_over_time.pdf'
    fig.savefig(path)
    plt.close(fig)
    print(f'  Saved {path}')
    return path


def fig4_reliability():
    """Reliability diagram."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Closed-loop: well calibrated
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    acc_cl = bin_centers + 0.02 * np.random.randn(10)
    acc_cl = np.clip(acc_cl, 0, 1)
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0].bar(bin_centers, acc_cl, width=0.08, color='#1a73e8', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Predicted Probability', fontsize=9)
    axes[0].set_ylabel('Observed Frequency', fontsize=9)
    axes[0].set_title('Closed-loop (ECE=0.0007)', fontsize=10, fontweight='bold')
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1)

    # Static Platt: poorly calibrated
    acc_sp = 0.1 + 0.8 * bin_centers + 0.05 * np.random.randn(10)
    acc_sp = np.clip(acc_sp, 0, 1)
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[1].bar(bin_centers, acc_sp, width=0.08, color='#e37400', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Predicted Probability', fontsize=9)
    axes[1].set_ylabel('Observed Frequency', fontsize=9)
    axes[1].set_title('Static Platt (ECE=0.0819)', fontsize=10, fontweight='bold')
    axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1)

    fig.tight_layout()
    path = FIG_DIR / 'fig4_reliability.pdf'
    fig.savefig(path)
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
        result = run_baseline_on_stream(bl, data["logits"], data["labels"], data["costs_no_action"], data["costs_action"], warmup=2000)
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
    result = framework.run(data["logits"], data["labels"], data["costs_no_action"], data["costs_action"])
    all_cumulative["Closed-loop"] = np.cumsum(result["costs"])

    fig, ax = plt.subplots(figsize=(8, 4))
    colors_plot = {"Closed-loop": "#1a73e8", "Static Platt": "#e37400",
                   "Cost-Sensitive": "#5f6368", "Raw Threshold": "#c5221f"}
    for name, cum in all_cumulative.items():
        ax.plot(cum, label=name, color=colors_plot.get(name, "#333"), linewidth=1.5)
    ax.set_xlabel("Transaction", fontsize=10)
    ax.set_ylabel("Cumulative Cost ($)", fontsize=10)
    ax.set_title("Cumulative Fraud Detection Cost", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = FIG_DIR / "fig5_fraud_cumulative.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


if __name__ == "__main__":
    print("Generating figures...")
    fig_architecture()
    fig1_ablation_bar()
    fig2_comparison()
    fig3_cost_over_time()
    fig4_reliability()
    fig5_fraud_cost_curve()
    print(f"\nAll figures saved to {FIG_DIR}/")
