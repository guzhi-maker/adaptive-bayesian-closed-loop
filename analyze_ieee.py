#!/usr/bin/env python3
"""Compute KL divergence and stratified cost analysis for IEEE-CIS data."""
import sys, json, warnings
import numpy as np

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from src.baselines.methods import get_baseline, run_baseline_on_stream

# Load data
d = np.load("data/processed/ieee_cis_processed.npz")
logits = d["logits"]
labels = d["labels"]
costs_fn = d["costs_fn"]
costs_fp = d["costs_fp"]
amounts = d["transaction_amt"]
weeks = d["week_labels"]
n_weeks = int(weeks.max()) + 1

print("=" * 60)
print("IEEE-CIS Data Analysis")
print("=" * 60)

# ============ 1. KL Divergence ============
print("\n--- Weekly KL Divergence ---")
kl_divs = []
for w in range(1, n_weeks):
    mask_prev = weeks == w - 1
    mask_cur = weeks == w
    p_prev = labels[mask_prev].mean() + 1e-10
    p_cur = labels[mask_cur].mean() + 1e-10
    # KL(P_cur || P_prev) for Bernoulli
    kl = p_cur * np.log(p_cur / p_prev) + (1 - p_cur) * np.log((1 - p_cur) / (1 - p_prev))
    kl_divs.append(kl)
    if w < 8 or w % 5 == 0:
        print(f"  Week {w-1}->{w}: KL={kl:.6f}, fraud_rate={p_cur:.4f}")

avg_kl = np.mean(kl_divs)
max_kl = np.max(kl_divs)
print(f"\n  Average weekly KL: {avg_kl:.6f}")
print(f"  Max weekly KL:     {max_kl:.6f}")
print(f"  Equivalent drift speed: ~{avg_kl:.6f}")
print(f"  (Synthetic gradual drift with speed 5e-4 produces KL ~ 0.03 per 1000 steps)")

# ============ 2. Stratified Cost Analysis ============
print("\n--- Stratified Cost Analysis ---")
# Define strata
low_mask = amounts < 50
mid_mask = (amounts >= 50) & (amounts < 500)
high_mask = amounts >= 500

strata = [
    ("Low (<$50)", low_mask),
    ("Medium ($50-$500)", mid_mask),
    ("High (>$500)", high_mask),
]

# Run a few key baselines + closed-loop on full data
baseline_names = ["static_platt", "raw", "bayesian"]
all_results = {}

for name in baseline_names:
    bl = get_baseline(name)
    result = run_baseline_on_stream(bl, logits, labels, costs_fn, costs_fp, warmup=500)
    all_results[name] = {
        "probs": result["probs"],
        "decisions": result["decisions"],
        "costs": result["costs"],
        "name": result["name"],
    }
    print(f"  {result['name']}: complete")

# Run closed-loop on first 50K
from src.core.closed_loop import ClosedLoopConfig, ClosedLoopFramework
from src.modules.ukf_tracker import UKFTracker
from src.modules.platt_calibrator import PlattCalibrator
from src.modules.three_layer_decision import ThreeLayerConfig, ThreeLayerDecisionMaker
from scipy.ndimage import uniform_filter1d

n_cl = 50000
tracker = UKFTracker(
    initial_state=np.array([0.0]),
    initial_covariance=np.eye(1) * 0.1,
    process_noise=np.eye(1) * 1e-3,
    observation_model=lambda s: (s.copy(), np.array([0.01])),
    state_clamp=(-8.0, 8.0),
)
calibrator = PlattCalibrator(use_scalar=True, n_bootstrap=50)
decision_config = ThreeLayerConfig(
    low_threshold=0.3, high_threshold=0.7,
    default_cost_no_action=10.0, default_cost_action=1.0,
)
decision_maker = ThreeLayerDecisionMaker(config=decision_config)
loop_config = ClosedLoopConfig(
    use_f1=True, use_f2=True, use_f3=True,
    calibrate_every=500, calibration_window=2000,
    warmup_steps=500, bootstrap_n=50,
)
framework = ClosedLoopFramework(
    tracker=tracker, calibrator=calibrator,
    decision_maker=decision_maker, config=loop_config,
)

obs = uniform_filter1d(logits[:n_cl], size=501, mode="nearest").reshape(-1, 1)
records = framework.process_stream(
    logits=logits[:n_cl], labels=labels[:n_cl],
    costs_no_action=costs_fn[:n_cl], costs_action=costs_fp[:n_cl],
    observations=obs, verbose=False,
)
cl_probs = np.array([r.calibrated_prob for r in records])
cl_decisions = np.array([1 if r.decision.action.name == "REJECT" else 0 for r in records])
cl_costs = np.array([r.decision.expected_cost for r in records])
all_results["closed_loop"] = {
    "probs": cl_probs,
    "decisions": cl_decisions,
    "costs": cl_costs,
    "name": "Closed-Loop",
}
print(f"  Closed-Loop: complete")

# Stratified analysis
print(f"\n{'Stratum':20s} {'Method':20s} {'Avg Cost':>10s} {'Fraud Rate':>12s} {'Miss Rate':>10s} {'Improvement':>12s}")
print("-" * 85)

stratum_results = {}
for sname, smask in strata:
    stratum_results[sname] = {}
    best_cost = float("inf")
    best_name = ""
    for mname, mres in all_results.items():
        # Use only overlapping samples
        n_strat = min(len(mres["costs"]), len(labels))
        m_costs = mres["costs"][:n_strat]
        m_decisions = mres["decisions"][:n_strat]
        m_labels = labels[:n_strat]
        m_smask = smask[:n_strat]
        
        s_costs = m_costs[m_smask]
        s_labels = m_labels[m_smask]
        s_decisions = m_decisions[m_smask]
        
        if len(s_costs) == 0:
            continue
        
        avg_cost = s_costs.mean()
        fraud_rate = s_labels.mean()
        miss_rate = ((s_decisions == 0) & (s_labels == 1)).sum() / max((s_labels == 1).sum(), 1)
        
        stratum_results[sname][mname] = {
            "avg_cost": float(avg_cost),
            "fraud_rate": float(fraud_rate),
            "miss_rate": float(miss_rate),
            "n": int(len(s_costs)),
        }
        
        if avg_cost < best_cost and mname != "closed_loop":
            best_cost = avg_cost
            best_name = mres["name"]
    
    # Print stratum results
    for mname, mres in all_results.items():
        r = stratum_results[sname][mname]
        impr = (best_cost - r["avg_cost"]) / best_cost * 100 if mname == "closed_loop" else 0
        impr_str = f"{impr:.1f}%" if mname == "closed_loop" else "-"
        marker = "*" if mname == "closed_loop" else " "
        print(f"{sname:20s} {marker}{mres['name']:19s} {r['avg_cost']:>10.4f} {r['fraud_rate']:>11.2%} {r['miss_rate']:>9.1%} {impr_str:>12s}")

# Summary
print(f"\n{'='*60}")
print("Summary")
print(f"{'='*60}")
print(f"Average weekly KL divergence: {avg_kl:.4f}")
print(f"Estimated drift speed: {avg_kl:.6f} per week")

for sname, _ in strata:
    cl = stratum_results[sname]["closed_loop"]
    best_base = min(
        (r for m, r in stratum_results[sname].items() if m != "closed_loop"),
        key=lambda x: x["avg_cost"]
    )
    impr = (best_base["avg_cost"] - cl["avg_cost"]) / best_base["avg_cost"] * 100
    print(f"{sname:20s}: Closed-loop cost={cl['avg_cost']:.3f}, "
          f"Best baseline cost={best_base['avg_cost']:.3f}, "
          f"Improvement={impr:.1f}%, "
          f"n={cl['n']:,}")

# Save
with open("results/ieee_analysis.json", "w") as f:
    json.dump({
        "kl_divergence": {
            "weekly": [float(x) for x in kl_divs],
            "average": float(avg_kl),
            "max": float(max_kl),
        },
        "stratified": stratum_results,
    }, f, indent=2)
print(f"\nSaved to results/ieee_analysis.json")
