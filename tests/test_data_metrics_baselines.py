"""Tests for synthetic data generator, metrics, and baselines."""
import sys
sys.path.insert(0, ".")

import numpy as np

from src.data.synthetic_generator import (
    DriftConfig, CostConfig, SyntheticDataConfig, SyntheticDataGenerator,
)
from src.evaluation.metrics import (
    compute_ece, compute_nll, compute_brier, compute_decision_costs,
    compute_all_decision_metrics,
)
from src.baselines.methods import (
    get_baseline, run_baseline_on_stream, BASELINE_REGISTRY
)


# =========================================================================
# Synthetic data generator tests
# =========================================================================

def test_generator_output_shape():
    """Generator produces correct shapes."""
    cfg = SyntheticDataConfig(n_samples=100)
    gen = SyntheticDataGenerator(cfg)
    data = gen.generate()
    assert data["logits"].shape == (100,)
    assert data["labels"].shape == (100,)
    assert data["costs_no_action"].shape == (100,)
    assert data["costs_action"].shape == (100,)
    assert data["drift_state"].shape == (100,)


def test_generator_drift_types():
    """All drift types produce valid output."""
    for dt in ["none", "gradual", "abrupt", "periodic"]:
        cfg = SyntheticDataConfig(
            n_samples=50,
            drift=DriftConfig(drift_type=dt),
        )
        gen = SyntheticDataGenerator(cfg)
        data = gen.generate()
        assert np.all(np.isfinite(data["logits"]))


def test_generator_cost_types():
    """All cost types produce valid output."""
    for ct in ["fixed", "dynamic", "stratified"]:
        cfg = SyntheticDataConfig(
            n_samples=50,
            cost=CostConfig(cost_type=ct),
        )
        gen = SyntheticDataGenerator(cfg)
        data = gen.generate()
        assert np.all(data["costs_no_action"] > 0)
        assert np.all(data["costs_action"] > 0)


def test_generator_labels_binary():
    """Labels are in {0, 1}."""
    cfg = SyntheticDataConfig(n_samples=100)
    gen = SyntheticDataGenerator(cfg)
    data = gen.generate()
    assert set(np.unique(data["labels"])) <= {0.0, 1.0}


# =========================================================================
# Metrics tests
# =========================================================================

def test_ece_perfect():
    """Perfect calibration gives ECE = 0."""
    n = 1000
    probs = np.full(n, 0.5)
    labels = np.random.binomial(1, 0.5, size=n).astype(float)
    ece, mce, _, _, _ = compute_ece(probs, labels, n_bins=10)
    assert ece < 0.05, f"ECE too high: {ece}"


def test_nll_perfect():
    """Perfect probabilities give low NLL."""
    n = 1000
    np.random.seed(42)
    probs = np.random.uniform(0.1, 0.9, size=n)
    labels = (np.random.rand(n) < probs).astype(float)
    nll = compute_nll(probs, labels)
    assert np.isfinite(nll)


def test_brier_range():
    """Brier score is in [0, 1]."""
    n = 500
    probs = np.random.uniform(0, 1, size=n)
    labels = np.random.binomial(1, 0.5, size=n).astype(float)
    brier = compute_brier(probs, labels)
    assert 0 <= brier <= 1


def test_decision_costs():
    """Decision costs are finite and non-negative."""
    n = 500
    np.random.seed(42)
    probs = np.random.uniform(0, 1, size=n)
    labels = np.random.binomial(1, 0.5, size=n).astype(float)
    costs_na = np.full(n, 10.0)
    costs_a = np.full(n, 1.0)

    cum, oracle, ratio = compute_decision_costs(probs, labels, costs_na, costs_a)
    assert np.isfinite(cum) and cum >= 0
    assert np.isfinite(oracle) and oracle >= 0
    assert ratio >= 1.0


def test_all_metrics():
    """compute_all_decision_metrics returns all expected fields."""
    n = 500
    np.random.seed(42)
    probs = np.random.uniform(0, 1, size=n)
    labels = np.random.binomial(1, 0.5, size=n).astype(float)
    costs_na = np.full(n, 10.0)
    costs_a = np.full(n, 1.0)
    decisions = (probs >= 0.5).astype(float)

    metrics = compute_all_decision_metrics(
        probs, labels, costs_na, costs_a, decisions, n_bootstrap=10
    )
    assert metrics.ece >= 0
    assert metrics.average_cost >= 0
    assert metrics.cumulative_cost >= 0
    assert metrics.cost_ratio >= 1.0


# =========================================================================
# Baseline tests
# =========================================================================

def test_all_baselines_run():
    """All baselines can run on a data stream without errors."""
    np.random.seed(42)
    n = 300
    logits = np.random.randn(n)
    labels = (np.random.rand(n) < 0.5).astype(float)
    costs_fn = np.full(n, 10.0)
    costs_fp = np.full(n, 1.0)

    for name in BASELINE_REGISTRY:
        bl = get_baseline(name)
        result = run_baseline_on_stream(bl, logits, labels, costs_fn, costs_fp, warmup=100)
        assert len(result["probs"]) == n
        assert len(result["decisions"]) == n
        assert len(result["costs"]) == n
        assert result["cumulative_cost"] > 0
        assert result["name"] is not None


def test_baseline_output_range():
    """Baseline probabilities are in [0, 1]."""
    np.random.seed(42)
    n = 200
    logits = np.random.randn(n)
    labels = (np.random.rand(n) < 0.5).astype(float)
    costs_fn = np.full(n, 10.0)
    costs_fp = np.full(n, 1.0)

    bl = get_baseline("raw")
    result = run_baseline_on_stream(bl, logits, labels, costs_fn, costs_fp, warmup=50)
    assert np.all((0 <= result["probs"]) & (result["probs"] <= 1))


def test_cost_sensitive_threshold():
    """Cost-sensitive baseline uses cost-ratio threshold."""
    bl = get_baseline("cost_sensitive")
    bl.initialize(np.array([-1.0, 0.0, 1.0]), np.array([0, 0, 1]))

    # When FN >> FP, threshold is very low -> more REJECT
    r1 = bl.predict(0.3, cost_fn=100.0, cost_fp=1.0)
    assert r1.decision == 1  # REJECT because threshold ~ 1/101 ~ 0.01

    # When FP >> FN, threshold is very high -> more ACCEPT
    r2 = bl.predict(0.3, cost_fn=1.0, cost_fp=100.0)
    assert r2.decision == 0  # ACCEPT because threshold ~ 100/101 ~ 0.99


if __name__ == "__main__":
    test_generator_output_shape()
    test_generator_drift_types()
    test_generator_cost_types()
    test_generator_labels_binary()
    test_ece_perfect()
    test_nll_perfect()
    test_brier_range()
    test_decision_costs()
    test_all_metrics()
    test_all_baselines_run()
    test_baseline_output_range()
    test_cost_sensitive_threshold()
    print("All unit tests passed!")
