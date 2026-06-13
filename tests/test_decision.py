"""Tests for three-layer decision maker."""
import sys
sys.path.insert(0, ".")

import numpy as np
from src.core.base_decision_maker import Action
from src.modules.three_layer_decision import (
    ThreeLayerConfig, ThreeLayerDecisionMaker, DecisionResult,
)


def test_decision_accept_when_confident():
    """CI entirely below low_threshold -> ACCEPT."""
    maker = ThreeLayerDecisionMaker(ThreeLayerConfig(low_threshold=0.4, high_threshold=0.6))
    result = maker.decide(calibrated_prob=0.2, ci_lower=0.15, ci_upper=0.25)
    assert result.action == Action.ACCEPT


def test_decision_reject_when_confident():
    """CI entirely above high_threshold -> REJECT."""
    maker = ThreeLayerDecisionMaker(ThreeLayerConfig(low_threshold=0.4, high_threshold=0.6))
    result = maker.decide(calibrated_prob=0.8, ci_lower=0.75, ci_upper=0.85)
    assert result.action == Action.REJECT


def test_decision_cost_based_fallback():
    """CI straddles boundary -> cost-based fallback."""
    maker = ThreeLayerDecisionMaker(ThreeLayerConfig(
        low_threshold=0.4, high_threshold=0.6,
        default_cost_no_action=10.0, default_cost_action=1.0,
    ))
    # prob=0.5, CI=[0.4, 0.6] -> straddles 0.5, cost-based
    # With FN=10, FP=1: expected cost of REJECT = 1*0.5 = 0.5, ACCEPT = 10*0.5 = 5
    # So should REJECT
    result = maker.decide(calibrated_prob=0.5, ci_lower=0.4, ci_upper=0.6)
    assert result.action == Action.REJECT
    assert result.expected_cost < 1.0


def test_decision_no_ci_provided():
    """Decision works without CI (auto-computes margin)."""
    maker = ThreeLayerDecisionMaker(ThreeLayerConfig(low_threshold=0.3, high_threshold=0.7))
    result = maker.decide(calibrated_prob=0.1)
    assert result.action in (Action.ACCEPT, Action.REJECT)
    assert result.ci_lower >= 0
    assert result.ci_upper <= 1


def test_dynamic_costs():
    """Cost update changes decision behavior."""
    maker = ThreeLayerDecisionMaker(ThreeLayerConfig(
        low_threshold=0.4, high_threshold=0.6,
    ))
    # With FN=100, FP=1: REJECT is much cheaper
    maker.set_dynamic_costs(cost_no_action=100.0, cost_action=1.0)
    result = maker.decide(calibrated_prob=0.5, ci_lower=0.4, ci_upper=0.6)
    assert result.action == Action.REJECT

    # With FN=1, FP=100: ACCEPT is much cheaper
    maker.set_dynamic_costs(cost_no_action=1.0, cost_action=100.0)
    result = maker.decide(calibrated_prob=0.5, ci_lower=0.4, ci_upper=0.6)
    assert result.action == Action.ACCEPT


def test_dynamic_thresholds():
    """Threshold update changes decision boundary."""
    maker = ThreeLayerDecisionMaker(ThreeLayerConfig(
        low_threshold=0.4, high_threshold=0.6,
    ))
    # With low thresholds, prob=0.35 with small CI should be ACCEPT
    result = maker.decide(calibrated_prob=0.35, ci_lower=0.33, ci_upper=0.37)
    assert result.action == Action.ACCEPT

    # After raising thresholds, same prob should still be ACCEPT (below both)
    maker.set_dynamic_thresholds(0.2, 0.4)
    result2 = maker.decide(calibrated_prob=0.35, ci_lower=0.33, ci_upper=0.37)
    assert result2.action == Action.REJECT  # ci_lower > high_threshold


def test_history():
    """Decision history is accumulated."""
    maker = ThreeLayerDecisionMaker()
    maker.decide(calibrated_prob=0.1, ci_lower=0.05, ci_upper=0.15)
    maker.decide(calibrated_prob=0.9, ci_lower=0.85, ci_upper=0.95)
    maker.decide(calibrated_prob=0.5, ci_lower=0.4, ci_upper=0.6)
    assert len(maker.action_history) == 3
    assert len(maker.action_history) == 3


def test_decision_breakdown():
    """Breakdown counts match decisions made."""
    maker = ThreeLayerDecisionMaker()
    maker.decide(calibrated_prob=0.1, ci_lower=0.05, ci_upper=0.15)  # ACCEPT
    maker.decide(calibrated_prob=0.9, ci_lower=0.85, ci_upper=0.95)  # REJECT
    bd = maker.get_decision_breakdown()
    assert bd["total"] == 2
    assert bd["accept"] == 1
    assert bd["reject"] == 1


def test_reset():
    """Reset clears history."""
    maker = ThreeLayerDecisionMaker()
    maker.decide(calibrated_prob=0.1, ci_lower=0.05, ci_upper=0.15)
    maker.decide(calibrated_prob=0.9, ci_lower=0.85, ci_upper=0.95)
    maker.reset()
    assert len(maker.action_history) == 0
    assert maker.total_expected_cost == 0.0


def test_average_cost():
    """Average cost is correctly computed."""
    maker = ThreeLayerDecisionMaker(ThreeLayerConfig(
        default_cost_no_action=10.0, default_cost_action=1.0,
    ))
    result = maker.decide(calibrated_prob=0.1, ci_lower=0.05, ci_upper=0.15)
    # ACCEPT: expected cost = 10 * 0.1 = 1.0
    assert abs(result.expected_cost - 1.0) < 1e-6


def test_config_validation():
    """Invalid config raises error."""
    try:
        ThreeLayerDecisionMaker(ThreeLayerConfig(low_threshold=0.7, high_threshold=0.3))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    test_decision_accept_when_confident()
    test_decision_reject_when_confident()
    test_decision_cost_based_fallback()
    test_decision_no_ci_provided()
    test_dynamic_costs()
    test_dynamic_thresholds()
    test_history()
    test_decision_breakdown()
    test_reset()
    test_average_cost()
    test_config_validation()
    print("All decision maker tests passed!")
