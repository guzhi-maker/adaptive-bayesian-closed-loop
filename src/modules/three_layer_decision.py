"""Three-layer adaptive decision maker with dynamic cost support.

Implements a three-layer decision rule that uses both calibrated
probabilities and their confidence intervals to make cost-sensitive
decisions under uncertainty.

Three layers:
    1. Low risk: ci_upper < low_threshold -> ACCEPT (no action)
    2. High risk: ci_lower > high_threshold -> REJECT (take action)
    3. High uncertainty: CI straddles decision boundary -> cost-based optimal

The thresholds can be dynamically adjusted via the F3 feedback loop
(state + uncertainty -> dynamic threshold).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from src.core.base_decision_maker import (
    Action,
    BaseDecisionMaker,
    DecisionResult,
)


@dataclass
class ThreeLayerConfig:
    """Configuration for the three-layer decision maker.

    Attributes:
        low_threshold: If ci_upper < low_threshold -> ACCEPT.
        high_threshold: If ci_lower > high_threshold -> REJECT.
        default_cost_no_action: Default false negative cost.
        default_cost_action: Default false positive cost.
        uncertainty_margin: Fallback CI half-width when bootstrap unavailable.
    """

    low_threshold: float = 0.4
    high_threshold: float = 0.6
    default_cost_no_action: float = 10.0
    default_cost_action: float = 1.0
    uncertainty_margin: float = 0.05

    def validate(self) -> None:
        if self.low_threshold >= self.high_threshold:
            raise ValueError(
                f"low_threshold ({self.low_threshold}) must be < "
                f"high_threshold ({self.high_threshold})"
            )
        if not (0 <= self.low_threshold <= 1):
            raise ValueError(f"low_threshold must be in [0, 1]")
        if not (0 <= self.high_threshold <= 1):
            raise ValueError(f"high_threshold must be in [0, 1]")


class ThreeLayerDecisionMaker(BaseDecisionMaker):
    """Three-layer decision maker with dynamic cost and thresholds.

    The decision rule uses both the calibrated probability and its
    confidence interval to make risk-aware decisions under dynamic costs.

    Decision layers:
        1. ACCEPT if ci_upper < low_threshold (confident negative)
        2. REJECT if ci_lower > high_threshold (confident positive)
        3. Cost-based fallback: choose action minimizing expected cost
    """

    def __init__(self, config: ThreeLayerConfig | None = None):
        self._config = config if config is not None else ThreeLayerConfig()
        self._config.validate()

        self._cost_no_action = self._config.default_cost_no_action
        self._cost_action = self._config.default_cost_action

        self._history: list[DecisionResult] = []
        self._total_expected_cost = 0.0

    # --- BaseDecisionMaker interface -----------------------------------------

    def decide(
        self,
        calibrated_prob: float,
        ci_lower: float | None = None,
        ci_upper: float | None = None,
        cost_no_action: float | None = None,
        cost_action: float | None = None,
    ) -> DecisionResult:
        """Make a decision using the three-layer rule.

        Args:
            calibrated_prob: Calibrated probability P(y=1 | x).
            ci_lower: Lower bound of CI. Auto-computed if None.
            ci_upper: Upper bound of CI. Auto-computed if None.
            cost_no_action: Cost of false negative. Uses config default if None.
            cost_action: Cost of false positive. Uses config default if None.

        Returns:
            DecisionResult with the chosen action.
        """
        # Use instance defaults if not overridden
        c_na = cost_no_action if cost_no_action is not None else self._cost_no_action
        c_a = cost_action if cost_action is not None else self._cost_action

        # Auto-compute CI if not provided
        if ci_lower is None or ci_upper is None:
            margin = self._config.uncertainty_margin
            ci_lower = max(0.0, calibrated_prob - margin)
            ci_upper = min(1.0, calibrated_prob + margin)

        # Layer 1: Confident negative
        if ci_upper < self._config.low_threshold:
            action = Action.ACCEPT
            expected_cost = c_na * calibrated_prob  # Only false negative cost

        # Layer 2: Confident positive
        elif ci_lower > self._config.high_threshold:
            action = Action.REJECT
            expected_cost = c_a * (1.0 - calibrated_prob)  # Only false positive cost

        # Layer 3: High uncertainty -> cost-based optimal
        else:
            expected_cost_no_action = c_na * calibrated_prob
            expected_cost_action = c_a * (1.0 - calibrated_prob)
            if expected_cost_action < expected_cost_no_action:
                action = Action.REJECT
                expected_cost = expected_cost_action
            else:
                action = Action.ACCEPT
                expected_cost = expected_cost_no_action

        result = DecisionResult(
            action=action,
            calibrated_prob=calibrated_prob,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            expected_cost=expected_cost,
            cost_no_action=c_na,
            cost_action=c_a,
        )

        self._history.append(result)
        self._total_expected_cost += expected_cost
        return result

    def set_dynamic_costs(
        self, cost_no_action: float, cost_action: float
    ) -> None:
        """Update the cost structure.

        Args:
            cost_no_action: Cost of false negative.
            cost_action: Cost of false positive.
        """
        self._cost_no_action = cost_no_action
        self._cost_action = cost_action

    def set_dynamic_thresholds(
        self, low_threshold: float, high_threshold: float
    ) -> None:
        """Update decision thresholds dynamically (used by F3 feedback).

        Args:
            low_threshold: New low (accept) threshold.
            high_threshold: New high (reject) threshold.
        """
        self._config.low_threshold = low_threshold
        self._config.high_threshold = high_threshold
        self._config.validate()

    def reset(self) -> None:
        self._history = []
        self._total_expected_cost = 0.0
        self._cost_no_action = self._config.default_cost_no_action
        self._cost_action = self._config.default_cost_action

    @property
    def action_history(self) -> list[DecisionResult]:
        return list(self._history)

    @property
    def total_expected_cost(self) -> float:
        return self._total_expected_cost

    @property
    def average_expected_cost(self) -> float:
        if not self._history:
            return 0.0
        return self._total_expected_cost / len(self._history)

    @property
    def config(self) -> ThreeLayerConfig:
        return self._config

    def get_decision_breakdown(self) -> dict:
        """Return counts of each decision type."""
        counts = {a: 0 for a in Action}
        for r in self._history:
            counts[r.action] += 1
        return {
            "total": len(self._history),
            "accept": counts[Action.ACCEPT],
            "reject": counts[Action.REJECT],
            "flag": counts[Action.FLAG],
        }

    def get_current_costs(self) -> Tuple[float, float]:
        """Return current cost structure."""
        return self._cost_no_action, self._cost_action
