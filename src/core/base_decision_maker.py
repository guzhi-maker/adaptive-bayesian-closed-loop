"""Abstract base class for decision makers.

A decision maker takes calibrated probabilities (with uncertainty)
and produces an action that minimizes expected cost under
dynamic cost structures.

Key outputs:
1. Decision action (e.g., reject/accept/flag for review)
2. Expected cost of the chosen action
3. Decision uncertainty (confidence in the decision)

The cost structure can be dynamic (time-varying), which is one
of the two dimensions of non-stationarity addressed by this framework.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np


class Action(Enum):
    """Possible decision actions. Domain-specific subclasses may extend."""

    REJECT = "reject"         # Predict positive / take action
    ACCEPT = "accept"         # Predict negative / no action
    FLAG = "flag"             # Request more information / defer


@dataclass
class DecisionResult:
    """Result of a single decision step."""

    action: Action
    calibrated_prob: float
    ci_lower: float
    ci_upper: float
    expected_cost: float
    cost_no_action: float
    cost_action: float


class BaseDecisionMaker(ABC):
    """Abstract base class for decision-making modules.

    Maps calibrated probabilities (with uncertainty) to optimal
    actions under potentially dynamic cost structures.
    """

    @abstractmethod
    def decide(
        self,
        calibrated_prob: float,
        ci_lower: float | None = None,
        ci_upper: float | None = None,
        cost_no_action: float | None = None,
        cost_action: float | None = None,
    ) -> DecisionResult:
        """Make a decision given calibrated probability and costs.

        Args:
            calibrated_prob: Calibrated probability P(y=1 | x).
            ci_lower: Lower bound of confidence interval (optional).
            ci_upper: Upper bound of confidence interval (optional).
            cost_no_action: Cost of false negative (optional, uses config default).
            cost_action: Cost of false positive (optional, uses config default).

        Returns:
            DecisionResult with the chosen action and metadata.
        """
        ...

    @abstractmethod
    def set_dynamic_costs(
        self, cost_no_action: float, cost_action: float
    ) -> None:
        """Update the cost structure for the current time step.

        Args:
            cost_no_action: Cost of false negative (e.g.,漏检成本).
            cost_action: Cost of false positive (e.g.,误检成本).
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset decision statistics."""
        ...

    @property
    @abstractmethod
    def action_history(self) -> list[DecisionResult]:
        """History of all decisions made."""
        ...
