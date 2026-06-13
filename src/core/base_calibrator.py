"""Abstract base class for all calibrators.

A calibrator maps raw model outputs (logits/probabilities) to
well-calibrated probability estimates. It optionally provides
uncertainty estimates (confidence intervals) around each prediction.

Key outputs:
1. Calibrated probability P(y=1 | raw_output)
2. Confidence interval [lower, upper] for uncertainty-aware decisions

All calibrators support:
    - fit(logits, labels): Train on validation data
    - predict(logits): Return calibrated probabilities
    - predict_with_ci(logits): Return (probs, ci_lower, ci_upper)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseCalibrator(ABC):
    """Abstract base class for probability calibration modules.

    Maps raw model outputs to well-calibrated probabilities,
    optionally with uncertainty quantification via confidence intervals.
    """

    @abstractmethod
    def fit(self, logits: np.ndarray, labels: np.ndarray) -> None:
        """Fit the calibrator on validation data.

        Args:
            logits: Raw model outputs (pre-sigmoid), shape (n_samples,).
            labels: True binary labels, shape (n_samples,), in {0, 1}.
        """
        ...

    @abstractmethod
    def predict(self, logits: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities.

        Args:
            logits: Raw model outputs, shape (n_samples,).

        Returns:
            Calibrated probabilities, shape (n_samples,), in [0, 1].
        """
        ...

    @abstractmethod
    def predict_with_ci(
        self, logits: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return calibrated probabilities with confidence intervals.

        Args:
            logits: Raw model outputs, shape (n_samples,).

        Returns:
            (probs, ci_lower, ci_upper), each shape (n_samples,),
            where ci_lower and ci_upper define a (1-alpha) CI.
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this calibrator."""
        ...

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Whether the calibrator has been fitted."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset the calibrator to its initial (unfitted) state."""
        ...
