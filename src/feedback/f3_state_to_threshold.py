"""F3 Feedback: State + Uncertainty -> Dynamic Decision Threshold.

Adjusts the decision maker's thresholds based on the current state
estimate and its uncertainty. When the state indicates high drift or
when uncertainty is large, thresholds widen to make decisions more
conservative. When stable, thresholds narrow for more decisive actions.

Key mechanism:
- State estimate (drift level) shifts the decision boundary
- State uncertainty widens/narrows the uncertainty zone
- Enables adaptive risk tolerance based on environmental stability
"""

from __future__ import annotations

import numpy as np


class F3StateToThreshold:
    """F3 feedback: dynamically adjusts decision thresholds from tracker state.

    Maps state + uncertainty -> decision thresholds.

    Parameters
    ----------
    base_low : float
        Default low (accept) threshold when state is stable.
    base_high : float
        Default high (reject) threshold when state is stable.
    min_margin : float
        Minimum margin between thresholds (most decisive).
    max_margin : float
        Maximum margin between thresholds (most conservative).
    drift_sensitivity : float
        How strongly state drift shifts the threshold center.
    uncertainty_sensitivity : float
        How strongly uncertainty widens the threshold margin.
    """

    def __init__(
        self,
        base_low: float = 0.4,
        base_high: float = 0.6,
        min_margin: float = 0.05,
        max_margin: float = 0.4,
        drift_sensitivity: float = 0.5,
        uncertainty_sensitivity: float = 2.0,
    ):
        self.base_low = base_low
        self.base_high = base_high
        self.min_margin = min_margin
        self.max_margin = max_margin
        self.drift_sensitivity = drift_sensitivity
        self.uncertainty_sensitivity = uncertainty_sensitivity

    def _compute_center_shift(
        self, state_estimate: np.ndarray
    ) -> float:
        """Compute how much the threshold center should shift.

        State drift shifts the decision boundary to compensate.
        E.g., if drift causes more positives, shift threshold up
        to maintain constant false positive rate.

        Args:
            state_estimate: Current state from tracker.

        Returns:
            Center shift amount.
        """
        # Use first component of state as drift signal
        drift_signal = float(state_estimate[0]) if len(state_estimate) > 0 else 0.0
        return -self.drift_sensitivity * drift_signal

    def _compute_margin(
        self, state_covariance: np.ndarray
    ) -> float:
        """Compute threshold margin from state uncertainty.

        Higher uncertainty -> wider margin (more conservative decisions).
        Lower uncertainty -> narrower margin (more decisive).

        Args:
            state_covariance: Tracker state covariance.

        Returns:
            Threshold margin, clamped to [min_margin, max_margin].
        """
        uncertainty = float(np.trace(state_covariance))
        norm_uncertainty = np.clip(
            uncertainty * self.uncertainty_sensitivity, 0.0, 1.0
        )
        margin = self.min_margin + (
            self.max_margin - self.min_margin
        ) * norm_uncertainty
        return float(np.clip(margin, self.min_margin, self.max_margin))

    def get_thresholds(
        self,
        state_estimate: np.ndarray,
        state_covariance: np.ndarray,
    ) -> tuple[float, float]:
        """Compute adaptive decision thresholds.

        Args:
            state_estimate: Current state from tracker, shape (n_states,).
            state_covariance: Current state covariance, shape (n_states, n_states).

        Returns:
            (low_threshold, high_threshold), each in [0, 1].
        """
        center = 0.5 + self._compute_center_shift(state_estimate)
        margin = self._compute_margin(state_covariance)

        low = float(np.clip(center - margin, 0.0, 1.0))
        high = float(np.clip(center + margin, 0.0, 1.0))

        # Ensure low < high
        if low >= high:
            mid = (low + high) / 2
            low = max(0.0, mid - self.min_margin / 2)
            high = min(1.0, mid + self.min_margin / 2)

        return low, high

    def get_feedback(
        self,
        state_estimate: np.ndarray,
        state_covariance: np.ndarray,
    ) -> dict:
        """Return full feedback signal.

        Args:
            state_estimate: Current state from tracker.
            state_covariance: Current state covariance.

        Returns:
            Dict with 'low_threshold', 'high_threshold', 'margin', 'center_shift'.
        """
        low, high = self.get_thresholds(state_estimate, state_covariance)
        return {
            "low_threshold": low,
            "high_threshold": high,
            "margin": high - low,
            "center_shift": self._compute_center_shift(state_estimate),
        }
