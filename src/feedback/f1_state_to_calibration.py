"""F1 Feedback: State -> Calibration (Dynamic Window / Regularization).

Adjusts the calibration module's behavior based on the current state
estimate from the tracker. When the state indicates high drift or
instability, the calibration window shrinks and regularization changes.

Key mechanism:
- State uncertainty (covariance trace) controls calibration window size
- State mean controls Platt regularization (more drift = less regularization)
- Enables fast adaptation during regime changes while maintaining
  stability during stationary periods
"""

from __future__ import annotations

import numpy as np


class F1StateToCalibration:
    """F1 feedback: dynamically adjusts calibration parameters from tracker state.

    Maps tracker state -> calibration parameters:
    1. Window size for online calibration (inverse to drift speed)
    2. Regularization strength for Platt scaling (inverse to uncertainty)

    Parameters
    ----------
    base_window : int
        Default calibration window size when state is stable.
    min_window : int
        Minimum window size during rapid drift (faster adaptation).
    max_window : int
        Maximum window size during stability (smoother estimates).
    base_reg : float
        Default Platt regularization strength.
    min_reg : float
        Minimum regularization (most adaptive).
    max_reg : float
        Maximum regularization (most stable).
    sensitivity : float
        How strongly state uncertainty modulates parameters.
    """

    def __init__(
        self,
        base_window: int = 2000,
        min_window: int = 500,
        max_window: int = 5000,
        base_reg: float = 1e-4,
        min_reg: float = 1e-6,
        max_reg: float = 1e-2,
        sensitivity: float = 1.0,
    ):
        self.base_window = base_window
        self.min_window = min_window
        self.max_window = max_window
        self.base_reg = base_reg
        self.min_reg = min_reg
        self.max_reg = max_reg
        self.sensitivity = sensitivity

    def compute_window_size(self, state_covariance: np.ndarray) -> int:
        """Compute adaptive window size from state uncertainty.

        Higher uncertainty -> smaller window (faster adaptation).
        Lower uncertainty -> larger window (smoother estimates).

        Args:
            state_covariance: Tracker state covariance, shape (n_states, n_states).

        Returns:
            Adaptive window size, clamped to [min_window, max_window].
        """
        uncertainty = float(np.trace(state_covariance))
        # Normalize uncertainty to [0, 1] range
        norm_uncertainty = np.clip(uncertainty * self.sensitivity, 0.0, 1.0)
        # Linear interpolation: high uncertainty -> small window
        window = int(
            self.max_window - (self.max_window - self.min_window) * norm_uncertainty
        )
        return int(np.clip(window, self.min_window, self.max_window))

    def compute_regularization(self, state_covariance: np.ndarray) -> float:
        """Compute adaptive regularization from state uncertainty.

        Higher uncertainty -> lower regularization (more adaptive).
        Lower uncertainty -> higher regularization (more stable).

        Args:
            state_covariance: Tracker state covariance.

        Returns:
            Adaptive regularization strength, clamped to [min_reg, max_reg].
        """
        uncertainty = float(np.trace(state_covariance))
        norm_uncertainty = np.clip(uncertainty * self.sensitivity, 0.0, 1.0)
        # Inverse: high uncertainty -> low regularization
        reg = self.max_reg - (self.max_reg - self.min_reg) * norm_uncertainty
        return float(np.clip(reg, self.min_reg, self.max_reg))

    def get_feedback(
        self, state_covariance: np.ndarray
    ) -> dict:
        """Return full feedback signal as a dict.

        Args:
            state_covariance: Tracker state covariance.

        Returns:
            Dict with 'window_size', 'regularization', 'uncertainty'.
        """
        uncertainty = float(np.trace(state_covariance))
        return {
            "window_size": self.compute_window_size(state_covariance),
            "regularization": self.compute_regularization(state_covariance),
            "uncertainty": uncertainty,
        }
