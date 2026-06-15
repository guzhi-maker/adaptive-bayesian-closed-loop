"""F2 Feedback: Calibration Residual -> UKF Process Noise.

Adjusts the UKF tracker's process noise based on calibration quality.
When calibration error is high, the UKF increases process noise to
allow faster state tracking. When calibration is good, process noise
decreases for smoother estimates.

Key mechanism:
- Calibration residual (mean absolute error) is the feedback signal
- High residual -> increase Q (trust observations more, trust model less)
- Low residual -> decrease Q (trust model more, smooth estimates)
- Prevents the tracker from being too slow during regime changes
"""

from __future__ import annotations

import numpy as np


class F2CalibrationResidualToUKF:
    """F2 feedback: adjusts UKF process noise from calibration quality.

    Maps calibration residual -> UKF process noise covariance Q.

    Parameters
    ----------
    base_Q : float
        Default process noise when calibration residual is at target.
    min_Q : float
        Minimum process noise (most stable tracking).
    max_Q : float
        Maximum process noise (fastest adaptation).
    target_residual : float
        Target calibration residual (below this, Q decreases).
        Should reflect expected irreducible calibration error.
    sensitivity : float
        How strongly the residual modulates Q.
    adaptation_rate : float
        Smoothing factor for Q updates (0=instant, 1=no update).
    """

    def __init__(
        self,
        base_Q: float = 1e-4,
        min_Q: float = 1e-6,
        max_Q: float = 1e-2,
        target_residual: float = 0.01,
        sensitivity: float = 10.0,
        adaptation_rate: float = 0.3,
    ):
        self.base_Q = base_Q
        self.min_Q = min_Q
        self.max_Q = max_Q
        self.target_residual = target_residual
        self.sensitivity = sensitivity
        self.adaptation_rate = adaptation_rate

        self._current_Q = base_Q

    def compute_process_noise(self, calibration_residual: float) -> float:
        """Compute adaptive process noise from calibration residual.

        Higher calibration residual -> higher Q (faster adaptation).
        Lower calibration residual -> lower Q (smoother tracking).

        Args:
            calibration_residual: Mean absolute calibration error.

        Returns:
            Adaptive process noise value, clamped to [min_Q, max_Q].
        """
        # Ratio of actual residual to target
        ratio = calibration_residual / max(self.target_residual, 1e-10)
        # Log-scale adjustment
        log_factor = np.log1p(ratio)
        # Compute target Q
        target_Q = self.base_Q * (1.0 + self.sensitivity * log_factor)
        target_Q = float(np.clip(target_Q, self.min_Q, self.max_Q))

        # Smooth update
        self._current_Q = (
            self.adaptation_rate * self._current_Q
            + (1.0 - self.adaptation_rate) * target_Q
        )

        return self._current_Q

    def get_feedback(
        self, calibration_residual: float
    ) -> dict:
        """Return full feedback signal.

        Args:
            calibration_residual: Current calibration residual (MAE).

        Returns:
            Dict with 'process_noise', 'residual_ratio', 'Q_change'.
        """
        old_Q = self._current_Q
        new_Q = self.compute_process_noise(calibration_residual)
        return {
            "process_noise": new_Q,
            "Q_change": new_Q - old_Q,
            "calibration_residual": calibration_residual,
            "residual_ratio": calibration_residual / max(self.target_residual, 1e-10),
        }

    def reset(self) -> None:
        """Reset to initial state."""
        self._current_Q = self.base_Q
