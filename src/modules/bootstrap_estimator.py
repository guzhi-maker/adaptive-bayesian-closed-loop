"""Bootstrap-based uncertainty estimator for calibrated probabilities.

Provides confidence intervals and uncertainty quantification via
bootstrapped resampling of the calibration data. This is a core
component that feeds uncertainty information to both:
1. The decision maker (for risk-aware decisions)
2. The F2 feedback loop (calibration residual -> UKF process noise)
"""

from __future__ import annotations

import numpy as np


class BootstrapEstimator:
    """Bootstrap estimator for calibrated probability uncertainty.

    Uses bootstrap resampling to estimate the distribution of calibrated
    probabilities, from which confidence intervals and variance estimates
    are derived.

    The estimator works with any callable calibrator that maps
    (logits, labels) -> fitted model and (logits) -> probabilities.
    """

    def __init__(
        self,
        calibrator_cls: type,
        n_bootstrap: int = 100,
        ci_alpha: float = 0.05,
        random_seed: int = 42,
        **calibrator_kwargs,
    ):
        self.calibrator_cls = calibrator_cls
        self.n_bootstrap = n_bootstrap
        self.ci_alpha = ci_alpha
        self.calibrator_kwargs = calibrator_kwargs

        self._rng = np.random.RandomState(random_seed)
        self._bootstrap_probs: np.ndarray | None = None
        self._fitted_calibrators: list = []
        self._main_calibrator = None

    def fit(
        self, logits: np.ndarray, labels: np.ndarray, eval_logits: np.ndarray
    ) -> None:
        """Fit bootstrap ensemble and compute CI on eval points.

        Args:
            logits: Calibration training logits, shape (n_train,).
            labels: Calibration training labels, shape (n_train,).
            eval_logits: Logits where CI is evaluated, shape (n_eval,).
        """
        n = len(logits)
        X = np.asarray(logits, dtype=float).reshape(-1, 1)
        y = np.asarray(labels, dtype=float)
        eval_l = np.asarray(eval_logits, dtype=float)

        # Fit main calibrator
        self._main_calibrator = self.calibrator_cls(**self.calibrator_kwargs)
        self._main_calibrator.fit(logits, labels)

        # Bootstrap ensemble
        bootstrap_probs = np.zeros((self.n_bootstrap, len(eval_logits)))
        for i in range(self.n_bootstrap):
            indices = self._rng.randint(0, n, size=n)
            X_b = X[indices].flatten()
            y_b = y[indices]

            if len(np.unique(y_b)) < 2:
                # Fallback: use main calibrator prediction
                bootstrap_probs[i] = self._main_calibrator.predict(eval_l)
                continue

            cal = self.calibrator_cls(**self.calibrator_kwargs)
            try:
                cal.fit(X_b, y_b)
                bootstrap_probs[i] = cal.predict(eval_l)
            except Exception:
                bootstrap_probs[i] = self._main_calibrator.predict(eval_l)

        self._bootstrap_probs = bootstrap_probs

    def predict(self, logits: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities from the main calibrator.

        Args:
            logits: Raw logits, shape (n_samples,).

        Returns:
            Calibrated probabilities, shape (n_samples,).
        """
        if self._main_calibrator is None:
            raise RuntimeError("Estimator not fitted. Call fit() first.")
        return self._main_calibrator.predict(logits)

    def predict_with_ci(
        self, logits: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return calibrated probabilities with CI.

        Args:
            logits: Raw logits, shape (n_samples,).

        Returns:
            (probs, ci_lower, ci_upper), each shape (n_samples,).
        """
        probs = self.predict(logits)

        if self._bootstrap_probs is None:
            # No bootstrap available; return symmetric margin
            margin = 0.05
            ci_lower = np.clip(probs - margin, 0.0, 1.0)
            ci_upper = np.clip(probs + margin, 0.0, 1.0)
            return probs, ci_lower, ci_upper

        alpha = self.ci_alpha
        ci_lower = np.clip(
            np.percentile(self._bootstrap_probs, 100 * alpha / 2, axis=0),
            0.0, 1.0,
        )
        ci_upper = np.clip(
            np.percentile(self._bootstrap_probs, 100 * (1 - alpha / 2), axis=0),
            0.0, 1.0,
        )

        return probs, ci_lower, ci_upper

    def get_uncertainty(self, logits: np.ndarray) -> np.ndarray:
        """Return uncertainty (std of bootstrap distribution) for each point.

        Args:
            logits: Raw logits, shape (n_samples,).

        Returns:
            Uncertainty estimate, shape (n_samples,).
        """
        if self._bootstrap_probs is None:
            return np.full(len(logits), 0.05)
        return np.std(self._bootstrap_probs, axis=0)

    def get_calibration_residual(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """Return the average calibration error as a scalar feedback signal.

        Used by the F2 feedback loop: calibration residual -> UKF process noise.

        Args:
            logits: Raw logits, shape (n_samples,).
            labels: True labels, shape (n_samples,).

        Returns:
            Scalar calibration error (mean absolute error).
        """
        probs = self.predict(logits)
        return float(np.mean(np.abs(probs - labels)))

    def reset(self) -> None:
        self._bootstrap_probs = None
        self._fitted_calibrators = []
        self._main_calibrator = None
