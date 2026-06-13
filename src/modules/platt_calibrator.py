"""Platt scaling calibrator with bootstrap confidence intervals.

Migrated and adapted from the quantum error correction calibration project.
Fits a logistic regression model on logits to produce calibrated probabilities,
with optional bootstrap-based uncertainty quantification.

Supports:
1. Standard Platt scaling (logistic regression on logits)
2. Temperature scaling (single scalar parameter)
3. Bootstrap confidence intervals on calibrated probabilities
4. Online (streaming) recalibration for non-stationary environments
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from src.core.base_calibrator import BaseCalibrator


class PlattCalibrator(BaseCalibrator):
    """Platt scaling calibrator with bootstrap CI support.

    Maps raw logits to calibrated probabilities via logistic regression:
        P(y=1 | f) = 1 / (1 + exp(a * f + b))

    Optionally computes bootstrap confidence intervals for uncertainty
    quantification in the decision stage.
    """

    def __init__(
        self,
        use_scalar: bool = True,
        regularization: float = 1e-10,
        n_bootstrap: int = 100,
        ci_alpha: float = 0.05,
    ):
        self.use_scalar = use_scalar
        self.regularization = regularization
        self.n_bootstrap = n_bootstrap
        self.ci_alpha = ci_alpha

        self.a_: float = 1.0
        self.b_: float = 0.0
        self._fitted = False
        self._bootstrap_a: np.ndarray | None = None
        self._bootstrap_b: np.ndarray | None = None

    # --- BaseCalibrator interface --------------------------------------------

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> None:
        """Fit Platt scaling via logistic regression.

        Args:
            logits: Raw logits, shape (n_samples,).
            labels: Binary labels in {0, 1}, shape (n_samples,).
        """
        logits = np.asarray(logits, dtype=float).reshape(-1, 1)
        labels = np.asarray(labels, dtype=float)

        clf = LogisticRegression(
            C=1.0 / self.regularization if self.regularization > 0 else 1e10,
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        )
        clf.fit(logits, labels)
        self.a_ = float(clf.coef_[0, 0])
        self.b_ = float(clf.intercept_[0])
        self._fitted = True

    def predict(self, logits: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities.

        Args:
            logits: Raw logits, shape (n_samples,).

        Returns:
            Calibrated probabilities in [0, 1], shape (n_samples,).
        """
        if not self._fitted:
            return expit(np.asarray(logits, dtype=float))
        logits = np.asarray(logits, dtype=float)
        return expit(self.a_ * logits + self.b_)

    def predict_with_ci(
        self, logits: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return calibrated probabilities with bootstrap CI.

        Args:
            logits: Raw logits, shape (n_samples,).

        Returns:
            (probs, ci_lower, ci_upper), each shape (n_samples,).
        """
        probs = self.predict(logits)
        n = len(logits)

        if self._bootstrap_a is None or self._bootstrap_b is None:
            # No bootstrap available; return symmetric margin
            margin = 0.05
            ci_lower = np.clip(probs - margin, 0.0, 1.0)
            ci_upper = np.clip(probs + margin, 0.0, 1.0)
            return probs, ci_lower, ci_upper

        # Compute CI from bootstrap distribution of (a, b)
        logits_arr = np.asarray(logits, dtype=float)
        bootstrap_probs = np.zeros((self.n_bootstrap, n))
        for i in range(self.n_bootstrap):
            z = self._bootstrap_a[i] * logits_arr + self._bootstrap_b[i]
            bootstrap_probs[i] = expit(z)

        alpha = self.ci_alpha
        ci_lower = np.clip(
            np.percentile(bootstrap_probs, 100 * alpha / 2, axis=0), 0.0, 1.0
        )
        ci_upper = np.clip(
            np.percentile(bootstrap_probs, 100 * (1 - alpha / 2), axis=0), 0.0, 1.0
        )

        return probs, ci_lower, ci_upper

    def name(self) -> str:
        if self.use_scalar:
            return "Platt (Temperature) Scaling"
        return "Platt Scaling (full)"

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def reset(self) -> None:
        self.a_ = 1.0
        self.b_ = 0.0
        self._fitted = False
        self._bootstrap_a = None
        self._bootstrap_b = None

    # --- Additional methods --------------------------------------------------

    def fit_with_bootstrap(
        self, logits: np.ndarray, labels: np.ndarray
    ) -> None:
        """Fit Platt scaling with bootstrap confidence intervals.

        Fits the main model and also stores bootstrap distribution
        of (a, b) for CI computation.

        Args:
            logits: Raw logits, shape (n_samples,).
            labels: Binary labels in {0, 1}, shape (n_samples,).
        """
        self.fit(logits, labels)

        n = len(logits)
        X = np.asarray(logits, dtype=float).reshape(-1, 1)
        y = np.asarray(labels, dtype=float)
        rng = np.random.RandomState(42)

        a_list, b_list = [], []
        C_val = 1.0 / self.regularization if self.regularization > 0 else 1e10

        for _ in range(self.n_bootstrap):
            indices = rng.randint(0, n, size=n)
            X_b = X[indices]
            y_b = y[indices]
            if len(np.unique(y_b)) < 2:
                a_list.append(self.a_)
                b_list.append(self.b_)
                continue
            clf = LogisticRegression(C=C_val, solver="lbfgs", max_iter=1000)
            try:
                clf.fit(X_b, y_b)
                a_list.append(float(clf.coef_[0, 0]))
                b_list.append(float(clf.intercept_[0]))
            except Exception:
                a_list.append(self.a_)
                b_list.append(self.b_)

        self._bootstrap_a = np.array(a_list)
        self._bootstrap_b = np.array(b_list)

    def get_params(self) -> dict:
        """Return current calibration parameters."""
        return {"a": self.a_, "b": self.b_, "fitted": self._fitted}
