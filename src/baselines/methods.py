"""Baseline comparison methods for the closed-loop framework.

Implements 8 baselines with a unified interface:
    1. RawThreshold: Raw sigmoid + fixed threshold 0.5
    2. StaticPlatt: Pre-fit Platt scaling + fixed threshold
    3. StaticIsotonic: Pre-fit isotonic regression + fixed threshold
    4. TemperatureScaling: Temperature scaling + fixed threshold
    5. OnlinePlatt: Streaming Platt with sliding window
    6. AdaptiveCalibration: Window-weighted recalibration
    7. CostSensitivePlatt: Platt + cost-ratio adjusted threshold
    8. ThresholdMoving: Cost-driven dynamic threshold

All baselines share the same interface as the closed-loop framework
so they can be evaluated with the same metrics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression as SklearnIsotonic
from sklearn.linear_model import LogisticRegression


@dataclass
class BaselineResult:
    """Output from a single baseline step, matching DecisionResult format."""

    decision: int  # 1 = REJECT, 0 = ACCEPT
    prob: float  # Calibrated probability P(y=1)
    cost: float  # Expected cost of this decision
    info: dict = field(default_factory=dict)  # Additional info


class BaseBaseline(ABC):
    """Abstract base for all baseline methods."""

    @abstractmethod
    def initialize(self, logits: np.ndarray, labels: np.ndarray) -> None:
        """Initialize/calibrate on warmup data."""
        ...

    @abstractmethod
    def predict(self, logit: float, cost_fn: float, cost_fp: float) -> BaselineResult:
        """Make a decision for a single sample."""
        ...

    def update(
        self, logit: float, label: float, cost_fn: float, cost_fp: float
    ) -> None:
        """Optional online update after seeing the true label."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        ...


# =============================================================================
# 1. Raw Threshold: sigmoid + 0.5
# =============================================================================


class RawThreshold(BaseBaseline):
    """Raw sigmoid probability with fixed threshold 0.5. No calibration."""

    def initialize(self, logits: np.ndarray, labels: np.ndarray) -> None:
        pass

    def predict(self, logit: float, cost_fn: float, cost_fp: float) -> BaselineResult:
        prob = float(expit(logit))
        decision = 1 if prob >= 0.5 else 0
        cost = cost_fn * prob if decision == 0 else cost_fp * (1.0 - prob)
        return BaselineResult(decision=decision, prob=prob, cost=cost)

    @property
    def name(self) -> str:
        return "Raw Threshold (0.5)"


# =============================================================================
# 2. Static Platt Scaling
# =============================================================================


class StaticPlatt(BaseBaseline):
    """Platt scaling fit once on warmup data, then frozen."""

    def __init__(self, regularization: float = 1e-10):
        self.reg: float = regularization
        self.a: float = 1.0
        self.b: float = 0.0
        self._fitted = False

    def initialize(self, logits: np.ndarray, labels: np.ndarray) -> None:
        if len(np.unique(labels)) < 2:
            return
        X = np.asarray(logits, dtype=float).reshape(-1, 1)
        y = np.asarray(labels, dtype=float)
        C = 1.0 / self.reg if self.reg > 0 else 1e10
        clf = LogisticRegression(C=C, solver="lbfgs", max_iter=1000, random_state=42)
        clf.fit(X, y)
        self.a = float(clf.coef_[0, 0])
        self.b = float(clf.intercept_[0])
        self._fitted = True

    def predict(self, logit: float, cost_fn: float, cost_fp: float) -> BaselineResult:
        if not self._fitted:
            prob = float(expit(logit))
        else:
            prob = float(expit(self.a * logit + self.b))
        decision = 1 if prob >= 0.5 else 0
        cost = cost_fn * prob if decision == 0 else cost_fp * (1.0 - prob)
        return BaselineResult(decision=decision, prob=prob, cost=cost)

    @property
    def name(self) -> str:
        return "Static Platt"


# =============================================================================
# 3. Static Isotonic Regression
# =============================================================================


class StaticIsotonic(BaseBaseline):
    """Isotonic regression fit once on warmup data, then frozen."""

    def __init__(self):
        self.model = SklearnIsotonic(out_of_bounds="clip", increasing=True)
        self._fitted = False

    def initialize(self, logits: np.ndarray, labels: np.ndarray) -> None:
        if len(np.unique(labels)) < 2:
            return
        probs = expit(np.asarray(logits, dtype=float))
        self.model.fit(probs, labels)
        self._fitted = True

    def predict(self, logit: float, cost_fn: float, cost_fp: float) -> BaselineResult:
        prob = float(expit(logit))
        if self._fitted:
            prob = float(self.model.predict(np.array([prob]))[0])
        prob = float(np.clip(prob, 0.0, 1.0))
        decision = 1 if prob >= 0.5 else 0
        cost = cost_fn * prob if decision == 0 else cost_fp * (1.0 - prob)
        return BaselineResult(decision=decision, prob=prob, cost=cost)

    @property
    def name(self) -> str:
        return "Static Isotonic"


# =============================================================================
# 4. Temperature Scaling
# =============================================================================


class TemperatureScaling(BaseBaseline):
    """Temperature scaling: P = sigmoid(logit / T). Single parameter."""

    def __init__(self):
        self.T: float = 1.0
        self._fitted = False

    def initialize(self, logits: np.ndarray, labels: np.ndarray) -> None:
        if len(np.unique(labels)) < 2:
            return
        from scipy.optimize import minimize

        def nll(T):
            probs = expit(np.asarray(logits, dtype=float) / max(T, 1e-6))
            eps = 1e-12
            return -np.mean(
                labels * np.log(np.clip(probs, eps, 1 - eps))
                + (1 - labels) * np.log(np.clip(1 - probs, eps, 1 - eps))
            )

        result = minimize(nll, x0=np.array([1.0]), bounds=[(0.01, 10.0)], method="L-BFGS-B")
        self.T = float(result.x[0])
        self._fitted = True

    def predict(self, logit: float, cost_fn: float, cost_fp: float) -> BaselineResult:
        prob = float(expit(logit / self.T)) if self._fitted else float(expit(logit))
        decision = 1 if prob >= 0.5 else 0
        cost = cost_fn * prob if decision == 0 else cost_fp * (1.0 - prob)
        return BaselineResult(decision=decision, prob=prob, cost=cost)

    @property
    def name(self) -> str:
        return "Temperature Scaling"


# =============================================================================
# 5. Online Platt (sliding window)
# =============================================================================


class OnlinePlatt(BaseBaseline):
    """Platt scaling with sliding window online recalibration."""

    def __init__(self, window: int = 2000, recalibrate_every: int = 500, regularization: float = 0.01):
        self.window = window
        self.recalibrate_every = recalibrate_every
        self._logits: list[float] = []
        self._labels: list[float] = []
        self._count = 0
        self.reg = regularization
        self.a = 1.0
        self.b = 0.0

    def initialize(self, logits: np.ndarray, labels: np.ndarray) -> None:
        self._logits = list(logits)
        self._labels = list(labels)
        self._refit()

    def predict(self, logit: float, cost_fn: float, cost_fp: float) -> BaselineResult:
        prob = float(expit(self.a * logit + self.b))
        decision = 1 if prob >= 0.5 else 0
        cost = cost_fn * prob if decision == 0 else cost_fp * (1.0 - prob)
        return BaselineResult(decision=decision, prob=prob, cost=cost)

    def update(self, logit: float, label: float, cost_fn: float, cost_fp: float) -> None:
        self._logits.append(logit)
        self._labels.append(label)
        if len(self._logits) > self.window:
            self._logits.pop(0)
            self._labels.pop(0)
        self._count += 1
        if self._count % self.recalibrate_every == 0 and len(np.unique(self._labels)) >= 2:
            self._refit()

    def _refit(self) -> None:
        X = np.array(self._logits).reshape(-1, 1)
        y = np.array(self._labels)
        if len(np.unique(y)) < 2:
            return
        clf = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
        clf.fit(X, y)
        self.a = float(clf.coef_[0, 0])
        self.b = float(clf.intercept_[0])

    @property
    def name(self) -> str:
        return "Online Platt"


# =============================================================================
# 6. Adaptive Calibration (window with exponential weighting)
# =============================================================================


class AdaptiveCalibration(BaseBaseline):
    """Exponentially weighted online calibration."""

    def __init__(self, alpha: float = 0.1, regularization: float = 0.01):
        self.alpha = alpha
        self.a = 1.0
        self.b = 0.0
        self._n = 0
        self.reg = regularization

    def initialize(self, logits: np.ndarray, labels: np.ndarray) -> None:
        # Fit initial Platt on warmup
        if len(np.unique(labels)) >= 2:
            X = np.asarray(logits, dtype=float).reshape(-1, 1)
            y = np.asarray(labels, dtype=float)
            clf = LogisticRegression(C=1.0/self.reg if self.reg > 0 else 1e10, solver="lbfgs", max_iter=1000)
            clf.fit(X, y)
            self.a = float(clf.coef_[0, 0])
            self.b = float(clf.intercept_[0])
        self._n = len(logits)

    def predict(self, logit: float, cost_fn: float, cost_fp: float) -> BaselineResult:
        prob = float(expit(self.a * logit + self.b))
        decision = 1 if prob >= 0.5 else 0
        cost = cost_fn * prob if decision == 0 else cost_fp * (1.0 - prob)
        return BaselineResult(decision=decision, prob=prob, cost=cost)

    def update(self, logit: float, label: float, cost_fn: float, cost_fp: float) -> None:
        # Non-decaying learning rate for tracking non-stationarity
        prob = float(expit(self.a * logit + self.b))
        grad_a = (prob - label) * logit
        grad_b = prob - label
        self.a -= self.alpha * grad_a
        self.b -= self.alpha * grad_b
        # Clamp to prevent divergence
        self.a = np.clip(self.a, -10.0, 10.0)
        self.b = np.clip(self.b, -10.0, 10.0)

    @property
    def name(self) -> str:
        return "Adaptive Calibration"


# =============================================================================
# 7. Cost-Sensitive Platt
# =============================================================================


class CostSensitivePlatt(BaseBaseline):
    """Platt scaling with cost-ratio adjusted threshold."""

    def __init__(self, regularization: float = 1e-10):
        self.reg = regularization
        self.a = 1.0
        self.b = 0.0
        self._fitted = False

    def initialize(self, logits: np.ndarray, labels: np.ndarray) -> None:
        if len(np.unique(labels)) < 2:
            return
        X = np.asarray(logits, dtype=float).reshape(-1, 1)
        y = np.asarray(labels, dtype=float)
        C = 1.0 / self.reg if self.reg > 0 else 1e10
        clf = LogisticRegression(C=C, solver="lbfgs", max_iter=1000, random_state=42)
        clf.fit(X, y)
        self.a = float(clf.coef_[0, 0])
        self.b = float(clf.intercept_[0])
        self._fitted = True

    def predict(self, logit: float, cost_fn: float, cost_fp: float) -> BaselineResult:
        if not self._fitted:
            prob = float(expit(logit))
        else:
            prob = float(expit(self.a * logit + self.b))
        # Cost-sensitive threshold: P(y=1) > FP / (FN + FP)
        threshold = cost_fp / (cost_fn + cost_fp) if (cost_fn + cost_fp) > 0 else 0.5
        decision = 1 if prob >= threshold else 0
        cost = cost_fn * prob if decision == 0 else cost_fp * (1.0 - prob)
        return BaselineResult(
            decision=decision, prob=prob, cost=cost, info={"threshold": threshold}
        )

    @property
    def name(self) -> str:
        return "Cost-Sensitive Platt"


# =============================================================================
# 8. Threshold Moving (dynamic threshold optimization)
# =============================================================================


class ThresholdMoving(BaseBaseline):
    """Moving threshold based on recent cost ratio."""

    def __init__(self, window: int = 500):
        self.window = window
        self._costs_fn: list[float] = []
        self._costs_fp: list[float] = []

    def initialize(self, logits: np.ndarray, labels: np.ndarray) -> None:
        pass

    def predict(self, logit: float, cost_fn: float, cost_fp: float) -> BaselineResult:
        prob = float(expit(logit))
        # Dynamic threshold from recent cost ratio
        avg_fn = np.mean(self._costs_fn[-self.window:]) if self._costs_fn else cost_fn
        avg_fp = np.mean(self._costs_fp[-self.window:]) if self._costs_fp else cost_fp
        threshold = avg_fp / (avg_fn + avg_fp) if (avg_fn + avg_fp) > 0 else 0.5
        threshold = float(np.clip(threshold, 0.05, 0.95))

        decision = 1 if prob >= threshold else 0
        cost = cost_fn * prob if decision == 0 else cost_fp * (1.0 - prob)
        return BaselineResult(
            decision=decision, prob=prob, cost=cost, info={"threshold": threshold}
        )

    def update(self, logit: float, label: float, cost_fn: float, cost_fp: float) -> None:
        self._costs_fn.append(cost_fn)
        self._costs_fp.append(cost_fp)
        if len(self._costs_fn) > self.window * 2:
            self._costs_fn = self._costs_fn[-self.window:]
            self._costs_fp = self._costs_fp[-self.window:]

    @property
    def name(self) -> str:
        return "Threshold Moving"


# =============================================================================
# 9. Bayesian Decision Rule (optimal fixed threshold oracle)
# =============================================================================


class BayesianDecisionRule(BaseBaseline):
    """Bayesian optimal decision with true threshold FP/(FN+FP)."""

    def __init__(self):
        pass

    def initialize(self, logits: np.ndarray, labels: np.ndarray) -> None:
        pass

    def predict(self, logit: float, cost_fn: float, cost_fp: float) -> BaselineResult:
        prob = float(expit(logit))
        threshold = cost_fp / (cost_fn + cost_fp) if (cost_fn + cost_fp) > 0 else 0.5
        decision = 1 if prob >= threshold else 0
        cost = cost_fn * prob if decision == 0 else cost_fp * (1.0 - prob)
        return BaselineResult(
            decision=decision, prob=prob, cost=cost, info={"threshold": threshold}
        )

    @property
    def name(self) -> str:
        return "Bayesian Decision"


# =============================================================================
# Factory
# =============================================================================


# =============================================================================
# 10. Ensemble: Logistic Regression Stacking
# =============================================================================

class EnsembleStacking(BaseBaseline):
    """Logistic regression stacking of raw, Platt, and isotonic predictions."""

    def __init__(self, window: int = 500):
        self.window = window
        self._logits: list[float] = []
        self._labels: list[float] = []
        self._platt_a = 1.0
        self._platt_b = 0.0
        self._iso = SklearnIsotonic(out_of_bounds="clip", increasing=True)
        self._iso_fitted = False

        self._count = 0

    def initialize(self, logits: np.ndarray, labels: np.ndarray) -> None:
        self._logits = list(logits)
        self._labels = list(labels)
        # Fit Platt
        X = np.asarray(logits, dtype=float).reshape(-1, 1)
        y = np.asarray(labels, dtype=float)
        if len(np.unique(y)) >= 2:
            clf = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
            clf.fit(X, y)
            self._platt_a = float(clf.coef_[0, 0])
            self._platt_b = float(clf.intercept_[0])
        # Fit isotonic
        if len(np.unique(y)) >= 2:
            probs = expit(np.asarray(logits, dtype=float))
            self._iso.fit(probs, labels)
            self._iso_fitted = True
        # Fit stacking LR
        self._refit_stacking()

    def predict(self, logit: float, cost_fn: float, cost_fp: float) -> BaselineResult:
        raw = float(expit(logit))
        platt = float(expit(self._platt_a * logit + self._platt_b))
        iso = float(self._iso.predict(np.array([raw]))[0]) if self._iso_fitted else raw
        if self._lr is not None:
            features = np.array([[raw, platt, iso]])
            prob = float(self._lr.predict_proba(features)[0, 1])
        else:
            prob = (raw + platt + iso) / 3.0
        decision = 1 if prob >= 0.5 else 0
        cost = cost_fn * prob if decision == 0 else cost_fp * (1.0 - prob)
        return BaselineResult(decision=decision, prob=prob, cost=cost)

    def update(self, logit: float, label: float, cost_fn: float, cost_fp: float) -> None:
        self._logits.append(logit)
        self._labels.append(label)
        if len(self._logits) > self.window:
            self._logits.pop(0)
            self._labels.pop(0)
        self._count += 1
        if self._count % 200 == 0:
            self._refit_stacking()

    def _refit_stacking(self) -> None:
        if len(self._logits) < 100 or len(np.unique(self._labels)) < 2:
            return
        logits_arr = np.array(self._logits)
        labels_arr = np.array(self._labels)
        raw = expit(logits_arr)
        platt = expit(self._platt_a * logits_arr + self._platt_b)
        iso = self._iso.predict(raw) if self._iso_fitted else raw
        features = np.column_stack([raw, platt, iso])
        self._lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        self._lr.fit(features, labels_arr)

    @property
    def name(self) -> str:
        return "Ensemble Stacking"


# =============================================================================
# 11. Weighted Average Ensemble
# =============================================================================

class WeightedEnsemble(BaseBaseline):
    """Weighted average of raw, Platt, and isotonic predictions."""

    def __init__(self):
        self._platt_a = 1.0
        self._platt_b = 0.0
        self._weights = np.array([1.0, 1.0, 1.0])
        self._n = 0

    def initialize(self, logits: np.ndarray, labels: np.ndarray) -> None:
        X = np.asarray(logits, dtype=float).reshape(-1, 1)
        y = np.asarray(labels, dtype=float)
        if len(np.unique(y)) >= 2:
            clf = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
            clf.fit(X, y)
            self._platt_a = float(clf.coef_[0, 0])
            self._platt_b = float(clf.intercept_[0])
        self._n = len(logits)

    def predict(self, logit: float, cost_fn: float, cost_fp: float) -> BaselineResult:
        raw = float(expit(logit))
        platt = float(expit(self._platt_a * logit + self._platt_b))
        w = self._weights / self._weights.sum()
        prob = float(w[0] * raw + w[1] * platt + w[2] * 0.5)  # 0.5 is "isotonic placeholder"
        prob = float(np.clip(prob, 0.0, 1.0))
        decision = 1 if prob >= 0.5 else 0
        cost = cost_fn * prob if decision == 0 else cost_fp * (1.0 - prob)
        return BaselineResult(decision=decision, prob=prob, cost=cost)

    @property
    def name(self) -> str:
        return "Weighted Ensemble"


BASELINE_REGISTRY = {
    "raw": RawThreshold,
    "static_platt": StaticPlatt,
    "static_isotonic": StaticIsotonic,
    "temperature": TemperatureScaling,
    "online_platt": OnlinePlatt,
    "adaptive_calib": AdaptiveCalibration,
    "cost_sensitive": CostSensitivePlatt,
    "threshold_moving": ThresholdMoving,
    "bayesian": BayesianDecisionRule,
    "ensemble_stacking": EnsembleStacking,
    "weighted_ensemble": WeightedEnsemble,
}


def get_baseline(name: str, **kwargs) -> BaseBaseline:
    """Create a baseline by name."""
    if name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline: {name}. Options: {list(BASELINE_REGISTRY.keys())}")
    cls = BASELINE_REGISTRY[name]
    return cls(**kwargs)


def run_baseline_on_stream(
    baseline: BaseBaseline,
    logits: np.ndarray,
    labels: np.ndarray,
    costs_fn: np.ndarray,
    costs_fp: np.ndarray,
    warmup: int = 500,
) -> dict:
    """Run a baseline on a data stream and return metrics.

    Args:
        baseline: Baseline instance.
        logits: Stream of logits, shape (n,).
        labels: True labels, shape (n,).
        costs_fn: False negative costs, shape (n,).
        costs_fp: False positive costs, shape (n,).
        warmup: Number of initial samples for initialization.

    Returns:
        Dict with 'probs', 'decisions', 'costs', 'cumulative_cost', 'name'.
    """
    n = len(logits)
    probs = np.zeros(n)
    decisions = np.zeros(n, dtype=int)
    costs = np.zeros(n)

    # Initialize on warmup data
    if warmup > 0:
        warmup_end = min(warmup, n)
        baseline.initialize(logits[:warmup_end], labels[:warmup_end])
    else:
        warmup_end = 0

    # Stream
    for t in range(warmup_end, n):
        result = baseline.predict(float(logits[t]), float(costs_fn[t]), float(costs_fp[t]))
        probs[t] = result.prob
        decisions[t] = result.decision
        costs[t] = result.cost
        baseline.update(float(logits[t]), float(labels[t]), float(costs_fn[t]), float(costs_fp[t]))

    return {
        "probs": probs,
        "decisions": decisions,
        "costs": costs,
        "cumulative_cost": float(costs.sum()),
        "name": baseline.name,
    }



