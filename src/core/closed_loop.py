"""Closed-Loop Framework: Estimation -> Calibration -> Decision with feedback.

This is the core orchestrator that combines:
1. State tracker (e.g., UKF)
2. Calibrator (e.g., Platt + Bootstrap)
3. Decision maker (e.g., three-layer)
4. Three feedback loops (F1, F2, F3)

The framework processes a stream of (logit, label, cost) tuples and
closes the loop by feeding back information at each step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from src.core.base_tracker import BaseTracker
from src.core.base_calibrator import BaseCalibrator
from src.core.base_decision_maker import BaseDecisionMaker, Action, DecisionResult
from src.feedback.f1_state_to_calibration import F1StateToCalibration
from src.feedback.f2_residual_to_ukf import F2CalibrationResidualToUKF
from src.feedback.f3_state_to_threshold import F3StateToThreshold


@dataclass
class ClosedLoopConfig:
    """Configuration for the closed-loop framework.

    Controls which feedback loops are active and their parameters.
    """

    use_f1: bool = True  # State -> Calibration window/reg
    use_f2: bool = True  # Calibration residual -> UKF process noise
    use_f3: bool = True  # State + uncertainty -> Decision thresholds
    calibrate_every: int = 500  # Re-fit calibrator every N steps
    ci_every: int = 1000  # Recompute bootstrap CI every N steps
    bootstrap_n: int = 50  # Bootstrap resamples for CI
    calibration_window: int = 2000  # Max samples in calibration buffer
    warmup_steps: int = 500  # Steps before activating feedback

    # F1 parameters
    f1_min_window: int = 500
    f1_max_window: int = 5000
    f1_base_reg: float = 1e-4
    f1_sensitivity: float = 1.0

    # F2 parameters
    f2_base_Q: float = 1e-4
    f2_min_Q: float = 1e-6
    f2_max_Q: float = 1e-2
    f2_target_residual: float = 0.01
    f2_sensitivity: float = 10.0

    # F3 parameters
    f3_base_low: float = 0.4
    f3_base_high: float = 0.6
    f3_drift_sensitivity: float = 0.5
    f3_uncertainty_sensitivity: float = 2.0


@dataclass
class StepRecord:
    """Record of a single closed-loop step."""

    step: int
    logit: float
    label: int
    calibrated_prob: float
    ci_lower: float
    ci_upper: float
    decision: DecisionResult
    state_estimate: np.ndarray
    state_uncertainty: float
    f1_feedback: dict | None
    f2_feedback: dict | None
    f3_feedback: dict | None


class ClosedLoopFramework:
    """Main closed-loop orchestrator.

    Processes a stream of data points through the estimation-calibration-
    decision pipeline with bidirectional feedback between modules.
    """

    def __init__(
        self,
        tracker: BaseTracker,
        calibrator: BaseCalibrator,
        decision_maker: BaseDecisionMaker,
        config: ClosedLoopConfig | None = None,
        observation_model: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]
        | None = None,
    ):
        self.tracker = tracker
        self.calibrator = calibrator
        self.decision_maker = decision_maker
        self.config = config if config is not None else ClosedLoopConfig()

        # Feedback modules
        self.f1 = F1StateToCalibration(
            base_window=self.config.calibration_window,
            min_window=self.config.f1_min_window,
            max_window=self.config.f1_max_window,
            base_reg=self.config.f1_base_reg,
            sensitivity=self.config.f1_sensitivity,
        )
        self.f2 = F2CalibrationResidualToUKF(
            base_Q=self.config.f2_base_Q,
            min_Q=self.config.f2_min_Q,
            max_Q=self.config.f2_max_Q,
            target_residual=self.config.f2_target_residual,
            sensitivity=self.config.f2_sensitivity,
        )
        self.f3 = F3StateToThreshold(
            base_low=self.config.f3_base_low,
            base_high=self.config.f3_base_high,
            drift_sensitivity=self.config.f3_drift_sensitivity,
            uncertainty_sensitivity=self.config.f3_uncertainty_sensitivity,
        )

        # Calibration buffer
        self._cal_logits: list[float] = []
        self._cal_labels: list[int] = []
        self._step_count = 0
        self._history: list[StepRecord] = []

        # Observation model (for UKF update)
        self._observation_model = observation_model

    # --- Public API ----------------------------------------------------------

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def history(self) -> list[StepRecord]:
        return list(self._history)

    @property
    def is_warmed_up(self) -> bool:
        return self._step_count >= self.config.warmup_steps

    def process_stream(
        self,
        logits: np.ndarray,
        labels: np.ndarray | None = None,
        costs_no_action: np.ndarray | None = None,
        costs_action: np.ndarray | None = None,
        observations: np.ndarray | None = None,
        verbose: bool = False,
    ) -> list[StepRecord]:
        """Process a stream of data points through the closed loop.

        Args:
            logits: Raw model outputs, shape (n_steps,).
            labels: True labels, shape (n_steps,). Optional for inference.
            costs_no_action: False negative costs, shape (n_steps,). Optional.
            costs_action: False positive costs, shape (n_steps,). Optional.
            observations: Tracker observations, shape (n_steps, n_obs). Optional.
            verbose: If True, print progress.

        Returns:
            List of StepRecord for each step.
        """
        n_steps = len(logits)

        for t in range(n_steps):
            logit = float(logits[t])
            label = int(labels[t]) if labels is not None else -1
            c_na = float(costs_no_action[t]) if costs_no_action is not None else None
            c_a = float(costs_action[t]) if costs_action is not None else None
            obs = observations[t] if observations is not None else None

            record = self._step(logit, label, c_na, c_a, obs)
            self._history.append(record)

            if verbose and (t + 1) % 1000 == 0:
                print(f"  Step {t + 1}/{n_steps} | "
                      f"State: {record.state_estimate[0]:.4f} | "
                      f"Decision: {record.decision.action.value}")

        return self._history

    def get_cumulative_cost(self) -> float:
        """Return total expected cost across all decisions."""
        return self.decision_maker.total_expected_cost

    def get_average_cost(self) -> float:
        """Return average expected cost per decision."""
        return self.decision_maker.average_expected_cost

    def get_decision_breakdown(self) -> dict:
        """Return decision type counts."""
        return self.decision_maker.get_decision_breakdown()

    def reset(self) -> None:
        """Reset the entire framework to initial state."""
        self.tracker.reset()
        self.calibrator.reset()
        self.decision_maker.reset()
        self._cal_logits = []
        self._cal_labels = []
        self._step_count = 0
        self._history = []
        self.f2.reset()

    # --- Internal ------------------------------------------------------------

    def _step(
        self,
        logit: float,
        label: int,
        cost_no_action: float | None,
        cost_action: float | None,
        observation: np.ndarray | None,
    ) -> StepRecord:
        """Execute one closed-loop step."""

        # --- 1. Tracker update -----------------------------------------------
        if observation is not None and self._observation_model is not None:
            self.tracker.step(observation)
        elif observation is not None:
            self.tracker.step(observation)

        state_est = self.tracker.state_estimate.copy()
        state_uncertainty = float(np.trace(self.tracker.state_covariance))

        # --- 2. Feedback signals ---------------------------------------------

        # F1: State -> Calibration
        f1_fb = None
        if self.config.use_f1 and self.is_warmed_up:
            f1_fb = self.f1.get_feedback(self.tracker.state_covariance)

        # F3: State -> Decision thresholds
        f3_fb = None
        if self.config.use_f3 and self.is_warmed_up:
            low_t, high_t = self.f3.get_thresholds(
                state_est, self.tracker.state_covariance
            )
            self.decision_maker.set_dynamic_thresholds(low_t, high_t)
            f3_fb = {"low_threshold": low_t, "high_threshold": high_t}

        # --- 3. Calibration --------------------------------------------------
        # Add to calibration buffer
        self._cal_logits.append(logit)
        self._cal_labels.append(label)

        # Apply F1 window size
        max_window = self.config.calibration_window
        if f1_fb is not None:
            max_window = f1_fb["window_size"]
        if len(self._cal_logits) > max_window:
            self._cal_logits.pop(0)
            self._cal_labels.pop(0)

        # Periodic recalibration
        if (
            self._step_count > 0
            and self._step_count % self.config.calibrate_every == 0
            and len(self._cal_logits) >= 100
        ):
            cal_logits = np.array(self._cal_logits)
            cal_labels = np.array(self._cal_labels)
            if len(np.unique(cal_labels)) >= 2:
                # Apply F1 regularization
                reg = self.config.f1_base_reg
                if f1_fb is not None:
                    reg = f1_fb["regularization"]
                self.calibrator.regularization = reg  # type: ignore
                self.calibrator.fit(cal_logits, cal_labels)

        # Predict calibrated probability and CI
        prob, ci_low, ci_high = self._predict_with_ci(logit)

        # --- 4. Cost update --------------------------------------------------
        if cost_no_action is not None and cost_action is not None:
            self.decision_maker.set_dynamic_costs(cost_no_action, cost_action)

        # --- 5. Decision -----------------------------------------------------
        result = self.decision_maker.decide(
            float(prob),
            ci_lower=float(ci_low),
            ci_upper=float(ci_high),
        )

        # --- 6. F2: Calibration residual -> UKF process noise ----------------
        f2_fb = None
        if (
            self.config.use_f2
            and self.is_warmed_up
            and len(self._cal_logits) >= 100
        ):
            residual = float(
                np.mean(np.abs(
                    self.calibrator.predict(np.array(self._cal_logits[-100:]))
                    - np.array(self._cal_labels[-100:])
                ))
            )
            f2_fb = self.f2.get_feedback(residual)
            Q = f2_fb["process_noise"]
            current_Q = self.tracker.state_covariance
            n_states = len(current_Q)
            self.tracker.set_process_noise(np.eye(n_states) * Q)

        # --- 7. Record -------------------------------------------------------
        self._step_count += 1

        return StepRecord(
            step=self._step_count,
            logit=logit,
            label=label,
            calibrated_prob=float(prob),
            ci_lower=float(ci_low),
            ci_upper=float(ci_high),
            decision=result,
            state_estimate=state_est,
            state_uncertainty=state_uncertainty,
            f1_feedback=f1_fb,
            f2_feedback=f2_fb,
            f3_feedback=f3_fb,
        )

    def _predict_with_ci(
        self, logit: float
    ) -> tuple[float, float, float]:
        """Get calibrated probability with CI for a single logit."""
        logits_arr = np.array([logit])

        if not self.calibrator.is_fitted:
            prob = float(1.0 / (1.0 + np.exp(-logit)))
            return prob, max(0, prob - 0.05), min(1, prob + 0.05)

        try:
            probs, ci_low, ci_high = self.calibrator.predict_with_ci(logits_arr)
            return float(probs[0]), float(ci_low[0]), float(ci_high[0])
        except Exception:
            probs = self.calibrator.predict(logits_arr)
            prob = float(probs[0])
            return prob, max(0, prob - 0.05), min(1, prob + 0.05)
