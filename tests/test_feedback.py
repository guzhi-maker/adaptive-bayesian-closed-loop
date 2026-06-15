"""Tests for feedback loop modules."""
import sys
sys.path.insert(0, ".")

import numpy as np
from src.feedback.f1_state_to_calibration import F1StateToCalibration
from src.feedback.f2_residual_to_ukf import F2CalibrationResidualToUKF
from src.feedback.f3_state_to_threshold import F3StateToThreshold


def test_f1_window_size():
    """Higher uncertainty -> smaller window."""
    f1 = F1StateToCalibration(base_window=2000, min_window=500, max_window=5000)
    cov_low = np.eye(1) * 0.001
    cov_high = np.eye(1) * 1.0
    w_low = f1.compute_window_size(cov_low)
    w_high = f1.compute_window_size(cov_high)
    assert w_low > w_high, f"Expected {w_low} > {w_high}"
    assert f1.min_window <= w_high <= f1.max_window


def test_f1_regularization():
    """Higher uncertainty -> lower regularization."""
    f1 = F1StateToCalibration(base_reg=1e-4, min_reg=1e-6, max_reg=1e-2)
    cov_low = np.eye(1) * 0.001
    cov_high = np.eye(1) * 1.0
    r_low = f1.compute_regularization(cov_low)
    r_high = f1.compute_regularization(cov_high)
    assert r_low > r_high, f"Expected {r_low} > {r_high}"


def test_f1_feedback():
    """get_feedback returns expected keys."""
    f1 = F1StateToCalibration()
    fb = f1.get_feedback(np.eye(1) * 0.1)
    assert "window_size" in fb
    assert "regularization" in fb
    assert "uncertainty" in fb


def test_f2_process_noise():
    """Higher calibration residual -> higher process noise."""
    f2 = F2CalibrationResidualToUKF(base_Q=1e-4, min_Q=1e-6, max_Q=1e-2)
    q_low = f2.compute_process_noise(0.001)
    q_high = f2.compute_process_noise(0.5)
    assert q_high >= q_low


def test_f2_clamping():
    """Process noise is clamped to [min_Q, max_Q]."""
    f2 = F2CalibrationResidualToUKF(min_Q=1e-6, max_Q=1e-2)
    q = f2.compute_process_noise(100.0)
    assert q <= f2.max_Q + 1e-10

    q = f2.compute_process_noise(1e-10)
    # Min noise
    assert q >= f2.min_Q - 1e-10


def test_f2_reset():
    """Reset restores base_Q."""
    f2 = F2CalibrationResidualToUKF(base_Q=1e-4)
    f2.compute_process_noise(0.5)
    f2.reset()
    assert abs(f2._current_Q - f2.base_Q) < 1e-10


def test_f3_threshold_direction():
    """Higher state (more drift) -> lower thresholds (more REJECT)."""
    f3 = F3StateToThreshold(base_low=0.4, base_high=0.6, drift_sensitivity=0.5)
    cov = np.eye(1) * 0.01

    low_0, high_0 = f3.get_thresholds(np.array([0.0]), cov)
    low_1, high_1 = f3.get_thresholds(np.array([1.0]), cov)

    # Higher drift should lower thresholds (more conservative)
    assert low_1 <= low_0, f"Expected {low_1} <= {low_0}"
    assert high_1 <= high_0, f"Expected {high_1} <= {high_0}"


def test_f3_wider_margin_with_uncertainty():
    """Higher uncertainty -> wider threshold margin."""
    f3 = F3StateToThreshold(
        min_margin=0.05, max_margin=0.4,
        uncertainty_sensitivity=2.0,
    )
    cov_low = np.eye(1) * 0.001
    cov_high = np.eye(1) * 1.0

    low_low, high_low = f3.get_thresholds(np.array([0.0]), cov_low)
    low_high, high_high = f3.get_thresholds(np.array([0.0]), cov_high)

    margin_low = high_low - low_low
    margin_high = high_high - low_high
    assert margin_high >= margin_low


def test_f3_thresholds_in_range():
    """Thresholds are always in [0, 1] with low < high."""
    f3 = F3StateToThreshold()
    for drift in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        for unc in [0.001, 0.1, 1.0]:
            low, high = f3.get_thresholds(np.array([drift]), np.eye(1) * unc)
            assert 0.0 <= low <= 1.0, f"low={low} out of range"
            assert 0.0 <= high <= 1.0, f"high={high} out of range"
            assert low < high, f"low={low} >= high={high}"


def test_f3_feedback():
    """get_feedback returns expected keys."""
    f3 = F3StateToThreshold()
    fb = f3.get_feedback(np.array([0.5]), np.eye(1) * 0.1)
    assert "low_threshold" in fb
    assert "high_threshold" in fb
    assert "margin" in fb
    assert "center_shift" in fb


if __name__ == "__main__":
    test_f1_window_size()
    test_f1_regularization()
    test_f1_feedback()
    test_f2_process_noise()
    test_f2_clamping()
    test_f2_reset()
    test_f3_threshold_direction()
    test_f3_wider_margin_with_uncertainty()
    test_f3_thresholds_in_range()
    test_f3_feedback()
    print("All feedback tests passed!")
