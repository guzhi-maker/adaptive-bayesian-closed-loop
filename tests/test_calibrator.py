"""Tests for Platt calibrator and bootstrap estimator."""
import sys
sys.path.insert(0, ".")

import numpy as np
from scipy.special import expit

from src.modules.platt_calibrator import PlattCalibrator
from src.modules.bootstrap_estimator import BootstrapEstimator


def test_platt_fit_predict():
    """Platt calibrator can fit and predict on simple data."""
    np.random.seed(42)
    n = 500
    logits = np.random.randn(n) * 2
    true_probs = expit(logits)
    labels = (np.random.rand(n) < true_probs).astype(float)

    cal = PlattCalibrator(use_scalar=True)
    cal.fit(logits, labels)
    probs = cal.predict(logits)
    assert probs.shape == (n,)
    assert np.all((0 <= probs) & (probs <= 1))
    assert cal.is_fitted


def test_platt_isotonic():
    """Platt prediction is monotonic in logits."""
    cal = PlattCalibrator(use_scalar=True)
    cal.fit(np.array([-2, -1, 0, 1, 2]), np.array([0, 0, 1, 1, 1]))
    probs = cal.predict(np.array([-3, -1, 1, 3]))
    assert probs[0] < probs[1] < probs[2] < probs[3]


def test_platt_unfitted():
    """Unfitted Platt returns raw sigmoid."""
    cal = PlattCalibrator()
    probs = cal.predict(np.array([0.0, 1.0, -1.0]))
    expected = expit(np.array([0.0, 1.0, -1.0]))
    assert np.allclose(probs, expected)


def test_platt_reset():
    """Reset clears fitted state."""
    cal = PlattCalibrator()
    cal.fit(np.array([0.0, 1.0]), np.array([0, 1]))
    assert cal.is_fitted
    cal.reset()
    assert not cal.is_fitted
    assert cal.a_ == 1.0 and cal.b_ == 0.0


def test_platt_predict_with_ci_fallback():
    """Predict with CI works even without bootstrap (fallback margin)."""
    cal = PlattCalibrator()
    probs, ci_low, ci_high = cal.predict_with_ci(np.array([0.0, 0.5]))
    assert np.all(ci_low <= probs) and np.all(probs <= ci_high)


def test_platt_fit_with_bootstrap():
    """Bootstrap fitting produces multiple (a, b) samples."""
    np.random.seed(42)
    n = 200
    logits = np.random.randn(n)
    labels = (np.random.rand(n) < expit(logits)).astype(float)
    cal = PlattCalibrator(n_bootstrap=20)
    cal.fit_with_bootstrap(logits, labels)
    assert cal._bootstrap_a is not None
    assert cal._bootstrap_b is not None
    assert len(cal._bootstrap_a) == 20


def test_bootstrap_estimator():
    """BootstrapEstimator produces valid CI."""
    np.random.seed(42)
    n = 200
    logits = np.random.randn(n)
    labels = (np.random.rand(n) < expit(logits)).astype(float)
    eval_logits = np.array([-1.0, 0.0, 1.0])

    est = BootstrapEstimator(PlattCalibrator, n_bootstrap=20, use_scalar=True)
    est.fit(logits, labels, eval_logits)
    probs, ci_low, ci_high = est.predict_with_ci(eval_logits)
    assert probs.shape == (3,)
    assert np.all(ci_low <= probs) and np.all(probs <= ci_high)


def test_bootstrap_uncertainty():
    """Uncertainty estimate is positive."""
    np.random.seed(42)
    n = 200
    logits = np.random.randn(n)
    labels = (np.random.rand(n) < expit(logits)).astype(float)

    est = BootstrapEstimator(PlattCalibrator, n_bootstrap=20, use_scalar=True)
    est.fit(logits, labels, np.array([0.0]))
    unc = est.get_uncertainty(np.array([0.0]))
    assert unc[0] >= 0


def test_bootstrap_calibration_residual():
    """Calibration residual returns a finite scalar."""
    np.random.seed(42)
    n = 200
    logits = np.random.randn(n)
    labels = (np.random.rand(n) < expit(logits)).astype(float)

    est = BootstrapEstimator(PlattCalibrator, n_bootstrap=10, use_scalar=True)
    est.fit(logits, labels, np.array([0.0]))
    residual = est.get_calibration_residual(logits[:50], labels[:50])
    assert np.isfinite(residual)
    assert 0 <= residual <= 1


if __name__ == "__main__":
    test_platt_fit_predict()
    test_platt_isotonic()
    test_platt_unfitted()
    test_platt_reset()
    test_platt_predict_with_ci_fallback()
    test_platt_fit_with_bootstrap()
    test_bootstrap_estimator()
    test_bootstrap_uncertainty()
    test_bootstrap_calibration_residual()
    print("All calibrator tests passed!")
