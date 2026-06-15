"""Tests for UKF tracker module."""
import sys
sys.path.insert(0, ".")

import numpy as np
from src.modules.ukf_tracker import UKFTracker


def test_ukf_initialization():
    """UKF initializes with correct state and covariance."""
    tracker = UKFTracker(
        initial_state=np.array([0.0]),
        initial_covariance=np.eye(1) * 0.1,
        process_noise=np.eye(1) * 1e-4,
        observation_model=lambda s: (s.copy(), np.array([0.01])),
    )
    assert tracker.state_estimate.shape == (1,)
    assert tracker.state_covariance.shape == (1, 1)
    assert abs(tracker.state_estimate[0]) < 1e-10
    assert abs(tracker.state_covariance[0, 0] - 0.1) < 1e-10


def test_ukf_predict_update_cycle():
    """UKF can run predict+update without errors."""
    tracker = UKFTracker(
        initial_state=np.array([0.0]),
        initial_covariance=np.eye(1) * 0.1,
        process_noise=np.eye(1) * 1e-4,
        observation_model=lambda s: (s.copy(), np.array([0.01])),
    )
    tracker.predict()
    state = tracker.update(np.array([0.5]))
    assert state.shape == (1,)
    assert np.isfinite(state[0])


def test_ukf_step():
    """UKF step (predict+update) returns finite state."""
    tracker = UKFTracker(
        initial_state=np.array([1.0]),
        initial_covariance=np.eye(1) * 0.1,
        process_noise=np.eye(1) * 1e-4,
        observation_model=lambda s: (s.copy(), np.array([0.01])),
    )
    for _ in range(100):
        state = tracker.step(np.array([1.0]))
        assert np.isfinite(state[0])


def test_ukf_tracks_constant():
    """UKF converges to a constant signal."""
    tracker = UKFTracker(
        initial_state=np.array([0.0]),
        initial_covariance=np.eye(1) * 1.0,
        process_noise=np.eye(1) * 1e-6,
        observation_model=lambda s: (s.copy(), np.array([0.01])),
    )
    true_value = 2.0
    for _ in range(200):
        state = tracker.step(np.array([true_value]))
    # Should converge close to true value
    assert abs(state[0] - true_value) < 0.5


def test_ukf_state_clamp():
    """State clamping works correctly."""
    tracker = UKFTracker(
        initial_state=np.array([0.0]),
        initial_covariance=np.eye(1) * 1.0,
        process_noise=np.eye(1) * 1.0,
        observation_model=lambda s: (s.copy(), np.array([0.01])),
        state_clamp=(-1.0, 1.0),
    )
    for _ in range(50):
        state = tracker.step(np.array([10.0]))
        assert -1.0 <= state[0] <= 1.0


def test_ukf_innovation():
    """Innovation is finite and has correct shape."""
    tracker = UKFTracker(
        initial_state=np.array([0.0]),
        initial_covariance=np.eye(1) * 0.1,
        process_noise=np.eye(1) * 1e-4,
        observation_model=lambda s: (s.copy(), np.array([0.01])),
    )
    tracker.step(np.array([0.5]))
    innov = tracker.innovation
    assert innov.shape == (1,)
    assert np.isfinite(innov[0])


def test_ukf_reset():
    """Reset restores initial state."""
    tracker = UKFTracker(
        initial_state=np.array([0.0]),
        initial_covariance=np.eye(1) * 0.1,
        process_noise=np.eye(1) * 1e-4,
        observation_model=lambda s: (s.copy(), np.array([0.01])),
    )
    tracker.step(np.array([1.0]))
    tracker.step(np.array([2.0]))
    tracker.reset()
    assert abs(tracker.state_estimate[0]) < 1e-10


def test_ukf_process_noise_update():
    """set_process_noise correctly updates Q."""
    tracker = UKFTracker(
        initial_state=np.array([0.0]),
        initial_covariance=np.eye(1) * 0.1,
        process_noise=np.eye(1) * 1e-4,
        observation_model=lambda s: (s.copy(), np.array([0.01])),
    )
    tracker.set_process_noise(np.eye(1) * 0.5)
    assert abs(tracker._Q[0, 0] - 0.5) < 1e-10


if __name__ == "__main__":
    test_ukf_initialization()
    test_ukf_predict_update_cycle()
    test_ukf_step()
    test_ukf_tracks_constant()
    test_ukf_state_clamp()
    test_ukf_innovation()
    test_ukf_reset()
    test_ukf_process_noise_update()
    print("All UKF tests passed!")
