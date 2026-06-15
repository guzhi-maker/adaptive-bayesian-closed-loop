"""UKF-based state tracker for non-stationary environments.

Migrated and adapted from the online Bayesian prior calibration project.
Tracks a latent state (e.g., distribution drift parameter) using an
Unscented Kalman Filter with a configurable observation model.

State variable:
    x = latent state (e.g., log-odds of error rate)

Process model:
    Random walk in state space: x_{t+1} = x_t + N(0, Q)

Observation model:
    Provided by caller, maps state -> expected observation + variance
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from src.core.base_tracker import BaseTracker


class UKFTracker(BaseTracker):
    """Unscented Kalman Filter tracker for non-stationary state tracking.

    Tracks a latent state x_t from noisy observations y_t using the
    unscented transform, which handles nonlinear observation models
    without linearization.

    Parameters
    ----------
    initial_state : np.ndarray
        Initial state estimate, shape (n_states,).
    initial_covariance : np.ndarray
        Initial state covariance, shape (n_states, n_states).
    process_noise : np.ndarray
        Process noise covariance Q, shape (n_states, n_states).
    observation_model : callable
        Function mapping state -> (expected_obs, obs_variance).
        obs_variance can be a scalar or array matching observation dim.
    alpha : float
        UKF spread parameter (default 1e-3).
    beta : float
        UKF prior knowledge parameter (beta=2 optimal for Gaussian).
    kappa : float
        UKF secondary scaling parameter (default 0).
    state_clamp : tuple[float, float] | None
        Optional (min, max) to clamp each state dimension after update.
    """

    def __init__(
        self,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
        process_noise: np.ndarray,
        observation_model: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
        state_clamp: tuple[float, float] | None = None,
    ):
        self._n = len(initial_state)
        self._x = initial_state.copy()
        self._P = initial_covariance.copy()
        self._Q = process_noise.copy()
        self._obs_model = observation_model
        self._state_clamp = state_clamp

        # UKF parameters
        self._alpha = alpha
        self._beta = beta
        self._kappa = kappa
        lam = alpha**2 * (self._n + kappa) - self._n
        self._lam = lam

        # Sigma-point weights
        self._Wm = np.full(2 * self._n + 1, 1.0 / (2.0 * (self._n + lam)))
        self._Wc = np.full(2 * self._n + 1, 1.0 / (2.0 * (self._n + lam)))
        self._Wm[0] = lam / (self._n + lam)
        self._Wc[0] = lam / (self._n + lam) + (1.0 - alpha**2 + beta)

        # Cached prediction
        self._sigma_points_pred: np.ndarray | None = None
        self._x_pred: np.ndarray | None = None
        self._P_pred: np.ndarray | None = None
        self._innovation: np.ndarray | None = None

    # --- BaseTracker interface -----------------------------------------------

    @property
    def state_estimate(self) -> np.ndarray:
        return self._x.copy()

    @property
    def state_covariance(self) -> np.ndarray:
        return self._P.copy()

    @property
    def innovation(self) -> np.ndarray:
        if self._innovation is None:
            return np.zeros(1)
        return self._innovation.copy()

    def predict(self) -> None:
        """Predict step: propagate sigma points through random walk."""
        sigma_points = self._generate_sigma_points()
        sigma_points_pred = sigma_points  # Random walk: x_{t+1} = x_t + noise

        # Predicted mean
        self._x_pred = np.dot(self._Wm, sigma_points_pred)

        # Predicted covariance
        self._P_pred = np.zeros((self._n, self._n))
        for i in range(2 * self._n + 1):
            diff = sigma_points_pred[i] - self._x_pred
            self._P_pred += self._Wc[i] * np.outer(diff, diff)
        self._P_pred += self._Q

        self._sigma_points_pred = sigma_points_pred
        self._P = self._P_pred

    def update(self, observation: np.ndarray) -> np.ndarray:
        """Update step: incorporate observation via UKF update.

        Args:
            observation: Observed value y_t, shape (n_obs,).

        Returns:
            Posterior state estimate, shape (n_states,).
        """
        if self._sigma_points_pred is None:
            raise RuntimeError("Must call predict() before update() or use step().")

        obs = np.asarray(observation, dtype=float).flatten()
        n_obs = len(obs)

        # Propagate sigma points through observation function
        obs_sigma = np.zeros((2 * self._n + 1, n_obs))
        obs_var_sigma = np.zeros((2 * self._n + 1, n_obs))
        for i in range(2 * self._n + 1):
            mu, var = self._obs_model(self._sigma_points_pred[i])
            obs_sigma[i] = np.asarray(mu).flatten()
            obs_var_sigma[i] = np.asarray(var).flatten()

        # Predicted observation mean
        y_pred = np.dot(self._Wm, obs_sigma)

        # Predicted observation covariance P_yy
        P_yy = np.zeros((n_obs, n_obs))
        for i in range(2 * self._n + 1):
            diff = obs_sigma[i] - y_pred
            P_yy += self._Wc[i] * np.outer(diff, diff)
        P_yy += np.diag(np.mean(obs_var_sigma, axis=0))

        # Cross-covariance P_xy
        P_xy = np.zeros((self._n, n_obs))
        for i in range(2 * self._n + 1):
            diff_x = self._sigma_points_pred[i] - self._x_pred
            diff_y = obs_sigma[i] - y_pred
            P_xy += self._Wc[i] * np.outer(diff_x, diff_y)

        # Kalman gain
        K = P_xy @ np.linalg.inv(P_yy)

        # Update
        innovation = obs - y_pred
        self._innovation = innovation.copy()
        self._x = self._x_pred + (K @ innovation).flatten()
        self._P = self._P_pred - K @ P_yy @ K.T

        # Ensure symmetry
        self._P = (self._P + self._P.T) / 2

        # Clamp state if configured
        if self._state_clamp is not None:
            self._x = np.clip(self._x, self._state_clamp[0], self._state_clamp[1])

        return self._x.copy()

    def step(self, observation: np.ndarray) -> np.ndarray:
        """Predict + update in one call."""
        self.predict()
        return self.update(observation)

    def reset(self, state: np.ndarray | None = None) -> None:
        """Reset tracker to initial state."""
        if state is not None:
            self._x = np.asarray(state, dtype=float).copy()
        else:
            self._x = np.zeros(self._n)
        self._P = np.eye(self._n) * 0.01
        self._sigma_points_pred = None
        self._x_pred = None
        self._P_pred = None
        self._innovation = None

    def set_process_noise(self, Q: np.ndarray) -> None:
        """Update the process noise covariance (used by F2 feedback)."""
        self._Q = np.asarray(Q, dtype=float)

    # --- Internal helpers ----------------------------------------------------

    def _generate_sigma_points(self) -> np.ndarray:
        """Generate sigma points from current state and covariance."""
        sqrt_P = np.linalg.cholesky((self._n + self._lam) * self._P)
        sigma_points = np.zeros((2 * self._n + 1, self._n))
        sigma_points[0] = self._x
        for i in range(self._n):
            sigma_points[i + 1] = self._x + sqrt_P[i]
            sigma_points[i + 1 + self._n] = self._x - sqrt_P[i]
        return sigma_points
