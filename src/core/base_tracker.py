"""Abstract base class for all state trackers.

A tracker estimates the latent state of a non-stationary system
from noisy observations. The key outputs are:

1. State estimate (e.g., mean of latent variable)
2. State uncertainty (e.g., covariance)
3. Prediction residual (innovation) for feedback

All trackers follow the same online interface:
    - predict(): Propagate state forward one step
    - update(observation): Incorporate a new observation
    - step(observation): Convenience: predict + update
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseTracker(ABC):
    """Abstract base class for state tracking modules.

    Tracks a latent state x_t from noisy observations y_t.
    Supports online (streaming) operation with predict-update cycles.
    """

    @abstractmethod
    def predict(self) -> None:
        """Propagate the state forward one time step (prior update)."""
        ...

    @abstractmethod
    def update(self, observation: np.ndarray) -> np.ndarray:
        """Incorporate a new observation to obtain the posterior state.

        Args:
            observation: New observation y_t, shape (n_obs,).

        Returns:
            Posterior state estimate, shape (n_states,).
        """
        ...

    @abstractmethod
    def step(self, observation: np.ndarray) -> np.ndarray:
        """Convenience: predict + update in one call.

        Args:
            observation: New observation y_t.

        Returns:
            Posterior state estimate.
        """
        ...

    @property
    @abstractmethod
    def state_estimate(self) -> np.ndarray:
        """Current posterior state estimate, shape (n_states,)."""
        ...

    @property
    @abstractmethod
    def state_covariance(self) -> np.ndarray:
        """Current posterior state covariance, shape (n_states, n_states)."""
        ...

    @property
    @abstractmethod
    def innovation(self) -> np.ndarray:
        """Most recent innovation (y_t - y_pred), shape (n_obs,).

        The innovation is the prediction residual, a key signal
        for feedback loops (e.g., F2: calibration residual -> UKF).
        """
        ...

    @abstractmethod
    def reset(self, state: np.ndarray | None = None) -> None:
        """Reset the tracker to initial state.

        Args:
            state: Optional initial state. Uses default if None.
        """
        ...
