"""Abstract base class for observation models.

An observation model defines how the latent state generates
observations. This is used by the tracker (e.g., UKF) to map
state -> expected observation and observation variance.

For the closed-loop framework, the observation model is a
critical abstraction because it decouples:
1. The tracker's internal dynamics from
2. The specific observation modality

This allows the framework to work with any base classifier
that outputs logits/probabilities.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseObservationModel(ABC):
    """Abstract base class for observation models.

    Maps latent state to expected observation and variance.
    Used by the tracker during the update step.
    """

    @abstractmethod
    def observe(self, state: np.ndarray) -> np.ndarray:
        """Compute expected observation given state.

        Args:
            state: Latent state vector, shape (n_states,).

        Returns:
            Expected observation, shape (n_obs,).
        """
        ...

    @abstractmethod
    def observe_with_noise(
        self, state: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute expected observation and variance.

        Args:
            state: Latent state vector, shape (n_states,).

        Returns:
            (expected_obs, obs_variance), where:
            - expected_obs: shape (n_obs,)
            - obs_variance: shape (n_obs,) or (n_obs, n_obs)
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this observation model."""
        ...
