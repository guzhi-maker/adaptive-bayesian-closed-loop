"""Synthetic data generator for non-stationary environments.

Generates synthetic data with configurable:
1. Distribution drift (gradual, abrupt, periodic)
2. Dynamic costs (fixed, time-varying, stratified)
3. Noise levels and base rates

The generator creates (logit, label, cost_no_action, cost_action) tuples
that simulate the output of a base classifier in a non-stationary
environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass
class DriftConfig:
    """Configuration for distribution drift.

    Attributes:
        drift_type: Type of drift ('gradual', 'abrupt', 'periodic', 'none').
        drift_speed: Rate of gradual drift per step.
        abrupt_freq: Frequency of abrupt shifts (in steps).
        abrupt_magnitude: Magnitude of abrupt shifts (in logit space).
        period: Period of cyclic drift (in steps).
        periodic_amplitude: Amplitude of cyclic drift.
    """

    drift_type: Literal["gradual", "abrupt", "periodic", "none"] = "gradual"
    drift_speed: float = 1e-4
    abrupt_freq: int = 10000
    abrupt_magnitude: float = 0.3
    period: int = 10000
    periodic_amplitude: float = 0.2


@dataclass
class CostConfig:
    """Configuration for cost structure.

    Attributes:
        cost_type: Type of cost ('fixed', 'dynamic', 'stratified').
        base_fp: Base false positive cost (cost of action when y=0).
        base_fn: Base false negative cost (cost of no action when y=1).
        dynamic_rate: Rate of cost change per step (for 'dynamic').
        n_strata: Number of cost strata (for 'stratified').
    """

    cost_type: Literal["fixed", "dynamic", "stratified"] = "dynamic"
    base_fp: float = 1.0
    base_fn: float = 10.0
    dynamic_rate: float = 1e-4
    n_strata: int = 3


@dataclass
class SyntheticDataConfig:
    """Top-level configuration for synthetic data generation.

    Attributes:
        n_samples: Total number of samples.
        drift: Drift configuration.
        cost: Cost configuration.
        noise_std: Standard deviation of observation noise.
        base_logit_mean: Mean of logit distribution at start.
        base_logit_std: Standard deviation of logit distribution.
        seed: Random seed.
    """

    n_samples: int = 100_000
    drift: DriftConfig = field(default_factory=DriftConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    noise_std: float = 0.1
    base_logit_mean: float = 0.0
    base_logit_std: float = 2.0
    seed: int = 42


class SyntheticDataGenerator:
    """Generates synthetic data for non-stationary experiments.

    Produces (logit, label, cost_no_action, cost_action) tuples where:
    - logit: Raw classifier output (pre-sigmoid)
    - label: True binary label (0 or 1)
    - cost_no_action: Cost of false negative
    - cost_action: Cost of false positive
    """

    def __init__(self, config: SyntheticDataConfig):
        self.config = config
        self._rng = np.random.RandomState(config.seed)

    def generate(self) -> dict:
        """Generate the synthetic dataset.

        Returns:
            Dict with keys:
            - 'logits': Raw classifier logits, shape (n_samples,)
            - 'labels': True labels, shape (n_samples,), in {0, 1}
            - 'costs_no_action': False negative costs, shape (n_samples,)
            - 'costs_action': False positive costs, shape (n_samples,)
            - 'drift_state': Latent drift state at each step, shape (n_samples,)
            - 'true_probs': True underlying probabilities, shape (n_samples,)
        """
        cfg = self.config
        n = cfg.n_samples

        # Generate latent state (drift process)
        drift_state = self._generate_drift(n)

        # Generate true probabilities from drift state
        logit_mean = cfg.base_logit_mean + drift_state
        logits = (
            logit_mean
            + self._rng.randn(n) * cfg.base_logit_std
            + self._rng.randn(n) * cfg.noise_std
        )
        true_probs = 1.0 / (1.0 + np.exp(-logits))

        # Generate labels
        labels = (self._rng.rand(n) < true_probs).astype(float)

        # Generate costs
        costs_na, costs_a = self._generate_costs(n)

        return {
            "logits": logits.astype(np.float32),
            "labels": labels.astype(np.float32),
            "costs_no_action": costs_na.astype(np.float32),
            "costs_action": costs_a.astype(np.float32),
            "drift_state": drift_state.astype(np.float32),
            "true_probs": true_probs.astype(np.float32),
        }

    def generate_dynamic_costs(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Standalone cost generation (for use in online experiments)."""
        return self._generate_costs(n)

    # --- Internal ------------------------------------------------------------

    def _generate_drift(self, n: int) -> np.ndarray:
        """Generate the latent drift process."""
        cfg = self.config.drift
        drift = np.zeros(n)

        if cfg.drift_type == "none":
            return drift

        elif cfg.drift_type == "gradual":
            drift = np.arange(n, dtype=float) * cfg.drift_speed

        elif cfg.drift_type == "abrupt":
            # Random abrupt shifts
            n_shifts = n // cfg.abrupt_freq
            for i in range(1, n_shifts + 1):
                idx = i * cfg.abrupt_freq
                if idx < n:
                    drift[idx:] += self._rng.choice(
                        [-1, 1]
                    ) * cfg.abrupt_magnitude

        elif cfg.drift_type == "periodic":
            drift = cfg.periodic_amplitude * np.sin(
                2 * np.pi * np.arange(n) / cfg.period
            )

        return drift

    def _generate_costs(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Generate cost sequences."""
        cfg = self.config.cost
        base_fn = cfg.base_fn
        base_fp = cfg.base_fp

        if cfg.cost_type == "fixed":
            return np.full(n, base_fn), np.full(n, base_fp)

        elif cfg.cost_type == "dynamic":
            # Costs grow over time, simulating increasing risk
            growth = 1.0 + np.arange(n, dtype=float) * cfg.dynamic_rate
            return base_fn * growth, base_fp * growth

        elif cfg.cost_type == "stratified":
            # Three cost strata with different base rates
            costs_na = np.full(n, base_fn)
            costs_a = np.full(n, base_fp)
            strata_size = n // cfg.n_strata
            for i in range(cfg.n_strata):
                start = i * strata_size
                end = start + strata_size if i < cfg.n_strata - 1 else n
                multiplier = 10.0 ** i  # 1x, 10x, 100x
                costs_na[start:end] *= multiplier
            return costs_na, costs_a

        return np.full(n, base_fn), np.full(n, base_fp)
