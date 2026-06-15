"""Unified dataset loader for all data sources.

Provides a consistent interface for loading:
1. Synthetic generated data
2. IEEE-CIS financial fraud data (after preprocessing)
3. NASA C-MAPSS industrial data (after preprocessing, optional)

All loaders return data in the same format:
    {'logits', 'labels', 'costs_no_action', 'costs_action'}
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np


class DatasetLoader:
    """Unified dataset loader.

    Provides a consistent interface across synthetic and real datasets.
    All loaded data is returned in the same dict format for seamless
    switching between datasets in experiments.
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)

    def load_synthetic(
        self,
        n_samples: int = 100_000,
        drift_type: Literal["gradual", "abrupt", "periodic", "none"] = "gradual",
        cost_type: Literal["fixed", "dynamic", "stratified"] = "dynamic",
        drift_speed: float = 1e-4,
        seed: int = 42,
    ) -> dict:
        """Generate and load a synthetic dataset on the fly.

        Args:
            n_samples: Number of samples.
            drift_type: Type of distribution drift.
            cost_type: Type of cost structure.
            drift_speed: Rate of drift.
            seed: Random seed.

        Returns:
            Dict with 'logits', 'labels', 'costs_no_action', 'costs_action'.
        """
        from src.data.synthetic_generator import (
            CostConfig,
            DriftConfig,
            SyntheticDataConfig,
            SyntheticDataGenerator,
        )

        cfg = SyntheticDataConfig(
            n_samples=n_samples,
            drift=DriftConfig(
                drift_type=drift_type,
                drift_speed=drift_speed,
            ),
            cost=CostConfig(cost_type=cost_type),
            seed=seed,
        )
        generator = SyntheticDataGenerator(cfg)
        data = generator.generate()
        return {
            "logits": data["logits"],
            "labels": data["labels"],
            "costs_no_action": data["costs_no_action"],
            "costs_action": data["costs_action"],
            "drift_state": data.get("drift_state"),
            "true_probs": data.get("true_probs"),
        }

    def load_from_disk(self, path: str | Path) -> dict:
        """Load a preprocessed dataset from disk.

        Expected format: npz file with keys
        'logits', 'labels', 'costs_no_action', 'costs_action'.

        Args:
            path: Path to .npz file.

        Returns:
            Dict with data arrays.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        data = np.load(path)
        return {
            "logits": data["logits"],
            "labels": data["labels"],
            "costs_no_action": data["costs_no_action"],
            "costs_action": data["costs_action"],
        }

    def save_to_disk(self, data: dict, path: str | Path) -> None:
        """Save a dataset to disk in .npz format.

        Args:
            data: Dict with data arrays.
            path: Output path (.npz).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **data)
        print(f"Dataset saved to {path}")

    @staticmethod
    def split_sequential(
        data: dict,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
    ) -> tuple[dict, dict, dict]:
        """Sequentially split time-ordered data into train/val/test.

        Preserves temporal order (no random shuffle).

        Args:
            data: Full dataset dict.
            train_ratio: Fraction for training set.
            val_ratio: Fraction for validation set.

        Returns:
            (train_data, val_data, test_data).
        """
        n = len(data["logits"])
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        def _slice(start: int, end: int) -> dict:
            return {k: v[start:end] for k, v in data.items() if v is not None}

        return (
            _slice(0, n_train),
            _slice(n_train, n_train + n_val),
            _slice(n_train + n_val, n),
        )
