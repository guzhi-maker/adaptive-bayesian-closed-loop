"""Realistic fraud detection data generator.

Simulates the key properties of the IEEE-CIS Fraud Detection dataset:
- Low base fraud rate (~2%)
- Temporal non-stationarity (fraud rate drifts over time)
- Transaction amounts as dynamic costs (FN = amount, FP = fixed)
- Weekly windows with gradual changes
- Realistic logit distributions from a pre-trained model
- Can generate multiple seeds for reproducibility

Paper reference: IEEE-CIS Fraud Detection, Kaggle 2019
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass
class FraudDataConfig:
    """Configuration for realistic fraud data generation.

    Attributes:
        n_weeks: Number of weekly windows (52 weeks = 1 year).
        samples_per_week: Average samples per week (default ~11,350 for 590K/yr).
        base_fraud_rate: Initial fraud rate (~2% is typical for credit cards).
        fraud_rate_drift: How fraud rate changes per week (simulates seasonal effects).
        amount_mean: Mean transaction amount in dollars.
        amount_std: Standard deviation of transaction amount.
        fp_cost: Fixed false positive cost (manual review cost, ~$15).
        logit_separation: How well logits separate fraud vs legitimate.
        weekly_volatility: Random volatility in fraud rate per week.
        seed: Random seed.
    """

    n_weeks: int = 52
    samples_per_week: int = 11350  # ~590K / 52
    base_fraud_rate: float = 0.02
    fraud_rate_drift: float = 0.002  # Per-week drift (up to ~10% by week 52)
    amount_mean: float = 150.0
    amount_std: float = 75.0
    fp_cost: float = 15.0
    logit_separation: float = 2.5
    weekly_volatility: float = 0.005
    seed: int = 42


class FraudDataGenerator:
    """Generates realistic fraud detection data with temporal non-stationarity.

    The data simulates what you'd get after running a pre-trained XGBoost
    model on the IEEE-CIS dataset: logits, labels, and transaction amounts
    organized in temporal order.
    """

    def __init__(self, config: FraudDataConfig):
        self.config = config
        self._rng = np.random.RandomState(config.seed)

    def generate(self) -> dict:
        """Generate the fraud dataset.

        Returns:
            Dict with keys:
            - 'logits': Raw classifier logits, shape (n,)
            - 'labels': True labels (1=fraud), shape (n,)
            - 'costs_no_action': Transaction amount (FN cost), shape (n,)
            - 'costs_action': Fixed FP cost, shape (n,)
            - 'fraud_rate': True fraud rate per week, shape (n_weeks,)
            - 'week_idx': Week index for each sample, shape (n,)
            - 'amounts': Transaction amounts, shape (n,)
        """
        cfg = self.config
        n_total = cfg.n_weeks * cfg.samples_per_week

        # Generate weekly fraud rates with drift and seasonality
        week_fraud_rates = self._generate_weekly_rates()

        # Pre-allocate
        logits = np.zeros(n_total)
        labels = np.zeros(n_total)
        amounts = np.zeros(n_total)
        costs_na = np.zeros(n_total)
        costs_a = np.full(n_total, cfg.fp_cost)
        week_idx = np.zeros(n_total, dtype=int)

        idx = 0
        for w in range(cfg.n_weeks):
            n_w = cfg.samples_per_week
            f_rate = week_fraud_rates[w]

            # Generate labels for this week
            week_labels = (self._rng.rand(n_w) < f_rate).astype(float)

            # Generate logits: well-separated for fraud, lower for legitimate
            # Fraud: positive logits, Legitimate: negative logits
            week_logits = np.zeros(n_w)
            fraud_mask = week_labels == 1
            legit_mask = ~fraud_mask

            n_fraud = fraud_mask.sum()
            n_legit = n_w - n_fraud

            if n_fraud > 0:
                week_logits[fraud_mask] = (
                    cfg.logit_separation + self._rng.randn(n_fraud) * 0.8
                )
            if n_legit > 0:
                week_logits[legit_mask] = (
                    -cfg.logit_separation + self._rng.randn(n_legit) * 1.0
                )

            # Add drift in logit quality (classifier degrades/improves over time)
            quality_drift = 0.1 * np.sin(2 * np.pi * w / 26)  # Semi-annual cycle
            week_logits += quality_drift

            # Generate transaction amounts (log-normal distribution)
            week_amounts = np.exp(
                np.log(cfg.amount_mean)
                + self._rng.randn(n_w) * 0.5
            )
            # Clip to reasonable range
            week_amounts = np.clip(week_amounts, 1.0, 10000.0)

            # Store
            start = idx
            end = idx + n_w
            logits[start:end] = week_logits
            labels[start:end] = week_labels
            amounts[start:end] = week_amounts
            costs_na[start:end] = week_amounts  # FN cost = transaction amount
            week_idx[start:end] = w
            idx = end

        return {
            "logits": logits.astype(np.float32),
            "labels": labels.astype(np.float32),
            "costs_no_action": costs_na.astype(np.float32),
            "costs_action": costs_a.astype(np.float32),
            "fraud_rate": week_fraud_rates.astype(np.float32),
            "week_idx": week_idx,
            "amounts": amounts.astype(np.float32),
        }

    def _generate_weekly_rates(self) -> np.ndarray:
        """Generate weekly fraud rates with drift + seasonality + volatility."""
        cfg = self.config
        rates = np.zeros(cfg.n_weeks)

        for w in range(cfg.n_weeks):
            # Base + drift
            base = cfg.base_fraud_rate + cfg.fraud_rate_drift * w
            # Seasonality: more fraud during holidays (weeks 45-52)
            seasonal = 0.01 * np.sin(2 * np.pi * (w + 10) / 52)
            # Random volatility
            noise = self._rng.randn() * cfg.weekly_volatility
            rate = base + seasonal + noise
            rates[w] = float(np.clip(rate, 0.001, 0.15))

        return rates


if __name__ == "__main__":
    gen = FraudDataGenerator(FraudDataConfig(n_weeks=10, samples_per_week=1000))
    data = gen.generate()
    print(f"Samples: {len(data['logits']):,}")
    print(f"Overall fraud rate: {data['labels'].mean():.4f}")
    print(f"Weekly fraud rates: {data['fraud_rate']}")
    print(f"Logit range: [{data['logits'].min():.2f}, {data['logits'].max():.2f}]")
    print(f"Amount range: [{data['amounts'].min():.2f}, {data['amounts'].max():.2f}]")
    print(f"FN cost range: [{data['costs_no_action'].min():.2f}, {data['costs_no_action'].max():.2f}]")
    print("Fraud data generator OK")
