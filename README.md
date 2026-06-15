# A Closed-Loop Adaptive Decision Framework for Cost-Sensitive Sequential Decision-Making in Non-Stationary Environments

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Repository for the manuscript submitted to *Expert Systems with Applications*.

This algorithm couples **state tracking** (unscented Kalman filter), **online probability calibration** (Platt scaling with sliding window), and **cost-sensitive decision-making** through three uncertainty-driven feedback mechanisms.

## Feedback Mechanisms

- **F1**: Posterior covariance scales the calibration window and regularization strength
- **F2**: Calibration residual inflates the UKF process noise for accelerated tracking during regime changes
- **F3**: Posterior mean and covariance define time-varying cost-sensitive decision thresholds

## Quick Start

```bash
pip install -r requirements.txt
python main.py --ablation
python run_comparison.py
python run_fraud.py
python generate_figures.py
```

## Reproducing Experiments

All experiments use fixed seeds (42, 43, 44). Run the experiment scripts in order:

```bash
python main.py --ablation           # Synthetic ablation study
python run_comparison.py            # Full baseline comparison
python run_fraud.py                 # IEEE-CIS fraud detection
python generate_figures.py          # Generate all figures and tables
```

## Reproducing Figures

```bash
python generate_figures.py
```

## Datasets

- **IEEE-CIS Fraud Detection**: Download from [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection)
- **MIMIC-III**: Requires [PhysioNet credentialing](https://mimic.mit.edu)
- **C-MAPSS**: Simulated, generated on the fly by the experiment scripts

## Results Summary

| Dataset | Cost Reduction | ECE |
|---------|---------------|-----|
| IEEE-CIS (fraud) | 6.3% over best baseline | 0.0007 |
| MIMIC-III (mortality) | 18.3% over best baseline | 0.082 |
| C-MAPSS (industrial) | 10.8% over best baseline | 0.015 |
| Synthetic (rapid drift) | 8.7% over cost-sensitive baseline | — |

## License

MIT
