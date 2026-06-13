# Adaptive Bayesian Closed-Loop Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**An Adaptive Bayesian Framework for Sequential Decision-Making under Non-Stationary Environments**

Repository for the JRSS-B submission: a unified Bayesian framework that jointly performs **state tracking**, **online probability calibration**, and **cost-sensitive decision-making** with bidirectional feedback between modules.

## Key Innovation

Traditional "estimate -> calibrate -> decide" pipelines suffer from cascading error accumulation under non-stationarity. Our closed-loop architecture introduces three feedback pathways:

- **F1**: State uncertainty -> calibration window/regularization
- **F2**: Calibration residual -> UKF process noise  
- **F3**: State drift -> dynamic decision thresholds

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Quick smoke test
python main.py --quick

# Run comparison with 11 baselines
python run_comparison.py
```

## Project Structure

```
.
|-- src/
|   |-- core/          # Framework orchestrator & base classes
|   |   +-- closed_loop.py    # ClosedLoopFramework main class
|   |-- modules/       # Core modules (UKF, Platt, Decision)
|   |-- feedback/      # F1, F2, F3 feedback pathways
|   |-- baselines/     # 11 baseline comparison methods
|   |-- data/          # Data generators & loaders
|   |-- evaluation/    # Metrics & visualization
|   +-- experiments/   # Experiment pipeline
|-- paper/             # LaTeX source, figures, compiled PDFs
|-- results/           # Experiment outputs
|-- data/              # Datasets
|-- run_comparison.py      # Main comparison experiment
|-- run_fraud.py           # IEEE-CIS fraud detection experiment
|-- generate_figures.py    # Paper-quality figures
+-- main.py                # Entry point
```

## Key Results

| Dataset | Metric | Improvement |
|---------|--------|-------------|
| Synthetic (rapid drift) | Avg decision cost | 8.7% over best cost-sensitive baseline |
| IEEE-CIS fraud (590K txns) | Avg cost | 6.3% over raw threshold |
| IEEE-CIS fraud (590K txns) | ECE | 0.0007 (two orders of magnitude improvement) |
| Industrial (simulated, 50K) | Avg cost | 10.8% over online Platt |

All three feedback pathways contribute positively, with F3 (dynamic thresholds) accounting for 37.8% of the total cost reduction.

## Reproducibility

All experiments use fixed random seeds (42, 43, 44) and can be reproduced with:

```bash
python run_comparison.py     # ~30 seconds
python run_fraud.py          # ~5 minutes
python generate_figures.py   # Paper figures
```

Results are saved as JSON in `results/`.

## Requirements

- Python 3.10+
- NumPy, SciPy, scikit-learn, matplotlib
- (Optional) PyTorch for fraud data logit extraction

## Citation

```bibtex
@article{adaptive-bayesian-closed-loop,
  title={An Adaptive Bayesian Framework for Sequential Decision-Making under Non-Stationary Environments},
  journal={JRSS-B (under review)},
  year={2026}
}
```

## License

MIT
