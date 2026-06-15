# Cover Letter

**Date:** June 15, 2026

**To:** The Editor, *Expert Systems with Applications*

**Subject:** Submission of "A Closed-Loop Adaptive Decision Framework for Cost-Sensitive Sequential Decision-Making in Non-Stationary Environments"

Dear Editor,

I am pleased to submit our manuscript entitled "A Closed-Loop Adaptive Decision Framework for Cost-Sensitive Sequential Decision-Making in Non-Stationary Environments" for consideration in *Expert Systems with Applications*.

**Problem and Motivation**

Real-world decision support systems in fraud detection, medical diagnosis, and industrial monitoring face a common challenge: they must make cost-sensitive decisions under non-stationary conditions where both the data distribution and cost structures evolve over time. Standard pipelines process estimation, calibration, and decision-making as independent stages, creating cascaded error propagation with no feedback mechanism. This limitation degrades decision quality precisely when adaptation is most needed.

**What We Propose**

We present a closed-loop adaptive decision framework that couples these three stages through three uncertainty-driven feedback pathways:

- **F1**: State uncertainty modulates the calibration window size and regularization strength
- **F2**: Calibration residuals adaptively inflate the state-space process noise to accelerate tracking during regime changes
- **F3**: The posterior state estimate shifts the cost-sensitive decision threshold to anticipate drift

The framework is modular, computationally efficient (<1 ms per decision), and fully open-source.

**Key Contributions**

1. A practical closed-loop architecture with principled feedback mechanisms that can be deployed with any probabilistic classifier
2. Theoretical analysis of system boundedness and convergence under stationarity
3. Comprehensive empirical validation across four settings (synthetic, fraud detection, medical diagnosis, industrial fault prediction) with consistent cost reductions of 8.7-18.3% over cost-sensitive baselines
4. Open-source implementation with reproducible experiments

**Why ESWA**

This work directly addresses the development of expert and intelligent decision support systems under uncertainty. The framework's modular design, computational efficiency, and demonstrated effectiveness across multiple real-world domains align well with the journal's scope. All code and data are publicly available.

I confirm that this work is original, has not been published elsewhere, and is not under consideration by another journal. All authors have approved the submission.

Thank you for your consideration.

Sincerely,

Lin Yuanbin
School of Economics, Xiamen University
15220242202178@stu.xmu.edu.cn
