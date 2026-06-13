# Cover Letter --JRSS-B Special Issue

# "Adaptive Bayesian Methods for Non-Stationary Data"



**Date:** June 2026

**Manuscript Title:** An Adaptive Bayesian Framework for Sequential Decision-Making under Non-Stationary Environments

**Authors:** Lin Yuanbin

**Journal:** Journal of the Royal Statistical Society, Series B (Statistical Methodology)

**Special Issue:** Adaptive Bayesian Methods for Non-Stationary Data



---



Dear Editor,



We are pleased to submit our manuscript entitled **"An Adaptive Bayesian Framework for Sequential Decision-Making under Non-Stationary Environments"** for consideration in the JRSS-B special issue on Adaptive Bayesian Methods for Non-Stationary Data.



**Summary of the work.** This paper addresses a fundamental limitation of existing pipeline architectures for sequential decision-making under non-stationarity. Standard approaches treat state estimation, probability calibration, and risk-sensitive decision-making as independent stages, leading to cascaded error accumulation when the environment drifts. We propose a unified Bayesian framework that couples these three tasks through posterior-dependent information flows--specifically, three feedback pathways that allow downstream discrepancies to correct upstream inference in real time.



**Methodological contributions.** Our framework is not an ad hoc engineering solution but a statistically principled coupling of three well-established components:

- A state-space tracker (UKF) whose diffusion is adaptively inflated by calibration residuals (F2);

- A Platt calibrator whose effective sample size is scaled by the tracker's posterior covariance (F1);

- A decision rule whose Bayes boundary is continuously adjusted by the tracker's posterior mean and variance (F3).



We prove that under bounded observations the coupled system satisfies local Lipschitz stability (small perturbations in the observation model lead to proportionally bounded deviations in the joint posterior-predictive distribution) and that under stationarity the adaptive decision rule converges to the optimal Bayes rule.



**Empirical validation.** Extensive experiments across synthetic data with gradual, abrupt, and periodic drift (n=10,000--200,000) and the IEEE-CIS credit card fraud detection dataset (n=590,540) demonstrate:

- 52.6% cost reduction over static calibration baselines and 8.7% over the best cost-sensitive baseline (p<0.0001, 20 random seeds);

- Two-orders-of-magnitude improvement in expected calibration error (ECE=0.0007 vs 0.082);

- All three feedback pathways contribute positively, with F3 (dynamic thresholds) accounting for 37.8% of the total improvement.



**Fit with the special issue.** This work directly addresses the special issue's focus on adaptive Bayesian methods for non-stationary data by providing a theoretically grounded, modular, and computationally efficient framework that jointly handles distribution drift and dynamic decision costs--a setting that existing methods cannot address without manual retuning.



**Prior publication and competing interests.** This manuscript has not been published or submitted elsewhere. The authors declare no competing interests.



**Suggested reviewers** (all experts in adaptive Bayesian methods or related areas):

1. Prof. David B. Dunson (Duke University) --Bayesian nonparametrics

2. Prof. Peter Buehlmann (ETH Zurich) --High-dimensional statistics, online learning

3. Prof. Yaniv Romano (Technion) --Probability calibration, uncertainty quantification

4. Prof. Ryan J. Tibshirani (CMU) --Nonparametric statistics, adaptive methods

5. Prof. Gareth O. Roberts (University of Warwick) --Bayesian computation, MCMC



We believe this work makes a substantial methodological contribution that will be of broad interest to the JRSS-B readership, and we hope you will consider it for publication.



Sincerely,



Lin Yuanbin

School of Economics, Xiamen University

15220242202178@stu.xmu.edu.cn

https://orcid.org/0009-0001-0193-4396

