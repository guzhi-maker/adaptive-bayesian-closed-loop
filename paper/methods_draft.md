# Methods Section — Adaptive Bayesian Closed-Loop Framework
## Draft for JRSS-B Special Issue Submission

---

## 1. Problem Formulation

### 1.1 Notation

Let $\{(x_t, y_t, c_t^{\text{FN}}, c_t^{\text{FP}})\}_{t=1}^T$ denote a sequential decision-making problem under non-stationarity, where:

- $x_t \in \mathbb{R}^d$: feature vector at time $t$
- $y_t \in \{0, 1\}$: binary outcome (1 = adverse event, e.g., fraud)
- $f(x_t) \in \mathbb{R}$: raw classifier output (logit) from a pre-trained model
- $c_t^{\text{FN}} > 0$: cost of false negative (missed detection) at time $t$
- $c_t^{\text{FP}} > 0$: cost of false positive (false alarm) at time $t$

The data-generating process is non-stationary in two ways:

1. **Distribution drift**: $P_t(y=1|x)$ evolves over time due to changes in the underlying process
2. **Dynamic costs**: $(c_t^{\text{FN}}, c_t^{\text{FP}})$ vary with time (e.g., transaction amounts in fraud detection)

### 1.2 Objective

We aim to learn a decision policy $\pi_t: \mathbb{R} \to \{0, 1\}$ that maps each logit $f(x_t)$ to a binary action $a_t \in \{0, 1\}$ (0 = accept, 1 = reject) to minimize the cumulative expected cost:

$$\mathcal{R}(\pi) = \sum_{t=1}^T \mathbb{E}\left[ c_t^{\text{FN}} \cdot \mathbb{1}\{a_t=0, y_t=1\} + c_t^{\text{FP}} \cdot \mathbb{1}\{a_t=1, y_t=0\} \right]$$

The optimal decision at time $t$ under known probability $p_t = P(y_t=1|x_t)$ is:

$$a_t^* = \mathbb{1}\left\{p_t > \frac{c_t^{\text{FP}}}{c_t^{\text{FN}} + c_t^{\text{FP}}}\right\}$$

In practice, $p_t$ is unknown and must be estimated from data. Traditional pipelines decompose this into three sequential stages:
1. **Estimation**: track latent state $z_t$ governing the non-stationarity
2. **Calibration**: map raw logits to calibrated probabilities $\hat{p}_t$
3. **Decision**: apply threshold rule with $\hat{p}_t$

### 1.3 The Cascaded Error Problem

Standard pipelines suffer from **unidirectional error propagation**:

$$\hat{z}_t \xrightarrow{\text{estimation}} \hat{p}_t \xrightarrow{\text{calibration}} a_t \xrightarrow{\text{decision}} \text{cost}$$

Errors at each stage propagate forward without correction: a biased state estimate $\hat{z}_t$ degrades calibration, which in turn produces suboptimal decisions. There is no mechanism for the decision cost to inform the calibration or estimation stage.

---

## 2. Closed-Loop Architecture

We propose a **closed-loop framework** with three feedback pathways that create bidirectional information flow between stages:

### 2.1 Core Modules

**Tracker** (State Estimation): Maintains a latent state $z_t \in \mathbb{R}^m$ capturing the non-stationary environment dynamics. Implemented via Unscented Kalman Filter (UKF) with state transition:

$$z_{t+1} = z_t + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, Q_t)$$

and observation model:

$$o_t = h(z_t) + \nu_t, \quad \nu_t \sim \mathcal{N}(0, R_t)$$

where $o_t$ is a summary statistic of the current data stream (e.g., smoothed logit mean).

**Calibrator**: Maps raw logits to calibrated probabilities. We use Platt scaling with parameters $(a_t, b_t)$:

$$\hat{p}_t = \sigma(a_t \cdot f(x_t) + b_t) = \frac{1}{1 + \exp(-a_t f(x_t) - b_t)}$$

The calibrator maintains a sliding window $\mathcal{W}_t$ of recent (logit, label) pairs and periodically refits $(a_t, b_t)$ via logistic regression.

**Decision Maker**: Implements a three-layer decision rule with confidence intervals:

$$a_t = \begin{cases}
0 \text{ (accept)} & \text{if } \hat{p}_t^{\text{CI+}} < \tau_t^{\text{low}} \\
1 \text{ (reject)} & \text{if } \hat{p}_t^{\text{CI-}} > \tau_t^{\text{high}} \\
\text{cost-based} & \text{otherwise}
\end{cases}$$

where $[\hat{p}_t^{\text{CI-}}, \hat{p}_t^{\text{CI+}}]$ is a $(1-\alpha)$ confidence interval for $\hat{p}_t$, estimated via bootstrap, and $(\tau_t^{\text{low}}, \tau_t^{\text{high}})$ are adaptive thresholds.

### 2.2 Feedback Pathways

**F1: State → Calibration**. The tracker's uncertainty modulates the calibrator's window size $W_t$ and regularization $\lambda_t$:

$$W_t = W_{\max} - (W_{\max} - W_{\min}) \cdot \phi(\text{tr}(P_t))$$
$$\lambda_t = \lambda_{\max} - (\lambda_{\max} - \lambda_{\min}) \cdot \phi(\text{tr}(P_t))$$

where $P_t$ is the UKF's posterior covariance and $\phi(u) = \min(1, \gamma_1 \cdot u)$ is a squashing function. Higher uncertainty → smaller window (faster adaptation) + weaker regularization (more responsive).

**F2: Calibration Residual → UKF**. The calibration residual $r_t = \text{MAE}(\hat{p}_t, y_t)$ over a recent window modulates the UKF's process noise:

$$Q_t = Q_{\text{base}} \cdot (1 + \gamma_2 \cdot \log(1 + r_t / r_{\text{target}}))$$

Higher calibration error → larger process noise → faster UKF adaptation.

**F3: State → Decision Threshold**. The tracker's state estimate shifts the decision thresholds:

$$\tau_t^{\text{low}} = \max(0, \mu_0 - \gamma_3 \cdot z_t - \delta(P_t))$$
$$\tau_t^{\text{high}} = \min(1, \mu_0 - \gamma_3 \cdot z_t + \delta(P_t))$$

where $\mu_0 = 0.5$ is the neutral center, $\gamma_3$ controls drift sensitivity, and $\delta(P_t) = \delta_{\min} + (\delta_{\max} - \delta_{\min}) \cdot \phi(\text{tr}(P_t))$ is an uncertainty-dependent margin.

### 2.3 The Closed Loop

At each time step $t$, the closed loop executes:

1. **Predict**: UKF propagates state forward: $z_{t|t-1} = z_{t-1}$
2. **Observe**: Compute observation $o_t$ from current data
3. **Update**: UKF incorporates observation: $(z_t, P_t) = \text{UKFUpdate}(z_{t|t-1}, P_{t|t-1}, o_t)$
4. **F1**: Adjust calibration window $W_t$ and regularization $\lambda_t$ from $P_t$
5. **Calibrate**: Compute $\hat{p}_t$ with current calibrator; periodically refit on $\mathcal{W}_t$
6. **F3**: Adjust thresholds $(\tau_t^{\text{low}}, \tau_t^{\text{high}})$ from $z_t$ and $P_t$
7. **Decide**: Apply three-layer rule to produce $a_t$ and incur cost
8. **F2**: Compute calibration residual $r_t$; adjust UKF process noise $Q_{t+1}$

The feedback loops create a dynamical system where each module's output influences the others' future behavior, breaking the cascaded error propagation.

---

## 3. Stability Analysis

### 3.1 Boundedness Under Parameter Clipping

Let all adaptive parameters be clipped to compact intervals:

$$W_t \in [W_{\min}, W_{\max}], \quad \lambda_t \in [\lambda_{\min}, \lambda_{\max}], \quad Q_t \in [Q_{\min}, Q_{\max}]$$

**Proposition 1** (State Boundedness). Under bounded observations $|o_t| \leq M_o$ and clipped process noise $Q_t \in [Q_{\min}, Q_{\max}]$, the UKF state estimate $z_t$ remains bounded:

$$|z_t| \leq \max\{|z_0|, M_o\} + C \cdot \frac{Q_{\max}}{Q_{\min}}$$

for some constant $C > 0$ depending on the UKF parameters.

*Proof sketch.* The UKF update is a linear combination of the predicted state and the innovation, weighted by the Kalman gain $K_t$. The Kalman gain satisfies $\|K_t\| \leq 1$ under standard conditions. Bounded observations imply bounded innovations, and clipped process noise prevents the state covariance from vanishing or exploding. The result follows by induction. ∎

**Proposition 2** (Calibration Stability). Under bounded logits $|f(x_t)| \leq M_f$ and sufficiently many calibration samples ($|\mathcal{W}_t| \geq N_{\min}$), the Platt parameters $(a_t, b_t)$ satisfy:

$$|a_t| \leq A_{\max}, \quad |b_t| \leq B_{\max}$$

for constants $A_{\max}, B_{\max}$ depending only on $M_f$ and $N_{\min}$.

*Proof sketch.* Logistic regression with bounded features and non-degenerate labels has finite MLE estimates. The sliding window ensures at least $N_{\min}$ samples per fit. ∎

### 3.2 Local Input-to-State Stability

Consider the closed-loop system as a discrete-time dynamical system with state $s_t = (z_t, a_t, b_t, Q_t, \tau_t^{\text{low}}, \tau_t^{\text{high}})$ and input $u_t = (o_t, y_t, c_t^{\text{FN}}, c_t^{\text{FP}})$.

**Proposition 3** (Local ISS). There exists a neighborhood $\mathcal{N}$ of the nominal (stationary) operating point such that, for all initial states $s_0 \in \mathcal{N}$ and bounded inputs $\|u_t\|_\infty \leq M_u$, the closed-loop system is input-to-state stable:

$$\|s_t\| \leq \beta(\|s_0\|, t) + \gamma(\|u\|_\infty)$$

where $\beta$ is a $\mathcal{KL}$ function and $\gamma$ is a $\mathcal{K}$ function.

*Proof sketch.* Each module (UKF, Platt calibrator, decision maker) is individually ISS when considered in isolation due to boundedness of parameters and contractive properties of the Kalman filter. The feedback interconnections are static gains (no integration), so the interconnected system inherits ISS by the small-gain theorem. The proof uses the fact that F1, F2, and F3 are bounded, memoryless functions of $P_t$ and $z_t$, which are themselves ISS. ∎

### 3.3 Convergence in Stationary Regimes

**Proposition 4** (Convergence). If the data-generating process is stationary ($P(y_t=1|x_t) = p^*$ constant, $c_t^{\text{FN}} = c^{\text{FN}}$, $c_t^{\text{FP}} = c^{\text{FP}}$), the closed-loop system converges to the optimal decision rule:

$$\lim_{t \to \infty} a_t = a_t^* \quad \text{almost surely}$$

*Proof sketch.* Under stationarity, the UKF state converges to a constant (the observation mean). The calibrator's sliding window fills with i.i.d. samples, so Platt scaling converges to the true log-odds ratio. The decision threshold converges to the optimal $c^{\text{FP}}/(c^{\text{FN}} + c^{\text{FP}})$. The three-layer rule then implements the Bayes-optimal decision. ∎

---

## 4. Algorithm Pseudocode

```
Algorithm 1: Closed-Loop Adaptive Bayesian Framework

Input: Stream {(f(x_t), c_t^FN, c_t^FP)}_{t=1}^T
       Warmup period T_w
       Feedback flags F1, F2, F3

Initialize:
  UKF: z_0 = 0, P_0 = 0.1 I, Q = Q_base
  Calibrator: a = 1, b = 0, window W = W_init
  Decision: tau_low = 0.4, tau_high = 0.6

for t = 1 ... T do
  // 1. State tracking
  o_t = Smooth(logit_t)                     // Compute observation
  z_t, P_t = UKF_step(z_{t-1}, P_{t-1}, o_t, Q_t)
  
  // 2. F1: Calibration update
  if F1 and t > T_w then
    W_t = f1_window(P_t)
    lambda_t = f1_reg(P_t)
  end if
  
  // 3. Calibrate
  if t mod calibrate_every == 0 and |W| >= 100 then
    (a, b) = Platt_fit(W, lambda_t)
  end if
  p_t = sigmoid(a * f(x_t) + b)
  
  // 4. F3: Threshold update
  if F3 and t > T_w then
    tau_low, tau_high = f3_thresholds(z_t, P_t)
  end if
  
  // 5. Decision
  CI_low, CI_high = bootstrap_CI(p_t, calibration_data)
  a_t = three_layer_decision(p_t, CI_low, CI_high, 
                              tau_low, tau_high,
                              c_t^FN, c_t^FP)
  
  // 6. F2: Process noise update
  if F2 and t > T_w then
    r_t = calibration_residual(W)
    Q_{t+1} = f2_process_noise(r_t)
  end if
  
  // 7. Update calibration buffer
  W.append((f(x_t), y_t))
  if |W| > W_t then W.pop_oldest() end if
end for
```

---

## 5. Relationship to Existing Work

[To be expanded with specific citations]

1. **Adaptive Bayesian state estimation** (JRSS-B 2023, 2024): Existing methods track non-stationarity via dynamic models (e.g., AR-HMM, dynamic GLM) but stop at estimation; they do not close the loop to calibration and decision.

2. **Online probability calibration** (NeurIPS 2022, ICML 2023): Methods like online Platt and adaptive temperature scaling track calibration parameters over time but are decoupled from state estimation and decision costs.

3. **Cost-sensitive decision theory** (JASA 2021, JRSS-B 2022): Optimal decision rules under known costs are well-understood, but existing work assumes stationary environments or known probability estimates.

4. **Closed-loop learning** (control theory + ML): The idea of using feedback to stabilize learning appears in adaptive control and bandit literature, but to our knowledge, this is the first work to close the estimation-calibration-decision loop for non-stationary Bayesian decision problems.
