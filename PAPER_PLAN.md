# Paper Plan: EcoFed — Lifecycle-Energy-Aware Federated Learning with Coverage-Constrained Depth, Precision and Client Control

**Target venue:** Sustainable Computing: Informatics and Systems (SUSCOM), VSI "Energy-aware Artificial Intelligence". Submission closes 15 Dec 2026.
**Target length:** ~14,000 words (main text, excluding references and appendix), 10 figures, 9 tables.
**Status of this document:** a blueprint. Nothing below is an experimental result. All "expected findings" are hypotheses with explicit falsification criteria. All theorems are *proposed* and need proofs written and checked before submission. All citations must be verified against the original sources before use.

---

## 0. Why this paper, and what makes it Q1-grade

### 0.1 Working title options
1. *EcoFed: Coverage-Constrained, Lifecycle-Energy-Aware Federated Learning on Heterogeneous Edge Devices*
2. *Spend Joules Where They Count: Joint Client, Depth and Precision Control for Energy-Bounded Federated Learning*
3. *From Training Joules to Inference Joules: A Lifecycle Energy View of Federated Multi-Exit Learning at the Edge*

Recommended: option 1 as the title, option 3 as the framing of the Introduction.

### 0.2 The research question
> Given a fleet of battery-limited, heterogeneous edge devices, how should a federated learning system choose *which clients train, how deep, at what precision, and for how many epochs*, so that total lifecycle energy (training + communication + later inference) is minimized subject to a target accuracy, a long-term per-device energy budget, and a guarantee that every part of the model is still trained?

### 0.3 Gap in existing work (to be substantiated in the literature review)
| Gap | What most papers do | What this paper does |
|---|---|---|
| G1. Energy models | Arbitrary constants, or FLOPs only | Energy model calibrated on real devices (Raspberry Pi, Jetson, optionally an MCU) with reported fit error |
| G2. Control knobs | Client selection *or* compression *or* depth, one at a time | Joint control of selection, depth, precision, local epochs |
| G3. Theory | Convergence proofs ignore partial-depth training bias | Convergence bound with an explicit *coverage floor*, which then drives the algorithm |
| G4. Horizon | Training energy only | Lifecycle view: training energy vs. downstream inference energy, with a break-even analysis |
| G5. Long-term budgets | Per-round budgets, or none | Lyapunov virtual queues for time-average energy constraints |
| G6. Evaluation | Accuracy-vs-rounds | Accuracy-vs-Joules, fleet lifetime, fairness, carbon |

### 0.4 Claimed contributions (keep to four, make each testable)
- **C1 (Model):** A calibrated, device-aware energy model for federated multi-exit training, with measured parameters on at least two hardware classes.
- **C2 (Theory):** A non-convex convergence bound for federated training with heterogeneous depth, quantization and partial participation, containing an explicit coverage-bias floor. A corollary gives the minimum participation coverage needed to reach a target gradient norm.
- **C3 (Algorithm):** EcoFed, a drift-plus-penalty controller with a coverage queue. It has O(N·|C|) per-round cost and a stated optimality-gap/backlog trade-off controlled by one parameter V.
- **C4 (Systems evidence):** A lifecycle energy analysis showing the break-even number of inferences at which extra training energy pays for itself, across four datasets, with a released measurement harness.

### 0.5 Honest risks a Q1 reviewer will raise
1. "The theorem is a routine combination of known bounds." Mitigation: the coverage-floor term and the way it produces a constraint in the algorithm must be shown to be non-trivial and *empirically tight enough to matter* (Fig. 6).
2. "Simulated energy is not real energy." Mitigation: calibration on hardware, fit error reported, sensitivity analysis to ±30% energy-parameter error.
3. "Datasets are small." Mitigation: include FEMNIST and Speech Commands as larger, naturally partitioned benchmarks; use CIFAR-10 with Dirichlet partitions only as a stress test, not the headline.
4. "Compared to weak baselines." Mitigation: include depth-heterogeneous FL methods and an Oort-style selector, tuned with the same budget as EcoFed.
5. "Gains vanish with tuning." Mitigation: report hyperparameter search budget equal for all methods.

---

## 1. Datasets and benchmarks

| ID | Dataset | Role | Partitioning (natural non-IID) | Model family | Notes |
|---|---|---|---|---|---|
| D1 | PAMAP2 | Headline, wearable | By subject (9 clients; split further by session to get ~50–100 virtual clients) | 1D multi-exit CNN | Check license/citation; leave-subject-out test set |
| D2 | UCI HAR | Wearable replication | By subject (30 clients) | 1D multi-exit CNN | Small, so use for ablations |
| D3 | WISDM | Optional wearable | By subject (~51) | 1D multi-exit CNN | Drop if time is short |
| D4 | FEMNIST (LEAF) | Scale and standard FL baseline | By writer (~3,500 clients) | 2D multi-exit CNN | Lets reviewers compare with prior FL papers |
| D5 | Speech Commands v2 | Always-on audio | By speaker where metadata allows, else Dirichlet(α) | DS-CNN multi-exit | Edge-relevant; ties to MLPerf Tiny |
| D6 | CIFAR-10 | Stress test only | Dirichlet(α ∈ {0.1, 0.5, 10}) | ResNet-8 multi-exit | Not the headline |

**Must-do protocol rules**
- Subject-wise / speaker-wise train–test separation. Window-level random splits leak and are the most common rejection reason in HAR.
- Five seeds minimum for all headline results; report mean ± 95% CI; paired Wilcoxon tests across seeds/datasets with Holm correction.
- Equal tuning budget for every method (e.g., 30 random-search trials each).
- Fix target accuracies per dataset *before* running comparisons (e.g., 90% of centralized multi-exit accuracy).

---

## 2. Mathematical framework (the core of the paper)

### 2.1 Notation
| Symbol | Meaning |
|---|---|
| $N$ | number of clients; $\mathcal{N}=\{1,\dots,N\}$ |
| $T$ | number of communication rounds |
| $L$ | number of blocks / exits in the multi-exit model |
| $\mathbf{w}=(\mathbf{w}_1,\dots,\mathbf{w}_L)$ | model parameters partitioned into $L$ blocks (each block includes its exit head) |
| $\mathcal{C}$ | discrete configuration set; $c=(d,b,\tau)$: depth $d\in\{1..L\}$, bit-width $b\in\{8,16,32\}$, local epochs $\tau\in\{1,2,\dots\}$ |
| $\mathcal{S}_t\subseteq\mathcal{N}$ | clients selected in round $t$, $|\mathcal{S}_t|\le m$ |
| $c_k^t$ | configuration chosen for client $k$ in round $t$ |
| $n_k$ | local sample count; $p_k=n_k/\sum_j n_j$ |
| $E_k^t$ | energy used by client $k$ in round $t$ (J) |
| $\bar E_k$ | long-term average energy budget of client $k$ (J/round) |
| $B_k^t$ | battery state of client $k$ (J) |
| $Q_k^t$ | energy virtual queue; $Z_l^t$ coverage virtual queue for block $l$ |
| $V$ | Lyapunov trade-off parameter |

### 2.2 Multi-exit model and objective
The model has $L$ blocks; exit $j$ uses blocks $1..j$ plus head $h_j$. Local loss at exit $j$ is $f_k^{j}(\mathbf{w})=\mathbb{E}_{(x,y)\sim\mathcal{D}_k}[\ell(h_j(\phi_{1:j}(x;\mathbf{w})),y)]$. The global objective is

$$
F(\mathbf{w})=\sum_{k=1}^{N}p_k\sum_{j=1}^{L}\lambda_j f_k^{j}(\mathbf{w}),\qquad \lambda_j\ge0,\ \sum_j\lambda_j=1.
$$

A client with depth $d$ computes forward/backward through blocks $1..d$ only and optimizes the truncated objective $F_k^{(d)}(\mathbf{w})=\sum_{j\le d}\lambda_j f_k^{j}(\mathbf{w})$. Consequently block $l$ receives gradient contributions from exits $j\ge l$ **only from clients with $d\ge l$**. This is the source of the coverage bias analysed in §2.6.

### 2.3 Energy model
For a client running configuration $c=(d,b,\tau)$ on local data of size $n_k$:

$$
E_k^{\text{comp}}(c)=\tau\,n_k\Big[\,\underbrace{\alpha_k(b)\,\mathrm{MAC}_{1:d}}_{\text{arithmetic}}+\underbrace{\beta_k\,\mathrm{Mem}_{1:d}(b)}_{\text{data movement}}\Big]+E_k^{\text{static}}\,t_k^{\text{comp}}(c),
$$

$$
E_k^{\text{comm}}(c)=P_k^{\text{tx}}\frac{S_{1:d}(b)}{R_k^t}+P_k^{\text{rx}}\frac{S_{\text{down}}}{R_k^t},\qquad E_k^t=E_k^{\text{comp}}(c_k^t)+E_k^{\text{comm}}(c_k^t).
$$

- $\mathrm{MAC}_{1:d}$ = multiply-accumulates for forward+backward through the first $d$ blocks; $\mathrm{Mem}_{1:d}(b)$ = bytes moved; $S_{1:d}(b)$ = upload size in bits (blocks $1..d$ at $b$ bits); $R_k^t$ = achievable rate in round $t$.
- The $\beta_k\,\mathrm{Mem}$ term is deliberate: it encodes the memory-bound behaviour highlighted in the call text. Calibration should *test* whether $\beta_k$ is significant on each device rather than assume it.
- **Calibration procedure:** run each configuration on each device for ≥30 trials, measure power with a USB power meter (Pi) / `tegrastats` or an external meter (Jetson), integrate over time, subtract idle power, fit $(\alpha_k,\beta_k,E^{\text{static}}_k)$ by non-negative least squares. Report $R^2$, MAPE and leave-one-configuration-out error (Table 3).

### 2.4 Battery dynamics and long-term constraints
$$
B_k^{t+1}=\min\{B_k^{\max},\,B_k^{t}-E_k^{t}\mathbb{1}[k\in\mathcal{S}_t]+H_k^{t}\},\qquad B_k^{t}\ge B_k^{\min}\ \ \forall t.
$$
$H_k^t\ge0$ is energy harvested/charged (zero in the strictest setting). The practical long-term constraint is

$$
\limsup_{T\to\infty}\frac1T\sum_{t=1}^{T}\mathbb{E}\big[E_k^t\mathbb{1}[k\in\mathcal{S}_t]\big]\le\bar E_k\quad\forall k.
$$

### 2.5 Lifecycle energy and break-even analysis (C4)
After training, the deployed multi-exit model runs inference with a confidence-threshold policy $\theta$ (exit at the first $j$ with max-softmax $\ge\theta_j$). Expected inference energy per sample:

$$
\bar e_{\text{inf}}(\theta)=\sum_{j=1}^{L}\pi_j(\theta)\,\Big(\sum_{i\le j}e_i^{\text{blk}}+e_j^{\text{head}}\Big),\qquad \pi_j(\theta)=\Pr[\text{sample exits at }j].
$$

Lifecycle energy for $M$ deployed inferences across the fleet: $E_{\text{life}}(M)=E_{\text{train}}+M\,\bar e_{\text{inf}}(\theta)$.

Comparing a method A (more training energy, lower inference energy) with a baseline B, the **break-even inference count** is

$$
M^{*}=\frac{E_{\text{train}}^{A}-E_{\text{train}}^{B}}{\bar e_{\text{inf}}^{B}-\bar e_{\text{inf}}^{A}}\qquad(\text{defined when the denominator}>0),
$$

and the paper reports $M^*$ with bootstrap CIs. This turns "is the extra training worth it?" into a number, and it is the figure most likely to be cited.

Optional extension for HAR datasets: add a sensing term $e_{\text{sense}}(r)=P_s\,r\,T_w$ with sampling rate $r\in\{10,25,50\}$ Hz and window length $T_w$, and let the deployment policy pick $(\theta,r)$ jointly.

### 2.6 Convergence analysis (C2) — proposed theorem

**Algorithm (analysed version):** each round, $m$ clients are sampled with probabilities $q_k^t$; client $k$ runs $\tau_k$ local SGD steps on $F_k^{(d_k)}$ with stochastic quantization of the update at $b_k$ bits; the server averages block $l$ over the participating clients with $d_k\ge l$ using weights proportional to $p_k/q_k$ (unbiased within the covered set).

**Assumptions**
- A1. Each $f_k^j$ is $L_s$-smooth.
- A2. Stochastic gradients are unbiased with variance $\le\sigma^2$.
- A3. Heterogeneity: $\sum_k p_k\|\nabla F_k^{(L)}(\mathbf{w})-\nabla F(\mathbf{w})\|^2\le\zeta^2$.
- A4. Bounded exit gradients: $\|\nabla_l f_k^j(\mathbf{w})\|\le G$.
- A5. Stochastic quantizer is unbiased with variance factor $\omega_b\propto 2^{-2b}$ (QSGD-type).
- A6. Coverage: block $l$ is trained with probability at least $\rho_l\in(0,1]$ in each round, i.e. $\Pr[\exists k\in\mathcal{S}_t:d_k\ge l]$ weighted so that the covered fraction of exit-$l$-and-deeper gradient mass is $\ge\rho_l$.

**Theorem 1 (proposed form).** With local step size $\eta=\Theta(1/(L_s\tau\sqrt{T}))$,

$$
\frac1T\sum_{t=0}^{T-1}\mathbb{E}\|\nabla F(\mathbf{w}^t)\|^2\ \le\ \underbrace{\frac{2\,(F(\mathbf{w}^0)-F^\star)}{\eta\,\bar\tau\,T}}_{\text{optimization}}\ +\ \underbrace{\eta L_s\bar\tau\Big(\tfrac{\sigma^2}{m}\big(1+\bar\omega\big)+\zeta^2\Big)}_{\text{noise, quantization, heterogeneity}}\ +\ \underbrace{\sum_{l=1}^{L}c_l\,(1-\rho_l)^2G^2}_{\text{coverage floor}}
$$

where $c_l$ depends on the exit weights $\{\lambda_j\}_{j\ge l}$, $\bar\tau$ is the average local step count and $\bar\omega$ the average quantization variance factor.

**Interpretation**
- The first two terms vanish at rate $O(1/\sqrt T)$.
- The third term is a **floor that does not vanish**: if deep blocks are rarely trained (because low-battery clients always run shallow), the model cannot converge to a stationary point of the full objective.
- **Corollary 1:** to reach $\epsilon$-stationarity one needs $\sum_l c_l(1-\rho_l)^2G^2\le\epsilon/2$, which gives a computable minimum coverage vector $\rho^{\min}(\epsilon)$. This is what Algorithm 1 enforces through the coverage queues $Z_l$.

**Proof outline (to be written in the appendix)**
1. Decompose the global update into a covered part and an uncovered part per block.
2. Bound the drift of local models over $\tau$ steps (standard client-drift lemma, with the truncated objective).
3. Bound the quantization contribution via variance additivity.
4. Bound the uncovered-gradient bias by $(1-\rho_l)G$ per block using A4 and A6.
5. Apply the descent lemma using $L_s$-smoothness; telescope; optimize $\eta$.

**Validation plan for the theory.** Run a synthetic experiment (controlled coverage $\rho$ swept from 0.2 to 1.0) and plot the final gradient norm against $(1-\rho)^2$. The bound is "useful" if the trend is monotone with a similar shape; it does not need to be numerically tight. If the plot shows no floor effect, drop the claim and reframe C2 as an upper bound only.

### 2.7 EcoFed control problem and Lyapunov solution (C3)

**Per-round decision.** Choose $\mathcal{S}_t$ and $\{c_k^t\}_{k\in\mathcal{S}_t}$.

**Utility of a (client, configuration) pair.** An estimate of progress per round:

$$
U_k(c)=\underbrace{|\mathcal{D}_k|\sqrt{\tfrac{1}{|\mathcal{D}_k|}\textstyle\sum_{i}\ell_i^2}}_{\text{statistical utility (Oort-style)}}\cdot\ \gamma_d\ \cdot\ (1-\kappa\,\omega_b),
$$

with $\gamma_d$ a depth-value factor (fraction of exit weight $\sum_{j\le d}\lambda_j$) and $\kappa$ a quantization penalty. Both factors are measured, not guessed (ablate them).

**Virtual queues**
$$
Q_k^{t+1}=\max\{Q_k^{t}+E_k^{t}\mathbb{1}[k\in\mathcal{S}_t]-\bar E_k,\ 0\},\qquad
Z_l^{t+1}=\max\{Z_l^{t}+\rho_l-\mathbb{1}[\text{block }l\text{ covered in round }t],\ 0\}.
$$

Stability of $Q_k$ implies the time-average energy constraint; stability of $Z_l$ implies coverage at least $\rho_l$ in the long run.

**Drift-plus-penalty objective.** Each round solve
$$
\max_{\mathcal{S},\{c_k\}}\ V\sum_{k\in\mathcal{S}}U_k(c_k)\ -\ \sum_{k\in\mathcal{S}}Q_k^{t}E_k(c_k)\ +\ \sum_{l=1}^{L}Z_l^{t}\,\mathbb{1}\big[\exists k\in\mathcal{S}:d_k\ge l\big]
$$
subject to $|\mathcal{S}|\le m$, round deadline $t_k(c_k)\le T_{\max}$ for all $k\in\mathcal{S}$, and $B_k^t-E_k(c_k)\ge B_k^{\min}$.

**Algorithm 1 (EcoFed), per round**
1. Server broadcasts the current model; clients report $(B_k,R_k,\text{loss statistic})$.
2. For each client $k$, compute the best feasible configuration $c_k^\star=\arg\max_{c\in\mathcal{C}_k}\{V\,U_k(c)-Q_kE_k(c)\}$ (O(|C|) each).
3. Rank clients by their resulting score; greedily pick $m$ clients, then repair coverage: if some block $l$ with $Z_l>0$ is uncovered, swap in the cheapest-per-coverage client able to train depth $\ge l$ (greedy submodular-style repair; state the approximation claim only if proven).
4. Selected clients train, quantize, upload; server aggregates block-wise (§2.6).
5. Update $B_k$, $Q_k$, $Z_l$.

**Proposition 1 (proposed, from standard Lyapunov optimization).** Under a Slater-type feasibility condition, EcoFed's time-average utility is within $O(1/V)$ of the optimum of the relaxed problem, while the average virtual-queue backlog (and hence the transient budget violation) is $O(V)$. The paper must state precisely which relaxation (e.g., ignoring the discrete repair step) this holds for. If the repair step breaks the guarantee, say so and present the claim for the relaxed controller only.

**Complexity:** O(N·|C|) per round for scoring plus O(N log N) for ranking; negligible next to local training. Report measured server-side controller time.

---

## 3. Experimental design

### 3.1 Hardware profiles (Table 3)
| Profile | Hardware | Measurement | Used for |
|---|---|---|---|
| P1 | Raspberry Pi 4 or 5 | USB power meter / INA219 at ≥10 Hz | Weak client |
| P2 | Jetson Nano or Orin Nano | `tegrastats` plus external meter if available | Strong client |
| P3 | MCU-class (optional) | Nordic PPK2 / Joulescope | Inference only |
| P4 | Server GPU | NVML via CodeCarbon or Zeus | Aggregation cost, centralized baseline |

Simulation uses the fitted models to emulate fleets of 100–3,500 clients with a mix such as 50% P1-like, 35% P2-like, 15% mid-tier (interpolated). Link rates drawn from a Markov channel model; report the assumed per-bit radio energy and run a sensitivity sweep.

### 3.2 Baselines (Table 4)
| Group | Method | Reason |
|---|---|---|
| Standard FL | FedAvg, FedProx, SCAFFOLD | Reference |
| Selection | Random, Power-of-Choice, Oort-style | Selection-only energy blindness |
| Depth/width-heterogeneous | HeteroFL, DepthFL, ScaleFL | Closest to partial-depth training |
| Compression | Quantized updates (QSGD-type), top-k sparsification | Communication-only |
| Energy-aware | Energy-greedy selection; a recent energy/carbon-aware FL method from the literature review | Direct competitors (identify and verify during review) |
| Ablations of EcoFed | No coverage queue; no depth control; no precision control; no energy queue; V swept | Attribution |

### 3.3 Metrics
| Metric | Definition |
|---|---|
| Accuracy / macro-F1 | Leave-subject-out test set |
| **Energy-to-Accuracy (ETA)** | Total fleet Joules until target accuracy is first reached |
| Accuracy@Budget | Test accuracy when cumulative fleet energy hits a fixed budget |
| Fleet lifetime | Rounds until the first client, and until 10% of clients, hit $B^{\min}$ |
| Energy fairness | Jain's index over per-client cumulative energy; also normalised by battery capacity |
| Accuracy fairness | Std/worst-decile of per-client accuracy |
| Coverage | Realised $\hat\rho_l$ per block |
| Inference energy | $\bar e_{\text{inf}}$ at matched accuracy; energy per correct prediction |
| Lifecycle break-even | $M^*$ with bootstrap CI |
| Carbon (optional) | gCO₂e using stated grid intensity; state as an assumption |

### 3.4 Experiments (each with a stated purpose)
| Exp | Question | Output |
|---|---|---|
| E1 | Does the energy model fit real devices? | Table 3, Fig. 3 |
| E2 | Main comparison on D1, D2, D4, D5 | Table 5, Fig. 4–5 |
| E3 | Does the coverage floor exist and matter? | Fig. 6 (synthetic + real) |
| E4 | Ablations | Table 6 |
| E5 | Sensitivity: non-IID level, participation $m$, V, energy-parameter error ±30%, channel quality | Fig. 7, Table 7 |
| E6 | Fleet lifetime and fairness | Fig. 8, Table 8 |
| E7 | Lifecycle break-even with cascaded inference (and sampling-rate control on D1/D2) | Fig. 9, Table 9 |
| E8 | Scalability to 3,500 clients (FEMNIST); controller overhead | Fig. 10 |
| E9 | Negative result search: settings where EcoFed does *not* win | Discussion §7 |

### 3.5 Hypotheses with falsification criteria (not results)
| ID | Hypothesis | Falsified if |
|---|---|---|
| H1 | Depth and precision control reduce ETA vs. selection-only and compression-only methods at equal target accuracy | ETA reduction is not significant (Holm-corrected p ≥ 0.05) on a majority of datasets |
| H2 | Removing the coverage queue degrades final accuracy or deep-exit accuracy, and the degradation grows as energy budgets tighten | Ablation shows no accuracy difference across budgets |
| H3 | The measured final gradient norm rises with $(1-\rho)^2$ in the controlled sweep | No monotone relationship |
| H4 | EcoFed extends fleet lifetime and improves energy fairness relative to Oort-style selection | Lifetime or Jain's index not improved |
| H5 | The break-even $M^*$ is finite and within a plausible deployment volume (e.g., ≤ $10^5$ inferences per device) for the wearable tasks | $M^*$ undefined or implausibly large |
| H6 | Memory-movement term $\beta$ is significant on at least one device class | Fit shows $\beta\approx0$ everywhere (then simplify the model and say so) |

If H1 fails, the paper becomes a measurement/negative-results paper and should be reframed; do not force it.

---

## 4. Figures (10) and tables (9)

### Figures
| # | Content | Type | Section |
|---|---|---|---|
| F1 | System overview: clients, server, controller, queues, lifecycle (train → deploy) | Architecture diagram | 3 |
| F2 | Multi-exit model with block-wise aggregation and which clients cover which blocks | Schematic | 3 |
| F3 | Energy-model fit: predicted vs. measured Joules per configuration, per device | Scatter with $y=x$ line | 4 |
| F4 | Accuracy vs. cumulative fleet energy (log-x), all methods, 4 datasets | Line plot, shaded CI | 6 |
| F5 | ETA bar chart with significance markers | Grouped bars | 6 |
| F6 | Coverage floor: gradient norm / error vs. $(1-\rho)^2$, synthetic and real | Scatter + fit | 6 |
| F7 | Sensitivity heatmaps: non-IID level × participation; V trade-off curve | Heatmap + Pareto | 7 |
| F8 | Fleet lifetime survival curves and per-client energy distribution | Survival / violin | 7 |
| F9 | Lifecycle energy vs. number of inferences, with break-even points | Line plot, annotated $M^*$ | 7 |
| F10 | Scalability: controller time and ETA vs. number of clients | Dual-axis line | 7 |

Follow the `dataviz` guidance when drawing them: colour-blind-safe palette, direct labels, consistent method colours across all figures, vector output (PDF/SVG), grayscale-legible markers.

### Tables
| # | Content |
|---|---|
| T1 | Related-work comparison matrix (energy model, knobs, theory, lifecycle, hardware validation) |
| T2 | Notation |
| T3 | Hardware profiles and fitted energy parameters with fit error |
| T4 | Baselines and hyperparameter search ranges |
| T5 | Main results: accuracy, ETA, lifetime, fairness (mean ± CI) per dataset |
| T6 | Ablation results |
| T7 | Sensitivity to energy-parameter error and channel quality |
| T8 | Fairness and lifetime statistics |
| T9 | Lifecycle break-even $M^*$ per dataset/hardware, with CIs |

---

## 5. Section-by-section outline with word budget (total ≈ 14,000)

| § | Section | Words | Key content |
|---|---|---|---|
| — | Abstract | 250 | Problem, method, 3 quantified findings (fill from results), artifact link |
| — | Highlights (3–5 bullets) | ~100 | Elsevier requirement |
| 1 | Introduction | 1,300 | Energy cost of AI; edge/FL energy blind spots; the lifecycle view; gap table G1–G6; contributions C1–C4; paper roadmap |
| 2 | Related work | 1,700 | (a) Energy-efficient edge AI and TinyML; (b) FL client selection and resource-aware FL; (c) Heterogeneous-depth/width FL; (d) Early-exit and adaptive inference; (e) Energy/carbon measurement and modelling; positioning (T1) |
| 3 | System model and problem formulation | 1,700 | §2.1–2.5 content: multi-exit objective, energy model, battery dynamics, lifecycle energy, formal problem (P1), why it is hard (mixed-integer, coupled over time) |
| 4 | Energy model calibration | 1,000 | Measurement setup, procedure, fit results, error analysis, limitations |
| 5 | Theory and algorithm | 3,000 | Theorem 1 with assumptions and interpretation; Corollary 1; proof sketch (full proof in Appendix A); Lyapunov formulation; Algorithm 1; Proposition 1; complexity |
| 6 | Experimental setup | 1,300 | Datasets/partitions, models, baselines, tuning protocol, metrics, statistics, reproducibility details |
| 7 | Results and analysis | 2,800 | E2–E8 in order; each subsection states hypothesis → result → interpretation; negative findings E9 |
| 8 | Discussion | 700 | Practical guidance (when to use depth vs. precision vs. selection), deployment, carbon implications, relation to SI themes |
| 9 | Limitations and threats to validity | 400 | Simulated fleets, energy-model transfer, dataset size, assumed radio energy |
| 10 | Conclusion and future work | 350 | Findings, 3 future directions (LLM-scale clients, hardware co-design, event-driven sensing) |
| — | Declarations | ~150 | CRediT, funding, competing interests, data and code availability |
| | **Total (main text)** | **≈14,750 incl. front matter; trim §2 and §7 to land near 14,000** | |

### Detailed notes per section

**1. Introduction (1,300).** Open with the sustainability stakes (training and deployment energy), then the specific blind spot: FL papers optimize accuracy-per-round while edge devices are limited in Joules, not rounds. Introduce the lifecycle view with the $M^*$ concept in two sentences. End with an explicit contribution list that maps one-to-one onto C1–C4, and a "what we do *not* claim" sentence (no claim of universal best; results are simulation-with-calibration).

**2. Related work (1,700).** Verify every citation. Candidate sources to look up and confirm (do not cite without reading): FedAvg (McMahan et al.), FedProx (Li et al.), SCAFFOLD (Karimireddy et al.), Oort (Lai et al.), HeteroFL (Diao et al.), DepthFL, ScaleFL, BranchyNet (Teerapittayanon et al.), shallow-deep networks (Kaya et al.), QSGD (Alistarh et al.), Lyapunov optimization (Neely), LEAF (Caldas et al.), MLPerf Tiny (Banbury et al.), Speech Commands (Warden), PAMAP2 (Reiss & Stricker), UCI HAR (Anguita et al.), Horowitz on operation energy, Strubell et al. and Patterson et al. on training carbon. Add 2024–2026 energy-aware FL papers found by a systematic search (record queries and dates for transparency).

**3. System model (1,700).** Present Fig. F1 and F2 here. Define the problem formally:

$$
\text{(P1)}\quad \min_{\{\mathcal{S}_t,c_k^t\}}\ \sum_{t=1}^{T}\sum_{k\in\mathcal{S}_t}E_k^t\ \ \text{s.t.}\ \ \text{Acc}(\mathbf{w}^T)\ge A^{\star},\ \ \text{(2.4)},\ \ \text{coverage}\ \hat\rho_l\ge\rho_l^{\min},
$$

then state the dual (budget-constrained) form used in experiments: maximize accuracy subject to a total energy budget.

**4. Calibration (1,000).** Transparent about measurement uncertainty: meter resolution, sampling rate, warm-up, thermal throttling, idle-power subtraction. Report fit quality honestly; if MAPE is large for a device class, widen the sensitivity sweep.

**5. Theory and algorithm (3,000).** Lead with intuition (the coverage floor in one paragraph and one small illustrative figure inset), then the formal statements. Put long proofs in Appendix A. Include a table of how each assumption is checked or approximated empirically (e.g., estimate $G$ and $\zeta$ from training logs).

**6. Setup (1,300).** All hyperparameters in a table. State exactly what is simulated and what is measured. Provide seeds, software versions, and the exact commit hash of the released code.

**7. Results (2,800).** Order: main comparison → coverage floor → ablations → sensitivity → fleet lifetime/fairness → lifecycle → scalability → where EcoFed does not win. For every claim, give effect size and CI, not only significance. Include a short "what this means for practitioners" box.

**8–10.** Keep claims proportionate; connect to the special-issue themes (runtime energy management, adaptive inference, federated learning under energy and communication constraints, always-on sensing and wearables).

---

## 6. Reproducibility and artifact plan

Repository layout to create in this repo:

```
/data_loaders        # PAMAP2, UCI HAR, WISDM, FEMNIST, Speech Commands, CIFAR-10 partitioners
/models              # multi-exit 1D-CNN, 2D-CNN, DS-CNN, ResNet-8
/energy              # measurement scripts (Pi/Jetson), fitting code, fitted-parameter JSON
/fl                  # FedAvg/FedProx/Oort/HeteroFL/DepthFL baselines, EcoFed controller
/theory              # synthetic coverage-floor experiment, constant-estimation scripts
/experiments         # YAML configs per experiment (E1–E9), seeds, launchers
/analysis            # stats (Wilcoxon + Holm), figure scripts, table generators
/paper               # LaTeX (Elsevier elsarticle), figures, bib
```

Release: code, fitted energy parameters, raw power traces, and config files (Zenodo DOI). State dataset licenses. Pin dependency versions. Provide a one-command script that regenerates every figure and table.

---

## 7. Timeline (today 1 Oct 2026 → submission before 15 Dec 2026, ≈10.5 weeks)

| Weeks | Dates (approx.) | Deliverable | Gate |
|---|---|---|---|
| 1 | 1–7 Oct | Literature search log; finalize baselines list; repo scaffold; data loaders for D1, D2, D4, D5 | Baselines list frozen |
| 2 | 8–14 Oct | Multi-exit models; centralized accuracy references; FedAvg/FedProx running | Centralized targets set |
| 3 | 15–21 Oct | Hardware measurements; energy-model fit (E1) | Fit error reported; H6 checked |
| 4 | 22–28 Oct | Fleet simulator; Oort, HeteroFL, DepthFL baselines | Baselines reproduce published trends |
| 5 | 29 Oct–4 Nov | EcoFed controller; coverage-floor synthetic experiment (E3) | **Go/no-go on theory claim** (H3) |
| 6–7 | 5–18 Nov | Main experiments E2, E4–E6 across seeds | H1/H2/H4 verdicts |
| 8 | 19–25 Nov | Lifecycle E7, scalability E8, negative search E9 | All results frozen |
| 9 | 26 Nov–2 Dec | Proofs finalized (Appendix A); figures and tables; full draft | Internal draft complete |
| 10 | 3–9 Dec | Co-author review; language edit; reference verification; cover letter; highlights | Submission package ready |
| 11 | 10–14 Dec | Buffer; submit via "VSI: SUSCOM_Energy-aware Artificial Intelligence" | Submitted by 14 Dec |

**Contingency:** if the Week 5 go/no-go fails (no floor effect), drop the "coverage queue" as a theoretical contribution and keep it as a heuristic, rebalancing the paper toward C1, C3 (energy-queue only) and C4.

---

## 8. Submission checklist (Elsevier / SUSCOM)

- Article type selected: "VSI: SUSCOM_Energy-aware Artificial Intelligence"
- Highlights (3–5 bullets, ≤85 characters each), graphical abstract, keywords (from the call: Energy-aware AI algorithms; Federated and decentralized AI; Adaptive inference; plus "multi-exit networks")
- Originality statement; no concurrent review; any earlier conference version disclosed with ≥30% new material
- CRediT author statement; data and code availability statement; declaration of competing interests
- Statement on use of generative AI tools in writing, as required by Elsevier
- All figures vector; every table self-contained; every symbol defined once (T2)
- Agreement to referee if requested (stated in the call)

---

## 9. Immediate next steps I can do in this repo

1. Create the repository scaffold above with a working data loader for UCI HAR and a multi-exit 1D-CNN.
2. Build the energy-measurement harness (power-meter logging, fitting code) so hardware time is not wasted later.
3. Implement the synthetic coverage-floor experiment first, since it decides whether the theory contribution survives.
4. Set up the LaTeX template (elsarticle) with the section skeleton and word-count tracker.
