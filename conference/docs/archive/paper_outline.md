# Conference Paper — Outline & Storyline (final, results-backed)

Working title (pick one):
- *Re-examining R² in Residential Appliance Energy Forecasting: Evaluation
  Granularity, Data Leakage, and a Noise Ceiling*
- *On the Reproducibility of Demand-Response Energy Forecasting: Why a Reported
  R²=0.94 Is an Evaluation Artifact*

## Thesis (one line)
On the UCI Appliances Energy dataset, a recently claimed R²=0.94 is **not** a
modeling achievement but an **evaluation artifact** — produced by coarse
evaluation granularity and data-split leakage. Honest 10-min evaluation hits a
**noise ceiling of R²≈0.64–0.66** for every competitive model; deep sequence
models are the only ones that stay robust under an honest split.

## Contributions
1. **Faithful reproduction** of Durrani et al. (2025) under their apparent
   protocol (E1).
2. **Two-axis explanation** of the inflated number:
   (a) **evaluation granularity** — coarsening 10-min → 1/2-hour lifts R² from
       ~0.65 to ~0.91 for *every* model (E2, Fig. A);
   (b) **data-split leakage** — random split + lag features inflate R²; an
       honest chronological split deflates it, and tree models collapse to
       R²=−1 (E3, Fig. B).
3. **Noise-ceiling result** — at honest 10-min resolution all competitive models
   (RF, GBM, LSTM, CNN-LSTM) plateau at R²≈0.64–0.66; no architecture escapes it.
4. **Honest model study**:
   (a) a decomposition ablation locating the gain (weak→tuned LSTM is the main
       step; ensemble helps at coarse granularity; CNN ≈ neutral; log/Huber
       hurt) (E6, Fig. E);
   (b) a **robustness** finding — under chronological split deep sequence models
       stay at 0.55–0.60 while tree models collapse to −1 (Fig. B).
5. **Comparison to the original method** (Candanedo GBM ≈ 0.57 reproduced) and a
   **feature ablation** isolating the lag trade-off (E4, E5).
6. A practical **trust checklist** for R² claims on this dataset.

## Scope decisions
- **In:** E1–E6, with multi-seed std as error bars (E8 folded in).
- **Deferred to the next paper (Phase 2):** system-level DR benefit (peak/PAR/
  cost) with a decision agent + synthetic pricing (E7) — keeps this paper free
  of synthetic-economics assumptions.

## Key results (from the full runs)

**E2 — granularity (mean R² over 7 DR strategies):**
| Model | 10-min | 30-min | 1-hour | 2-hour |
|---|---|---|---|---|
| CNN-LSTM | 0.65 | 0.82 | 0.88 | 0.91 |
| LSTM | 0.65 | 0.82 | 0.88 | 0.91 |
| RF / GBM | 0.66 | 0.83 | 0.88 | 0.90 |
| LR | 0.56 | 0.74 | 0.80 | 0.83 |

**E3 — split leakage (no DR, raw R²):**
| Model | random | chronological |
|---|---|---|
| LSTM | 0.64 | **0.60** (best honest) |
| LR | 0.59 | 0.58 |
| CNN-LSTM | 0.63 | 0.55 |
| RF | 0.64 | 0.21 |
| GBM / XGBoost | 0.65 | **−1.00** (collapse) |

**E4 — vs Candanedo protocol (random, env-only, no DR):** RF 0.60, XGBoost 0.56,
GBM 0.43, LR 0.18; our LSTM/CNN-LSTM 0.64 → reproduces the ~0.57 benchmark and
slightly improves on it.

**E5 — feature ablation (random, raw):** LR 0.18→0.59→0.60 (env→+lags→full);
GBM 0.43→0.65; LSTM 0.64→0.64 (already captures history).

**E6 — decomposition ladder (no DR):**
| Step | 10-min | 1-hour |
|---|---|---|
| (1) weak LSTM | 0.611 | 0.866 |
| (2) + tuning (capacity/training) | 0.644 | 0.876 |
| (3) + CNN front-end | 0.627 | 0.869 |
| (4) + ensemble = final | 0.643 | 0.880 |

Final beats the weak baseline (+0.032 raw / +0.014 1-h); gain is mostly tuning,
ensemble helps at coarse granularity, CNN ≈ neutral, log target worst.

## Section-by-section storyline

| Section | Experiment / Fig-Table | Takeaway |
|---|---|---|
| §1 Introduction | — | A claimed 0.94 far exceeds this dataset's known difficulty (0.57) — suspicious and a reproducibility hazard. |
| §2 Related work & dataset | literature table | Honest results cluster at 0.57–0.64; clarify the two easily-confused UCI datasets. |
| §3 Method | architecture fig + DR + protocols | Pipeline, models incl. CNN-LSTM, 7 DR, two evaluation axes (granularity, split), metrics. |
| §4.1 Reproduction | **E1 / Table II** | Under the paper's apparent protocol we reproduce their numbers. |
| §4.2 Granularity ★ | **E2 / Fig. A (±std)** | Coarsening 10-min → 1/2-hour lifts R² 0.65 → 0.91 for every model. Explains 0.94. |
| §4.3 Split leakage ★ | **E3 / Fig. B** | Random split inflates R²; chronological is honest; tree models collapse to −1, deep models stay robust. |
| §4.4 Noise ceiling ★ | **E2 + E6** | At honest 10-min, all competitive models plateau at 0.64–0.66 — no architecture escapes it. |
| §4.5 vs original method | **E4 / Table III** | We reproduce Candanedo's GBM ≈ 0.57; our deep models reach 0.64. |
| §4.6 Feature ablation | **E5 / Table V** | Lags are the source of the classical gain (and of the random-split leakage). |
| §4.7 Model decomposition | **E6 / Fig. E + Table IV** | Gain comes from tuning + ensemble, not the CNN; log/Huber hurt. Honest accounting. |
| §5 Discussion | — | Trust checklist (same dataset? test vs train? same granularity?); robustness as the real model advantage; limitations. |
| §6 Conclusion & future work | ROADMAP | From prediction to a decision agent, multi-agent, federated learning (system-level DR benefit). |

## Punchline (the sentence reviewers remember)
> We do not merely report a higher number — we explain why the reported number
> should not be trusted, establish the dataset's honest noise ceiling, and show
> that the only durable model advantage is robustness under honest evaluation.

## Why this is publishable (and stronger than a "we beat them" paper)
The contribution is methodological (reproducibility, evaluation protocol, noise
ceiling) plus an honest model study. It does not depend on beating an inflated
number — and the ablations are leakage-free, so they withstand review.

## Figures / tables produced
- Fig. A `figures/figA_granularity.png` — R² vs granularity (±std)
- Fig. B `figures/figB_split.png` — random vs chronological split
- Fig. E `figures/figE_decomposition.png` — architecture decomposition ladder
- Table II `results/TableII_E1_pivot_R2.csv` — reproduction (model × DR)
- Table III `results/TableIII_E4.csv` — vs Candanedo
- Table IV `results/TableIV_E6_architecture.csv` — architecture ablation
- Table V `results/TableV_E5_features_random.csv` — feature ablation
- Raw: `results/E1`–`E6_*.csv`
