# Results Summary — All Experiments (E1–E6)

Dataset: UCI Appliances Energy Prediction (19,735 × 29, 10-min, Wh).
Metric: R² (test set). Default: random 80/20 split, env+lag features.

---

## E1 — Reproduction (random split, raw 10-min, R² by model × DR)

| model | behavioral | load_level | load_shift | peak_clip | price | tou | valley |
|---|---|---|---|---|---|---|---|
| LR | 0.579 | 0.612 | 0.505 | 0.586 | 0.519 | 0.552 | 0.576 |
| kNN | 0.492 | 0.568 | 0.464 | 0.685 | 0.453 | 0.470 | 0.471 |
| SVR | 0.548 | 0.602 | 0.498 | 0.736 | 0.484 | 0.531 | 0.543 |
| RF | 0.639 | 0.678 | 0.629 | 0.784 | 0.627 | 0.633 | 0.622 |
| GBM | 0.646 | 0.680 | 0.632 | 0.792 | 0.627 | 0.636 | 0.631 |
| XGBoost | 0.652 | 0.692 | 0.644 | 0.791 | 0.638 | 0.640 | 0.636 |
| Persistence | 0.414 | 0.022 | 0.009 | −1.00 | 0.286 | 0.385 | 0.484 |
| Seasonal-Naive | −0.770 | −1.00 | −1.00 | −1.00 | −0.910 | −0.795 | −0.592 |
| ETS | −0.159 | −0.441 | −0.732 | −1.00 | −0.366 | −0.239 | −0.082 |
| LSTM | 0.636 | 0.688 | 0.603 | 0.786 | 0.592 | 0.613 | 0.627 |
| CNN-LSTM | 0.627 | 0.690 | 0.607 | 0.790 | 0.598 | 0.617 | 0.624 |

→ All strong models cluster ~0.6–0.66 at raw; naive baselines collapse to −1
on Load-Leveling/Shifting (matches Durrani Tables 7–9).

---

## E2 — Granularity (mean R² over 7 DR strategies) ★ key finding

| model | 10-min | 30-min | 1-hour | 2-hour |
|---|---|---|---|---|
| LR | 0.561 | 0.738 | 0.799 | 0.827 |
| kNN | 0.515 | 0.699 | 0.783 | 0.820 |
| SVR | 0.563 | 0.731 | 0.788 | 0.815 |
| RF | 0.659 | 0.826 | 0.881 | 0.906 |
| GBM | 0.664 | 0.827 | 0.881 | 0.904 |
| XGBoost | 0.671 | 0.830 | 0.884 | 0.909 |
| LSTM | 0.649 | 0.817 | 0.875 | 0.907 |
| CNN-LSTM | 0.651 | 0.822 | 0.880 | 0.909 |

→ Coarsening evaluation alone lifts R² from ~0.65 to ~0.91 (→ explains 0.94).

---

## E3 — Split leakage (no DR, raw R²) ★ key finding

| model | random | chronological |
|---|---|---|
| LSTM | 0.644 | **0.597** (best honest) |
| LR | 0.594 | 0.582 |
| CNN-LSTM | 0.633 | 0.552 |
| kNN | 0.494 | 0.298 |
| RF | 0.642 | 0.208 |
| SVR | 0.570 | 0.119 |
| GBM | 0.647 | **−1.00** (collapse) |
| XGBoost | 0.654 | **−1.00** (collapse) |

→ Random split inflates R²; tree models collapse under an honest chronological
split; deep sequence models (LSTM/CNN-LSTM) stay robust.

---

## E4 — vs Candanedo (2017) original method (no DR)

**Candanedo protocol (random split, env-only features):**
| model | R² | MAE | RMSE |
|---|---|---|---|
| LR | 0.184 | 46.70 | 75.51 |
| SVR | 0.394 | 40.02 | 65.08 |
| GBM | 0.430 | 36.21 | 63.10 |
| XGBoost | 0.560 | 30.25 | 55.48 |
| RF | 0.595 | 27.69 | 53.18 |
| LSTM | 0.642 | 24.31 | 49.94 |
| CNN-LSTM | 0.643 | 24.43 | 49.89 |

→ Reproduces Candanedo's GBM/RF ≈ 0.5–0.6 benchmark; our deep models reach 0.64.

**Honest chronological split (env-only):** trees collapse to −1; LSTM 0.593,
CNN-LSTM 0.571, LR 0.104.

---

## E5 — Feature ablation (raw R²)

**Random split:**
| model | env | +lags | +cyclical(full) |
|---|---|---|---|
| LR | 0.184 | 0.594 | 0.598 |
| GBM | 0.430 | 0.647 | 0.650 |
| RF | 0.595 | 0.642 | 0.642 |
| LSTM | 0.642 | 0.644 | 0.642 |

→ Lags drive the classical gain (LR 0.18→0.59); LSTM already captures history.

**Chronological split:** GBM/RF still collapse (−1 / 0.21); LSTM robust (~0.59).

---

## E6 — Architecture decomposition (no DR) ★ where the gain comes from

| step | variant | 10-min | 1-hour |
|---|---|---|---|
| 1 | LSTM (weak: h32, lb12, 15ep) | 0.611 | 0.866 |
| 2 | LSTM (tuned: h128, lb36, 60ep) | 0.644 | 0.876 |
| 3 | + CNN front-end | 0.627 | 0.869 |
| 4 | + ensemble = **final** | 0.643 | 0.880 |
| — | CNN-LSTM (Huber loss) | 0.629 | 0.880 |
| — | CNN-LSTM (log target) | 0.616 | 0.841 |

→ Final beats weak baseline (+0.032 raw / +0.014 1-h). Gain is mostly **tuning**;
ensemble helps at coarse granularity; CNN ≈ neutral; log target worst.

---

## Cross-references

**vs Durrani 2025 (LSTM, claimed):** their R² 0.89–0.94 across DR; our raw
0.59–0.79; our 1-hour 0.85–0.91; our 2-hour ~0.91. Matched once granularity
matches.

**vs literature (verified, see literature.md):** Candanedo GBM 0.57; Kulkarni
LSTM/GRU 0.60/0.62; ours (raw) 0.64. The honest ceiling is ~0.6–0.66.

## Noise ceiling (the headline)
At honest 10-min resolution, **every competitive model plateaus at R²≈0.64–0.66**.
No architecture escapes it. Higher numbers come only from coarser evaluation
(E2) or split leakage (E3).
