# Literature — Verified R² on the UCI Appliances Energy Prediction dataset

Only numbers we have **verified against the source** are listed. The honest
test-set ceiling on this dataset is ~0.6; R²=0.94 is an outlier.

| Study | Type | Model | Test R² | Verified |
|---|---|---|---|---|
| Candanedo et al. (2017) | Journal (Energy & Buildings, Q1) | GBM | **0.57** | ✅ (train 0.97 → test 0.57, classic overfitting) |
| Candanedo et al. (2017) | " | RF | ~0.54 | ✅ |
| Candanedo et al. (2017) | " | LR | ~0.16–0.19 | ✅ |
| Kulkarni (2025) | MTSU Master's thesis | GRU | 0.62 | ✅ (best in that thesis) |
| Kulkarni (2025) | " | LSTM | 0.60 | ✅ |
| Kulkarni (2025) | " | GBR | 0.61 | ✅ |
| **This work (v2)** | — | CNN-LSTM (ensemble) | **0.64** | ✅ (raw 10-min, our run) |
| This work (v2) | — | CNN-LSTM, hourly eval | ~0.91 | ✅ (MA12) |
| Durrani et al. (2025) | Journal (SAGE) | LSTM | **0.94** | claim — outlier; we trace it to coarse-grained evaluation |

## NOT comparable / do not cite as same-task evidence

- **Moon et al. (2024), PLOS One** — uses the AEP dataset but **resampled to
  hourly** and reports **MAPE/CVRMSE/NMAE, not R²** (CatBoost CVRMSE 68.69%).
  The "ETR R²=0.74" seen in search summaries is **NOT in the paper** — retracted.
- **CNN-LSTM "R²=0.97–0.99" papers** — almost all are on a **different dataset**
  (Individual Household Electric Power Consumption, kW-scale; or MW-scale grid
  load), or report training metrics. Different task — not comparable.
- **Blog posts (e.g., XGBoost 0.9865)** — not peer-reviewed; almost certainly a
  training-set score or leakage.

## Two UCI datasets that are easy to confuse

| | Ours | The confusable one |
|---|---|---|
| Name | **Appliances Energy Prediction** (Candanedo 2017) | Individual Household Electric Power Consumption |
| Unit / scale | appliance Wh, 10-min | whole-house kW, 1-min |
| Difficulty | hard (R² ≈ 0.6) | easier (often 0.9+) |

## Rule of thumb before trusting any "0.9+" on this dataset

1. **Same dataset?** (not the Individual-Household one)
2. **Test or train?** (train 0.97 is common = overfitting)
3. **Same time granularity?** (some silently resample to hourly)

All three must hold before the number is comparable.

## Sources

- Candanedo et al. 2017 — Energy and Buildings 140:81–97. DOI: 10.1016/j.enbuild.2017.01.083
  - free PDF: https://orbi.umons.ac.be/bitstream/20.500.12907/23357/1/1-s2.0-S0378778816308970-main.pdf
  - dataset/code: https://github.com/LuisM78/Appliances-energy-prediction-data
- Kulkarni, P. (2025). *Appliance Energy Prediction using Machine Learning Techniques.*
  Master's thesis, Middle Tennessee State University.
  https://jewlscholar.mtsu.edu/items/918b968e-4c23-465b-81fd-376258d21609
- Moon et al. (2024). PLOS One 19(11):e0307654.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11563398/
- Durrani et al. (2025). Energy Exploration & Exploitation 44(3):1382–1419.
  DOI: 10.1177/01445987251403607
- Dataset: https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction
