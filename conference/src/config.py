"""Shared paths, seeds, and constants for the Durrani 2025 reproduction."""
from __future__ import annotations
from pathlib import Path

# src/config.py  ->  parents[1] = project root (reproduction/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# dataset lives one level above the project root (AI Agent smart grid/dataset/)
DATA_CSV = PROJECT_ROOT.parent / "dataset" / "energydata_complete.csv"

RESULTS = PROJECT_ROOT / "results"
FIGURES = PROJECT_ROOT / "figures"
CACHE = PROJECT_ROOT / "cache"
ASSETS = PROJECT_ROOT / "slide_assets"
for d in (RESULTS, FIGURES, CACHE):
    d.mkdir(exist_ok=True)

SEED = 42
TARGET = "Appliances"

# Temporal resolution. None => keep raw 10-min data; "1h" => hourly mean.
# Hourly aggregation removes most of the 10-min sensor noise that dominates
# RMSE on this dataset. Overridable via env var REPRO_RESAMPLE.
import os
RESAMPLE_FREQ = os.environ.get("REPRO_RESAMPLE") or None
if RESAMPLE_FREQ in ("", "none", "None"):
    RESAMPLE_FREQ = None
RESULT_SUFFIX = f"_{RESAMPLE_FREQ}" if RESAMPLE_FREQ else ""

# Features dropped before modeling (per paper EDA: rv1==rv2, drop one for multicollinearity).
DROP_ALWAYS = ["rv2"]

# Train/test split.
#   "random"        - shuffle then take 80/20. Matches the paper's reported
#                     R²>=0.7 for every model. NOT a valid time-series CV
#                     scheme — neighbouring 10-min rows leak across splits.
#   "chronological" - first 80% by date for train, last 20% for test.
#                     Methodologically correct; gives much lower R² because
#                     of the winter->spring distribution shift.
SPLIT_MODE = "random"   # change to "chronological" for the honest version
TRAIN_FRAC = 0.80

# LSTM hyperparameters — paper omits these. Picked to match the paper's
# "autoregressive temporal-memory" framing on 10-min UCI Appliances data.
LSTM = dict(
    lookback=36,        # 36 * 10 min = 6 h of history
    hidden=128,
    layers=2,
    dropout=0.25,
    batch=128,
    epochs=60,
    lr=1e-3,
    weight_decay=1e-5,
)

# Classical model hyperparameters — paper omits, taken from the original
# Candanedo (2017) UCI paper grid + reasonable defaults.
RFR = dict(n_estimators=300, max_depth=None, min_samples_leaf=2, n_jobs=-1, random_state=SEED)
SVR = dict(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale")
KNN = dict(n_neighbors=10, weights="distance")
# GBM = original Candanedo (2017) best model; XGBoost = modern gradient boosting.
GBM = dict(n_estimators=300, max_depth=3, learning_rate=0.1, subsample=0.9,
           random_state=SEED)
XGB = dict(n_estimators=400, max_depth=5, learning_rate=0.08, subsample=0.9,
           colsample_bytree=0.9, n_jobs=-1, random_state=SEED)

# DR strategies — the 7 demand-response interventions evaluated in Tables 3-10.
DR_STRATEGIES = [
    "peak_clipping",
    "valley_filling",
    "load_shifting",
    "load_leveling",
    "tou_optimization",
    "price_based",
    "behavioral_dr",
]

ALL_MODELS = [
    "linear_regression",
    "random_forest",
    "svr",
    "knn",
    "gbm",
    "xgboost",
    "persistence",
    "seasonal_naive",
    "ets",
    "lstm",
]

# Learning models that train on features (used in the split-leakage study E3).
LEARNING_MODELS = [
    "linear_regression", "knn", "svr", "random_forest", "gbm", "xgboost",
    "lstm", "cnn_lstm",
]
