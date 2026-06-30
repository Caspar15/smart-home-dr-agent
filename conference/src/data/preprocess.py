"""
Step 1-3 of the paper: Data preprocessing, feature engineering, missing-value
and outlier handling.

Paper says (p.1392):
  - forward fill + interpolation + mean imputation for missing values
  - drop duplicates
  - split timestamps into hour / day / weekend
  - add apparent temperature and discomfort index
  - z-score >3 sigma => outlier (smooth or discard)
  - Min-Max scaling
  - drop rv2 because corr(rv1, rv2)=1
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.config import (DATA_CSV, CACHE, TARGET, DROP_ALWAYS, TRAIN_FRAC,
                        SPLIT_MODE, SEED, RESAMPLE_FREQ)


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV, parse_dates=["date"])
    if RESAMPLE_FREQ:
        df = df.set_index("date").resample(RESAMPLE_FREQ).mean().reset_index()
    return df


def _apparent_temperature(t: pd.Series, rh: pd.Series, wind: pd.Series) -> pd.Series:
    """Australian BoM apparent temperature (AT) formula. Works for typical
    Belgian outdoor conditions in this dataset."""
    e = (rh / 100.0) * 6.105 * np.exp(17.27 * t / (237.7 + t))
    return t + 0.33 * e - 0.70 * wind - 4.00


def _discomfort_index(t: pd.Series, rh: pd.Series) -> pd.Series:
    """Thom's discomfort index using indoor T1/RH_1 in degC."""
    return t - 0.55 * (1.0 - 0.01 * rh) * (t - 14.5)


def feature_engineer(df: pd.DataFrame, feature_set: str = "env_lags") -> pd.DataFrame:
    """feature_set controls which feature groups are built (for ablation E5):
        "env"       -> environmental + basic time + comfort  (NO lags)
        "env_lags"  -> env + history-of-consumption lags     (DEFAULT, = current)
        "full"      -> env_lags + cyclical sin/cos time encodings
    """
    df = df.copy()
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.sort_values("date").reset_index(drop=True)

    # Timestamp features.
    ts = df["date"]
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    df["month"] = ts.dt.month
    df["minute_of_day"] = ts.dt.hour * 60 + ts.dt.minute

    # Comfort features mentioned in the paper.
    df["apparent_temp"] = _apparent_temperature(df["T_out"], df["RH_out"], df["Windspeed"])
    df["discomfort_index"] = _discomfort_index(df["T1"], df["RH_1"])

    # "History of energy consumption" — paper p.1393.  Used by every model.
    # Lags scale with sampling frequency: at 10-min, lag-6 = 1h, lag-144 = 24h;
    # at hourly, lag-1 = 1h, lag-24 = 24h.
    if feature_set in ("env_lags", "full"):
        if RESAMPLE_FREQ == "1h":
            df["Appliances_lag1"] = df[TARGET].shift(1)
            df["Appliances_lag2"] = df[TARGET].shift(2)
            df["Appliances_lag3"] = df[TARGET].shift(3)
            df["Appliances_lag6"] = df[TARGET].shift(6)
            df["Appliances_lag24"] = df[TARGET].shift(24)
            df["Appliances_roll3_mean"] = df[TARGET].shift(1).rolling(3).mean()
            df["Appliances_roll3_std"] = df[TARGET].shift(1).rolling(3).std()
        else:
            df["Appliances_lag1"] = df[TARGET].shift(1)
            df["Appliances_lag2"] = df[TARGET].shift(2)
            df["Appliances_lag3"] = df[TARGET].shift(3)
            df["Appliances_lag6"] = df[TARGET].shift(6)        # 1 h ago
            df["Appliances_lag144"] = df[TARGET].shift(144)    # 24 h ago
            df["Appliances_roll6_mean"] = df[TARGET].shift(1).rolling(6).mean()
            df["Appliances_roll6_std"] = df[TARGET].shift(1).rolling(6).std()

    # Cyclical sin/cos time encodings (only in "full").
    if feature_set == "full":
        import numpy as _np
        df["hour_sin"] = _np.sin(2 * _np.pi * df["hour"] / 24)
        df["hour_cos"] = _np.cos(2 * _np.pi * df["hour"] / 24)
        df["dow_sin"] = _np.sin(2 * _np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = _np.cos(2 * _np.pi * df["day_of_week"] / 7)

    # Multicollinearity: rv1 == rv2.
    df = df.drop(columns=[c for c in DROP_ALWAYS if c in df.columns])
    return df


def handle_missing_and_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Forward fill + interpolate + mean impute (paper p.1392-1393).
    df[num_cols] = df[num_cols].ffill().interpolate(method="linear", limit_direction="both")
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

    # z-score > 3 sigma outlier removal on the TARGET only (so we don't lose
    # all the high-energy events on every feature).  Paper says "smooth or
    # discard"; we cap at 3*sigma which is the standard "winsorize" reading
    # of that sentence and matches Candanedo (2017).
    y = df[TARGET]
    mu, sd = y.mean(), y.std()
    upper = mu + 3 * sd
    lower = max(0.0, mu - 3 * sd)
    df[TARGET] = y.clip(lower=lower, upper=upper)
    return df


def split_xy(df: pd.DataFrame, split_mode: str | None = None):
    """Train/test split. split_mode overrides config.SPLIT_MODE (for E3).

    The DataFrame is always sorted by date first so the full chronological
    series is available for time-dependent baselines (Persistence, Seasonal
    Naive, ETS, LSTM). We keep `train_idx`/`test_idx` (positions in the
    chronological series) for downstream time-aware operations.
    """
    mode = split_mode or SPLIT_MODE
    df = df.sort_values("date").reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in (TARGET, "date")]

    n_total = len(df)
    n_train = int(n_total * TRAIN_FRAC)
    if mode == "random":
        rng = np.random.default_rng(SEED)
        order = rng.permutation(n_total)
        train_idx = np.sort(order[:n_train])
        test_idx = np.sort(order[n_train:])
    elif mode == "chronological":
        train_idx = np.arange(n_train)
        test_idx = np.arange(n_train, n_total)
    else:
        raise ValueError(f"unknown split_mode: {mode}")

    train, test = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train = scaler_X.fit_transform(train[feature_cols])
    X_test = scaler_X.transform(test[feature_cols])
    y_train = scaler_y.fit_transform(train[[TARGET]]).ravel()
    y_test = scaler_y.transform(test[[TARGET]]).ravel()

    return dict(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        y_train_raw=train[TARGET].to_numpy(),
        y_test_raw=test[TARGET].to_numpy(),
        feature_names=feature_cols,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        date_train=train["date"].to_numpy(),
        date_test=test["date"].to_numpy(),
        df_clean=df,                  # full chronological frame
        train_idx=train_idx,           # positions of train rows in df_clean
        test_idx=test_idx,             # positions of test rows in df_clean
        split_mode=mode,
    )


def build_dataset(verbose: bool = True, split_mode: str | None = None,
                  feature_set: str = "env_lags"):
    raw = load_raw()
    if verbose:
        print(f"[data_prep] raw shape: {raw.shape}")
    fe = feature_engineer(raw, feature_set=feature_set)
    cleaned = handle_missing_and_outliers(fe)
    if verbose:
        print(f"[data_prep] post-clean shape: {cleaned.shape}, target stats: "
              f"mean={cleaned[TARGET].mean():.2f}, std={cleaned[TARGET].std():.2f}, "
              f"max={cleaned[TARGET].max():.0f}")
    split = split_xy(cleaned, split_mode=split_mode)
    if verbose:
        print(f"[data_prep] X_train {split['X_train'].shape}  X_test {split['X_test'].shape}")
        print(f"[data_prep] features ({len(split['feature_names'])}): {split['feature_names']}")
    return split


if __name__ == "__main__":
    d = build_dataset()
    out = CACHE / "split.npz"
    np.savez_compressed(
        out,
        X_train=d["X_train"], X_test=d["X_test"],
        y_train=d["y_train"], y_test=d["y_test"],
        y_train_raw=d["y_train_raw"], y_test_raw=d["y_test_raw"],
        feature_names=np.array(d["feature_names"]),
        date_train=d["date_train"].astype("datetime64[ns]"),
        date_test=d["date_test"].astype("datetime64[ns]"),
    )
    print(f"[data_prep] saved -> {out}")
