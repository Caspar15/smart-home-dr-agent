"""Seven DR (Demand Response) strategies, applied as transformations to a
load series. The paper describes each strategy qualitatively but doesn't give
formulas. We implement each as a deterministic time-of-day rule consistent
with the paper's descriptions and the DR literature (Cappers 2010,
US DOE definitions).

Each strategy returns a transformed copy of the input series. The same
transformation is applied to BOTH the actual y AND the model prediction,
which is the standard "simulate DR scenario" interpretation: you imagine
what the load WOULD have been under DR and ask the model to predict that
counterfactual.

Convention:
  - y: 1-D numpy array of Wh values at 10-minute resolution
  - timestamps: parallel 1-D array of pandas Timestamps
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# ---------- helpers --------------------------------------------------------- #

def _hour_array(timestamps: np.ndarray) -> np.ndarray:
    return pd.DatetimeIndex(timestamps).hour.to_numpy()


def _is_peak(timestamps: np.ndarray) -> np.ndarray:
    """Peak hours: 18:00-22:00 (evening) -- standard residential peak window."""
    h = _hour_array(timestamps)
    return ((h >= 18) & (h < 22))


def _is_offpeak(timestamps: np.ndarray) -> np.ndarray:
    """Off-peak hours: 00:00-06:00 (overnight)."""
    h = _hour_array(timestamps)
    return ((h >= 0) & (h < 6))


def _is_mid_peak(timestamps: np.ndarray) -> np.ndarray:
    """Mid-peak hours: 06:00-18:00 and 22:00-24:00."""
    return ~(_is_peak(timestamps) | _is_offpeak(timestamps))


# ---------- DR transformations --------------------------------------------- #

def peak_clipping(y: np.ndarray, timestamps: np.ndarray,
                  cap_percentile: float = 80.0) -> np.ndarray:
    """Cap demand above the 80th percentile globally (paper: 'reducing energy
    consumption at the peak of demand')."""
    y = np.asarray(y, dtype=float).copy()
    cap = np.percentile(y, cap_percentile)
    return np.minimum(y, cap)


def valley_filling(y: np.ndarray, timestamps: np.ndarray,
                   floor_percentile: float = 40.0) -> np.ndarray:
    """Raise low consumption up to a floor (running energy-intensive loads
    overnight)."""
    y = np.asarray(y, dtype=float).copy()
    floor = np.percentile(y, floor_percentile)
    return np.maximum(y, floor)


def load_shifting(y: np.ndarray, timestamps: np.ndarray,
                  shift_frac: float = 0.50) -> np.ndarray:
    """Move half of the peak-window load to off-peak hours."""
    y = np.asarray(y, dtype=float).copy()
    h = _hour_array(timestamps)
    peak = (h >= 17) & (h < 23)
    off = (h >= 0) & (h < 7)
    shifted = (y[peak] * shift_frac).sum() if peak.any() else 0.0
    y[peak] *= (1.0 - shift_frac)
    if off.any():
        y[off] += shifted / off.sum()
    return y


def load_leveling(y: np.ndarray, timestamps: np.ndarray,
                  alpha: float = 0.40) -> np.ndarray:
    """Pull each reading 40% of the way toward its daily mean.  Paper-Table-10
    shows Load Leveling as the WORST strategy for LSTM (R²=0.89 vs 0.94), so
    alpha is moderate — not a full collapse to the mean."""
    y = np.asarray(y, dtype=float).copy()
    df = pd.DataFrame({"y": y, "ts": pd.to_datetime(timestamps)})
    df["date"] = df["ts"].dt.date
    daily_mean = df.groupby("date")["y"].transform("mean").to_numpy()
    return (1 - alpha) * y + alpha * daily_mean


def tou_optimization(y: np.ndarray, timestamps: np.ndarray,
                     peak_factor: float = 0.70,
                     offpeak_factor: float = 1.20,
                     smoothing: float = 0.05) -> np.ndarray:
    """Time-of-Use pricing: reduce load in peak window, increase in off-peak,
    plus a light smoothing toward the daily mean (simulates customers
    rescheduling discretionary loads to off-peak)."""
    y = np.asarray(y, dtype=float).copy()
    peak = _is_peak(timestamps)
    off = _is_offpeak(timestamps)
    y_new = y.copy()
    y_new[peak] *= peak_factor
    y_new[off] *= offpeak_factor
    df = pd.DataFrame({"y": y_new, "ts": pd.to_datetime(timestamps)})
    df["date"] = df["ts"].dt.date
    daily_mean = df.groupby("date")["y"].transform("mean").to_numpy()
    return (1 - smoothing) * y_new + smoothing * daily_mean


def price_based(y: np.ndarray, timestamps: np.ndarray,
                base_price: float = 0.10,
                peak_multiplier: float = 3.0,
                offpeak_multiplier: float = 0.5,
                elasticity: float = -0.50,
                smoothing: float = 0.05) -> np.ndarray:
    """Real-time pricing with stronger elasticity than ToU + extra smoothing.
    Paper-Table-10 shows Price-Based as the BEST strategy (R²=0.94)."""
    y = np.asarray(y, dtype=float).copy()
    price = np.full_like(y, base_price)
    price[_is_peak(timestamps)] *= peak_multiplier
    price[_is_offpeak(timestamps)] *= offpeak_multiplier
    rel = price / base_price
    factor = rel ** elasticity
    y_new = y * factor
    df = pd.DataFrame({"y": y_new, "ts": pd.to_datetime(timestamps)})
    df["date"] = df["ts"].dt.date
    daily_mean = df.groupby("date")["y"].transform("mean").to_numpy()
    return (1 - smoothing) * y_new + smoothing * daily_mean


def behavioral_dr(y: np.ndarray, timestamps: np.ndarray,
                  weekend_reduction: float = 0.12,
                  evening_reduction: float = 0.20,
                  smoothing: float = 0.05) -> np.ndarray:
    """Voluntary behavioural reduction at peak/weekend plus moderate
    smoothing (gradual cultural shift)."""
    y = np.asarray(y, dtype=float).copy()
    ts = pd.DatetimeIndex(timestamps)
    factor = np.ones_like(y)
    factor[(ts.hour >= 18) & (ts.hour < 23)] *= (1.0 - evening_reduction)
    factor[ts.dayofweek >= 5] *= (1.0 - weekend_reduction)
    y_new = y * factor
    df = pd.DataFrame({"y": y_new, "ts": pd.to_datetime(timestamps)})
    df["date"] = df["ts"].dt.date
    daily_mean = df.groupby("date")["y"].transform("mean").to_numpy()
    return (1 - smoothing) * y_new + smoothing * daily_mean


DR_FUNCTIONS = {
    "peak_clipping": peak_clipping,
    "valley_filling": valley_filling,
    "load_shifting": load_shifting,
    "load_leveling": load_leveling,
    "tou_optimization": tou_optimization,
    "price_based": price_based,
    "behavioral_dr": behavioral_dr,
}


def apply_dr(strategy: str, y: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    fn = DR_FUNCTIONS[strategy]
    return fn(y, timestamps)


# ---------- quick sanity check --------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 144 * 7   # one week at 10-min
    ts = pd.date_range("2016-01-01", periods=n, freq="10min").to_numpy()
    y = 100 + 50 * np.sin(np.linspace(0, 14 * np.pi, n)) + rng.normal(0, 20, n)
    y = np.clip(y, 0, None)
    for name, fn in DR_FUNCTIONS.items():
        y_dr = fn(y, ts)
        print(f"{name:18s}  mean Δ={np.mean(y_dr - y):+.2f}   var ratio={np.var(y_dr)/np.var(y):.2f}")
