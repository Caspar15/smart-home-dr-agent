"""Reproduce Figures 2-5 from the paper:
  Fig 2 — Appliance energy usage over time
  Fig 3 — Feature correlation heatmap
  Fig 4 — Histogram of appliance energy usage
  Fig 5 — Pair plot of selected variables
Plus Table 2 (descriptive stats)."""
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import FIGURES, RESULTS, TARGET
from src.data.preprocess import load_raw, feature_engineer, handle_missing_and_outliers

sns.set_theme(style="whitegrid")


def plot_timeseries(df):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df[TARGET].values, lw=0.5)
    ax.set_title("Figure 2. Appliance energy usage over time.")
    ax.set_xlabel("Time Index"); ax.set_ylabel("Energy (Wh)")
    fig.tight_layout(); fig.savefig(FIGURES / "fig02_timeseries.png", dpi=140)
    plt.close(fig)


def plot_correlation(df):
    raw_corr_cols = [c for c in df.columns if c not in ("date", "hour", "day_of_week",
                                                        "is_weekend", "month",
                                                        "minute_of_day",
                                                        "apparent_temp",
                                                        "discomfort_index")]
    corr = df[raw_corr_cols].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-0.7, vmax=1.0, square=True,
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Figure 3. Correlation heat map of the environment and energy variables.")
    fig.tight_layout(); fig.savefig(FIGURES / "fig03_correlation.png", dpi=140)
    plt.close(fig)


def plot_histogram(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df[TARGET], bins=50, kde=True, ax=ax, color="steelblue")
    ax.set_title("Figure 4. Energy consumption of appliances histogram.")
    ax.set_xlabel("Energy (Wh)"); ax.set_ylabel("Count")
    fig.tight_layout(); fig.savefig(FIGURES / "fig04_histogram.png", dpi=140)
    plt.close(fig)


def plot_pairplot(df):
    cols = ["T1", "RH_1", "T2", "RH_2", "T3", "RH_3", TARGET]
    sample = df[cols].sample(n=min(3000, len(df)), random_state=0)
    g = sns.pairplot(sample, plot_kws=dict(s=6, alpha=0.4), diag_kind="hist", height=1.4)
    g.fig.suptitle("Figure 5. Pair plot of scatter relationship among the selected "
                   "temperature, humidity, and energy consumption variables.",
                   y=1.01, fontsize=10)
    g.fig.savefig(FIGURES / "fig05_pairplot.png", dpi=140)
    plt.close(g.fig)


def descriptive_stats_table(df):
    cols = [TARGET, "T1", "RH_1", "T_out", "Windspeed"]
    tbl = df[cols].describe().T[["mean", "std", "min", "max"]].round(2)
    tbl.to_csv(RESULTS / "table02_descriptive.csv")
    print("\n[EDA] Table 2 — descriptive statistics")
    print(tbl.to_string())


def main():
    raw = load_raw()
    fe = feature_engineer(raw)
    df_pre_clip = fe.copy()   # Fig 2/4 should show original distribution incl. peaks
    df_post = handle_missing_and_outliers(fe)

    plot_timeseries(df_pre_clip)
    plot_correlation(df_pre_clip)
    plot_histogram(df_pre_clip)
    plot_pairplot(df_pre_clip)
    descriptive_stats_table(df_pre_clip)
    print(f"[EDA] saved figures -> {FIGURES}")


if __name__ == "__main__":
    main()
