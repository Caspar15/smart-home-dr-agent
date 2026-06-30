# -*- coding: utf-8 -*-
"""Paper figures from the E1–E6 experiment CSVs.

  Fig. A  R² vs evaluation granularity (the key finding), with cnn_lstm std band
  Fig. B  R² under random vs chronological split (leakage)
Plus markdown tables for E1, E4, E5, E6.

Run:  python -m src.viz.experiment_figures
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results"
FIGS = PROJECT_ROOT / "figures"
FIGS.mkdir(exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

NAVY="#124163"; TEAL="#58B6C0"; GREEN="#75BDA7"; DARK="#373545"; GREY="#9AA7AD"; ACCENT="#E07A5F"
GRAN_ORDER = ["raw", "ma3", "ma6", "ma12"]
GRAN_LABEL = {"raw": "10-min", "ma3": "30-min", "ma6": "1-hour", "ma12": "2-hour"}
MODEL_LABEL = {"linear_regression": "LR", "random_forest": "RF", "gbm": "GBM",
               "xgboost": "XGBoost", "svr": "SVR", "knn": "kNN",
               "lstm": "LSTM", "cnn_lstm": "CNN-LSTM"}


def fig_a_granularity():
    f = RESULTS / "E2_granularity.csv"
    if not f.exists():
        print("[figA] missing E2_granularity.csv"); return
    d = pd.read_csv(f)
    show = ["linear_regression", "random_forest", "gbm", "lstm", "cnn_lstm"]
    colors = {"linear_regression": GREY, "random_forest": GREEN, "gbm": "#B5894A",
              "lstm": TEAL, "cnn_lstm": NAVY}
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    x = np.arange(len(GRAN_ORDER))
    for m in show:
        sub = d[d.model == m]
        means = [sub[sub.granularity == g]["R2"].mean() for g in GRAN_ORDER]
        ax.plot(x, means, "-o", color=colors[m], lw=2, label=MODEL_LABEL[m])
        if m == "cnn_lstm" and "R2_std" in sub.columns:
            stds = [sub[sub.granularity == g]["R2_std"].mean() for g in GRAN_ORDER]
            ax.fill_between(x, np.array(means)-np.array(stds),
                            np.array(means)+np.array(stds), color=NAVY, alpha=0.15)
    ax.axhline(0.94, ls="--", color=ACCENT, lw=1.5)
    ax.text(0.02, 0.945, "Durrani 2025 claim (0.94)", color=ACCENT, fontsize=9,
            transform=ax.get_yaxis_transform(), va="bottom")
    ax.set_xticks(x); ax.set_xticklabels([GRAN_LABEL[g] for g in GRAN_ORDER])
    ax.set_xlabel("Evaluation granularity (averaging window)", fontsize=12)
    ax.set_ylabel("R²  (mean over 7 DR strategies)", fontsize=12)
    ax.set_title("Fig. A  R² rises with coarser evaluation granularity\n"
                 "(same models — explains the 0.94 claim)", fontsize=12.5, fontweight="bold")
    ax.set_ylim(0, 1.0); ax.grid(alpha=0.3); ax.legend(fontsize=10, loc="lower right")
    fig.tight_layout(); fig.savefig(FIGS / "figA_granularity.png", dpi=160); plt.close(fig)
    print("[figA] saved")


def fig_b_split():
    f = RESULTS / "E3_split.csv"
    if not f.exists():
        print("[figB] missing E3_split.csv"); return
    d = pd.read_csv(f)
    d = d[d.granularity == "raw"]
    models = ["linear_regression", "knn", "svr", "random_forest", "gbm", "xgboost",
              "lstm", "cnn_lstm"]
    rnd = [d[(d.model == m) & (d.split == "random")]["R2"].mean() for m in models]
    chr = [d[(d.model == m) & (d.split == "chronological")]["R2"].mean() for m in models]
    x = np.arange(len(models)); w = 0.38
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.bar(x-w/2, rnd, w, label="Random split (leak-prone)", color=GREY)
    ax.bar(x+w/2, chr, w, label="Chronological split (honest)", color=NAVY)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels([MODEL_LABEL[m] for m in models], fontsize=10)
    ax.set_ylabel("R²  (10-min, no DR)", fontsize=12)
    ax.set_title("Fig. B  Random-split leakage inflates R²\n"
                 "(tree models collapse under an honest chronological split)",
                 fontsize=12.5, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGS / "figB_split.png", dpi=160); plt.close(fig)
    print("[figB] saved")


def fig_e_decomposition():
    f = RESULTS / "E6_architecture.csv"
    if not f.exists():
        print("[figE] missing E6_architecture.csv"); return
    d = pd.read_csv(f)
    ladder = d[d.step.isin([1, 2, 3, 4])].sort_values("step")
    variants = ladder.drop_duplicates("step")["variant"].tolist()
    raw = [ladder[(ladder.variant == v) & (ladder.granularity == "raw")]["R2"].iloc[0] for v in variants]
    ma6 = [ladder[(ladder.variant == v) & (ladder.granularity == "ma6")]["R2"].iloc[0] for v in variants]
    x = np.arange(len(variants)); w = 0.38
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    b1 = ax.bar(x-w/2, raw, w, label="10-min (raw)", color=GREY)
    b2 = ax.bar(x+w/2, ma6, w, label="1-hour (MA6)", color=NAVY)
    for b, v in list(zip(b1, raw)) + list(zip(b2, ma6)):
        ax.text(b.get_x()+b.get_width()/2, v+0.008, f"{v:.3f}", ha="center",
                va="bottom", fontsize=8.5)
    short = ["(1) weak\nLSTM", "(2) +tuning\n(capacity/train)", "(3) +CNN", "(4) +ensemble\n= final"]
    ax.set_xticks(x); ax.set_xticklabels(short[:len(variants)], fontsize=9.5)
    ax.set_ylabel("R²  (no DR)", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.set_title("Fig. E  Where the improvement comes from\n"
                 "(weak LSTM → tuned → +CNN → +ensemble)", fontsize=12.5, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGS / "figE_decomposition.png", dpi=160); plt.close(fig)
    print("[figE] saved")


def _md_table(df, path):
    path.write_text(df.to_markdown(index=False), encoding="utf-8")


def tables():
    # E1 reproduction (raw), pivot model × DR on R2
    f = RESULTS / "E1_reproduction.csv"
    if f.exists():
        d = pd.read_csv(f)
        p = d.pivot_table(index="model", columns="DR", values="R2")
        p.round(3).to_csv(RESULTS / "TableII_E1_pivot_R2.csv")
    # E4
    f = RESULTS / "E4_vs_candanedo.csv"
    if f.exists():
        d = pd.read_csv(f)
        d[["experiment", "model", "R2", "MAE", "RMSE"]].to_csv(
            RESULTS / "TableIII_E4.csv", index=False)
    # E5 features pivot
    f = RESULTS / "E5_features.csv"
    if f.exists():
        d = pd.read_csv(f)
        p = d[d.split == "random"].pivot_table(index="model", columns="features", values="R2")
        p.round(3).to_csv(RESULTS / "TableV_E5_features_random.csv")
    # E6 architecture
    f = RESULTS / "E6_architecture.csv"
    if f.exists():
        d = pd.read_csv(f)
        d.pivot_table(index="variant", columns="granularity", values="R2").round(4).to_csv(
            RESULTS / "TableIV_E6_architecture.csv")
    print("[tables] saved Table II/III/IV/V CSVs")


def main():
    fig_a_granularity()
    fig_b_split()
    fig_e_decomposition()
    tables()


if __name__ == "__main__":
    main()
