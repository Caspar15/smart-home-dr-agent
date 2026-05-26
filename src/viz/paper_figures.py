"""Reproduce the headline figures of the paper:

  Figures 12, 15, 16 — heatmaps of MAE / RMSE / R^2 across (model, DR)
  Figure 17         — grouped bar plot of MAE & RMSE
  Figure 18         — grouped bar plot of R^2

Plus per-model "actual vs DR-adjusted prediction" line plots (Figures 6-14).

Uses the retrain-per-DR metrics produced by evaluate_retrain.py.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import RESULTS, FIGURES, ALL_MODELS, DR_STRATEGIES, TARGET, CACHE, RESULT_SUFFIX
from src.data.preprocess import build_dataset
from src.dr.strategies import apply_dr

# Pretty display names (match the paper).
MODEL_LABEL = {
    "linear_regression": "Linear Regression",
    "random_forest": "Random Forest",
    "svr": "SVR",
    "knn": "kNN",
    "persistence": "Persistence",
    "seasonal_naive": "Seasonal Naive",
    "ets": "ETS",
    "lstm": "LSTM",
}
DR_LABEL = {
    "behavioral_dr": "behavioral_dr",
    "load_leveling": "load_leveling",
    "load_shifting": "load_shifting",
    "peak_clipping": "peak_clipping",
    "price_based": "price_based",
    "tou_optimization": "tou_optimization",
    "valley_filling": "valley_filling",
}


def _pivot(metric: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "metrics_retrain.csv")
    p = df.pivot(index="model", columns="DR", values=metric)
    p = p.reindex(ALL_MODELS)
    p.index = [MODEL_LABEL[m] for m in p.index]
    return p


def heatmap_mae():
    p = _pivot("MAE")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(p, annot=True, fmt=".2f", cmap="Blues", cbar_kws={"label": "MAE"}, ax=ax)
    ax.set_title("Figure 12. Heat map of MAE comparison across models and DR strategies.")
    ax.set_xlabel("DR Strategy"); ax.set_ylabel("Model")
    fig.tight_layout(); fig.savefig(FIGURES / "fig12_heatmap_mae.png", dpi=140)
    plt.close(fig)


def heatmap_rmse():
    p = _pivot("RMSE")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(p, annot=True, fmt=".2f", cmap="Oranges", cbar_kws={"label": "RMSE"}, ax=ax)
    ax.set_title("Figure 15. Heat map of RMSE comparison across models and DR strategies.")
    ax.set_xlabel("DR Strategy"); ax.set_ylabel("Model")
    fig.tight_layout(); fig.savefig(FIGURES / "fig15_heatmap_rmse.png", dpi=140)
    plt.close(fig)


def heatmap_r2():
    p = _pivot("R2")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(p, annot=True, fmt=".2f", cmap="Greens", center=0.5,
                vmin=-1, vmax=1, cbar_kws={"label": "R²"}, ax=ax)
    ax.set_title("Figure 16. Heat map of R² score comparison across models and DR strategies.")
    ax.set_xlabel("DR Strategy"); ax.set_ylabel("Model")
    fig.tight_layout(); fig.savefig(FIGURES / "fig16_heatmap_r2.png", dpi=140)
    plt.close(fig)


def grouped_bar_mae_rmse():
    df = pd.read_csv(RESULTS / "metrics_retrain.csv")
    fig, ax = plt.subplots(figsize=(14, 6))
    melt = df.melt(id_vars=["model", "DR"], value_vars=["MAE", "RMSE"],
                   var_name="metric", value_name="val")
    sns.barplot(data=melt, x="DR", y="val", hue="model",
                hue_order=ALL_MODELS, ax=ax)
    ax.set_title("Figure 17. Grouped bar plot comparing MAE and RMSE across models and DR strategies.")
    ax.set_xlabel("DR Strategy"); ax.set_ylabel("Wh")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "fig17_grouped_mae_rmse.png", dpi=140)
    plt.close(fig)


def grouped_bar_r2():
    df = pd.read_csv(RESULTS / "metrics_retrain.csv")
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(data=df, x="DR", y="R2", hue="model",
                hue_order=ALL_MODELS, ax=ax)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_title("Figure 18. Grouped bar plot comparing R² scores across models and DR strategies.")
    ax.set_xlabel("DR Strategy"); ax.set_ylabel("R²")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "fig18_grouped_r2.png", dpi=140)
    plt.close(fig)


def per_model_dr_lineplots(model_name: str, fig_index: int, honest: bool = True):
    """For one model, plot Actual vs DR-adjusted prediction across 7 DR strats.

    honest=True  -> orange line = our model's ACTUAL prediction (trained on
                    DR-modified target). Visually shows how good the model
                    really is — peaks under-shoot when R² is low.
    honest=False -> orange line = DR(actual). Shows the canonical "DR shape"
                    that the paper effectively plots (their R²=0.94 LSTM
                    produces something visually identical to DR(actual)).
    """
    split = build_dataset(verbose=False)
    df_full = split["df_clean"]
    full_y = df_full[TARGET].to_numpy()
    full_ts = df_full["date"].to_numpy()
    test_idx = split["test_idx"]
    y_test_raw = full_y[test_idx]
    ts_test = full_ts[test_idx]

    # Per-(model, DR) actual predictions, written by evaluate_retrain.py.
    per_dr_file = CACHE / f"preds_per_dr{RESULT_SUFFIX}.npz"
    per_dr = np.load(per_dr_file, allow_pickle=False) if per_dr_file.exists() else None

    order = np.argsort(ts_test)
    show = order[:300]
    fig, axes = plt.subplots(4, 2, figsize=(13, 10), sharex=False)
    axes = axes.ravel()

    for i, strat in enumerate(DR_STRATEGIES):
        ax = axes[i]
        y_actual_raw = full_y[test_idx[show]]

        if honest and per_dr is not None and f"{model_name}__{strat}" in per_dr.files:
            y_pred_full = per_dr[f"{model_name}__{strat}"]
            # LSTM predictions cover fewer test rows (lookback offset).
            # Align by index in test_idx.
            if model_name == "lstm" and f"_lstm_endidx__{strat}" in per_dr.files:
                end_idx_test = per_dr[f"_lstm_endidx__{strat}"]
                pos_map = {pos: i_ for i_, pos in enumerate(test_idx)}
                full_pred = np.full(len(test_idx), np.nan)
                for j, pos in enumerate(end_idx_test):
                    if pos in pos_map:
                        full_pred[pos_map[pos]] = y_pred_full[j]
                y_pred_curve = full_pred[show]
            else:
                y_pred_curve = y_pred_full[show]
            label_suffix = ""
        else:
            # paper-shape fallback
            y_pred_curve = apply_dr(strat, full_y, full_ts)[test_idx[show]]
            label_suffix = " [target]"

        ax.plot(np.arange(len(show)), y_actual_raw,
                label="Actual", lw=0.9, color="C0", alpha=0.85, linestyle="--")
        ax.plot(np.arange(len(show)), y_pred_curve,
                label=f"{MODEL_LABEL[model_name]} ({strat}){label_suffix}",
                lw=1.1, color="C1", alpha=0.95)
        ax.set_title(f"DR Strategy: {strat}", fontsize=9)
        ax.set_ylabel("Energy (Wh)", fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(loc="upper right", fontsize=7)
    axes[-1].set_visible(False)
    title = (f"Figure {fig_index}. {MODEL_LABEL[model_name]} model evaluation "
             f"with DR strategies.")
    if not honest:
        title += "  [orange = DR(actual), reference shape]"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    suffix = "" if honest else "_papershape"
    fig.savefig(FIGURES / f"fig{fig_index:02d}_{model_name}_dr_lines{suffix}.png", dpi=140)
    plt.close(fig)


def main():
    sns.set_theme(style="whitegrid")

    heatmap_mae()
    heatmap_rmse()
    heatmap_r2()
    grouped_bar_mae_rmse()
    grouped_bar_r2()

    # Per-model line plots — paper figures 6,7,8,9,10,11,13,14
    line_specs = [
        ("linear_regression", 6),
        ("random_forest", 7),
        ("svr", 8),
        ("knn", 9),
        ("persistence", 10),
        ("seasonal_naive", 11),
        ("ets", 13),
        ("lstm", 14),
    ]
    for name, idx in line_specs:
        per_model_dr_lineplots(name, idx, honest=True)
        per_model_dr_lineplots(name, idx, honest=False)

    print(f"[viz] all figures saved to {FIGURES}")


if __name__ == "__main__":
    main()
