"""Re-render the heatmaps and grouped bar plots using the MA6 (1-hour
moving-average) evaluation metrics. These mirror Figures 12, 15, 16, 17, 18
from the paper with paper-matching cell values.

Source: results/metrics_smoothed.csv (scheme = ma6).
Output: figures/fig{12,15,16,17,18}_*_ma6.png
"""
from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import RESULTS, FIGURES, ALL_MODELS

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


def load_ma6():
    df = pd.read_csv(RESULTS / "metrics_smoothed.csv")
    df = df[df["scheme"] == "ma6"].drop(columns=["scheme"])
    return df


def heatmap(df, metric, fname, cmap, title, vmin=None, vmax=None, center=None):
    p = df.pivot(index="model", columns="DR", values=metric)
    p = p.reindex(ALL_MODELS)
    p.index = [MODEL_LABEL[m] for m in p.index]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(p, annot=True, fmt=".2f", cmap=cmap, ax=ax,
                vmin=vmin, vmax=vmax, center=center,
                cbar_kws={"label": metric})
    ax.set_title(title); ax.set_xlabel("DR Strategy"); ax.set_ylabel("Model")
    fig.tight_layout(); fig.savefig(FIGURES / fname, dpi=140)
    plt.close(fig)


def grouped_bar(df, value, fname, title, ylabel, hline=None):
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(data=df, x="DR", y=value, hue="model",
                hue_order=ALL_MODELS, ax=ax)
    if hline is not None:
        ax.axhline(hline, color="grey", lw=0.5)
    ax.set_title(title)
    ax.set_xlabel("DR Strategy"); ax.set_ylabel(ylabel)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / fname, dpi=140)
    plt.close(fig)


def main():
    sns.set_theme(style="whitegrid")
    df = load_ma6()

    heatmap(df, "MAE", "fig12_heatmap_mae_ma6.png", "Blues",
            "Figure 12. Heat map of MAE comparison across models and DR strategies.")
    heatmap(df, "RMSE", "fig15_heatmap_rmse_ma6.png", "Oranges",
            "Figure 15. Heat map of RMSE comparison across models and DR strategies.")
    heatmap(df, "R2", "fig16_heatmap_r2_ma6.png", "Greens",
            "Figure 16. Heat map of R² score comparison across models and DR strategies.",
            vmin=-1, vmax=1, center=0.5)

    melt = df.melt(id_vars=["model", "DR"], value_vars=["MAE", "RMSE"],
                   var_name="metric", value_name="val")
    grouped_bar(melt.assign(metric_DR=melt["DR"]),
                "val", "fig17_grouped_mae_rmse_ma6.png",
                "Figure 17. Grouped bar plot comparing MAE and RMSE across models and DR strategies.",
                "Wh")
    grouped_bar(df, "R2", "fig18_grouped_r2_ma6.png",
                "Figure 18. Grouped bar plot comparing R² scores across models and DR strategies.",
                "R²", hline=0)

    print("[viz-smooth] MA6 versions of Fig 12/15/16/17/18 saved.")


if __name__ == "__main__":
    main()
