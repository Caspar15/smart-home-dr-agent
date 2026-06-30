# -*- coding: utf-8 -*-
"""Generate English comparison figures for the 0522 presentation.

Outputs to reproduction/slide_assets/:
  s_week_over_week.png   last-week vs this-week LSTM (raw 10-min eval)
  s_literature.png       UCI Appliances dataset R^2 across the literature
  s_v2_vs_paper.png      our v2 LSTM (raw/ma6/ma12) vs paper Table 10
  s_arch_stage.png       Local Household Intelligence pipeline progress
  s_full_arch.png        Full system architecture (0508 p.6) with progress
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

# src/viz/comparison_figures.py -> parents[2] = project root (reproduction/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS = PROJECT_ROOT / "slide_assets"
ASSETS.mkdir(exist_ok=True)
RESULTS = PROJECT_ROOT / "results"

plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

NAVY = "#124163"
TEAL = "#58B6C0"
GREEN = "#75BDA7"
DARK = "#373545"
GREY = "#9AA7AD"
ACCENT = "#E07A5F"
LGREY = "#C9D2D6"


def fig_week_over_week():
    labels = ["Last week v1\n(0515)", "This week v2\n(improved)", "Paper\n(claimed)"]
    vals = [0.579, 0.644, 0.94]
    colors = [GREY, GREEN, NAVY]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    bars = ax.bar(labels, vals, color=colors, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.015, f"{v:.3f}",
                ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("R²  (raw 10-min evaluation)", fontsize=12)
    ax.set_title("LSTM improvement: last week vs this week\n(same honest 10-min evaluation)",
                 fontsize=12.5, fontweight="bold")
    ax.annotate("", xy=(1, 0.66), xytext=(0, 0.60),
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=2))
    ax.text(0.5, 0.71, "+0.065\n(CNN-LSTM\n+cyclical feats\n+ensemble)", ha="center",
            color=ACCENT, fontsize=9.5, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(ASSETS/"s_week_over_week.png", dpi=160); plt.close(fig)


def fig_literature():
    labels = ["LR\n(lit.)", "Candanedo\nGBM 2017", "LSTM\n(lit.)", "GRU\n(lit.)",
              "Ours v2\nraw 10-min", "Ours v2\nhourly eval", "Durrani\n2025 (claim)"]
    vals = [0.19, 0.57, 0.60, 0.62, 0.644, 0.91, 0.94]
    colors = [GREY, GREY, GREY, GREY, GREEN, TEAL, NAVY]
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    bars = ax.bar(labels, vals, color=colors, width=0.66)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.015, f"{v:.2f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.axhspan(0.57, 0.64, color=GREEN, alpha=0.12)
    ax.text(0.99, 0.50, "Literature consensus  0.57–0.64", color="#3E8E7E",
            fontsize=10, transform=ax.get_yaxis_transform(), va="center",
            ha="right", fontweight="bold")
    ax.set_ylim(0, 1.05); ax.set_ylabel("R²", fontsize=12)
    ax.set_title("UCI Appliances dataset: reported R² across the literature",
                 fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(ASSETS/"s_literature.png", dpi=160); plt.close(fig)


def fig_v2_vs_paper():
    df = pd.read_csv(RESULTS/"metrics_lstm_v2.csv")
    order = ["peak_clipping","valley_filling","load_shifting","load_leveling",
             "tou_optimization","price_based","behavioral_dr"]
    paper = {"peak_clipping":0.91,"valley_filling":0.92,"load_shifting":0.93,
             "load_leveling":0.89,"tou_optimization":0.93,"price_based":0.94,
             "behavioral_dr":0.93}
    raw = [df[(df.DR==d)&(df.scheme=="raw")].R2.iloc[0] for d in order]
    ma6 = [df[(df.DR==d)&(df.scheme=="ma6")].R2.iloc[0] for d in order]
    ma12 = [df[(df.DR==d)&(df.scheme=="ma12")].R2.iloc[0] for d in order]
    pap = [paper[d] for d in order]
    x = np.arange(len(order)); w = 0.2
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(x-1.5*w, raw, w, label="Ours raw 10-min", color=GREY)
    ax.bar(x-0.5*w, ma6, w, label="Ours MA6 (1h)", color=GREEN)
    ax.bar(x+0.5*w, ma12, w, label="Ours MA12 (2h)", color=TEAL)
    ax.bar(x+1.5*w, pap, w, label="Durrani 2025", color=NAVY)
    ax.set_xticks(x); ax.set_xticklabels([d.replace("_","\n") for d in order], fontsize=9)
    ax.set_ylim(0, 1.05); ax.set_ylabel("R²", fontsize=12)
    ax.set_title("Improved LSTM v2: evaluation granularity vs paper (7 DR strategies)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, ncol=2, loc="lower center")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(ASSETS/"s_v2_vs_paper.png", dpi=160); plt.close(fig)


def _box(ax, x, y, w, h, fc, text, sub, tc="white", fs=10.5, subfs=7.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.06",
                 linewidth=1.0, edgecolor="white", facecolor=fc))
    ax.text(x+w/2, y+h*0.62, text, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=tc)
    if sub:
        ax.text(x+w/2, y+h*0.26, sub, ha="center", va="center", fontsize=subfs, color=tc)


def fig_arch_stage():
    stages = [
        ("Data", "load / weather / price", "done"),
        ("LSTM Forecast", "future load / peak risk", "done"),
        ("DR Simulation", "7 strategies (rule-based)", "partial"),
        ("State", "load+price+device+comfort", "todo"),
        ("Decision Core", "Workflow / Multi-Agent", "todo"),
        ("Objective", "cost / peak / comfort / grid", "todo"),
    ]
    cmap = {"done":GREEN,"partial":TEAL,"todo":LGREY}
    txtc = {"done":"white","partial":"white","todo":DARK}
    tag = {"done":"Done","partial":"Partial (rule-based)","todo":"Not started"}
    fig, ax = plt.subplots(figsize=(11.5, 3.4))
    ax.set_xlim(0, len(stages)*2.0); ax.set_ylim(0, 3); ax.axis("off")
    bw, bh, y0 = 1.7, 1.3, 1.0
    for i,(label,sub,st) in enumerate(stages):
        x = i*2.0+0.1
        _box(ax, x, y0, bw, bh, cmap[st], label, sub, tc=txtc[st])
        ax.text(x+bw/2, y0-0.28, tag[st], ha="center", va="center",
                fontsize=8.4, color=cmap[st] if st!="todo" else GREY, fontweight="bold")
        if i < len(stages)-1:
            ax.annotate("", xy=(x+bw+0.28, y0+bh/2), xytext=(x+bw+0.02, y0+bh/2),
                        arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.6))
    ax.text(len(stages), 2.75, "Local Household Intelligence  —  current progress",
            ha="center", fontsize=13, fontweight="bold", color=NAVY)
    fig.tight_layout(); fig.savefig(ASSETS/"s_arch_stage.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_full_arch():
    """Full system architecture (0508 p.6) with current-progress highlight."""
    fig, ax = plt.subplots(figsize=(12.2, 7.0))
    ax.set_xlim(0, 12); ax.set_ylim(0, 11); ax.axis("off")

    # Layer 1: Global coordination (future)
    _box(ax, 0.5, 9.4, 11.0, 1.2, LGREY,
         "Global Grid Coordination & Optimization Layer",
         "Dynamic Pricing / DR Signal   ·   Coordination Core   ·   Federated Learning   ·   Safety / Fairness",
         tc=DARK, fs=12, subfs=8.5)
    ax.text(11.45, 10.7, "Phase 3 (not started)", ha="right", fontsize=8.5,
            color=GREY, fontweight="bold")

    # Layer 2: Household agents layer wrapper
    ax.add_patch(FancyBboxPatch((0.5, 3.4), 11.0, 5.4,
                 boxstyle="round,pad=0.02,rounding_size=0.06",
                 linewidth=1.2, edgecolor=NAVY, facecolor="#EAF1F4"))
    ax.text(6.0, 8.45, "Multiple Household AI Agents Layer", ha="center",
            fontsize=12.5, fontweight="bold", color=NAVY)
    ax.text(1.6, 7.85, "Household A", ha="center", fontsize=9, color=GREY)
    ax.text(6.0, 7.85, "Household B", ha="center", fontsize=9, color=GREY)
    ax.text(10.4, 7.85, "Household N", ha="center", fontsize=9, color=GREY)

    # Inner: Local Household Intelligence pipeline
    ax.text(6.0, 7.3, "Local Household Intelligence", ha="center",
            fontsize=11, fontweight="bold", color=DARK)
    stages = [
        ("Data", "load/weather/price", GREEN, "white"),
        ("LSTM\nForecast", "future load / peak", GREEN, "white"),
        ("DR Sim.", "env physics", GREEN, "white"),
        ("State", "load+price+buffer", TEAL, "white"),
        ("Decision\nCore", "rule-based", TEAL, "white"),
        ("Objective", "cost/peak/comfort", TEAL, "white"),
    ]
    bw = 1.62; gap = 0.18; x0 = 0.8; y = 4.9; bh = 1.5
    for i,(t,s,fc,tc) in enumerate(stages):
        x = x0 + i*(bw+gap)
        _box(ax, x, y, bw, bh, fc, t, s, tc=tc, fs=10, subfs=7)
        if i < len(stages)-1:
            ax.annotate("", xy=(x+bw+gap-0.01, y+bh/2), xytext=(x+bw+0.01, y+bh/2),
                        arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.4))
    # "You are here" callout
    ax.annotate("You are here\n(Phase 2 Step 1: single agent done, rule-based)",
                xy=(x0+4*(bw+gap)+bw/2, y), xytext=(x0+2.4, 3.7), fontsize=9.5,
                color=ACCENT, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.8))

    # Layer 3: Physical device execution
    _box(ax, 0.5, 1.9, 11.0, 1.0, LGREY, "Physical Device Execution",
         "HVAC   ·   EV   ·   ESS   ·   Appliances   ·   PV", tc=DARK, fs=11.5, subfs=8.5)
    # Layer 4: Feedback
    _box(ax, 0.5, 0.6, 11.0, 0.95, LGREY, "Feedback & Continuous Learning",
         "actual load / cost / comfort / peak / model update", tc=DARK, fs=11.5, subfs=8.5)

    # vertical arrows between layers
    for (yt, yb) in [(9.4, 8.8), (3.4, 2.9), (1.9, 1.55)]:
        ax.annotate("", xy=(6.0, yb), xytext=(6.0, yt),
                    arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.6))
    # feedback up arrow (right side)
    ax.annotate("", xy=(11.2, 9.4), xytext=(11.2, 1.05),
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.2,
                                connectionstyle="arc3,rad=0.0", linestyle="--"))

    # legend
    for i,(c,lab) in enumerate([(GREEN,"Done"),(TEAL,"Partial (rule-based)"),(LGREY,"Future work")]):
        ax.add_patch(FancyBboxPatch((0.6+i*2.6, 0.02), 0.28, 0.22,
                     boxstyle="round,pad=0.01", facecolor=c, edgecolor="white"))
        ax.text(0.95+i*2.6, 0.13, lab, fontsize=8.5, va="center", color=DARK)

    fig.tight_layout(); fig.savefig(ASSETS/"s_full_arch.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_cnn_lstm_arch():
    """Standalone CNN-LSTM forecasting architecture diagram."""
    fig, ax = plt.subplots(figsize=(13.5, 5.6))
    ax.set_xlim(0, 13.5); ax.set_ylim(0, 5.6); ax.axis("off")

    stages = [
        ("Input window", "48 timesteps × ~40 feat.\n(env + lags + time +\npast load)",
         "(48, 40)", GREY, "white", "8 h of history"),
        ("Conv1D × 2", "64 filters, kernel=3\nReLU", "(48, 64)", GREEN, "white",
         "local pattern\nextraction (denoise)"),
        ("LSTM × 2", "128 hidden units\ndropout 0.2", "(48, 128)", NAVY, "white",
         "long-term memory\n/ trend"),
        ("Dense head", "128 → 64 → 1\nReLU + dropout", "(1)", TEAL, "white",
         "summarize → predict"),
        ("Output", "next-step\nappliance energy", "(Wh)", ACCENT, "white",
         "forecast t+1"),
    ]
    n = len(stages); bw = 2.0; bh = 1.7; gap = 0.55
    total = n*bw + (n-1)*gap
    x0 = (13.5 - total)/2; y = 2.4
    for i,(name, cfg, shape, fc, tc, role) in enumerate(stages):
        x = x0 + i*(bw+gap)
        ax.add_patch(FancyBboxPatch((x, y), bw, bh,
                     boxstyle="round,pad=0.03,rounding_size=0.10",
                     linewidth=1.2, edgecolor="white", facecolor=fc))
        ax.text(x+bw/2, y+bh*0.72, name, ha="center", va="center",
                fontsize=12.5, fontweight="bold", color=tc)
        ax.text(x+bw/2, y+bh*0.34, cfg, ha="center", va="center",
                fontsize=8.8, color=tc)
        # tensor shape above
        ax.text(x+bw/2, y+bh+0.22, shape, ha="center", va="center",
                fontsize=9.5, color=DARK, style="italic", fontweight="bold")
        # role below
        ax.text(x+bw/2, y-0.45, role, ha="center", va="center",
                fontsize=9, color=DARK)
        if i < n-1:
            ax.annotate("", xy=(x+bw+gap-0.04, y+bh/2), xytext=(x+bw+0.04, y+bh/2),
                        arrowprops=dict(arrowstyle="-|>", color=DARK, lw=2.0))

    ax.text(6.75, 5.25, "CNN-LSTM Forecasting Architecture", ha="center",
            fontsize=16, fontweight="bold", color=NAVY)
    # ensemble note
    ax.annotate("× 4 seeds, predictions averaged  (ensemble)",
                xy=(x0+total, y+bh/2), xytext=(x0+total-0.2, y-1.15),
                fontsize=10.5, color=ACCENT, fontweight="bold", ha="right",
                arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.6))
    ax.text(6.75, 0.55, "Trained with MSE loss · Adam + ReduceLROnPlateau · early stopping",
            ha="center", fontsize=9.5, color=GREY)

    fig.tight_layout(); fig.savefig(ASSETS/"s_cnn_lstm_arch.png", dpi=180,
                                    bbox_inches="tight")
    plt.close(fig)


def fig_agent_loop():
    """Single-household agent decision loop (deferrable-load MVP)."""
    fig, ax = plt.subplots(figsize=(12.0, 5.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5.2); ax.axis("off")
    boxes = [
        ("LSTM Forecast", "next-hour load\n/ peak risk", GREEN, "white"),
        ("State", "demand · forecast ·\nprice · time · buffer", TEAL, "white"),
        ("Decision Core\n(rule-based)", "high price + peak\n→ defer; off-peak\n→ release", NAVY, "white"),
        ("Action", "defer none /\nhalf / all (3)", "#B5894A", "white"),
        ("Environment", "reshape load via\ndeferrable buffer\n+ ToU price", TEAL, "white"),
        ("Reward", "−(cost + peak\n+ comfort + switch)", ACCENT, "white"),
    ]
    n = len(boxes); bw = 1.66; bh = 1.5; gap = 0.28
    total = n*bw + (n-1)*gap; x0 = (12-total)/2; y = 2.6
    for i,(name, sub, fc, tc) in enumerate(boxes):
        x = x0 + i*(bw+gap)
        ax.add_patch(FancyBboxPatch((x, y), bw, bh,
                     boxstyle="round,pad=0.03,rounding_size=0.09",
                     linewidth=1.0, edgecolor="white", facecolor=fc))
        ax.text(x+bw/2, y+bh*0.70, name, ha="center", va="center",
                fontsize=10, fontweight="bold", color=tc)
        ax.text(x+bw/2, y+bh*0.28, sub, ha="center", va="center", fontsize=7, color=tc)
        if i < n-1:
            ax.annotate("", xy=(x+bw+gap-0.02, y+bh/2), xytext=(x+bw+0.02, y+bh/2),
                        arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.5))
    # feedback arrow from Reward back to Decision (learning loop, dashed)
    xr = x0 + (n-1)*(bw+gap) + bw/2          # reward center
    xd = x0 + 2*(bw+gap) + bw/2              # decision center
    ax.annotate("", xy=(xd, y), xytext=(xr, y),
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.2, linestyle="--",
                                connectionstyle="arc3,rad=0.35"))
    ax.text((xr+xd)/2, y-1.1, "feedback (for future RL)", ha="center", fontsize=8.5,
            color=GREY, style="italic")
    ax.text(6.0, 4.7, "Single-Household Agent — Decision Loop", ha="center",
            fontsize=15, fontweight="bold", color=NAVY)
    ax.text(6.0, 4.25, "predict → state → decide → act → reward", ha="center",
            fontsize=10.5, color=DARK, style="italic")
    fig.tight_layout(); fig.savefig(ASSETS/"s_agent_loop.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_week_over_week()
    fig_literature()
    fig_v2_vs_paper()
    fig_arch_stage()
    fig_full_arch()
    fig_cnn_lstm_arch()
    fig_agent_loop()
    print("slide figures ->", ASSETS)
