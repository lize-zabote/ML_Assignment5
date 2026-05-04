"""
plotting.py
-----------
All plotting functions for Assignment 3.

Functions
---------
- build_figure   : create the full 3x2 figure layout
- plot_results   : fill stability + error axes for one dataset
- plot_size_study: fill the dataset-size panel
- plot_l1_vs_l2  : fill the L1 vs L2 stability panel
- save_figure    : save to PDF and PNG in the outputs/ folder
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 130,
})

COLORS = {
    "ridge":  "#2563eb",   # blue
    "ols":    "#dc2626",   # red
    "lasso":  "#16a34a",   # green
    "small":  "#7c3aed",   # purple
    "medium": "#d97706",   # amber
    "large":  "#0891b2",   # cyan
}

_LABELS = {
    "ridge": "Ridge (L2)",
    "ols":   "OLS",
    "lasso": "Lasso (L1)",
}


# ──────────────────────────────────────────────────────────────────────────────

def build_figure():
    """
    Create the full 3×2 figure layout.

    Layout
    ------
    Row 0 : Synthetic   — stability | error
    Row 1 : Diabetes    — stability | error
    Row 2 : size study  | L1 vs L2

    Returns
    -------
    fig, axes (2×2), ax_size, ax_l1l2
    """
    fig = plt.figure(figsize=(15, 15))
    fig.suptitle(
        "Assignment 3 — Regularization and Stability\n"
        "Bousquet & Elisseeff (JMLR, 2002)",
        fontsize=15, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.32)

    axes = [
        [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
    ]
    ax_size  = fig.add_subplot(gs[2, 0])
    ax_l1l2  = fig.add_subplot(gs[2, 1])

    return fig, axes, ax_size, ax_l1l2


def _style_ax(ax, xlabel="Regularization strength  λ"):
    ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.legend(framealpha=0.9)
    ax.grid(True, which="both", alpha=0.25)


def plot_results(lambdas, results, dataset_name, ax_stab, ax_err):
    """
    Plot stability β̂ and train/test MSE vs λ for a single dataset.

    Parameters
    ----------
    lambdas     : regularization grid (x-axis)
    results     : dict returned by stability.run_experiment
    dataset_name: label for axis titles
    ax_stab     : axis for stability curve
    ax_err      : axis for error curves
    """
    for mode, res in results.items():
        color = COLORS.get(mode, "black")
        label = _LABELS.get(mode, mode)

        ax_stab.plot(lambdas, res["stabilities"],
                     color=color, lw=2.0, label=label, marker="o", markersize=3)
        ax_err.plot(lambdas, res["test_errs"],
                    color=color, lw=2.0, label=label, marker="o", markersize=3)
        ax_err.plot(lambdas, res["train_errs"],
                    color=color, lw=1.2, linestyle="--", alpha=0.55)

    _style_ax(ax_stab)
    _style_ax(ax_err)

    ax_stab.set_ylabel("Stability  β̂\n(avg |Δprediction| on test set)", fontsize=10)
    ax_err.set_ylabel("MSE", fontsize=10)
    ax_stab.set_title(f"{dataset_name} — Stability vs λ", fontsize=12)
    ax_err.set_title(f"{dataset_name} — Train (--) & Test (─) MSE vs λ", fontsize=12)


def plot_size_study(lambdas, size_results, ax):
    """
    Plot stability vs λ for different training set sizes.

    Parameters
    ----------
    lambdas      : regularization grid (x-axis)
    size_results : dict[n -> np.ndarray] returned by stability.run_size_study
    ax           : matplotlib axis
    """
    palette = [COLORS["small"], COLORS["medium"], COLORS["large"]]
    for n, color in zip(sorted(size_results), palette):
        ax.plot(lambdas, size_results[n],
                color=color, lw=2.0, label=f"n = {n}",
                marker="s", markersize=3)

    _style_ax(ax)
    ax.set_ylabel("Stability  β̂", fontsize=10)
    ax.set_title(
        "Effect of Training Set Size on Stability\n(Ridge, Synthetic, d=10)",
        fontsize=12,
    )


def plot_l1_vs_l2(lambdas, results, ax):
    """
    Plot L1 vs L2 stability comparison.

    Parameters
    ----------
    lambdas : regularization grid (x-axis)
    results : dict returned by stability.run_experiment for ridge + lasso
    ax      : matplotlib axis
    """
    for mode, res in results.items():
        color = COLORS[mode]
        label = _LABELS.get(mode, mode)
        ax.plot(lambdas, res["stabilities"],
                color=color, lw=2.0, label=label, marker="^", markersize=3)

    _style_ax(ax)
    ax.set_ylabel("Stability  β̂", fontsize=10)
    ax.set_title("L1 vs L2 Stability Comparison\n(Synthetic Dataset)", fontsize=12)


def save_figure(fig, out_dir="outputs", stem="assignment3_plots"):
    """
    Save figure to PDF and PNG inside `out_dir/`.

    Parameters
    ----------
    fig     : matplotlib Figure
    out_dir : output directory (created if it does not exist)
    stem    : filename without extension
    """
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, f"{stem}.pdf")
    png_path = os.path.join(out_dir, f"{stem}.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"\nPlots saved → {pdf_path}")
    print(f"             → {png_path}")
