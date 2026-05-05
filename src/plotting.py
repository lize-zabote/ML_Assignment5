"""
plotting.py — individual figure saving for Assignment 3.
Each experiment produces a separate PDF in outputs/.

Figures
-------
- fig1_synthetic_stability.pdf
- fig2_synthetic_error.pdf
- fig3_diabetes_stability.pdf
- fig4_diabetes_error.pdf
- fig5_size_study.pdf
"""

import os
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 11,
    "legend.fontsize": 10, "figure.dpi": 130,
})

COLORS = {
    "ridge": "#2563eb", "ols": "#dc2626",
    "lasso": "#16a34a", "small": "#7c3aed",
    "medium": "#d97706", "large": "#0891b2",
}
_LABELS = {"ridge": "Ridge (L2)", "ols": "OLS", "lasso": "Lasso (L1)"}


def _style(ax, xlabel=r"Regularization strength $\lambda$"):
    ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.legend(framealpha=0.9)
    ax.grid(True, which="both", alpha=0.25)


def _save(fig, filename, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {path}")


def plot_stability(lambdas, results, dataset_name, filename, out_dir="outputs"):
    fig, ax = plt.subplots(figsize=(6, 4))
    for mode, res in results.items():
        ax.plot(lambdas, res["stabilities"], color=COLORS.get(mode, "k"),
                lw=2, label=_LABELS.get(mode, mode), marker="o", markersize=3)
    _style(ax)
    ax.set_ylabel(r"Stability $\hat{\beta}$  (avg $|\Delta$pred$|$ on test)", fontsize=11)
    ax.set_title(f"{dataset_name} — Stability vs $\\lambda$")
    fig.tight_layout()
    _save(fig, filename, out_dir)


def plot_errors(lambdas, results, dataset_name, filename, out_dir="outputs"):
    fig, ax = plt.subplots(figsize=(6, 4))
    for mode, res in results.items():
        c = COLORS.get(mode, "k")
        lbl = _LABELS.get(mode, mode)
        ax.plot(lambdas, res["test_errs"],  color=c, lw=2,   label=f"{lbl} (test)",  marker="o", markersize=3)
        ax.plot(lambdas, res["train_errs"], color=c, lw=1.2, label=f"{lbl} (train)", linestyle="--", alpha=0.55)
    _style(ax)
    ax.set_ylabel("MSE", fontsize=11)
    ax.set_title(f"{dataset_name} — Train / Test MSE vs $\\lambda$")
    fig.tight_layout()
    _save(fig, filename, out_dir)


def plot_size_study(lambdas, size_results, filename, out_dir="outputs"):
    palette = [COLORS["small"], COLORS["medium"], COLORS["large"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    for n, color in zip(sorted(size_results), palette):
        ax.plot(lambdas, size_results[n], color=color, lw=2,
                label=f"$n={n}$", marker="s", markersize=3)
    _style(ax)
    ax.set_ylabel(r"Stability $\hat{\beta}$", fontsize=11)
    ax.set_title("Effect of Training Set Size on Stability\n(Ridge, Synthetic, $d=10$)")
    fig.tight_layout()
    _save(fig, filename, out_dir)

