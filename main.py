"""
main.py
-------
Entry point for Assignment 3 — Regularization and Stability.

Statistical Methods for Machine Learning — A.Y. 2025/26
Instructor: Nicolò Cesa-Bianchi

Based on: Bousquet & Elisseeff, "Stability and Generalization", JMLR 2002.

Usage
-----
    python main.py

Outputs (written to outputs/)
------------------------------
    assignment3_plots.pdf
    assignment3_plots.png

Declaration
-----------
I declare that this material, which I now submit for assessment, is entirely
my own work and has not been taken from the work of others, save and to the
extent that such work has been cited and acknowledged within the text of my
work. I understand that plagiarism, collusion, and copying are grave and
serious offences in the university and accept the penalties that would be
imposed should I engage in plagiarism, collusion or copying. This assignment,
or any part of it, has not been previously submitted by me or any other person
for assessment on this or any other course of study.
"""

import numpy as np

from src.datasets  import make_synthetic, make_diabetes
from src.stability import run_experiment, run_size_study
from src.plotting  import (
    build_figure, plot_results,
    plot_size_study, plot_l1_vs_l2, save_figure,
)
from src.models import fit, predict, mse

# ── reproducibility ───────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── λ grids ───────────────────────────────────────────────────────────────────
LAMBDAS_MAIN  = np.logspace(-4, 3, 25)   # Ridge vs OLS
LAMBDAS_LASSO = np.logspace(-4, 1, 20)   # L1 vs L2  (smaller range for Lasso)
LAMBDAS_SIZE  = np.logspace(-3, 2, 20)   # size study


def print_summary(datasets, lambdas_summary):
    """Print a concise numerical summary to the console."""
    print("\n" + "=" * 60)
    print("NUMERICAL SUMMARY  (Ridge regression)")
    print("=" * 60)
    for X_tr, X_te, y_tr, y_te, dname in datasets:
        print(f"\n  Dataset: {dname}")
        res = run_experiment(
            X_tr, y_tr, X_te, y_te,
            lambdas=lambdas_summary,
            dataset_name=dname + " (summary)",
            modes=["ridge"],
        )
        for lam_val, stab, test_e in zip(
            lambdas_summary,
            res["ridge"]["stabilities"],
            res["ridge"]["test_errs"],
        ):
            print(f"    λ={lam_val:.0e}  →  stability={stab:.5f}  "
                  f"test_MSE={test_e:.5f}")


def main():
    print("=" * 60)
    print("Assignment 3 — Regularization and Stability")
    print("=" * 60)

    # ── load datasets ─────────────────────────────────────────────────────────
    print("\nLoading datasets …")
    datasets = [make_synthetic(), make_diabetes()]

    # ── build figure ──────────────────────────────────────────────────────────
    fig, axes, ax_size, ax_l1l2 = build_figure()

    # ── main experiments: Ridge vs OLS on each dataset ────────────────────────
    for row_idx, (X_tr, X_te, y_tr, y_te, dname) in enumerate(datasets):
        print(f"\nDataset: {dname}")
        results = run_experiment(
            X_tr, y_tr, X_te, y_te,
            lambdas=LAMBDAS_MAIN,
            dataset_name=dname,
            modes=["ridge", "ols"],
        )
        plot_results(LAMBDAS_MAIN, results, dname,
                     axes[row_idx][0], axes[row_idx][1])

    # ── dataset-size study (Ridge, synthetic) ─────────────────────────────────
    size_results = run_size_study(LAMBDAS_SIZE, sizes=[50, 150, 300], d=10)
    plot_size_study(LAMBDAS_SIZE, size_results, ax_size)

    # ── L1 vs L2 (synthetic) ─────────────────────────────────────────────────
    print("\nL1 vs L2 experiment (synthetic data) …")
    X_tr, X_te, y_tr, y_te, dname = make_synthetic()
    results_l1l2 = run_experiment(
        X_tr, y_tr, X_te, y_te,
        lambdas=LAMBDAS_LASSO,
        dataset_name=dname,
        modes=["ridge", "lasso"],
    )
    plot_l1_vs_l2(LAMBDAS_LASSO, results_l1l2, ax_l1l2)

    # ── save plots ────────────────────────────────────────────────────────────
    save_figure(fig, out_dir="outputs", stem="assignment3_plots")

    # ── numerical summary ─────────────────────────────────────────────────────
    print_summary(datasets, lambdas_summary=np.array([1e-4, 1.0, 1e3]))

    print("\nDone.")


if __name__ == "__main__":
    main()
