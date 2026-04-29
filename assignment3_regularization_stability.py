"""
Assignment 3 — Regularization and Stability
Statistical Methods for Machine Learning — A.Y. 2025/26
Instructor: Nicolò Cesa-Bianchi

Based on: "Stability and Generalization" by Bousquet & Elisseeff (JMLR, 2002)

This script empirically studies the relationship between:
  - Regularization strength (lambda)
  - Algorithmic stability (sensitivity to removing/replacing training points)
  - Generalization performance (test error)

I declare that this material, which I now submit for assessment, is entirely my own
work and has not been taken from the work of others, save and to the extent that such
work has been cited and acknowledged within the text of my work.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ── reproducibility ──────────────────────────────────────────────────────────
RNG = np.random.default_rng(42)

# ── plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 130,
})

COLORS = {
    "ridge":        "#2563eb",   # blue
    "ols":          "#dc2626",   # red
    "lasso":        "#16a34a",   # green
    "small":        "#7c3aed",   # purple  (dataset-size study)
    "medium":       "#d97706",   # amber
    "large":        "#0891b2",   # cyan
}

# =============================================================================
# 1.  Core linear-algebra helpers (no sklearn estimators)
# =============================================================================

def ridge_weights(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Closed-form ridge regression: w = (X'X + λI)^{-1} X'y."""
    n, d = X.shape
    A = X.T @ X + lam * np.eye(d)
    return np.linalg.solve(A, X.T @ y)


def ols_weights(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ordinary least squares via the Moore-Penrose pseudoinverse."""
    return np.linalg.lstsq(X, y, rcond=None)[0]


def lasso_weights(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    n_iter: int = 1000,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Lasso (L1 regularization) via coordinate descent.
    Minimises  (1/2n)||Xw - y||^2 + λ||w||_1
    """
    n, d = X.shape
    w = np.zeros(d)
    # pre-compute column norms squared  (used in every update)
    col_norms_sq = np.einsum("ij,ij->j", X, X) / n   # shape (d,)

    for _ in range(n_iter):
        w_old = w.copy()
        for j in range(d):
            # partial residual (exclude feature j)
            r_j = y - X @ w + X[:, j] * w[j]
            rho_j = X[:, j] @ r_j / n
            # soft-thresholding
            if col_norms_sq[j] == 0:
                w[j] = 0.0
            else:
                w[j] = np.sign(rho_j) * max(abs(rho_j) - lam, 0.0) / col_norms_sq[j]
        if np.max(np.abs(w - w_old)) < tol:
            break
    return w


def predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return X @ w


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


# =============================================================================
# 2.  Stability estimation
# =============================================================================

def estimate_stability(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    lam: float,
    mode: str = "ridge",         # "ridge" | "ols" | "lasso"
    replace: bool = True,        # True = replace i-th point; False = remove it
) -> float:
    """
    For each training point i:
      1. Remove / replace it.
      2. Retrain the model.
      3. Measure the average change in predictions on X_test.

    Returns the stability metric β̂:
        β̂ = (1/n) Σ_i  (1/|test|) Σ_z |f_S(z) - f_{S\i}(z)|
    which estimates uniform stability as defined by Bousquet & Elisseeff (2002).
    """
    n = len(y_train)

    # baseline model trained on the full training set
    w_full = _fit(X_train, y_train, lam, mode)
    pred_full = predict(X_test, w_full)

    total_change = 0.0
    for i in range(n):
        if replace:
            # replace i-th point with a new sample drawn from an empirical bootstrap
            idx_replace = RNG.integers(0, n)
            X_mod = X_train.copy()
            y_mod = y_train.copy()
            X_mod[i] = X_train[idx_replace]
            y_mod[i] = y_train[idx_replace]
        else:
            # remove i-th point
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            X_mod = X_train[mask]
            y_mod = y_train[mask]

        w_mod = _fit(X_mod, y_mod, lam, mode)
        pred_mod = predict(X_test, w_mod)

        # average |Δf| over the test set
        total_change += float(np.mean(np.abs(pred_full - pred_mod)))

    return total_change / n


def _fit(X, y, lam, mode):
    if mode == "ridge":
        return ridge_weights(X, y, lam)
    elif mode == "lasso":
        return lasso_weights(X, y, lam)
    else:   # ols
        return ols_weights(X, y)


# =============================================================================
# 3.  Datasets
# =============================================================================

def make_synthetic(
    n: int = 300,
    d: int = 20,
    noise_std: float = 1.0,
    test_size: float = 0.25,
):
    """
    Synthetic regression:
      w* drawn from N(0,1), X ~ N(0,I), y = Xw* + ε, ε ~ N(0, noise_std²).
    """
    w_star = RNG.standard_normal(d)
    X = RNG.standard_normal((n, d))
    y = X @ w_star + noise_std * RNG.standard_normal(n)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=0
    )
    return X_tr, X_te, y_tr, y_te, "Synthetic"


def make_diabetes(test_size: float = 0.25):
    """
    Real-world dataset: Diabetes (sklearn, bundled – no download needed).
    442 samples, 10 features. Features are already standardised; target is
    normalised here for comparability across experiments.
    """
    data = load_diabetes()
    X, y = data.data, data.target

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=0
    )
    # standardise X (diabetes features are already scaled, but let's be safe)
    scaler_X = StandardScaler()
    X_tr = scaler_X.fit_transform(X_tr)
    X_te = scaler_X.transform(X_te)

    # normalise target
    scaler_y = StandardScaler()
    y_tr = scaler_y.fit_transform(y_tr.reshape(-1, 1)).ravel()
    y_te = scaler_y.transform(y_te.reshape(-1, 1)).ravel()

    return X_tr, X_te, y_tr, y_te, "Diabetes (UCI / sklearn)"


# =============================================================================
# 4.  Main experiment: stability & error vs λ
# =============================================================================

def run_experiment(
    X_tr, X_te, y_tr, y_te,
    lambdas: np.ndarray,
    dataset_name: str,
    modes: list,
):
    """
    For each mode in modes and each λ, compute:
      - train MSE
      - test  MSE
      - stability β̂
    Returns a dict of results.
    """
    results = {}
    for mode in modes:
        print(f"  [{dataset_name}]  mode={mode:6s}  ({len(lambdas)} λ values)")
        train_errs, test_errs, stabilities = [], [], []

        for lam in lambdas:
            # ── errors ──────────────────────────────────────────────────────
            w = _fit(X_tr, y_tr, lam, mode)
            train_errs.append(mse(y_tr, predict(X_tr, w)))
            test_errs.append(mse(y_te, predict(X_te, w)))

            # ── stability (remove, not replace, for clarity) ─────────────
            beta = estimate_stability(
            X_tr, y_tr, X_te, y_te, lam, mode=mode, replace=False
        )
            stabilities.append(beta)

        results[mode] = {
            "train_errs":   np.array(train_errs),
            "test_errs":    np.array(test_errs),
            "stabilities":  np.array(stabilities),
        }
    return results


# =============================================================================
# 5.  Dataset-size study (ridge only)
# =============================================================================

def run_size_study(lambdas, sizes=(50, 150, 300)):
    """
    Fix d=10, vary n.  Study how stability scales with training set size.
    """
    print("\n[Size study] Ridge regression on synthetic data")
    size_results = {}
    for n in sizes:
        X_tr, X_te, y_tr, y_te, _ = make_synthetic(n=n, d=10)
        stabs = []
        for lam in lambdas:
            beta = estimate_stability(
                X_tr, y_tr, X_te, y_te, lam, mode="ridge", replace=False
            )
            stabs.append(beta)
        size_results[n] = np.array(stabs)
    return size_results


# =============================================================================
# 6.  Plotting helpers
# =============================================================================

def plot_results(lambdas, results, dataset_name, ax_stab, ax_err):
    """Fill a pair of axes: stability and test-error vs λ."""
    for mode, res in results.items():
        color = COLORS.get(mode, "black")
        label = {"ridge": "Ridge (L2)", "ols": "OLS", "lasso": "Lasso (L1)"}.get(mode, mode)

        ax_stab.plot(lambdas, res["stabilities"], color=color, lw=2.0, label=label, marker="o", markersize=3)
        ax_err.plot(lambdas, res["test_errs"],    color=color, lw=2.0, label=label, marker="o", markersize=3)
        ax_err.plot(lambdas, res["train_errs"],   color=color, lw=1.2, linestyle="--", alpha=0.55)

    for ax in (ax_stab, ax_err):
        ax.set_xscale("log")
        ax.set_xlabel("Regularization strength  λ", fontsize=10)
        ax.legend(framealpha=0.9)
        ax.grid(True, which="both", alpha=0.25)

    ax_stab.set_ylabel("Stability  β̂\n(avg |Δprediction| on test set)", fontsize=10)
    ax_err.set_ylabel("MSE", fontsize=10)
    ax_stab.set_title(f"{dataset_name} — Stability vs λ", fontsize=12)
    ax_err.set_title(f"{dataset_name} — Train (--) & Test (─) MSE vs λ", fontsize=12)


def plot_size_study(lambdas, size_results, ax):
    sizes_sorted = sorted(size_results)
    palette = [COLORS["small"], COLORS["medium"], COLORS["large"]]
    for n, color in zip(sizes_sorted, palette):
        ax.plot(lambdas, size_results[n], color=color, lw=2.0,
                label=f"n = {n}", marker="s", markersize=3)
    ax.set_xscale("log")
    ax.set_xlabel("Regularization strength  λ", fontsize=10)
    ax.set_ylabel("Stability  β̂", fontsize=10)
    ax.set_title("Effect of Training Set Size on Stability\n(Ridge, Synthetic, d=10)", fontsize=12)
    ax.legend(framealpha=0.9)
    ax.grid(True, which="both", alpha=0.25)


# =============================================================================
# 7.  Entry point
# =============================================================================

def main():
    # ── λ grids ───────────────────────────────────────────────────────────────
    lambdas_main  = np.logspace(-4, 3, 25)   # for ridge vs OLS comparison
    lambdas_lasso = np.logspace(-4, 1, 20)   # lasso needs smaller λ range
    lambdas_size  = np.logspace(-3, 2, 20)

    # ── datasets ──────────────────────────────────────────────────────────────
    print("=== Assignment 3: Regularization and Stability ===\n")
    print("Loading datasets …")
    datasets = [make_synthetic(), make_diabetes()]

    # ── figure layout: 3 rows × 2 cols ───────────────────────────────────────
    # Row 0: synthetic   — stability | error
    # Row 1: california  — stability | error
    # Row 2: size study (span full width) | L1 vs L2 (synthetic)
    fig = plt.figure(figsize=(15, 15))
    fig.suptitle(
        "Assignment 3 — Regularization and Stability\n"
        "Bousquet & Elisseeff (JMLR, 2002)",
        fontsize=15, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.32)

    axes = [
        [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],  # row 0
        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],  # row 1
    ]
    ax_size   = fig.add_subplot(gs[2, 0])
    ax_l1_l2  = fig.add_subplot(gs[2, 1])

    # ── main experiments (ridge + OLS) ────────────────────────────────────────
    for row_idx, (X_tr, X_te, y_tr, y_te, dname) in enumerate(datasets):
        print(f"\nDataset: {dname}")
        res = run_experiment(
            X_tr, X_te, y_tr, y_te,
            lambdas=lambdas_main,
            dataset_name=dname,
            modes=["ridge", "ols"],
        )
        plot_results(lambdas_main, res, dname, axes[row_idx][0], axes[row_idx][1])

        # mark OLS error as horizontal dashed line (constant across λ)
        ols_test = res["ols"]["test_errs"][0]   # same for all λ
        axes[row_idx][1].axhline(ols_test, color=COLORS["ols"],
                                  lw=1.0, linestyle=":", alpha=0.7)

    # ── dataset-size study ────────────────────────────────────────────────────
    size_results = run_size_study(lambdas_size, sizes=[50, 150, 300])
    plot_size_study(lambdas_size, size_results, ax_size)

    # ── L1 vs L2 on synthetic data ────────────────────────────────────────────
    print("\nL1 vs L2 experiment (synthetic data) …")
    X_tr, X_te, y_tr, y_te, dname = make_synthetic()
    res_l1l2 = run_experiment(
        X_tr, X_te, y_tr, y_te,
        lambdas=lambdas_lasso,
        dataset_name=dname,
        modes=["ridge", "lasso"],
    )
    # plot stability only for the L1 vs L2 panel
    for mode, res in res_l1l2.items():
        color = COLORS[mode]
        label = "Ridge (L2)" if mode == "ridge" else "Lasso (L1)"
        ax_l1_l2.plot(lambdas_lasso, res["stabilities"],
                      color=color, lw=2.0, label=label, marker="^", markersize=3)
    ax_l1_l2.set_xscale("log")
    ax_l1_l2.set_xlabel("Regularization strength  λ", fontsize=10)
    ax_l1_l2.set_ylabel("Stability  β̂", fontsize=10)
    ax_l1_l2.set_title("L1 vs L2 Stability Comparison\n(Synthetic Dataset)", fontsize=12)
    ax_l1_l2.legend(framealpha=0.9)
    ax_l1_l2.grid(True, which="both", alpha=0.25)

    plt.savefig("assignment3_plots.pdf", bbox_inches="tight")
    plt.savefig("assignment3_plots.png", dpi=150, bbox_inches="tight")
    print("\nPlots saved → assignment3_plots.pdf / .png")

    # ── console summary ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for row_idx, (_, _, _, _, dname) in enumerate(datasets):
        print(f"\n  Dataset: {dname}")
        # re-use last computed results per dataset (stored inside loop above)
    # Re-run a quick summary pass
    for X_tr, X_te, y_tr, y_te, dname in datasets:
        print(f"\n  [{dname}]")
        res = run_experiment(
            X_tr, X_te, y_tr, y_te,
            lambdas=np.array([1e-4, 1.0, 1e3]),
            dataset_name=dname + " (summary)",
            modes=["ridge"],
        )
        for lam_val, stab, test_e in zip(
            [1e-4, 1.0, 1e3],
            res["ridge"]["stabilities"],
            res["ridge"]["test_errs"],
        ):
            print(f"    λ={lam_val:.0e}  →  stability={stab:.5f}  test_MSE={test_e:.5f}")

    print("\nDone.")


if __name__ == "__main__":
    main()