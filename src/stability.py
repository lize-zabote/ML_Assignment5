"""
stability.py
------------
Empirical estimation of algorithmic stability following:

    Bousquet & Elisseeff, "Stability and Generalization",
    JMLR 2 (2002) 499-526.

The stability metric estimated here is:

    β̂ = (1/n) Σ_{i=1}^{n}  mean_{z ∈ S_test} |f_S(x) - f_{S\\i}(x)|

which approximates *uniform stability* (Definition 6 in the paper):
for each training point i the model is retrained without that point,
and the average absolute change in predictions on the fixed test set
is measured.

Functions
---------
- estimate_stability : compute β̂ for a single (model, λ) pair
- run_experiment     : sweep over λ values and record errors + stability
- run_size_study     : vary training set size, record stability
"""

import numpy as np
from src.models import fit, predict, mse


def estimate_stability(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    lam: float,
    mode: str = "ridge",
    replace: bool = False,
    rng: np.random.Generator = None,
) -> float:
    """
    Estimate uniform stability β̂ via leave-one-out retraining.

    For each training index i:
      1. Remove (or replace) the i-th sample.
      2. Retrain the model on the modified set.
      3. Record mean |Δprediction| on the fixed test set.
    β̂ is the average over all i.

    Parameters
    ----------
    X_train : (n, d)  training features
    y_train : (n,)    training targets
    X_test  : (m, d)  test features (fixed throughout)
    y_test  : (m,)    test targets  (unused — kept for API symmetry)
    lam     : regularization parameter
    mode    : "ridge" | "ols" | "lasso"
    replace : if True, replace the i-th point with a random bootstrap sample
              instead of removing it (keeps training set size constant)
    rng     : numpy random generator (used only when replace=True)

    Returns
    -------
    beta_hat : float — empirical stability estimate
    """
    n = len(y_train)
    rng = rng or np.random.default_rng(0)

    w_full    = fit(X_train, y_train, lam, mode)
    pred_full = predict(X_test, w_full)

    total_change = 0.0
    for i in range(n):
        if replace:
            idx = rng.integers(0, n)
            X_mod, y_mod = X_train.copy(), y_train.copy()
            X_mod[i] = X_train[idx]
            y_mod[i] = y_train[idx]
        else:
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            X_mod, y_mod = X_train[mask], y_train[mask]

        w_mod    = fit(X_mod, y_mod, lam, mode)
        pred_mod = predict(X_test, w_mod)

        total_change += float(np.mean(np.abs(pred_full - pred_mod)))

    return total_change / n


def run_experiment(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    lambdas: np.ndarray,
    dataset_name: str,
    modes: list,
) -> dict:
    """
    For each model mode and each λ, compute train/test MSE and β̂.

    Parameters
    ----------
    X_tr, y_tr   : training data
    X_te, y_te   : test data
    lambdas      : array of regularization values to sweep
    dataset_name : label for progress printing
    modes        : list of mode strings, e.g. ["ridge", "ols"]

    Returns
    -------
    results : dict[mode -> {"train_errs", "test_errs", "stabilities"}]
    """
    results = {}
    for mode in modes:
        print(f"  [{dataset_name}]  mode={mode:<6s}  ({len(lambdas)} λ values)")
        train_errs, test_errs, stabilities = [], [], []

        for lam in lambdas:
            w = fit(X_tr, y_tr, lam, mode)
            train_errs.append(mse(y_tr, predict(X_tr, w)))
            test_errs.append(mse(y_te, predict(X_te, w)))

            beta = estimate_stability(
                X_tr, y_tr, X_te, y_te, lam, mode=mode, replace=False
            )
            stabilities.append(beta)

        results[mode] = {
            "train_errs":  np.array(train_errs),
            "test_errs":   np.array(test_errs),
            "stabilities": np.array(stabilities),
        }
    return results


def run_size_study(
    lambdas: np.ndarray,
    sizes: tuple = (50, 150, 300),
    d: int = 10,
) -> dict:
    """
    Study how stability scales with training set size for Ridge regression.

    For each n in `sizes`, generates a fresh synthetic dataset and computes
    β̂ across all λ values.

    Parameters
    ----------
    lambdas : regularization grid
    sizes   : training set sizes to compare
    d       : feature dimensionality

    Returns
    -------
    size_results : dict[n -> np.ndarray of β̂ values]
    """
    from src.datasets import make_synthetic

    print("\n[Size study] Ridge regression on synthetic data")
    size_results = {}
    for n in sizes:
        X_tr, X_te, y_tr, y_te, _ = make_synthetic(n=n, d=d)
        stabs = [
            estimate_stability(X_tr, y_tr, X_te, y_te, lam, mode="ridge")
            for lam in lambdas
        ]
        size_results[n] = np.array(stabs)
    return size_results
