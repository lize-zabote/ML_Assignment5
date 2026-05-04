"""
models.py
---------
From-scratch implementations of linear regression models.
No sklearn estimator .fit() calls are used — only numpy for linear algebra.

Models
------
- ridge_weights  : closed-form L2-regularized regression
- ols_weights    : ordinary least squares (pseudoinverse)
- lasso_weights  : L1-regularized regression via coordinate descent
- fit            : unified dispatcher
- predict        : linear prediction
- mse            : mean squared error
"""

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Fitting
# ──────────────────────────────────────────────────────────────────────────────

def ridge_weights(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """
    Closed-form Ridge regression.

        w* = (X'X + λI)^{-1} X'y

    Uses numpy.linalg.solve (more stable than explicit inversion).

    Parameters
    ----------
    X   : (n, d) design matrix
    y   : (n,)   target vector
    lam : regularization strength λ ≥ 0

    Returns
    -------
    w : (d,) weight vector
    """
    _, d = X.shape
    A = X.T @ X + lam * np.eye(d)
    return np.linalg.solve(A, X.T @ y)


def ols_weights(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Ordinary Least Squares via the Moore-Penrose pseudoinverse.
    Equivalent to Ridge with λ → 0.

    Parameters
    ----------
    X : (n, d) design matrix
    y : (n,)   target vector

    Returns
    -------
    w : (d,) weight vector
    """
    return np.linalg.lstsq(X, y, rcond=None)[0]


def lasso_weights(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    n_iter: int = 1000,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Lasso (L1) via coordinate descent with soft-thresholding.

    Minimises:  (1/2n) ||Xw - y||^2  +  λ ||w||_1

    Update rule for coordinate j:
        ρ_j  = X[:,j]' r_j / n          (partial correlation)
        w_j  = sign(ρ_j) * max(|ρ_j| - λ, 0) / (||X[:,j]||^2 / n)

    where r_j = y - Xw + X[:,j] w_j  is the partial residual.

    Parameters
    ----------
    X      : (n, d) design matrix
    y      : (n,)   target vector
    lam    : regularization strength λ ≥ 0
    n_iter : maximum number of passes over all coordinates
    tol    : convergence tolerance on max coordinate change

    Returns
    -------
    w : (d,) weight vector
    """
    n, d = X.shape
    w = np.zeros(d)
    col_norms_sq = np.einsum("ij,ij->j", X, X) / n   # ||X[:,j]||^2 / n

    for _ in range(n_iter):
        w_old = w.copy()
        for j in range(d):
            r_j   = y - X @ w + X[:, j] * w[j]       # partial residual
            rho_j = X[:, j] @ r_j / n
            if col_norms_sq[j] == 0:
                w[j] = 0.0
            else:
                w[j] = (np.sign(rho_j) * max(abs(rho_j) - lam, 0.0)
                        / col_norms_sq[j])
        if np.max(np.abs(w - w_old)) < tol:
            break
    return w


# ──────────────────────────────────────────────────────────────────────────────
# Unified interface
# ──────────────────────────────────────────────────────────────────────────────

def fit(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    mode: str = "ridge",
) -> np.ndarray:
    """
    Dispatch to the correct fitting function.

    Parameters
    ----------
    X    : (n, d) design matrix
    y    : (n,)   target vector
    lam  : regularization strength (ignored for OLS)
    mode : one of "ridge", "ols", "lasso"

    Returns
    -------
    w : (d,) weight vector
    """
    if mode == "ridge":
        return ridge_weights(X, y, lam)
    elif mode == "lasso":
        return lasso_weights(X, y, lam)
    elif mode == "ols":
        return ols_weights(X, y)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'ridge', 'ols', or 'lasso'.")


# ──────────────────────────────────────────────────────────────────────────────
# Prediction and evaluation
# ──────────────────────────────────────────────────────────────────────────────

def predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Linear prediction:  ŷ = Xw."""
    return X @ w


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error."""
    return float(np.mean((y_true - y_pred) ** 2))
