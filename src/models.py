"""
models.py
---------
From-scratch implementations of linear regression models.
No sklearn estimator .fit() calls are used — only numpy for linear algebra.

Models
------
- ridge_weights  : closed-form L2-regularized regression
- ols_weights    : ordinary least squares (pseudoinverse)
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
    mode : one of "ridge", "ols"

    Returns
    -------
    w : (d,) weight vector
    """
    if mode == "ridge":
        return ridge_weights(X, y, lam)
    elif mode == "ols":
        return ols_weights(X, y)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'ridge' or 'ols'.")


# ──────────────────────────────────────────────────────────────────────────────
# Prediction and evaluation
# ──────────────────────────────────────────────────────────────────────────────

def predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Linear prediction:  ŷ = Xw."""
    return X @ w


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error."""
    return float(np.mean((y_true - y_pred) ** 2))