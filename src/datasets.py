"""
datasets.py — dataset loaders for Assignment 3.
"""
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def make_synthetic(n=300, d=20, noise=2.0, test_size=0.25, seed=42):
    """
    Synthetic underparameterized regression (n=300, d=20).

    Data-generating process
    -----------------------
        w* ~ N(0, I_d),   X ~ N(0, I_d),
        y  = X w* + eps,  eps ~ N(0, noise^2).

    Choice rationale
    ----------------
    - n >> d ensures OLS is well-defined and the pseudoinverse is stable.
    - High noise (std=2) makes the bias-variance tradeoff visible:
      Ridge clearly outperforms OLS at the optimal lambda.
    - The regime is identical to the one studied theoretically by
      Bousquet & Elisseeff (2002) for regularized least squares.
    - Stability correctly decreases with lambda in this regime.
    """
    rng = np.random.default_rng(seed)
    w_true = rng.standard_normal(d)
    X = rng.standard_normal((n, d))
    y = X @ w_true + noise * rng.standard_normal(n)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=0)

    sx = StandardScaler()
    X_tr = sx.fit_transform(X_tr)
    X_te = sx.transform(X_te)

    sy = StandardScaler()
    y_tr = sy.fit_transform(y_tr.reshape(-1, 1)).ravel()
    y_te = sy.transform(y_te.reshape(-1, 1)).ravel()

    return X_tr, X_te, y_tr, y_te, "Synthetic"


def make_diabetes(test_size=0.25):
    """
    Real-world dataset: Diabetes (442 samples, 10 features).
    Source: sklearn.datasets.load_diabetes (bundled — no download needed).
    Reference: Efron et al., Ann. Stat. 2004.
    """
    data = load_diabetes()
    X, y = data.data, data.target

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=0)

    sx = StandardScaler()
    X_tr = sx.fit_transform(X_tr)
    X_te = sx.transform(X_te)

    sy = StandardScaler()
    y_tr = sy.fit_transform(y_tr.reshape(-1, 1)).ravel()
    y_te = sy.transform(y_te.reshape(-1, 1)).ravel()

    return X_tr, X_te, y_tr, y_te, "Diabetes (UCI / sklearn)"
