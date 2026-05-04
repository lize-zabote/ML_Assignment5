"""
datasets.py
-----------
Dataset loaders for Assignment 3.

Functions
---------
- make_synthetic : controlled Gaussian regression with known ground truth
- make_diabetes  : real-world Diabetes benchmark (sklearn, no download needed)
"""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# shared RNG — seeded in main.py via np.random.default_rng(seed)
_RNG = np.random.default_rng(42)


def make_synthetic(
    n: int = 300,
    d: int = 20,
    noise_std: float = 1.0,
    test_size: float = 0.25,
    rng: np.random.Generator = None,
):
    """
    Synthetic regression dataset.

    Data-generating process
    -----------------------
        w*  ~ N(0, I_d)
        x_i ~ N(0, I_d)
        y_i  = x_i' w*  +  ε_i,    ε_i ~ N(0, noise_std²)

    Parameters
    ----------
    n         : total number of samples
    d         : number of features
    noise_std : standard deviation of additive Gaussian noise
    test_size : fraction of samples held out for testing
    rng       : numpy random generator (uses module-level RNG if None)

    Returns
    -------
    X_tr, X_te, y_tr, y_te : train/test arrays
    name                    : human-readable dataset label
    """
    rng = rng or _RNG
    w_star = rng.standard_normal(d)
    X = rng.standard_normal((n, d))
    y = X @ w_star + noise_std * rng.standard_normal(n)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=0
    )
    return X_tr, X_te, y_tr, y_te, "Synthetic"


def make_diabetes(test_size: float = 0.25):
    """
    Real-world dataset: Diabetes (442 samples, 10 features).

    Source: sklearn.datasets.load_diabetes  (bundled — no download needed).
    Reference: Efron et al., "Least Angle Regression", Ann. Stat. 2004.

    Pre-processing
    --------------
    - Features standardised (zero mean, unit variance) on the training set;
      same transform applied to the test set.
    - Target standardised identically to allow cross-dataset MSE comparison.

    Parameters
    ----------
    test_size : fraction of samples held out for testing

    Returns
    -------
    X_tr, X_te, y_tr, y_te : train/test arrays
    name                    : human-readable dataset label
    """
    data = load_diabetes()
    X, y = data.data, data.target

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=0
    )

    scaler_X = StandardScaler()
    X_tr = scaler_X.fit_transform(X_tr)
    X_te = scaler_X.transform(X_te)

    scaler_y = StandardScaler()
    y_tr = scaler_y.fit_transform(y_tr.reshape(-1, 1)).ravel()
    y_te = scaler_y.transform(y_te.reshape(-1, 1)).ravel()

    return X_tr, X_te, y_tr, y_te, "Diabetes (UCI / sklearn)"
