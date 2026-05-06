import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Fitting
# ──────────────────────────────────────────────────────────────────────────────

def ridge_weights(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    _, d = X.shape
    A = X.T @ X + lam * np.eye(d)
    return np.linalg.solve(A, X.T @ y)


def ols_weights(X: np.ndarray, y: np.ndarray) -> np.ndarray:
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
    return X @ w


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))