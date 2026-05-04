# Assignment 3 — Regularization and Stability

**Statistical Methods for Machine Learning — A.Y. 2025/26**  
Instructor: Nicolò Cesa-Bianchi  

Based on: *Stability and Generalization* — Bousquet & Elisseeff, JMLR 2002.

---

## Overview

This project empirically studies the relationship between **regularization strength**,
**algorithmic stability**, and **generalization performance** in linear regression.

Key findings:
- Increasing λ raises stability (smaller β̂) and, up to a threshold, reduces test error
- OLS (λ = 0) is consistently the least stable model
- Larger training sets improve stability for a fixed λ, consistent with β = O(1/λm)
- L2 regularization (Ridge) is marginally more stable than L1 (Lasso) at the same λ

---

## Project Structure

```
assignment3/
│
├── main.py               # entry point — run this
│
├── src/
│   ├── __init__.py
│   ├── models.py         # Ridge, OLS, Lasso implemented from scratch
│   ├── datasets.py       # synthetic + real-world data loaders
│   ├── stability.py      # stability estimation + experiment runners
│   └── plotting.py       # all matplotlib visualization
│
├── outputs/              # generated plots (created on first run)
│   ├── assignment3_plots.pdf
│   └── assignment3_plots.png
│
├── report/
│   └── report_assignment3.tex   # LaTeX report
│
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/assignment3.git
cd assignment3
```

### 2. Create a virtual environment and install dependencies
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the experiments
```bash
python main.py
```

Plots are saved to `outputs/assignment3_plots.pdf` and `.png`.

---

## Algorithms (implemented from scratch)

| Model | Method | Regularization |
|---|---|---|
| Ridge regression | Closed-form `(X'X + λI)⁻¹ X'y` | L2 |
| OLS | Moore-Penrose pseudoinverse | None |
| Lasso | Coordinate descent + soft-thresholding | L1 |

---

## Stability Metric

Following Bousquet & Elisseeff (2002), stability is estimated as:

$$\hat{\beta} = \frac{1}{n} \sum_{i=1}^{n} \frac{1}{|S_{\text{test}}|} \sum_{z \in S_{\text{test}}} |f_S(x) - f_{S \setminus i}(x)|$$

For each training point *i*, the model is retrained without it and the average
absolute change in predictions on the held-out test set is recorded.

---

## Datasets

| Dataset | Type | Samples | Features | Source |
|---|---|---|---|---|
| Synthetic | Gaussian linear | 300 | 20 | Generated |
| Diabetes | Real-world | 442 | 10 | `sklearn.datasets` |

---

## Declaration

*I declare that this material, which I now submit for assessment, is entirely my own
work and has not been taken from the work of others, save and to the extent that such
work has been cited and acknowledged within the text of my work.*
