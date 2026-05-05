# Assignment 3 — Regularization and Stability

**Statistical Methods for Machine Learning — A.Y. 2025/26**  
Instructor: Nicolò Cesa-Bianchi  
Student: Lize Ana Zabote (V13380)

Based on: *Stability and Generalization* — Bousquet & Elisseeff, JMLR 2002.

---

## Overview

This project empirically studies the relationship between **regularization strength**, **algorithmic stability**, and **generalization performance** in linear regression.

Key findings:
- Increasing λ raises stability (smaller β̂) and, up to a threshold, reduces test error in correlated datasets (e.g., Diabetes).
- OLS (λ = 0) is consistently the least stable model.
- Larger training sets improve stability for a fixed λ, consistent with the theoretical bound β = O(1/λm).
- In well-conditioned, underparameterized regimes (n ≫ d), OLS achieves lower test MSE than Ridge at moderate λ.

---

## Project Structure
```text
assignment3/
│
├── main.py               # entry point — run this
│
├── src/
│   ├── __init__.py
│   ├── models.py         # Ridge and OLS implemented from scratch
│   ├── datasets.py       # synthetic + real-world data loaders
│   ├── stability.py      # stability estimation + experiment runners
│   └── plotting.py       # all matplotlib visualization
│
├── outputs/              # generated plots (created on first run)
│   ├── fig1_synthetic_stability.pdf
│   ├── fig2_synthetic_error.pdf
│   ├── fig3_diabetes_stability.pdf
│   ├── fig4_diabetes_error.pdf
│   └── fig5_size_study.pdf
│
├── report/
│   └── report_assignment3.pdf   # Final report
│
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Clone the repository
```bash
git clone [https://github.com/](https://github.com/)<your-username>/assignment3.git
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

Plots are saved as individual PDF files in the `outputs/` directory.

---

## Algorithms (implemented from scratch)

| Model | Method | Regularization |
|---|---|---|
| Ridge regression | Closed-form `(X'X + λI)⁻¹ X'y` | L2 |
| OLS | Moore-Penrose pseudoinverse | None |

---

## Stability Metric

Following Bousquet & Elisseeff (2002), stability is estimated as:

$$\hat{\beta} = \frac{1}{n} \sum_{i=1}^{n} \frac{1}{|S_{\text{test}}|} \sum_{z \in S_{\text{test}}} |f_S(x) - f_{S \setminus i}(x)|$$

For each training point *i*, the model is retrained without it and the average absolute change in predictions on the held-out test set is recorded.

---

## Datasets

| Dataset | Type | Samples | Features | Source |
|---|---|---|---|---|
| Synthetic | Gaussian linear | 300 | 20 | Generated |
| Diabetes | Real-world | 442 | 10 | `sklearn.datasets` |

---

## Declaration

*I declare that this material, which I now submit for assessment, is entirely my own work and has not been taken from the work of others, save and to the extent that such work has been cited and acknowledged within the text of my work. I understand that plagiarism, collusion, and copying are grave and serious offences in the university and accept the penalties that would be imposed should I engage in plagiarism, collusion or copying. This assignment, or any part of it, has not been previously submitted by me or any other person for assessment on this or any other course of study.*