import numpy as np
from src.datasets  import make_synthetic, make_diabetes
from src.stability import run_experiment, run_size_study
from src.plotting  import plot_stability, plot_errors, plot_size_study

SEED = 42
np.random.seed(SEED)

LAMBDAS_MAIN = np.logspace(-4, 3, 25)
LAMBDAS_SIZE = np.logspace(-3, 2, 20)
OUT = "outputs"

def main():
    print("=" * 55)
    print("Assignment 3 — Regularization and Stability")
    print("=" * 55)

    datasets = [make_synthetic(), make_diabetes()]
    fig_names = [("fig1_synthetic_stability.pdf", "fig2_synthetic_error.pdf"),
                 ("fig3_diabetes_stability.pdf",  "fig4_diabetes_error.pdf")]

    for (X_tr, X_te, y_tr, y_te, dname), (fn_stab, fn_err) in zip(datasets, fig_names):
        print(f"\nDataset: {dname}")
        res = run_experiment(X_tr, y_tr, X_te, y_te,
                             lambdas=LAMBDAS_MAIN,
                             dataset_name=dname,
                             modes=["ridge", "ols"])
        plot_stability(LAMBDAS_MAIN, res, dname, fn_stab, OUT)
        plot_errors   (LAMBDAS_MAIN, res, dname, fn_err,  OUT)

    print("\nSize study …")
    size_res = run_size_study(LAMBDAS_SIZE, sizes=[50, 150, 300], d=10)
    plot_size_study(LAMBDAS_SIZE, size_res, "fig5_size_study.pdf", OUT)

    print("\nDone. All figures in outputs/")

if __name__ == "__main__":
    main()
