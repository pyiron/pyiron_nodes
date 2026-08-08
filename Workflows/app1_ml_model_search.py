"""
Application 1 — Machine-learning model search with aiflow
=========================================================
Backs the "machine-learning model search" application in ``docs/paper_draft.md``.

Demonstrates the functional primitives on a real ML task:

1. **Pluggable algorithm** — a single higher-order ``benchmark`` node evaluates
   whatever estimator is plugged into its ``Node``-typed port. Swapping the model
   is a one-line change; the outer workflow topology is unchanged.
2. **Hyperparameter sweep** — ``IterToDataFrame`` sweeps polynomial degree and
   collects cross-validated validation error into a DataFrame in one call.

Runnable headless (numpy + scikit-learn only). Produces two tidy results:
``model_comparison`` (DataFrame) and ``degree_sweep`` (DataFrame), reused by the
figure script.
"""

import sys

# sys.path.insert(0, "pyiron_core/src")
# sys.path.insert(0, "/Users/jorgneugebauer/git_libs/pyiron_nodes")

import numpy as np
import pandas as pd

from core import Workflow, as_function_node, Node
from pyiron_nodes.controls import IterToDataFrame


# --------------------------------------------------------------------------- #
#  Data: a noisy nonlinear target sampled once, shared by every experiment.    #
# --------------------------------------------------------------------------- #

def make_dataset(n=120, noise=0.15, seed=0):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-3.0, 3.0, n))
    y_true = np.sin(1.5 * x) + 0.3 * x
    y = y_true + rng.normal(0.0, noise, n)
    return x.reshape(-1, 1), y


X, Y = make_dataset()


# --------------------------------------------------------------------------- #
#  Estimator nodes — all expose the same interface: fit/predict on (X, y).      #
#  Because they share a port interface, any of them plugs into `benchmark`.     #
# --------------------------------------------------------------------------- #

@as_function_node("model")
def PolynomialRidge(degree: int = 3, alpha: float = 1.0):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.linear_model import Ridge

    model = make_pipeline(
        PolynomialFeatures(degree), StandardScaler(), Ridge(alpha=alpha)
    )
    return model


@as_function_node("model")
def RandomForest(n_estimators: int = 100, max_depth: int = 6):
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth, random_state=0
    )
    return model


@as_function_node("model")
def KNeighbors(n_neighbors: int = 7):
    from sklearn.neighbors import KNeighborsRegressor

    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    return model


# --------------------------------------------------------------------------- #
#  Higher-order benchmark node: takes an estimator *node* as a Node-typed port. #
#  Swapping the plugged-in model does not change this node or the workflow.     #
# --------------------------------------------------------------------------- #

@as_function_node(["name", "cv_rmse", "cv_std"])
def benchmark(estimator: Node, folds: int = 5):
    from sklearn.model_selection import cross_val_score

    model = estimator.pull()  # compute the plugged-in estimator on demand
    scores = cross_val_score(
        model, X, Y, cv=folds, scoring="neg_root_mean_squared_error"
    )
    name = type(model).__name__
    if hasattr(model, "steps"):  # pipeline → name by its final estimator
        name = "PolynomialRidge"
    cv_rmse = float(-scores.mean())
    cv_std = float(scores.std())
    return name, cv_rmse, cv_std


def model_comparison():
    """Pluggable-algorithm comparison — one line per model swap."""
    rows = []
    for est in (PolynomialRidge(degree=5), RandomForest(), KNeighbors()):
        wf = Workflow("ml_benchmark")
        wf.estimator = est
        wf.bench = benchmark(estimator=wf.estimator, folds=5)
        name, rmse, std = wf.run()
        rows.append({"model": name, "cv_rmse": rmse, "cv_std": std})
    return pd.DataFrame(rows).sort_values("cv_rmse").reset_index(drop=True)


# --------------------------------------------------------------------------- #
#  Hyperparameter sweep with IterToDataFrame: degree -> CV RMSE, one call.      #
# --------------------------------------------------------------------------- #

@as_function_node(["degree", "cv_rmse"])
def poly_cv(degree: int = 1, alpha: float = 1.0, folds: int = 5):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score

    model = make_pipeline(
        PolynomialFeatures(int(degree)), StandardScaler(), Ridge(alpha=alpha)
    )
    scores = cross_val_score(
        model, X, Y, cv=folds, scoring="neg_root_mean_squared_error"
    )
    cv_rmse = float(-scores.mean())
    return int(degree), cv_rmse


# def degree_sweep(degrees=range(1, 13)):
"""Sweep polynomial degree; collect validation RMSE into a DataFrame."""
wf = Workflow("degree_sweep")
wf.template = poly_cv(alpha=1.0, folds=5)
wf.sweep = IterToDataFrame(
    node=wf.template, input_label="degree" # , values=list(degrees)
    )
    # return wf.run()


# if __name__ == "__main__":
#     cmp = model_comparison()
#     print("=== Model comparison (5-fold CV RMSE) ===")
#     print(cmp.to_string(index=False))
#     sweep = degree_sweep()
#     print("\n=== Polynomial-degree sweep ===")
#     print(sweep.to_string(index=False))
#     best = sweep.loc[sweep["cv_rmse"].idxmin()]
#     print(f"\nBest degree = {int(best['degree'])}  (RMSE = {best['cv_rmse']:.4f})")
