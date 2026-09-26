from pyiron_nodes.controls import IterToDataFrame
from pyiron_nodes.math_utils import Arange
from core import Workflow
from core import group_node
from core import as_function_node

# ── Local node definitions ──────────────────────


@as_function_node(["degree", "cv_rmse"])
def poly_cv(degree: int = 1, alpha: float = 1.0, folds: int = 5):
    """Cross-validated error of a polynomial-ridge regressor of a given degree.

    A self-contained template node for a hyperparameter sweep. It builds a fixed,
    seeded synthetic regression dataset (a noisy nonlinear target,
    ``y = sin(1.5 x) + 0.3 x + noise``), fits a
    ``PolynomialFeatures → StandardScaler → Ridge`` pipeline of the requested
    ``degree``, and returns the mean k-fold cross-validated RMSE. Because it
    depends on no external state, it can be dropped straight into
    ``IterToDataFrame`` and swept over ``degree`` in the GUI — each run becomes
    one row of the results DataFrame.

    Parameters
    ----------
    degree : int
        Degree of the polynomial feature expansion — the swept hyperparameter.
        Low degrees underfit; very high degrees overfit, so the RMSE-vs-degree
        curve has a minimum (here around degree 3).
    alpha : float
        L2 (ridge) regularisation strength applied to the linear model.
    folds : int
        Number of cross-validation folds.

    Returns
    -------
    degree : int
        The degree evaluated, echoed so it becomes a column in the sweep table.
    cv_rmse : float
        Mean cross-validated root-mean-square error (lower is better).
    """
    import numpy as np
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score

    # Fixed, seeded dataset so the node is fully self-contained and reproducible.
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(-3.0, 3.0, 120))
    y = np.sin(1.5 * x) + 0.3 * x + rng.normal(0.0, 0.15, x.size)
    X = x.reshape(-1, 1)

    model = make_pipeline(
        PolynomialFeatures(int(degree)), StandardScaler(), Ridge(alpha=alpha)
    )
    scores = cross_val_score(
        model, X, y, cv=folds, scoring="neg_root_mean_squared_error"
    )
    cv_rmse = float(-scores.mean())
    return int(degree), cv_rmse


wf = Workflow("degree_sweep")

wf.Arange = Arange(start=1, stop=13)

wf.template = poly_cv()

wf.sweep = IterToDataFrame(
    node=wf.template,
    input_label="degree",
    values=wf.Arange,
    debug=False,
    executor=None,
    store=False,
)
