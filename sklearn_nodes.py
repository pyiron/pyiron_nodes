"""
Scikit-Learn Machine Learning Nodes for PyIron.

This module contains nodes for integrating scikit-learn machine learning models
into the PyIron workflow system. Each node represents a specific sklearn model,
allowing users to easily build, train, and evaluate ML workflows through the GUI.

Supported Models:
- Linear Models: LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
- Tree-based: DecisionTreeRegressor, DecisionTreeClassifier
- Ensemble: RandomForest, GradientBoosting, AdaBoost
- Neighbors: KNeighborsRegressor, KNeighborsClassifier
- Support Vector: SVC, SVR
"""

from typing import Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np

from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
)
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    AdaBoostRegressor, AdaBoostClassifier
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

from core import as_function_node


# =============================================================================
# LINEAR REGRESSION MODELS
# =============================================================================

@as_function_node("model")
def LinearRegressionNode(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    fit_intercept: bool = True,
    copy_X: bool = True,
    n_jobs: Optional[int] = None
) -> Dict[str, Any]:
    """
    Trains a Linear Regression model.

    Linear regression fits a linear model with coefficients w to minimize the
    residual sum of squares between observed and predicted targets.

    Parameters:
        X_train: Training feature matrix
        y_train: Training target values
        fit_intercept: Whether to calculate intercept (default: True)
        copy_X: Whether to copy X matrix (default: True)
        n_jobs: Number of parallel jobs (default: None)

    Returns:
        Dictionary containing the fitted model and metadata
    """
    model = LinearRegression(
        fit_intercept=fit_intercept,
        copy_X=copy_X,
        n_jobs=n_jobs
    )
    model.fit(X_train, y_train)
    return {
        "model": model,
        "model_type": "LinearRegression",
        "coefficients": model.coef_,
        "intercept": model.intercept_
    }


@as_function_node("model")
def RidgeRegressionNode(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    alpha: float = 1.0,
    fit_intercept: bool = True,
    max_iter: Optional[int] = None,
    tol: float = 1e-4,
    solver: str = "auto"
) -> Dict[str, Any]:
    """
    Trains a Ridge Regression model.

    Ridge regression addresses overfitting by adding a regularization term
    (L2 penalty) to the loss function.

    Parameters:
        X_train: Training feature matrix
        y_train: Training target values
        alpha: Regularization strength (default: 1.0)
        fit_intercept: Whether to calculate intercept (default: True)
        max_iter: Maximum iterations for solver (default: None)
        tol: Tolerance for convergence (default: 1e-4)
        solver: Solver to use - 'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'

    Returns:
        Dictionary containing the fitted model and metadata
    """
    model = Ridge(
        alpha=alpha,
        fit_intercept=fit_intercept,
        max_iter=max_iter,
        tol=tol,
        solver=solver
    )
    model.fit(X_train, y_train)
    return {
        "model": model,
        "model_type": "Ridge",
        "alpha": alpha,
        "coefficients": model.coef_,
        "intercept": model.intercept_
    }


@as_function_node("model")
def LassoRegressionNode(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    alpha: float = 1.0,
    fit_intercept: bool = True,
    max_iter: int = 1000,
    tol: float = 1e-4,
    warm_start: bool = False
) -> Dict[str, Any]:
    """
    Trains a Lasso Regression model.

    Lasso (Least Absolute Shrinkage and Selection Operator) uses L1 penalty
    for feature selection and regularization.

    Parameters:
        X_train: Training feature matrix
        y_train: Training target values
        alpha: Regularization strength (default: 1.0)
        fit_intercept: Whether to calculate intercept (default: True)
        max_iter: Maximum iterations (default: 1000)
        tol: Tolerance for convergence (default: 1e-4)
        warm_start: Reuse solution from previous call (default: False)

    Returns:
        Dictionary containing the fitted model and metadata
    """
    model = Lasso(
        alpha=alpha,
        fit_intercept=fit_intercept,
        max_iter=max_iter,
        tol=tol,
        warm_start=warm_start
    )
    model.fit(X_train, y_train)
    return {
        "model": model,
        "model_type": "Lasso",
        "alpha": alpha,
        "coefficients": model.coef_,
        "intercept": model.intercept_,
        "n_iter": model.n_iter_
    }


@as_function_node("model")
def ElasticNetRegressionNode(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    fit_intercept: bool = True,
    max_iter: int = 1000,
    tol: float = 1e-4
) -> Dict[str, Any]:
    """
    Trains an ElasticNet Regression model.

    ElasticNet combines L1 and L2 penalties (Lasso + Ridge) for
    regularization and feature selection.

    Parameters:
        X_train: Training feature matrix
        y_train: Training target values
        alpha: Regularization strength (default: 1.0)
        l1_ratio: Balance between L1 and L2 (0=Ridge, 1=Lasso, default: 0.5)
        fit_intercept: Whether to calculate intercept (default: True)
        max_iter: Maximum iterations (default: 1000)
        tol: Tolerance for convergence (default: 1e-4)

    Returns:
        Dictionary containing the fitted model and metadata
    """
    model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=fit_intercept,
        max_iter=max_iter,
        tol=tol
    )
    model.fit(X_train, y_train)
    return {
        "model": model,
        "model_type": "ElasticNet",
        "alpha": alpha,
        "l1_ratio": l1_ratio,
        "coefficients": model.coef_,
        "intercept": model.intercept_,
        "n_iter": model.n_iter_
    }


@as_function_node("model")
def LogisticRegressionNode(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    penalty: str = "l2",
    C: float = 1.0,
    fit_intercept: bool = True,
    max_iter: int = 100,
    solver: str = "lbfgs",
    multi_class: str = "auto",
    class_weight: Optional[str] = None
) -> Dict[str, Any]:
    """
    Trains a Logistic Regression model for classification.

    Logistic regression is a linear model for binary and multiclass
    classification using the logistic function.

    Parameters:
        X_train: Training feature matrix
        y_train: Training target values
        penalty: Type of penalty - 'l1', 'l2', 'elasticnet', 'none' (default: 'l2')
        C: Inverse of regularization strength (default: 1.0)
        fit_intercept: Whether to calculate intercept (default: True)
        max_iter: Maximum iterations (default: 100)
        solver: Optimization algorithm - 'lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'
        multi_class: Strategy for multiclass - 'auto', 'ovr', 'multinomial' (default: 'auto')
        class_weight: Weight classes - 'balanced' or None (default: None)

    Returns:
        Dictionary containing the fitted model and metadata
    """
    model = LogisticRegression(
        penalty=penalty,
        C=C,
        fit_intercept=fit_intercept,
        max_iter=max_iter,
        solver=solver,
        multi_class=multi_class,
        class_weight=class_weight
    )
    model.fit(X_train, y_train)
    return {
        "model": model,
        "model_type": "LogisticRegression",
        "coefficients": model.coef_,
        "intercept": model.intercept_,
        "classes": model.classes_,
        "n_iter": model.n_iter_
    }


# =============================================================================
# TREE-BASED MODELS
# =============================================================================

@as_function_node("model")
def DecisionTreeRegressorNode(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    criterion: str = "squared_error",
    splitter: str = "best",
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Trains a Decision Tree Regressor.

    A decision tree regressor is a non-parametric supervised learning method
    that recursively partitions the feature space.

    Parameters:
        X_train: Training feature matrix
        y_train: Training target values
        criterion: Function to measure split quality - 'squared_error', 'friedman_mse', 'mae', 'poisson'
        splitter: Strategy for selecting splits - 'best' or 'random' (default: 'best')
        max_depth: Maximum tree depth (default: None)
        min_samples_split: Minimum samples to split a node (default: 2)
        min_samples_leaf: Minimum samples for leaf node (default: 1)
        random_state: Random seed (default: None)

    Returns:
        Dictionary containing the fitted model and metadata
    """
    model = DecisionTreeRegressor(
        criterion=criterion,
        splitter=splitter,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return {
        "model": model,
        "model_type": "DecisionTreeRegressor",
        "feature_importances": model.feature_importances_,
        "tree_depth": model.get_depth(),
        "n_leaves": model.get_n_leaves()
    }


@as_function_node("model")
def DecisionTreeClassifierNode(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    criterion: str = "gini",
    splitter: str = "best",
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    random_state: Optional[int] = None,
    class_weight: Optional[str] = None
) -> Dict[str, Any]:
    """
    Trains a Decision Tree Classifier.

    A decision tree classifier is a non-parametric supervised learning method
    for classification tasks.

    Parameters:
        X_train: Training feature matrix
        y_train: Training target values
        criterion: Function to measure split quality - 'gini' or 'entropy' (default: 'gini')
        splitter: Strategy for selecting splits - 'best' or 'random' (default: 'best')
        max_depth: Maximum tree depth (default: None)
        min_samples_split: Minimum samples to split a node (default: 2)
        min_samples_leaf: Minimum samples for leaf node (default: 1)
        random_state: Random seed (default: None)
        class_weight: Weight classes - 'balanced' or None (default: None)

    Returns:
        Dictionary containing the fitted model and metadata
    """
    model = DecisionTreeClassifier(
        criterion=criterion,
        splitter=splitter,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        class_weight=class_weight
    )
    model.fit(X_train, y_train)
    return {
        "model": model,
        "model_type": "DecisionTreeClassifier",
        "feature_importances": model.feature_importances_,
        "classes": model.classes_,
        "tree_depth": model.get_depth(),
        "n_leaves": model.get_n_leaves()
    }


# =============================================================================
# ENSEMBLE MODELS - RANDOM FOREST
# =============================================================================

@as_function_node("model")
def RandomForestRegressorNode(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    criterion: str = "squared_error",
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    random_state: Optional[int] = None,
    n_jobs: Optional[int] = None
) -> Dict[str, Any]:
    """
    Trains a Random Forest Regressor.

    Random Forest is an ensemble method that constructs multiple decision trees
    and averages their predictions for improved generalization.

    Parameters:
        X_train: Training feature matrix
        y_train: Training target values
        n_estimators: Number of trees (default: 100)
        criterion: Function to measure split quality - 'squared_error', 'friedman_mse', 'mae', 'poisson'
        max_depth: Maximum tree depth (default: None)
        min_samples_split: Minimum samples to split a node (default: 2)
        min_samples_leaf: Minimum samples for leaf node (default: 1)
        random_state: Random seed (default: None)
        n_jobs: Number of parallel jobs (default: None)

    Returns:
        Dictionary containing the fitted model and metadata
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs
    )
    model.fit(X_train, y_train)
    return {
        "model": model,
        "model_type": "RandomForestRegressor",
        "feature_importances": model.feature_importances_,
        "n_estimators": n_estimators
    }


@as_function_node("model")
def RandomForestClassifierNode(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    criterion: str = "gini",
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    random_state: Optional[int] = None,
    n_jobs: Optional[int] = None,
    class_weight: Optional[str] = None
) -> Dict[str, Any]:
    """
    Trains a Random Forest Classifier.

    Random Forest is an ensemble method for classification that constructs
    multiple decision trees and uses majority voting.

    Parameters:
        X_train: Training feature matrix
        y_train: Training target values
        n_estimators: Number of trees (default: 100)
        criterion: Function to measure split quality - 'gini' or 'entropy' (default: 'gini')
        max_depth: Maximum tree depth (default: None)
        min_samples_split: Minimum samples to split a node (default: 2)
        min_samples_leaf: Minimum samples for leaf node (default: 1)
        random_state: Random seed (default: None)
        n_jobs: Number of parallel jobs (default: None)
        class_weight: Weight classes - 'balanced' or None (default: None)

    Returns:
        Dictionary containing the fitted model and metadata
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs,
        class_weight=class_weight
    )
    model.fit(X_train, y_train)
    return {
        "model": model,
        "model_type": "RandomForestClassifier",
        "feature_importances": model.feature_importances_,
        "classes": model.classes_,
        "n_estimators": n_estimators
    }


# =============================================================================
# ENSEMBLE MODELS - GRADIENT BOOSTING
# =============================================================================

@as_function_node("model")
def GradientBoostingRegressorNode(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    subsample: float = 1.0,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Trains a Gradient Boosting Regressor.

    Gradient Boosting builds an ensemble of weak learners (typically shallow trees)
    sequentially, where each learner corrects previous errors.

    Parameters:
        X_train: Training feature matrix
        y_train: Training target values
        n_estimators: Number of boosting stages (default: 100)
        learning_rate: Step size for each boosting step (default: 0.1)
        max_depth: Maximum tree depth (default: 3)
        min_samples_split: Minimum samples to split a node (default: 2)
        min_samples_leaf: Minimum samples for leaf node (default: 1)
        subsample: Fraction of samples for training each tree (default: 1.0)
        random_state: Random seed (default: None)

    Returns:
        Dictionary containing the fitted model and metadata
    """
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        subsample=subsample,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return {
        "model": model,
        "model_type": "GradientBoostingRegressor",
        "feature_importances": model.feature_importances_,
        "train_score": model.train_score_,
        "n_estimators": n_estimators
    }


@as_function_node("model")
def GradientBoostingClassifierNode(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    subsample: float = 1.0,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Trains a Gradient Boosting Classifier.

    Gradient Boosting builds an ensemble of weak learners sequentially for
    classification tasks, where each learner corrects previous errors.

    Parameters:
        X_train: Training feature matrix
        y_train: Training target values
        n_estimators: Number of boosting stages (default: 100)
        learning_rate: Step size for each boosting step (default: 0.1)
        max_depth: Maximum tree depth (default: 3)
        min_samples_split: Minimum samples to split a node (default: 2)
        min_samples_leaf: Minimum samples for leaf node (default: 1)
        subsample: Fraction of samples for training each tree (default: 1.0)
        random_state: Random seed (default: None)

    Returns:
        Dictionary containing the fitted model and metadata
    """
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        subsample=subsample,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return {
        "model": model,
        "model_type": "GradientBoostingClassifier",
        "feature_importances": model.feature_importances_,
        "classes": model.classes_,
        "train_score": model.train_score_,
        "n_estimators": n_estimators
    }


# =============================================================================
# ENSEMBLE MODELS - ADABOOST
# =============================================================================

@as_function_node("model")
def AdaBoostRegressorNode(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 50,
    learning_rate: float = 1.0,
    loss: str = "linear",
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Trains an AdaBoost Regressor.

    AdaBoost is an ensemble method that combines weak learners sequentially,
    focusing on samples that previous learners misclassified.

    Parameters:
        X_train: Training feature matrix
        y_train: Training target values
        n_estimators: Number of boosting stages (default: 50)
        learning_rate: Weight shrinkage parameter (default: 1.0)
        loss: Loss function - 'linear', 'square', 'exponential' (default: 'linear')
        random_state: Random seed (default: None)

    Returns:
        Dictionary containing the fitted model and metadata
    """
    model = AdaBoostRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        loss=loss,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return {
        "model": model,
        "model_type": "AdaBoostRegressor",
        "feature_importances": model.feature_importances_,
        "n_estimators": n_estimators
    }


@as_function_node("model")
def AdaBoostClassifierNode(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 50,
    learning_rate: float = 1.0,
    algorithm: str = "SAMME.R",
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Trains an AdaBoost Classifier.

    AdaBoost is an ensemble method for classification that combines weak learners,
    focusing on samples that previous learners misclassified.

    Parameters:
        X_train: Training feature matrix
        y_train: Training target values
        n_estimators: Number of boosting stages (default: 50)
        learning_rate: Weight shrinkage parameter (default: 1.0)
        algorithm: Boosting algorithm - 'SAMME', 'SAMME.R' (default: 'SAMME.R')
        random_state: Random seed (default: None)

    Returns:
        Dictionary containing the fitted model and metadata
    """
    model = AdaBoostClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        algorithm=algorithm,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return {
        "model": model,
        "model_type": "AdaBoostClassifier",
        "feature_importances": model.feature_importances_,
        "classes": model.classes_,
        "n_estimators": n_estimators
    }


# =============================================================================
# NEIGHBOR-BASED MODELS
# =============================================================================

@as_function_node("model")
def KNeighborsRegressorNode(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_neighbors: int = 5,
    weights: str = "uniform",
    algorithm: str = "auto",
    leaf_size: int = 30,
    p: float = 2,
    metric: str = "minkowski",
    n_jobs: Optional[int] = None
) -> Dict[str, Any]:
    """
    Trains a K-Neighbors Regressor.

    K-Neighbors Regressor predicts target values based on the average of
    the k nearest neighbors in the feature space.

    Parameters:
        X_train: Training feature matrix
        y_train: Training target values
        n_neighbors: Number of neighbors (default: 5)
        weights: Weight function - 'uniform', 'distance' (default: 'uniform')
        algorithm: Algorithm for neighbor search - 'auto', 'ball_tree', 'kd_tree', 'brute' (default: 'auto')
        leaf_size: Leaf size for tree algorithms (default: 30)
        p: Power parameter for Minkowski distance (default: 2)
        metric: Distance metric (default: 'minkowski')
        n_jobs: Number of parallel jobs (default: None)

    Returns:
        Dictionary containing the fitted model and metadata
    """
    model = KNeighborsRegressor(
        n_neighbors=n_neighbors,
        weights=weights,
        algorithm=algorithm,
        leaf_size=leaf_size,
        p=p,
        metric=metric,
        n_jobs=n_jobs
    )
    model.fit(X_train, y_train)
    return {
        "model": model,
        "model_type": "KNeighborsRegressor",
        "n_neighbors": n_neighbors
    }


@as_function_node("model")
def KNeighborsClassifierNode(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_neighbors: int = 5,
    weights: str = "uniform",
    algorithm: str = "auto",
    leaf_size: int = 30,
    p: float = 2,
    metric: str = "minkowski",
    n_jobs: Optional[int] = None
) -> Dict[str, Any]:
    """
    Trains a K-Neighbors Classifier.

    K-Neighbors Classifier predicts class labels based on majority voting
    among the k nearest neighbors.

    Parameters:
        X_train: Training feature matrix
        y_train: Training target values
        n_neighbors: Number of neighbors (default: 5)
        weights: Weight function - 'uniform', 'distance' (default: 'uniform')
        algorithm: Algorithm for neighbor search - 'auto', 'ball_tree', 'kd_tree', 'brute' (default: 'auto')
        leaf_size: Leaf size for tree algorithms (default: 30)
        p: Power parameter for Minkowski distance (default: 2)
        metric: Distance metric (default: 'minkowski')
        n_jobs: Number of parallel jobs (default: None)

    Returns:
        Dictionary containing the fitted model and metadata
    """
    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        algorithm=algorithm,
        leaf_size=leaf_size,
        p=p,
        metric=metric,
        n_jobs=n_jobs
    )
    model.fit(X_train, y_train)
    return {
        "model": model,
        "model_type": "KNeighborsClassifier",
        "classes": model.classes_,
        "n_neighbors": n_neighbors
    }


# =============================================================================
# SUPPORT VECTOR MACHINES
# =============================================================================

@as_function_node("model")
def SupportVectorClassifierNode(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: str = "scale",
    degree: int = 3,
    probability: bool = False,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Trains a Support Vector Classifier (SVC).

    Support Vector Machine classifier finds the optimal hyperplane that
    maximizes the margin between classes.

    Parameters:
        X_train: Training feature matrix
        y_train: Training target values
        kernel: Kernel type - 'linear', 'poly', 'rbf', 'sigmoid' (default: 'rbf')
        C: Regularization parameter (default: 1.0)
        gamma: Kernel coefficient - 'scale', 'auto' (default: 'scale')
        degree: Polynomial degree (default: 3)
        probability: Enable probability estimates (default: False)
        random_state: Random seed (default: None)

    Returns:
        Dictionary containing the fitted model and metadata
    """
    model = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        degree=degree,
        probability=probability,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return {
        "model": model,
        "model_type": "SVC",
        "classes": model.classes_,
        "n_support": model.n_support_,
        "kernel": kernel
    }


@as_function_node("model")
def SupportVectorRegressorNode(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: str = "scale",
    degree: int = 3,
    epsilon: float = 0.1
) -> Dict[str, Any]:
    """
    Trains a Support Vector Regressor (SVR).

    Support Vector Regression applies SVM methodology to regression problems,
    finding a hyperplane with maximum tolerance to errors.

    Parameters:
        X_train: Training feature matrix
        y_train: Training target values
        kernel: Kernel type - 'linear', 'poly', 'rbf', 'sigmoid' (default: 'rbf')
        C: Regularization parameter (default: 1.0)
        gamma: Kernel coefficient - 'scale', 'auto' (default: 'scale')
        degree: Polynomial degree (default: 3)
        epsilon: Epsilon-tube parameter (default: 0.1)

    Returns:
        Dictionary containing the fitted model and metadata
    """
    model = SVR(
        kernel=kernel,
        C=C,
        gamma=gamma,
        degree=degree,
        epsilon=epsilon
    )
    model.fit(X_train, y_train)
    return {
        "model": model,
        "model_type": "SVR",
        "n_support": model.n_support_,
        "kernel": kernel
    }


# =============================================================================
# MODEL EVALUATION NODES
# =============================================================================

@as_function_node("metrics")
def EvaluateRegressionModelSklearn(
    model: Dict,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, float]:
    """
    Evaluates a regression model on test data.

    Parameters:
        model: Fitted regression model
        X_test: Test feature matrix
        y_test: Test target values

    Returns:
        Dictionary containing R2, MSE, MAE, and RMSE scores
    """
    y_pred = model["model"].predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    return {
        "R2": r2,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae
    }


@as_function_node("metrics")
def EvaluateClassificationModelSklearn(
    model: Dict,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    """
    Evaluates a classification model on test data.

    Parameters:
        model: Fitted classification model
        X_test: Test feature matrix
        y_test: Test target values

    Returns:
        Dictionary containing accuracy, precision, recall, F1-score, and confusion matrix
    """
    y_pred = model["model"].predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm.tolist()
    }


# =============================================================================
# MODEL PREDICTION NODES
# =============================================================================

@as_function_node("predictions")
def PredictRegression(
    model: Dict,
    X: pd.DataFrame
) -> np.ndarray:
    """
    Makes predictions using a fitted regression model.

    Parameters:
        model: Fitted regression model
        X: Feature matrix for prediction

    Returns:
        Predicted values as numpy array
    """
    return model["model"].predict(X)


@as_function_node("predictions")
def PredictClassification(
    model: Dict,
    X: pd.DataFrame,
    return_probabilities: bool = False
) -> Dict[str, Any]:
    """
    Makes predictions using a fitted classification model.

    Parameters:
        model: Fitted classification model
        X: Feature matrix for prediction
        return_probabilities: Whether to return class probabilities (default: False)

    Returns:
        Dictionary containing predictions and optionally probabilities
    """
    predictions = model["model"].predict(X)
    result = {"predictions": predictions}

    if return_probabilities and hasattr(model["model"], "predict_proba"):
        probabilities = model["model"].predict_proba(X)
        result["probabilities"] = probabilities

    return result


# =============================================================================
# MODEL COMPARISON NODES
# =============================================================================

@as_function_node("comparison_results")
def CompareRegressionModels(
    model_1: Dict,
    model_2: Dict,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    metric: str = "r2"
) -> Dict[str, Any]:
    """
    Compares two regression models on validation data.

    Selection Priority:
        1. Higher R2 (if metric='r2')
        2. Lower RMSE (if metric='rmse')
        3. Lower MAE (if metric='mae')

    Parameters:
        model_1: First fitted regression model
        model_2: Second fitted regression model
        X_validation: Validation feature matrix
        y_validation: Validation target values
        metric: Metric for comparison - 'r2', 'rmse', 'mae' (default: 'r2')

    Returns:
        Dictionary containing best model and comparison results
    """
    pred_1 = model_1["model"].predict(X_validation)
    pred_2 = model_2["model"].predict(X_validation)

    r2_1 = r2_score(y_validation, pred_1)
    rmse_1 = np.sqrt(mean_squared_error(y_validation, pred_1))
    mae_1 = mean_absolute_error(y_validation, pred_1)

    r2_2 = r2_score(y_validation, pred_2)
    rmse_2 = np.sqrt(mean_squared_error(y_validation, pred_2))
    mae_2 = mean_absolute_error(y_validation, pred_2)

    if metric == "r2":
        best_model = model_1 if r2_1 > r2_2 else model_2
    elif metric == "rmse":
        best_model = model_1 if rmse_1 < rmse_2 else model_2
    elif metric == "mae":
        best_model = model_1 if mae_1 < mae_2 else model_2
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return {
        "best_model": best_model,
        "model_1_metrics": {"R2": r2_1, "RMSE": rmse_1, "MAE": mae_1},
        "model_2_metrics": {"R2": r2_2, "RMSE": rmse_2, "MAE": mae_2},
        "winning_model": 1 if best_model is model_1 else 2
    }


@as_function_node("comparison_results")
def CompareClassificationModels(
    model_1: Dict,
    model_2: Dict,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    metric: str = "f1"
) -> Dict[str, Any]:
    """
    Compares two classification models on validation data.

    Parameters:
        model_1: First fitted classification model
        model_2: Second fitted classification model
        X_validation: Validation feature matrix
        y_validation: Validation target values
        metric: Metric for comparison - 'accuracy', 'f1', 'precision', 'recall' (default: 'f1')

    Returns:
        Dictionary containing best model and comparison results
    """
    pred_1 = model_1["model"].predict(X_validation)
    pred_2 = model_2["model"].predict(X_validation)

    acc_1 = accuracy_score(y_validation, pred_1)
    f1_1 = f1_score(y_validation, pred_1, average="weighted", zero_division=0)
    prec_1 = precision_score(y_validation, pred_1, average="weighted", zero_division=0)
    rec_1 = recall_score(y_validation, pred_1, average="weighted", zero_division=0)

    acc_2 = accuracy_score(y_validation, pred_2)
    f1_2 = f1_score(y_validation, pred_2, average="weighted", zero_division=0)
    prec_2 = precision_score(y_validation, pred_2, average="weighted", zero_division=0)
    rec_2 = recall_score(y_validation, pred_2, average="weighted", zero_division=0)

    if metric == "accuracy":
        best_model = model_1 if acc_1 > acc_2 else model_2
    elif metric == "f1":
        best_model = model_1 if f1_1 > f1_2 else model_2
    elif metric == "precision":
        best_model = model_1 if prec_1 > prec_2 else model_2
    elif metric == "recall":
        best_model = model_1 if rec_1 > rec_2 else model_2
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return {
        "best_model": best_model,
        "model_1_metrics": {
            "accuracy": acc_1,
            "f1": f1_1,
            "precision": prec_1,
            "recall": rec_1
        },
        "model_2_metrics": {
            "accuracy": acc_2,
            "f1": f1_2,
            "precision": prec_2,
            "recall": rec_2
        },
        "winning_model": 1 if best_model is model_1 else 2
    }
