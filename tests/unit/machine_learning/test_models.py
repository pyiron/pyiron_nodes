"""
Comprehensive Unit Tests for machine_learning/models.py

This module provides extensive unit tests for all scikit-learn model nodes
in the pyiron_nodes package. Tests cover initialization, training, prediction,
edge cases, and integration with the pyiron workflow system.
"""

import unittest
from typing import Tuple

import numpy as np
import pandas as pd

from pyiron_nodes.machine_learning.models import (
    # Linear models
    LinearRegressionModel,
    RidgeRegressionModel,
    LassoRegressionModel,
    ElasticNetRegressionModel,
    LogisticClassificationModel,
    # Tree models
    DecisionTreeRegressionModel,
    DecisionTreeClassificationModel,
    # Random Forest
    RandomForestRegressionModel,
    RandomForestClassificationModel,
    # Gradient Boosting
    GradientBoostingRegressionModel,
    GradientBoostingClassificationModel,
    # AdaBoost
    AdaBoostRegressionModel,
    AdaBoostClassificationModel,
    # KNeighbors
    KNeighborsRegressionModel,
    KNeighborsClassificationModel,
    # SVM
    SupportVectorClassificationModel,
    SupportVectorRegressionModel,
    # Evaluation
    EvaluateRegressionModelSklearn,
    EvaluateClassificationModelSklearn,
    # Prediction
    PredictRegressionModel,
    PredictClassificationModel,
    # Comparison
    CompareRegressionModels,
    CompareClassificationModels,
)

# =============================================================================
# SYNTHETIC TEST DATA HELPERS
# =============================================================================


def regression_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Generate synthetic regression dataset: (X_train, y_train, X_test, y_test)."""
    np.random.seed(42)
    n_train, n_test, n_features = 50, 20, 5

    X_train = pd.DataFrame(
        np.random.randn(n_train, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    y_train = pd.Series(
        3 * X_train.iloc[:, 0] + 2 * X_train.iloc[:, 1] + np.random.randn(n_train),
        name="target",
    )

    X_test = pd.DataFrame(
        np.random.randn(n_test, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    y_test = pd.Series(
        3 * X_test.iloc[:, 0] + 2 * X_test.iloc[:, 1] + np.random.randn(n_test),
        name="target",
    )

    return X_train, y_train, X_test, y_test


def binary_classification_data() -> (
    Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]
):
    """Generate synthetic binary classification dataset."""
    np.random.seed(42)
    n_train, n_test, n_features = 50, 20, 5

    X_train = pd.DataFrame(
        np.random.randn(n_train, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    y_train = pd.Series(
        (X_train.iloc[:, 0] + X_train.iloc[:, 1] > 0).astype(int), name="target"
    )

    X_test = pd.DataFrame(
        np.random.randn(n_test, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    y_test = pd.Series(
        (X_test.iloc[:, 0] + X_test.iloc[:, 1] > 0).astype(int), name="target"
    )

    return X_train, y_train, X_test, y_test


def multiclass_classification_data() -> (
    Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]
):
    """Generate synthetic multiclass (3-class) classification dataset."""
    np.random.seed(42)
    n_train, n_test, n_features = 60, 25, 5

    X_train = pd.DataFrame(
        np.random.randn(n_train, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    y_train = pd.Series(
        np.digitize(X_train.iloc[:, 0] + X_train.iloc[:, 1], bins=[-1, 1]) - 1,
        name="target",
    )

    X_test = pd.DataFrame(
        np.random.randn(n_test, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    y_test = pd.Series(
        np.digitize(X_test.iloc[:, 0] + X_test.iloc[:, 1], bins=[-1, 1]) - 1,
        name="target",
    )

    return X_train, y_train, X_test, y_test


def empty_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Generate an empty dataset."""
    X_empty = pd.DataFrame(np.empty((0, 5)), columns=[f"feature_{i}" for i in range(5)])
    y_empty = pd.Series([], dtype=float, name="target")
    return X_empty, y_empty


def single_sample_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Generate a single-sample dataset."""
    np.random.seed(42)
    X_single = pd.DataFrame(
        np.random.randn(1, 5), columns=[f"feature_{i}" for i in range(5)]
    )
    y_single = pd.Series([1.5], name="target")
    return X_single, y_single


def trained_linear_model():
    """Fit a LinearRegressionModel for reuse in evaluate/predict tests."""
    X_train, y_train, _, _ = regression_data()
    return LinearRegressionModel._original_func(X_train, y_train)


def trained_rf_classifier():
    """Fit a RandomForestClassificationModel for reuse in evaluate/predict tests."""
    X_train, y_train, _, _ = binary_classification_data()
    return RandomForestClassificationModel._original_func(
        X_train, y_train, n_estimators=10, random_state=42
    )


# =============================================================================
# LINEAR REGRESSION TESTS
# =============================================================================


class TestLinearRegressionModel(unittest.TestCase):
    def test_default_initialization(self):
        X_train, y_train, _, _ = regression_data()
        result = LinearRegressionModel._original_func(X_train, y_train)

        self.assertIsInstance(result, dict)
        self.assertIn("model", result)
        self.assertIn("model_type", result)
        self.assertEqual(result["model_type"], "LinearRegression")
        self.assertIn("coefficients", result)
        self.assertIn("intercept", result)

    def test_custom_parameters(self):
        X_train, y_train, _, _ = regression_data()
        result = LinearRegressionModel._original_func(
            X_train, y_train, fit_intercept=False, copy_X=False
        )

        self.assertEqual(result["model_type"], "LinearRegression")
        self.assertFalse(result["model"].fit_intercept)

    def test_coefficients_shape(self):
        X_train, y_train, _, _ = regression_data()
        result = LinearRegressionModel._original_func(X_train, y_train)

        self.assertEqual(result["coefficients"].shape[0], X_train.shape[1])

    def test_with_empty_data(self):
        X_empty, y_empty = empty_data()

        with self.assertRaises((ValueError, RuntimeError)):
            LinearRegressionModel._original_func(X_empty, y_empty)

    def test_with_single_sample(self):
        X_single, y_single = single_sample_data()
        result = LinearRegressionModel._original_func(X_single, y_single)

        self.assertIsNotNone(result["model"])
        self.assertIn("coefficients", result)


class TestRidgeRegressionModel(unittest.TestCase):
    def test_default_initialization(self):
        X_train, y_train, _, _ = regression_data()
        result = RidgeRegressionModel._original_func(X_train, y_train)

        self.assertEqual(result["model_type"], "Ridge")
        self.assertEqual(result["alpha"], 1.0)
        self.assertIn("coefficients", result)

    def test_custom_alpha(self):
        X_train, y_train, _, _ = regression_data()
        alpha_value = 5.0
        result = RidgeRegressionModel._original_func(
            X_train, y_train, alpha=alpha_value
        )

        self.assertEqual(result["alpha"], alpha_value)
        self.assertEqual(result["model"].alpha, alpha_value)

    def test_custom_solver(self):
        X_train, y_train, _, _ = regression_data()
        result = RidgeRegressionModel._original_func(X_train, y_train, solver="svd")

        self.assertIsNotNone(result["model"])


class TestLassoRegressionModel(unittest.TestCase):
    def test_default_initialization(self):
        X_train, y_train, _, _ = regression_data()
        result = LassoRegressionModel._original_func(X_train, y_train)

        self.assertEqual(result["model_type"], "Lasso")
        self.assertEqual(result["alpha"], 1.0)
        self.assertIn("n_iter", result)

    def test_custom_alpha(self):
        X_train, y_train, _, _ = regression_data()
        result = LassoRegressionModel._original_func(X_train, y_train, alpha=0.5)

        self.assertEqual(result["alpha"], 0.5)
        self.assertEqual(result["model"].alpha, 0.5)

    def test_feature_selection(self):
        X_train, y_train, _, _ = regression_data()
        result = LassoRegressionModel._original_func(X_train, y_train, alpha=10.0)

        coef_zeros = np.sum(result["coefficients"] == 0)
        self.assertGreater(coef_zeros, 0)


class TestElasticNetRegressionModel(unittest.TestCase):
    def test_default_initialization(self):
        X_train, y_train, _, _ = regression_data()
        result = ElasticNetRegressionModel._original_func(X_train, y_train)

        self.assertEqual(result["model_type"], "ElasticNet")
        self.assertEqual(result["alpha"], 1.0)
        self.assertEqual(result["l1_ratio"], 0.5)

    def test_l1_ratio_ridge(self):
        X_train, y_train, _, _ = regression_data()
        result = ElasticNetRegressionModel._original_func(
            X_train, y_train, l1_ratio=0.0
        )

        self.assertEqual(result["l1_ratio"], 0.0)

    def test_l1_ratio_lasso(self):
        X_train, y_train, _, _ = regression_data()
        result = ElasticNetRegressionModel._original_func(
            X_train, y_train, l1_ratio=1.0
        )

        self.assertEqual(result["l1_ratio"], 1.0)


class TestLogisticClassificationModel(unittest.TestCase):
    def test_default_initialization(self):
        X_train, y_train, _, _ = binary_classification_data()
        result = LogisticClassificationModel._original_func(X_train, y_train)

        self.assertEqual(result["model_type"], "LogisticRegression")
        self.assertIn("classes", result)
        self.assertEqual(len(result["classes"]), 2)

    def test_multiclass_support(self):
        X_train, y_train, _, _ = multiclass_classification_data()
        result = LogisticClassificationModel._original_func(
            X_train, y_train, max_iter=200
        )

        self.assertEqual(len(result["classes"]), 3)

    def test_custom_penalty(self):
        X_train, y_train, _, _ = binary_classification_data()
        result = LogisticClassificationModel._original_func(
            X_train, y_train, penalty="l1", solver="liblinear"
        )

        self.assertEqual(result["model"].penalty, "l1")


# =============================================================================
# TREE-BASED MODEL TESTS
# =============================================================================


class TestDecisionTreeRegressionModel(unittest.TestCase):
    def test_default_initialization(self):
        X_train, y_train, _, _ = regression_data()
        result = DecisionTreeRegressionModel._original_func(X_train, y_train)

        self.assertEqual(result["model_type"], "DecisionTreeRegressor")
        self.assertIn("feature_importances", result)
        self.assertIn("tree_depth", result)
        self.assertIn("n_leaves", result)

    def test_max_depth_constraint(self):
        X_train, y_train, _, _ = regression_data()
        result = DecisionTreeRegressionModel._original_func(
            X_train, y_train, max_depth=3
        )

        self.assertLessEqual(result["tree_depth"], 3)

    def test_feature_importances(self):
        X_train, y_train, _, _ = regression_data()
        result = DecisionTreeRegressionModel._original_func(X_train, y_train)

        self.assertEqual(len(result["feature_importances"]), X_train.shape[1])
        self.assertGreater(np.sum(result["feature_importances"]), 0)


class TestDecisionTreeClassificationModel(unittest.TestCase):
    def test_default_initialization(self):
        X_train, y_train, _, _ = binary_classification_data()
        result = DecisionTreeClassificationModel._original_func(X_train, y_train)

        self.assertEqual(result["model_type"], "DecisionTreeClassifier")
        self.assertIn("classes", result)
        self.assertEqual(len(result["classes"]), 2)

    def test_gini_criterion(self):
        X_train, y_train, _, _ = binary_classification_data()
        result = DecisionTreeClassificationModel._original_func(
            X_train, y_train, criterion="gini"
        )

        self.assertEqual(result["model"].criterion, "gini")

    def test_entropy_criterion(self):
        X_train, y_train, _, _ = binary_classification_data()
        result = DecisionTreeClassificationModel._original_func(
            X_train, y_train, criterion="entropy"
        )

        self.assertEqual(result["model"].criterion, "entropy")


# =============================================================================
# RANDOM FOREST TESTS
# =============================================================================


class TestRandomForestRegressionModel(unittest.TestCase):
    def test_default_initialization(self):
        X_train, y_train, _, _ = regression_data()
        result = RandomForestRegressionModel._original_func(X_train, y_train)

        self.assertEqual(result["model_type"], "RandomForestRegressor")
        self.assertEqual(result["n_estimators"], 100)
        self.assertIn("feature_importances", result)

    def test_custom_n_estimators(self):
        X_train, y_train, _, _ = regression_data()
        n_trees = 50
        result = RandomForestRegressionModel._original_func(
            X_train, y_train, n_estimators=n_trees
        )

        self.assertEqual(result["n_estimators"], n_trees)
        self.assertEqual(len(result["model"].estimators_), n_trees)

    def test_feature_importances(self):
        X_train, y_train, _, _ = regression_data()
        result = RandomForestRegressionModel._original_func(
            X_train, y_train, n_estimators=10
        )

        self.assertEqual(len(result["feature_importances"]), X_train.shape[1])


class TestRandomForestClassificationModel(unittest.TestCase):
    def test_default_initialization(self):
        X_train, y_train, _, _ = binary_classification_data()
        result = RandomForestClassificationModel._original_func(X_train, y_train)

        self.assertEqual(result["model_type"], "RandomForestClassifier")
        self.assertIn("classes", result)

    def test_class_weight_balanced(self):
        X_train, y_train, _, _ = binary_classification_data()
        result = RandomForestClassificationModel._original_func(
            X_train, y_train, class_weight="balanced"
        )

        self.assertEqual(result["model"].class_weight, "balanced")


# =============================================================================
# GRADIENT BOOSTING TESTS
# =============================================================================


class TestGradientBoostingRegressionModel(unittest.TestCase):
    def test_default_initialization(self):
        X_train, y_train, _, _ = regression_data()
        result = GradientBoostingRegressionModel._original_func(X_train, y_train)

        self.assertEqual(result["model_type"], "GradientBoostingRegressor")
        self.assertEqual(result["n_estimators"], 100)

    def test_custom_learning_rate(self):
        X_train, y_train, _, _ = regression_data()
        lr = 0.05
        result = GradientBoostingRegressionModel._original_func(
            X_train, y_train, learning_rate=lr
        )

        self.assertEqual(result["model"].learning_rate, lr)

    def test_train_score_attribute(self):
        X_train, y_train, _, _ = regression_data()
        result = GradientBoostingRegressionModel._original_func(
            X_train, y_train, n_estimators=10
        )

        self.assertIn("train_score", result)
        self.assertGreater(len(result["train_score"]), 0)


class TestGradientBoostingClassificationModel(unittest.TestCase):
    def test_default_initialization(self):
        X_train, y_train, _, _ = binary_classification_data()
        result = GradientBoostingClassificationModel._original_func(X_train, y_train)

        self.assertEqual(result["model_type"], "GradientBoostingClassifier")
        self.assertIn("classes", result)


# =============================================================================
# ADABOOST TESTS
# =============================================================================


class TestAdaBoostRegressionModel(unittest.TestCase):
    def test_default_initialization(self):
        X_train, y_train, _, _ = regression_data()
        result = AdaBoostRegressionModel._original_func(X_train, y_train)

        self.assertEqual(result["model_type"], "AdaBoostRegressor")
        self.assertEqual(result["n_estimators"], 50)

    def test_custom_loss(self):
        X_train, y_train, _, _ = regression_data()
        result = AdaBoostRegressionModel._original_func(
            X_train, y_train, loss="square"
        )

        self.assertEqual(result["model"].loss, "square")


class TestAdaBoostClassificationModel(unittest.TestCase):
    def test_default_initialization(self):
        X_train, y_train, _, _ = binary_classification_data()
        result = AdaBoostClassificationModel._original_func(X_train, y_train)

        self.assertEqual(result["model_type"], "AdaBoostClassifier")
        self.assertIn("classes", result)


# =============================================================================
# K-NEIGHBORS TESTS
# =============================================================================


class TestKNeighborsRegressionModel(unittest.TestCase):
    def test_default_initialization(self):
        X_train, y_train, _, _ = regression_data()
        result = KNeighborsRegressionModel._original_func(X_train, y_train)

        self.assertEqual(result["model_type"], "KNeighborsRegressor")
        self.assertEqual(result["n_neighbors"], 5)

    def test_custom_n_neighbors(self):
        X_train, y_train, _, _ = regression_data()
        result = KNeighborsRegressionModel._original_func(
            X_train, y_train, n_neighbors=3
        )

        self.assertEqual(result["n_neighbors"], 3)

    def test_distance_weights(self):
        X_train, y_train, _, _ = regression_data()
        result = KNeighborsRegressionModel._original_func(
            X_train, y_train, weights="distance"
        )

        self.assertEqual(result["model"].weights, "distance")


class TestKNeighborsClassificationModel(unittest.TestCase):
    def test_default_initialization(self):
        X_train, y_train, _, _ = binary_classification_data()
        result = KNeighborsClassificationModel._original_func(X_train, y_train)

        self.assertEqual(result["model_type"], "KNeighborsClassifier")
        self.assertIn("classes", result)


# =============================================================================
# SUPPORT VECTOR MACHINE TESTS
# =============================================================================


class TestSupportVectorClassificationModel(unittest.TestCase):
    def test_default_initialization(self):
        X_train, y_train, _, _ = binary_classification_data()
        result = SupportVectorClassificationModel._original_func(X_train, y_train)

        self.assertEqual(result["model_type"], "SVC")
        self.assertEqual(result["kernel"], "rbf")
        self.assertIn("n_support", result)

    def test_linear_kernel(self):
        X_train, y_train, _, _ = binary_classification_data()
        result = SupportVectorClassificationModel._original_func(
            X_train, y_train, kernel="linear"
        )

        self.assertEqual(result["kernel"], "linear")

    def test_polynomial_kernel(self):
        X_train, y_train, _, _ = binary_classification_data()
        result = SupportVectorClassificationModel._original_func(
            X_train, y_train, kernel="poly", degree=3
        )

        self.assertEqual(result["kernel"], "poly")


class TestSupportVectorRegressionModel(unittest.TestCase):
    def test_default_initialization(self):
        X_train, y_train, _, _ = regression_data()
        result = SupportVectorRegressionModel._original_func(X_train, y_train)

        self.assertEqual(result["model_type"], "SVR")
        self.assertEqual(result["kernel"], "rbf")

    def test_custom_epsilon(self):
        X_train, y_train, _, _ = regression_data()
        epsilon = 0.5
        result = SupportVectorRegressionModel._original_func(
            X_train, y_train, epsilon=epsilon
        )

        self.assertEqual(result["model"].epsilon, epsilon)


# =============================================================================
# EVALUATION NODE TESTS
# =============================================================================


class TestEvaluateRegressionModelSklearn(unittest.TestCase):
    def test_evaluation_output(self):
        _, _, X_test, y_test = regression_data()
        model = trained_linear_model()

        metrics = EvaluateRegressionModelSklearn._original_func(model, X_test, y_test)

        self.assertIsInstance(metrics, dict)
        self.assertIn("R2", metrics)
        self.assertIn("MSE", metrics)
        self.assertIn("MAE", metrics)
        self.assertIn("RMSE", metrics)

    def test_metric_values_valid(self):
        _, _, X_test, y_test = regression_data()
        model = trained_linear_model()

        metrics = EvaluateRegressionModelSklearn._original_func(model, X_test, y_test)

        self.assertIsInstance(metrics["R2"], (int, float))
        self.assertIsInstance(metrics["MSE"], (int, float))
        self.assertIsInstance(metrics["MAE"], (int, float))
        self.assertIsInstance(metrics["RMSE"], (int, float))
        self.assertGreaterEqual(metrics["RMSE"], 0)


class TestEvaluateClassificationModelSklearn(unittest.TestCase):
    def test_evaluation_output(self):
        _, _, X_test, y_test = binary_classification_data()
        model = trained_rf_classifier()

        metrics = EvaluateClassificationModelSklearn._original_func(
            model, X_test, y_test
        )

        self.assertIsInstance(metrics, dict)
        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1_score", metrics)
        self.assertIn("confusion_matrix", metrics)

    def test_metric_ranges(self):
        _, _, X_test, y_test = binary_classification_data()
        model = trained_rf_classifier()

        metrics = EvaluateClassificationModelSklearn._original_func(
            model, X_test, y_test
        )

        self.assertTrue(0 <= metrics["accuracy"] <= 1)
        self.assertTrue(0 <= metrics["precision"] <= 1)
        self.assertTrue(0 <= metrics["recall"] <= 1)
        self.assertTrue(0 <= metrics["f1_score"] <= 1)


# =============================================================================
# PREDICTION NODE TESTS
# =============================================================================


class TestPredictRegressionModel(unittest.TestCase):
    def test_prediction_shape(self):
        _, _, X_test, _ = regression_data()
        model = trained_linear_model()

        predictions = PredictRegressionModel._original_func(model, X_test)

        self.assertEqual(predictions.shape[0], X_test.shape[0])

    def test_prediction_values_numeric(self):
        _, _, X_test, _ = regression_data()
        model = trained_linear_model()

        predictions = PredictRegressionModel._original_func(model, X_test)

        self.assertTrue(np.issubdtype(predictions.dtype, np.number))


class TestPredictClassificationModel(unittest.TestCase):
    def test_prediction_output(self):
        _, _, X_test, _ = binary_classification_data()
        model = trained_rf_classifier()

        result = PredictClassificationModel._original_func(
            model, X_test, return_probabilities=False
        )

        self.assertIsInstance(result, dict)
        self.assertIn("predictions", result)
        self.assertEqual(result["predictions"].shape[0], X_test.shape[0])

    def test_prediction_with_probabilities(self):
        _, _, X_test, _ = binary_classification_data()
        model = trained_rf_classifier()

        result = PredictClassificationModel._original_func(
            model, X_test, return_probabilities=True
        )

        self.assertIn("predictions", result)
        self.assertIn("probabilities", result)
        self.assertEqual(result["probabilities"].shape[0], X_test.shape[0])


# =============================================================================
# COMPARISON NODE TESTS
# =============================================================================


class TestCompareRegressionModels(unittest.TestCase):
    def test_comparison_output(self):
        X_train, y_train, X_test, y_test = regression_data()
        X_val, y_val = X_test, y_test  # Use test as validation for simplicity

        model1 = LinearRegressionModel._original_func(X_train, y_train)
        model2 = RidgeRegressionModel._original_func(X_train, y_train)

        result = CompareRegressionModels._original_func(model1, model2, X_val, y_val)

        self.assertIn("best_model", result)
        self.assertIn("model_1_metrics", result)
        self.assertIn("model_2_metrics", result)
        self.assertIn("winning_model", result)
        self.assertIn(result["winning_model"], [1, 2])

    def test_r2_metric(self):
        X_train, y_train, X_test, y_test = regression_data()
        X_val, y_val = X_test, y_test

        model1 = LinearRegressionModel._original_func(X_train, y_train)
        model2 = RidgeRegressionModel._original_func(X_train, y_train, alpha=10.0)

        result = CompareRegressionModels._original_func(
            model1, model2, X_val, y_val, metric="r2"
        )

        self.assertIn("best_model", result)

    def test_rmse_metric(self):
        X_train, y_train, X_test, y_test = regression_data()
        X_val, y_val = X_test, y_test

        model1 = LinearRegressionModel._original_func(X_train, y_train)
        model2 = RidgeRegressionModel._original_func(X_train, y_train, alpha=10.0)

        result = CompareRegressionModels._original_func(
            model1, model2, X_val, y_val, metric="rmse"
        )

        self.assertIn(result["best_model"], (model1, model2))

    def test_mae_metric(self):
        X_train, y_train, X_test, y_test = regression_data()
        X_val, y_val = X_test, y_test

        model1 = LinearRegressionModel._original_func(X_train, y_train)
        model2 = RidgeRegressionModel._original_func(X_train, y_train, alpha=10.0)

        result = CompareRegressionModels._original_func(
            model1, model2, X_val, y_val, metric="mae"
        )

        self.assertIn(result["best_model"], (model1, model2))

    def test_invalid_metric_raises(self):
        X_train, y_train, X_test, y_test = regression_data()
        X_val, y_val = X_test, y_test

        model1 = LinearRegressionModel._original_func(X_train, y_train)
        model2 = RidgeRegressionModel._original_func(X_train, y_train)

        with self.assertRaises(ValueError):
            CompareRegressionModels._original_func(
                model1, model2, X_val, y_val, metric="bogus"
            )


class TestCompareClassificationModels(unittest.TestCase):
    def test_comparison_output(self):
        X_train, y_train, X_test, y_test = binary_classification_data()
        X_val, y_val = X_test, y_test

        model1 = LogisticClassificationModel._original_func(X_train, y_train)
        model2 = RandomForestClassificationModel._original_func(
            X_train, y_train, n_estimators=10
        )

        result = CompareClassificationModels._original_func(
            model1, model2, X_val, y_val
        )

        self.assertIn("best_model", result)
        self.assertIn("model_1_metrics", result)
        self.assertIn("model_2_metrics", result)
        self.assertIn("winning_model", result)

    def test_f1_metric(self):
        X_train, y_train, X_test, y_test = binary_classification_data()
        X_val, y_val = X_test, y_test

        model1 = LogisticClassificationModel._original_func(X_train, y_train)
        model2 = KNeighborsClassificationModel._original_func(X_train, y_train)

        result = CompareClassificationModels._original_func(
            model1, model2, X_val, y_val, metric="f1"
        )

        self.assertIn("model_1_metrics", result)
        self.assertIn("f1", result["model_1_metrics"])

    def test_accuracy_metric(self):
        X_train, y_train, X_test, y_test = binary_classification_data()
        X_val, y_val = X_test, y_test

        model1 = LogisticClassificationModel._original_func(X_train, y_train)
        model2 = KNeighborsClassificationModel._original_func(X_train, y_train)

        result = CompareClassificationModels._original_func(
            model1, model2, X_val, y_val, metric="accuracy"
        )

        self.assertIn(result["best_model"], (model1, model2))

    def test_precision_metric(self):
        X_train, y_train, X_test, y_test = binary_classification_data()
        X_val, y_val = X_test, y_test

        model1 = LogisticClassificationModel._original_func(X_train, y_train)
        model2 = KNeighborsClassificationModel._original_func(X_train, y_train)

        result = CompareClassificationModels._original_func(
            model1, model2, X_val, y_val, metric="precision"
        )

        self.assertIn(result["best_model"], (model1, model2))

    def test_recall_metric(self):
        X_train, y_train, X_test, y_test = binary_classification_data()
        X_val, y_val = X_test, y_test

        model1 = LogisticClassificationModel._original_func(X_train, y_train)
        model2 = KNeighborsClassificationModel._original_func(X_train, y_train)

        result = CompareClassificationModels._original_func(
            model1, model2, X_val, y_val, metric="recall"
        )

        self.assertIn(result["best_model"], (model1, model2))

    def test_invalid_metric_raises(self):
        X_train, y_train, X_test, y_test = binary_classification_data()
        X_val, y_val = X_test, y_test

        model1 = LogisticClassificationModel._original_func(X_train, y_train)
        model2 = KNeighborsClassificationModel._original_func(X_train, y_train)

        with self.assertRaises(ValueError):
            CompareClassificationModels._original_func(
                model1, model2, X_val, y_val, metric="bogus"
            )


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases(unittest.TestCase):
    def test_mismatched_shapes(self):
        X_train, y_train, _, _ = regression_data()
        y_wrong = pd.Series(np.random.randn(len(y_train) + 5))

        with self.assertRaises((ValueError, RuntimeError)):
            LinearRegressionModel._original_func(X_train, y_wrong)

    def test_non_numeric_features(self):
        X_train, y_train, _, _ = regression_data()
        X_non_numeric = X_train.copy()
        X_non_numeric.iloc[0, 0] = "string"

        with self.assertRaises((TypeError, ValueError)):
            LinearRegressionModel._original_func(X_non_numeric, y_train)

    def test_all_nan_column(self):
        X_train, y_train, _, _ = regression_data()
        X_nan = X_train.copy()
        X_nan.iloc[:, 0] = np.nan

        # Should either handle gracefully or raise appropriate error
        try:
            LinearRegressionModel._original_func(X_nan, y_train)
        except (ValueError, RuntimeError):
            pass  # Expected behavior

    def test_infinity_values(self):
        X_train, y_train, _, _ = regression_data()
        X_inf = X_train.copy()
        X_inf.iloc[0, 0] = np.inf

        try:
            LinearRegressionModel._original_func(X_inf, y_train)
        except (ValueError, RuntimeError):
            pass  # Expected behavior


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration(unittest.TestCase):
    def test_full_regression_workflow(self):
        X_train, y_train, X_test, y_test = regression_data()

        model = LinearRegressionModel._original_func(X_train, y_train)

        metrics = EvaluateRegressionModelSklearn._original_func(model, X_test, y_test)
        self.assertIsNotNone(metrics["R2"])

        predictions = PredictRegressionModel._original_func(model, X_test)
        self.assertEqual(predictions.shape[0], X_test.shape[0])

    def test_full_classification_workflow(self):
        X_train, y_train, X_test, y_test = binary_classification_data()

        model = RandomForestClassificationModel._original_func(
            X_train, y_train, n_estimators=10
        )

        metrics = EvaluateClassificationModelSklearn._original_func(
            model, X_test, y_test
        )
        self.assertIsNotNone(metrics["accuracy"])

        result = PredictClassificationModel._original_func(model, X_test)
        self.assertEqual(result["predictions"].shape[0], X_test.shape[0])

    def test_model_comparison_workflow(self):
        X_train, y_train, X_test, y_test = regression_data()

        model1 = LinearRegressionModel._original_func(X_train, y_train)
        model2 = RidgeRegressionModel._original_func(X_train, y_train)

        comparison = CompareRegressionModels._original_func(
            model1, model2, X_test, y_test
        )
        self.assertIsNotNone(comparison["best_model"])
        self.assertIn(comparison["winning_model"], [1, 2])


# =============================================================================
# PARAMETER VALIDATION TESTS
# =============================================================================


class TestParameterValidation(unittest.TestCase):
    def test_invalid_n_estimators_type(self):
        X_train, y_train, _, _ = regression_data()

        with self.assertRaises((TypeError, ValueError)):
            RandomForestRegressionModel._original_func(
                X_train, y_train, n_estimators="invalid"
            )

    def test_invalid_learning_rate(self):
        X_train, y_train, _, _ = regression_data()

        with self.assertRaises((ValueError, TypeError)):
            GradientBoostingRegressionModel._original_func(
                X_train, y_train, learning_rate=-1.0
            )

    def test_invalid_alpha(self):
        X_train, y_train, _, _ = regression_data()

        with self.assertRaises((ValueError, TypeError)):
            RidgeRegressionModel._original_func(X_train, y_train, alpha=-1.0)


# =============================================================================
# REPRODUCIBILITY TESTS
# =============================================================================


class TestReproducibility(unittest.TestCase):
    def test_random_forest_reproducibility(self):
        X_train, y_train, X_test, y_test = regression_data()

        result1 = RandomForestRegressionModel._original_func(
            X_train, y_train, n_estimators=10, random_state=42
        )
        result2 = RandomForestRegressionModel._original_func(
            X_train, y_train, n_estimators=10, random_state=42
        )

        pred1 = result1["model"].predict(X_test)
        pred2 = result2["model"].predict(X_test)

        np.testing.assert_array_equal(pred1, pred2)

    def test_gradient_boosting_reproducibility(self):
        X_train, y_train, X_test, y_test = regression_data()

        result1 = GradientBoostingRegressionModel._original_func(
            X_train, y_train, random_state=42
        )
        result2 = GradientBoostingRegressionModel._original_func(
            X_train, y_train, random_state=42
        )

        pred1 = result1["model"].predict(X_test)
        pred2 = result2["model"].predict(X_test)

        np.testing.assert_array_equal(pred1, pred2)


if __name__ == "__main__":
    unittest.main()
