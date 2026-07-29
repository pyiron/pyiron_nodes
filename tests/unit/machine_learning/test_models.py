"""
Comprehensive Unit Tests for machine_learning/models.py

This module provides extensive unit tests for all scikit-learn model nodes
in the pyiron_nodes package. Tests cover initialization, training, prediction,
edge cases, and integration with the pyiron workflow system.

Test Structure:
- Fixtures for synthetic data generation
- Parameterized tests for similar node types
- Edge case testing (empty data, single samples, invalid parameters)
- Output structure validation
- Model attribute verification
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from typing import Tuple

# Import all nodes from machine_learning.models
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
# FIXTURES FOR TEST DATA
# =============================================================================


@pytest.fixture
def regression_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Generate synthetic regression dataset.

    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
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


@pytest.fixture
def binary_classification_data() -> (
    Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]
):
    """
    Generate synthetic binary classification dataset.

    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    np.random.seed(42)
    n_train, n_test, n_features = 50, 20, 5

    X_train = pd.DataFrame(
        np.random.randn(n_train, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    # Binary target based on feature sum
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


@pytest.fixture
def multiclass_classification_data() -> (
    Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]
):
    """
    Generate synthetic multiclass classification dataset.

    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    np.random.seed(42)
    n_train, n_test, n_features = 60, 25, 5

    X_train = pd.DataFrame(
        np.random.randn(n_train, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    # Multiclass target (3 classes)
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


@pytest.fixture
def empty_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generate empty dataset.

    Returns:
        Tuple of (X_empty, y_empty)
    """
    X_empty = pd.DataFrame(np.empty((0, 5)), columns=[f"feature_{i}" for i in range(5)])
    y_empty = pd.Series([], dtype=float, name="target")
    return X_empty, y_empty


@pytest.fixture
def single_sample_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generate single sample dataset.

    Returns:
        Tuple of (X_single, y_single)
    """
    np.random.seed(42)
    X_single = pd.DataFrame(
        np.random.randn(1, 5), columns=[f"feature_{i}" for i in range(5)]
    )
    y_single = pd.Series([1.5], name="target")
    return X_single, y_single


@pytest.fixture
def trained_linear_model(regression_data):
    """Provide a fitted LinearRegression model for reuse."""
    X_train, y_train, _, _ = regression_data
    return LinearRegressionModel(X_train, y_train).run()


@pytest.fixture
def trained_rf_classifier(binary_classification_data):
    """Provide a fitted RandomForestClassifier model for reuse."""
    X_train, y_train, _, _ = binary_classification_data
    return RandomForestClassificationModel(
        X_train, y_train, n_estimators=10, random_state=42
    ).run()


# =============================================================================
# LINEAR REGRESSION TESTS
# =============================================================================


class TestLinearRegressionModel:
    """Tests for LinearRegressionModel."""

    def test_default_initialization(self, regression_data):
        """Test LinearRegressionModel with default parameters."""
        X_train, y_train, _, _ = regression_data
        result = LinearRegressionModel(X_train, y_train).run()

        assert isinstance(result, dict)
        assert "model" in result
        assert "model_type" in result
        assert result["model_type"] == "LinearRegression"
        assert "coefficients" in result
        assert "intercept" in result

    def test_custom_parameters(self, regression_data):
        """Test LinearRegressionModel with custom parameters."""
        X_train, y_train, _, _ = regression_data
        result = LinearRegressionModel(
            X_train, y_train, fit_intercept=False, copy_X=False
        ).run()

        assert result["model_type"] == "LinearRegression"
        assert not result["model"].fit_intercept

    def test_coefficients_shape(self, regression_data):
        """Test that coefficients have correct shape."""
        X_train, y_train, _, _ = regression_data
        result = LinearRegressionModel(X_train, y_train).run()

        assert result["coefficients"].shape[0] == X_train.shape[1]

    def test_with_empty_data(self, empty_data):
        """Test that empty data raises appropriate error."""
        X_empty, y_empty = empty_data

        with pytest.raises((ValueError, RuntimeError)):
            LinearRegressionModel(X_empty, y_empty).run()

    def test_with_single_sample(self, single_sample_data):
        """Test LinearRegressionModel with single sample."""
        X_single, y_single = single_sample_data
        result = LinearRegressionModel(X_single, y_single).run()

        assert result["model"] is not None
        assert "coefficients" in result


class TestRidgeRegressionModel:
    """Tests for RidgeRegressionModel."""

    def test_default_initialization(self, regression_data):
        """Test RidgeRegressionModel with default parameters."""
        X_train, y_train, _, _ = regression_data
        result = RidgeRegressionModel(X_train, y_train).run()

        assert result["model_type"] == "Ridge"
        assert result["alpha"] == 1.0
        assert "coefficients" in result

    def test_custom_alpha(self, regression_data):
        """Test RidgeRegressionModel with custom alpha value."""
        X_train, y_train, _, _ = regression_data
        alpha_value = 5.0
        result = RidgeRegressionModel(X_train, y_train, alpha=alpha_value).run()

        assert result["alpha"] == alpha_value
        assert result["model"].alpha == alpha_value

    def test_custom_solver(self, regression_data):
        """Test RidgeRegressionModel with different solvers."""
        X_train, y_train, _, _ = regression_data
        result = RidgeRegressionModel(X_train, y_train, solver="svd").run()

        assert result["model"] is not None


class TestLassoRegressionModel:
    """Tests for LassoRegressionModel."""

    def test_default_initialization(self, regression_data):
        """Test LassoRegressionModel with default parameters."""
        X_train, y_train, _, _ = regression_data
        result = LassoRegressionModel(X_train, y_train).run()

        assert result["model_type"] == "Lasso"
        assert result["alpha"] == 1.0
        assert "n_iter" in result

    def test_custom_alpha(self, regression_data):
        """Test LassoRegressionModel with custom alpha."""
        X_train, y_train, _, _ = regression_data
        result = LassoRegressionModel(X_train, y_train, alpha=0.5).run()

        assert result["alpha"] == 0.5
        assert result["model"].alpha == 0.5

    def test_feature_selection(self, regression_data):
        """Test that Lasso performs feature selection (zero coefficients)."""
        X_train, y_train, _, _ = regression_data
        result = LassoRegressionModel(X_train, y_train, alpha=10.0).run()

        # High alpha should result in some zero coefficients
        coef_zeros = np.sum(result["coefficients"] == 0)
        assert coef_zeros > 0


class TestElasticNetRegressionModel:
    """Tests for ElasticNetRegressionModel."""

    def test_default_initialization(self, regression_data):
        """Test ElasticNetRegressionModel with default parameters."""
        X_train, y_train, _, _ = regression_data
        result = ElasticNetRegressionModel(X_train, y_train).run()

        assert result["model_type"] == "ElasticNet"
        assert result["alpha"] == 1.0
        assert result["l1_ratio"] == 0.5

    def test_l1_ratio_ridge(self, regression_data):
        """Test ElasticNetRegressionModel with l1_ratio=0 (Ridge-like)."""
        X_train, y_train, _, _ = regression_data
        result = ElasticNetRegressionModel(X_train, y_train, l1_ratio=0.0).run()

        assert result["l1_ratio"] == 0.0

    def test_l1_ratio_lasso(self, regression_data):
        """Test ElasticNetRegressionModel with l1_ratio=1 (Lasso-like)."""
        X_train, y_train, _, _ = regression_data
        result = ElasticNetRegressionModel(X_train, y_train, l1_ratio=1.0).run()

        assert result["l1_ratio"] == 1.0


class TestLogisticClassificationModel:
    """Tests for LogisticClassificationModel."""

    def test_default_initialization(self, binary_classification_data):
        """Test LogisticClassificationModel with default parameters."""
        X_train, y_train, _, _ = binary_classification_data
        result = LogisticClassificationModel(X_train, y_train).run()

        assert result["model_type"] == "LogisticRegression"
        assert "classes" in result
        assert len(result["classes"]) == 2

    def test_multiclass_support(self, multiclass_classification_data):
        """Test LogisticClassificationModel with multiclass data."""
        X_train, y_train, _, _ = multiclass_classification_data
        result = LogisticClassificationModel(X_train, y_train, max_iter=200).run()

        assert len(result["classes"]) == 3

    def test_custom_penalty(self, binary_classification_data):
        """Test LogisticClassificationModel with custom penalty."""
        X_train, y_train, _, _ = binary_classification_data
        result = LogisticClassificationModel(
            X_train, y_train, penalty="l1", solver="liblinear"
        ).run()

        assert result["model"].penalty == "l1"


# =============================================================================
# TREE-BASED MODEL TESTS
# =============================================================================


class TestDecisionTreeRegressionModel:
    """Tests for DecisionTreeRegressionModel."""

    def test_default_initialization(self, regression_data):
        """Test DecisionTreeRegressionModel with default parameters."""
        X_train, y_train, _, _ = regression_data
        result = DecisionTreeRegressionModel(X_train, y_train).run()

        assert result["model_type"] == "DecisionTreeRegressor"
        assert "feature_importances" in result
        assert "tree_depth" in result
        assert "n_leaves" in result

    def test_max_depth_constraint(self, regression_data):
        """Test DecisionTreeRegressionModel with max_depth constraint."""
        X_train, y_train, _, _ = regression_data
        result = DecisionTreeRegressionModel(X_train, y_train, max_depth=3).run()

        assert result["tree_depth"] <= 3

    def test_feature_importances(self, regression_data):
        """Test that feature importances are calculated."""
        X_train, y_train, _, _ = regression_data
        result = DecisionTreeRegressionModel(X_train, y_train).run()

        assert len(result["feature_importances"]) == X_train.shape[1]
        assert np.sum(result["feature_importances"]) > 0


class TestDecisionTreeClassificationModel:
    """Tests for DecisionTreeClassificationModel."""

    def test_default_initialization(self, binary_classification_data):
        """Test DecisionTreeClassificationModel with default parameters."""
        X_train, y_train, _, _ = binary_classification_data
        result = DecisionTreeClassificationModel(X_train, y_train).run()

        assert result["model_type"] == "DecisionTreeClassifier"
        assert "classes" in result
        assert len(result["classes"]) == 2

    def test_gini_criterion(self, binary_classification_data):
        """Test DecisionTreeClassificationModel with gini criterion."""
        X_train, y_train, _, _ = binary_classification_data
        result = DecisionTreeClassificationModel(
            X_train, y_train, criterion="gini"
        ).run()

        assert result["model"].criterion == "gini"

    def test_entropy_criterion(self, binary_classification_data):
        """Test DecisionTreeClassificationModel with entropy criterion."""
        X_train, y_train, _, _ = binary_classification_data
        result = DecisionTreeClassificationModel(
            X_train, y_train, criterion="entropy"
        ).run()

        assert result["model"].criterion == "entropy"


# =============================================================================
# RANDOM FOREST TESTS
# =============================================================================


class TestRandomForestRegressionModel:
    """Tests for RandomForestRegressionModel."""

    def test_default_initialization(self, regression_data):
        """Test RandomForestRegressionModel with default parameters."""
        X_train, y_train, _, _ = regression_data
        result = RandomForestRegressionModel(X_train, y_train).run()

        assert result["model_type"] == "RandomForestRegressor"
        assert result["n_estimators"] == 100
        assert "feature_importances" in result

    def test_custom_n_estimators(self, regression_data):
        """Test RandomForestRegressionModel with custom n_estimators."""
        X_train, y_train, _, _ = regression_data
        n_trees = 50
        result = RandomForestRegressionModel(
            X_train, y_train, n_estimators=n_trees
        ).run()

        assert result["n_estimators"] == n_trees
        assert len(result["model"].estimators_) == n_trees

    def test_feature_importances(self, regression_data):
        """Test RandomForestRegressionModel feature importances."""
        X_train, y_train, _, _ = regression_data
        result = RandomForestRegressionModel(X_train, y_train, n_estimators=10).run()

        assert len(result["feature_importances"]) == X_train.shape[1]


class TestRandomForestClassificationModel:
    """Tests for RandomForestClassificationModel."""

    def test_default_initialization(self, binary_classification_data):
        """Test RandomForestClassificationModel with default parameters."""
        X_train, y_train, _, _ = binary_classification_data
        result = RandomForestClassificationModel(X_train, y_train).run()

        assert result["model_type"] == "RandomForestClassifier"
        assert "classes" in result

    def test_class_weight_balanced(self, binary_classification_data):
        """Test RandomForestClassificationModel with balanced class weights."""
        X_train, y_train, _, _ = binary_classification_data
        result = RandomForestClassificationModel(
            X_train, y_train, class_weight="balanced"
        ).run()

        assert result["model"].class_weight == "balanced"


# =============================================================================
# GRADIENT BOOSTING TESTS
# =============================================================================


class TestGradientBoostingRegressionModel:
    """Tests for GradientBoostingRegressionModel."""

    def test_default_initialization(self, regression_data):
        """Test GradientBoostingRegressionModel with default parameters."""
        X_train, y_train, _, _ = regression_data
        result = GradientBoostingRegressionModel(X_train, y_train).run()

        assert result["model_type"] == "GradientBoostingRegressor"
        assert result["n_estimators"] == 100

    def test_custom_learning_rate(self, regression_data):
        """Test GradientBoostingRegressionModel with custom learning_rate."""
        X_train, y_train, _, _ = regression_data
        lr = 0.05
        result = GradientBoostingRegressionModel(
            X_train, y_train, learning_rate=lr
        ).run()

        assert result["model"].learning_rate == lr

    def test_train_score_attribute(self, regression_data):
        """Test that train_score_ is available."""
        X_train, y_train, _, _ = regression_data
        result = GradientBoostingRegressionModel(
            X_train, y_train, n_estimators=10
        ).run()

        assert "train_score" in result
        assert len(result["train_score"]) > 0


class TestGradientBoostingClassificationModel:
    """Tests for GradientBoostingClassificationModel."""

    def test_default_initialization(self, binary_classification_data):
        """Test GradientBoostingClassificationModel with default parameters."""
        X_train, y_train, _, _ = binary_classification_data
        result = GradientBoostingClassificationModel(X_train, y_train).run()

        assert result["model_type"] == "GradientBoostingClassifier"
        assert "classes" in result


# =============================================================================
# ADABOOST TESTS
# =============================================================================


class TestAdaBoostRegressionModel:
    """Tests for AdaBoostRegressionModel."""

    def test_default_initialization(self, regression_data):
        """Test AdaBoostRegressionModel with default parameters."""
        X_train, y_train, _, _ = regression_data
        result = AdaBoostRegressionModel(X_train, y_train).run()

        assert result["model_type"] == "AdaBoostRegressor"
        assert result["n_estimators"] == 50

    def test_custom_loss(self, regression_data):
        """Test AdaBoostRegressionModel with custom loss function."""
        X_train, y_train, _, _ = regression_data
        result = AdaBoostRegressionModel(X_train, y_train, loss="square").run()

        assert result["model"].loss == "square"


class TestAdaBoostClassificationModel:
    """Tests for AdaBoostClassificationModel."""

    def test_default_initialization(self, binary_classification_data):
        """Test AdaBoostClassificationModel with default parameters."""
        X_train, y_train, _, _ = binary_classification_data
        result = AdaBoostClassificationModel(X_train, y_train).run()

        assert result["model_type"] == "AdaBoostClassifier"
        assert "classes" in result


# =============================================================================
# K-NEIGHBORS TESTS
# =============================================================================


class TestKNeighborsRegressionModel:
    """Tests for KNeighborsRegressionModel."""

    def test_default_initialization(self, regression_data):
        """Test KNeighborsRegressionModel with default parameters."""
        X_train, y_train, _, _ = regression_data
        result = KNeighborsRegressionModel(X_train, y_train).run()

        assert result["model_type"] == "KNeighborsRegressor"
        assert result["n_neighbors"] == 5

    def test_custom_n_neighbors(self, regression_data):
        """Test KNeighborsRegressionModel with custom n_neighbors."""
        X_train, y_train, _, _ = regression_data
        result = KNeighborsRegressionModel(X_train, y_train, n_neighbors=3).run()

        assert result["n_neighbors"] == 3

    def test_distance_weights(self, regression_data):
        """Test KNeighborsRegressionModel with distance weights."""
        X_train, y_train, _, _ = regression_data
        result = KNeighborsRegressionModel(X_train, y_train, weights="distance").run()

        assert result["model"].weights == "distance"


class TestKNeighborsClassificationModel:
    """Tests for KNeighborsClassificationModel."""

    def test_default_initialization(self, binary_classification_data):
        """Test KNeighborsClassificationModel with default parameters."""
        X_train, y_train, _, _ = binary_classification_data
        result = KNeighborsClassificationModel(X_train, y_train).run()

        assert result["model_type"] == "KNeighborsClassifier"
        assert "classes" in result


# =============================================================================
# SUPPORT VECTOR MACHINE TESTS
# =============================================================================


class TestSupportVectorClassificationModel:
    """Tests for SupportVectorClassificationModel."""

    def test_default_initialization(self, binary_classification_data):
        """Test SupportVectorClassificationModel with default parameters."""
        X_train, y_train, _, _ = binary_classification_data
        result = SupportVectorClassificationModel(X_train, y_train).run()

        assert result["model_type"] == "SVC"
        assert result["kernel"] == "rbf"
        assert "n_support" in result

    def test_linear_kernel(self, binary_classification_data):
        """Test SupportVectorClassificationModel with linear kernel."""
        X_train, y_train, _, _ = binary_classification_data
        result = SupportVectorClassificationModel(
            X_train, y_train, kernel="linear"
        ).run()

        assert result["kernel"] == "linear"

    def test_polynomial_kernel(self, binary_classification_data):
        """Test SupportVectorClassificationModel with polynomial kernel."""
        X_train, y_train, _, _ = binary_classification_data
        result = SupportVectorClassificationModel(
            X_train, y_train, kernel="poly", degree=3
        ).run()

        assert result["kernel"] == "poly"


class TestSupportVectorRegressionModel:
    """Tests for SupportVectorRegressionModel."""

    def test_default_initialization(self, regression_data):
        """Test SupportVectorRegressionModel with default parameters."""
        X_train, y_train, _, _ = regression_data
        result = SupportVectorRegressionModel(X_train, y_train).run()

        assert result["model_type"] == "SVR"
        assert result["kernel"] == "rbf"

    def test_custom_epsilon(self, regression_data):
        """Test SupportVectorRegressionModel with custom epsilon."""
        X_train, y_train, _, _ = regression_data
        epsilon = 0.5
        result = SupportVectorRegressionModel(X_train, y_train, epsilon=epsilon).run()

        assert result["model"].epsilon == epsilon


# =============================================================================
# EVALUATION NODE TESTS
# =============================================================================


class TestEvaluateRegressionModelSklearn:
    """Tests for EvaluateRegressionModelSklearn."""

    def test_evaluation_output(self, regression_data, trained_linear_model):
        """Test that evaluation returns all required metrics."""
        _, _, X_test, y_test = regression_data
        model_result = trained_linear_model
        model = model_result

        metrics = EvaluateRegressionModelSklearn(model, X_test, y_test).run()

        assert isinstance(metrics, dict)
        assert "R2" in metrics
        assert "MSE" in metrics
        assert "MAE" in metrics
        assert "RMSE" in metrics

    def test_metric_values_valid(self, regression_data, trained_linear_model):
        """Test that metric values are valid numbers."""
        _, _, X_test, y_test = regression_data
        model_result = trained_linear_model
        model = model_result

        metrics = EvaluateRegressionModelSklearn(model, X_test, y_test).run()

        assert isinstance(metrics["R2"], (int, float))
        assert isinstance(metrics["MSE"], (int, float))
        assert isinstance(metrics["MAE"], (int, float))
        assert isinstance(metrics["RMSE"], (int, float))
        assert metrics["RMSE"] >= 0


class TestEvaluateClassificationModelSklearn:
    """Tests for EvaluateClassificationModelSklearn."""

    def test_evaluation_output(self, binary_classification_data, trained_rf_classifier):
        """Test that classification evaluation returns all required metrics."""
        _, _, X_test, y_test = binary_classification_data
        model_result = trained_rf_classifier
        model = model_result

        metrics = EvaluateClassificationModelSklearn(model, X_test, y_test).run()

        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "confusion_matrix" in metrics

    def test_metric_ranges(self, binary_classification_data, trained_rf_classifier):
        """Test that classification metrics are in valid ranges."""
        _, _, X_test, y_test = binary_classification_data
        model_result = trained_rf_classifier
        model = model_result

        metrics = EvaluateClassificationModelSklearn(model, X_test, y_test).run()

        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1_score"] <= 1


# =============================================================================
# PREDICTION NODE TESTS
# =============================================================================


class TestPredictRegressionModel:
    """Tests for PredictRegressionModel."""

    def test_prediction_shape(self, regression_data, trained_linear_model):
        """Test that predictions have correct shape."""
        _, _, X_test, _ = regression_data
        model_result = trained_linear_model
        model = model_result

        predictions = PredictRegressionModel(model, X_test).run()

        assert predictions.shape[0] == X_test.shape[0]

    def test_prediction_values_numeric(self, regression_data, trained_linear_model):
        """Test that predictions are numeric."""
        _, _, X_test, _ = regression_data
        model_result = trained_linear_model
        model = model_result

        predictions = PredictRegressionModel(model, X_test).run()

        assert np.issubdtype(predictions.dtype, np.number)


class TestPredictClassificationModel:
    """Tests for PredictClassificationModel."""

    def test_prediction_output(self, binary_classification_data, trained_rf_classifier):
        """Test that classification predictions have correct output."""
        _, _, X_test, _ = binary_classification_data
        model_result = trained_rf_classifier
        model = model_result

        result = PredictClassificationModel(
            model, X_test, return_probabilities=False
        ).run()

        assert isinstance(result, dict)
        assert "predictions" in result
        assert result["predictions"].shape[0] == X_test.shape[0]

    def test_prediction_with_probabilities(
        self, binary_classification_data, trained_rf_classifier
    ):
        """Test classification predictions with probability estimates."""
        _, _, X_test, _ = binary_classification_data
        model_result = trained_rf_classifier
        model = model_result

        result = PredictClassificationModel(
            model, X_test, return_probabilities=True
        ).run()

        assert "predictions" in result
        assert "probabilities" in result
        assert result["probabilities"].shape[0] == X_test.shape[0]


# =============================================================================
# COMPARISON NODE TESTS
# =============================================================================


class TestCompareRegressionModels:
    """Tests for CompareRegressionModels."""

    def test_comparison_output(self, regression_data):
        """Test that model comparison returns valid results."""
        X_train, y_train, X_test, y_test = regression_data
        X_val, y_val = X_test, y_test  # Use test as validation for simplicity

        model1_result = LinearRegressionModel(X_train, y_train).run()
        model2_result = RidgeRegressionModel(X_train, y_train).run()
        model1 = model1_result
        model2 = model2_result

        result = CompareRegressionModels(model1, model2, X_val, y_val).run()

        assert "best_model" in result
        assert "model_1_metrics" in result
        assert "model_2_metrics" in result
        assert "winning_model" in result
        assert result["winning_model"] in [1, 2]

    def test_r2_metric(self, regression_data):
        """Test comparison using R2 metric."""
        X_train, y_train, X_test, y_test = regression_data
        X_val, y_val = X_test, y_test

        model1_result = LinearRegressionModel(X_train, y_train).run()
        model2_result = RidgeRegressionModel(X_train, y_train, alpha=10.0).run()
        model1 = model1_result
        model2 = model2_result

        result = CompareRegressionModels(
            model1, model2, X_val, y_val, metric="r2"
        ).run()

        assert "best_model" in result

    def test_rmse_metric(self, regression_data):
        """Test comparison using RMSE metric."""
        X_train, y_train, X_test, y_test = regression_data
        X_val, y_val = X_test, y_test

        model1 = LinearRegressionModel(X_train, y_train).run()
        model2 = RidgeRegressionModel(X_train, y_train, alpha=10.0).run()

        result = CompareRegressionModels(
            model1, model2, X_val, y_val, metric="rmse"
        ).run()

        assert result["best_model"] in (model1, model2)

    def test_mae_metric(self, regression_data):
        """Test comparison using MAE metric."""
        X_train, y_train, X_test, y_test = regression_data
        X_val, y_val = X_test, y_test

        model1 = LinearRegressionModel(X_train, y_train).run()
        model2 = RidgeRegressionModel(X_train, y_train, alpha=10.0).run()

        result = CompareRegressionModels(
            model1, model2, X_val, y_val, metric="mae"
        ).run()

        assert result["best_model"] in (model1, model2)

    def test_invalid_metric_raises(self, regression_data):
        """Test that an unknown metric raises ValueError."""
        X_train, y_train, X_test, y_test = regression_data
        X_val, y_val = X_test, y_test

        model1 = LinearRegressionModel(X_train, y_train).run()
        model2 = RidgeRegressionModel(X_train, y_train).run()

        with pytest.raises(ValueError):
            CompareRegressionModels(model1, model2, X_val, y_val, metric="bogus").run()


class TestCompareClassificationModels:
    """Tests for CompareClassificationModels."""

    def test_comparison_output(self, binary_classification_data):
        """Test that classification model comparison returns valid results."""
        X_train, y_train, X_test, y_test = binary_classification_data
        X_val, y_val = X_test, y_test

        model1_result = LogisticClassificationModel(X_train, y_train).run()
        model2_result = RandomForestClassificationModel(
            X_train, y_train, n_estimators=10
        ).run()
        model1 = model1_result
        model2 = model2_result

        result = CompareClassificationModels(model1, model2, X_val, y_val).run()

        assert "best_model" in result
        assert "model_1_metrics" in result
        assert "model_2_metrics" in result
        assert "winning_model" in result

    def test_f1_metric(self, binary_classification_data):
        """Test comparison using F1 metric."""
        X_train, y_train, X_test, y_test = binary_classification_data
        X_val, y_val = X_test, y_test

        model1_result = LogisticClassificationModel(X_train, y_train).run()
        model2_result = KNeighborsClassificationModel(X_train, y_train).run()
        model1 = model1_result
        model2 = model2_result

        result = CompareClassificationModels(
            model1, model2, X_val, y_val, metric="f1"
        ).run()

        assert "model_1_metrics" in result
        assert "f1" in result["model_1_metrics"]

    def test_accuracy_metric(self, binary_classification_data):
        """Test comparison using accuracy metric."""
        X_train, y_train, X_test, y_test = binary_classification_data
        X_val, y_val = X_test, y_test

        model1 = LogisticClassificationModel(X_train, y_train).run()
        model2 = KNeighborsClassificationModel(X_train, y_train).run()

        result = CompareClassificationModels(
            model1, model2, X_val, y_val, metric="accuracy"
        ).run()

        assert result["best_model"] in (model1, model2)

    def test_precision_metric(self, binary_classification_data):
        """Test comparison using precision metric."""
        X_train, y_train, X_test, y_test = binary_classification_data
        X_val, y_val = X_test, y_test

        model1 = LogisticClassificationModel(X_train, y_train).run()
        model2 = KNeighborsClassificationModel(X_train, y_train).run()

        result = CompareClassificationModels(
            model1, model2, X_val, y_val, metric="precision"
        ).run()

        assert result["best_model"] in (model1, model2)

    def test_recall_metric(self, binary_classification_data):
        """Test comparison using recall metric."""
        X_train, y_train, X_test, y_test = binary_classification_data
        X_val, y_val = X_test, y_test

        model1 = LogisticClassificationModel(X_train, y_train).run()
        model2 = KNeighborsClassificationModel(X_train, y_train).run()

        result = CompareClassificationModels(
            model1, model2, X_val, y_val, metric="recall"
        ).run()

        assert result["best_model"] in (model1, model2)

    def test_invalid_metric_raises(self, binary_classification_data):
        """Test that an unknown metric raises ValueError."""
        X_train, y_train, X_test, y_test = binary_classification_data
        X_val, y_val = X_test, y_test

        model1 = LogisticClassificationModel(X_train, y_train).run()
        model2 = KNeighborsClassificationModel(X_train, y_train).run()

        with pytest.raises(ValueError):
            CompareClassificationModels(
                model1, model2, X_val, y_val, metric="bogus"
            ).run()


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_mismatched_shapes(self, regression_data):
        """Test that mismatched X and y shapes raise errors."""
        X_train, y_train, _, _ = regression_data
        y_wrong = pd.Series(np.random.randn(len(y_train) + 5))

        with pytest.raises((ValueError, RuntimeError)):
            LinearRegressionModel(X_train, y_wrong).run()

    def test_non_numeric_features(self, regression_data):
        """Test that non-numeric features raise appropriate errors."""
        X_train, y_train, _, _ = regression_data
        X_non_numeric = X_train.copy()
        X_non_numeric.iloc[0, 0] = "string"

        with pytest.raises((TypeError, ValueError)):
            LinearRegressionModel(X_non_numeric, y_train).run()

    def test_all_nan_column(self, regression_data):
        """Test that all-NaN columns are handled."""
        X_train, y_train, _, _ = regression_data
        X_nan = X_train.copy()
        X_nan.iloc[:, 0] = np.nan

        # Should either handle gracefully or raise appropriate error
        try:
            LinearRegressionModel(X_nan, y_train).run()
        except (ValueError, RuntimeError):
            pass  # Expected behavior

    def test_infinity_values(self, regression_data):
        """Test that infinity values are handled."""
        X_train, y_train, _, _ = regression_data
        X_inf = X_train.copy()
        X_inf.iloc[0, 0] = np.inf

        try:
            LinearRegressionModel(X_inf, y_train).run()
        except (ValueError, RuntimeError):
            pass  # Expected behavior


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_regression_workflow(self, regression_data):
        """Test complete regression workflow: train, evaluate, predict."""
        X_train, y_train, X_test, y_test = regression_data

        # Train model
        model_result = LinearRegressionModel(X_train, y_train).run()
        model = model_result

        # Evaluate model
        metrics = EvaluateRegressionModelSklearn(model, X_test, y_test).run()
        assert metrics["R2"] is not None

        # Make predictions
        predictions = PredictRegressionModel(model, X_test).run()
        assert predictions.shape[0] == X_test.shape[0]

    def test_full_classification_workflow(self, binary_classification_data):
        """Test complete classification workflow: train, evaluate, predict."""
        X_train, y_train, X_test, y_test = binary_classification_data

        # Train model
        model_result = RandomForestClassificationModel(
            X_train, y_train, n_estimators=10
        ).run()
        model = model_result

        # Evaluate model
        metrics = EvaluateClassificationModelSklearn(model, X_test, y_test).run()
        assert metrics["accuracy"] is not None

        # Make predictions
        result = PredictClassificationModel(model, X_test).run()
        assert result["predictions"].shape[0] == X_test.shape[0]

    def test_model_comparison_workflow(self, regression_data):
        """Test model comparison workflow."""
        X_train, y_train, X_test, y_test = regression_data

        # Train two models
        model1_result = LinearRegressionModel(X_train, y_train).run()
        model2_result = RidgeRegressionModel(X_train, y_train).run()
        model1 = model1_result
        model2 = model2_result

        # Compare models
        comparison = CompareRegressionModels(model1, model2, X_test, y_test).run()
        assert comparison["best_model"] is not None
        assert comparison["winning_model"] in [1, 2]


# =============================================================================
# PARAMETER VALIDATION TESTS
# =============================================================================


class TestParameterValidation:
    """Tests for parameter validation and type checking."""

    def test_invalid_n_estimators_type(self, regression_data):
        """Test that invalid n_estimators type is handled."""
        X_train, y_train, _, _ = regression_data

        with pytest.raises((TypeError, ValueError)):
            RandomForestRegressionModel(X_train, y_train, n_estimators="invalid").run()

    def test_invalid_learning_rate(self, regression_data):
        """Test that invalid learning_rate values are handled."""
        X_train, y_train, _, _ = regression_data

        with pytest.raises((ValueError, TypeError)):
            GradientBoostingRegressionModel(X_train, y_train, learning_rate=-1.0).run()

    def test_invalid_alpha(self, regression_data):
        """Test that invalid alpha values are handled."""
        X_train, y_train, _, _ = regression_data

        with pytest.raises((ValueError, TypeError)):
            RidgeRegressionModel(X_train, y_train, alpha=-1.0).run()


# =============================================================================
# REPRODUCIBILITY TESTS
# =============================================================================


class TestReproducibility:
    """Tests for reproducibility with random_state."""

    def test_random_forest_reproducibility(self, regression_data):
        """Test that RandomForest with same seed produces same results."""
        X_train, y_train, X_test, y_test = regression_data

        result1 = RandomForestRegressionModel(
            X_train, y_train, n_estimators=10, random_state=42
        ).run()
        result2 = RandomForestRegressionModel(
            X_train, y_train, n_estimators=10, random_state=42
        ).run()

        pred1 = result1["model"].predict(X_test)
        pred2 = result2["model"].predict(X_test)

        np.testing.assert_array_equal(pred1, pred2)

    def test_gradient_boosting_reproducibility(self, regression_data):
        """Test that GradientBoosting with same seed produces same results."""
        X_train, y_train, X_test, y_test = regression_data

        result1 = GradientBoostingRegressionModel(
            X_train, y_train, random_state=42
        ).run()
        result2 = GradientBoostingRegressionModel(
            X_train, y_train, random_state=42
        ).run()

        pred1 = result1["model"].predict(X_test)
        pred2 = result2["model"].predict(X_test)

        np.testing.assert_array_equal(pred1, pred2)


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
