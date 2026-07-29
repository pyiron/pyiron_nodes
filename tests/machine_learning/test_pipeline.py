"""
Unit tests for machine_learning/pipeline.py
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from pyiron_nodes.machine_learning.pipeline import (
    ChooseBestModel,
    EvaluateRegressionModel,
    MLDataSplitter,
    TrainRegressor,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def splitter_df() -> pd.DataFrame:
    """Dataframe with numeric features, a non-numeric column, and a target."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame(
        {
            "feature_0": np.random.randn(n),
            "feature_1": np.random.randn(n),
            "category": ["group_a"] * n,
            "target": np.random.randn(n),
        }
    )


@pytest.fixture
def regression_data():
    """(X_train, y_train, X_test, y_test) for a simple linear relationship."""
    np.random.seed(42)
    n_train, n_test, n_features = 50, 20, 3

    X_train = pd.DataFrame(
        np.random.randn(n_train, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    y_train = pd.Series(
        3 * X_train.iloc[:, 0] + 2 * X_train.iloc[:, 1] + np.random.randn(n_train) * 0.1,
        name="target",
    )
    X_test = pd.DataFrame(
        np.random.randn(n_test, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    y_test = pd.Series(
        3 * X_test.iloc[:, 0] + 2 * X_test.iloc[:, 1] + np.random.randn(n_test) * 0.1,
        name="target",
    )
    return X_train, y_train, X_test, y_test


# =============================================================================
# MLDataSplitter
# =============================================================================


class TestMLDataSplitter:
    def test_default_split_shapes(self, splitter_df):
        X_train, X_val, X_test, y_train, y_val, y_test = MLDataSplitter(
            splitter_df, "target"
        ).run()

        assert len(X_train) + len(X_val) + len(X_test) == len(splitter_df)
        assert len(X_train) == len(y_train)
        assert len(X_val) == len(y_val)
        assert len(X_test) == len(y_test)

    def test_non_numeric_columns_dropped(self, splitter_df):
        X_train, X_val, X_test, y_train, y_val, y_test = MLDataSplitter(
            splitter_df, "target"
        ).run()

        assert set(X_train.columns) == {"feature_0", "feature_1"}

    def test_rows_with_missing_values_dropped(self, splitter_df):
        df_with_nan = splitter_df.copy()
        df_with_nan.loc[0, "feature_0"] = np.nan

        X_train, X_val, X_test, y_train, y_val, y_test = MLDataSplitter(
            df_with_nan, "target"
        ).run()

        assert len(X_train) + len(X_val) + len(X_test) == len(splitter_df) - 1

    def test_custom_fractions(self, splitter_df):
        X_train, X_val, X_test, y_train, y_val, y_test = MLDataSplitter(
            splitter_df,
            "target",
            train_fraction=0.5,
            validation_fraction=0.3,
            test_fraction=0.2,
        ).run()

        n = len(splitter_df)
        assert len(X_train) == pytest.approx(0.5 * n, abs=2)
        assert len(X_val) == pytest.approx(0.3 * n, abs=2)
        assert len(X_test) == pytest.approx(0.2 * n, abs=2)

    def test_invalid_fractions_raise(self, splitter_df):
        with pytest.raises(ValueError):
            MLDataSplitter(
                splitter_df,
                "target",
                train_fraction=0.5,
                validation_fraction=0.3,
                test_fraction=0.3,
            ).run()

    def test_reproducible_with_random_state(self, splitter_df):
        result_1 = MLDataSplitter(splitter_df, "target", random_state=1).run()
        result_2 = MLDataSplitter(splitter_df, "target", random_state=1).run()

        pd.testing.assert_frame_equal(result_1[0], result_2[0])


# =============================================================================
# TrainRegressor
# =============================================================================


class TestTrainRegressor:
    def test_linear(self, regression_data):
        X_train, y_train, _, _ = regression_data
        reg = TrainRegressor(X_train, y_train, r_type="linear").run()

        assert isinstance(reg, LinearRegression)

    def test_tree(self, regression_data):
        X_train, y_train, _, _ = regression_data
        reg = TrainRegressor(X_train, y_train, r_type="tree").run()

        assert isinstance(reg, RandomForestRegressor)

    def test_default_r_type_raises(self, regression_data):
        X_train, y_train, _, _ = regression_data
        with pytest.raises(ValueError):
            TrainRegressor(X_train, y_train).run()

    def test_invalid_r_type_raises(self, regression_data):
        X_train, y_train, _, _ = regression_data
        with pytest.raises(ValueError):
            TrainRegressor(X_train, y_train, r_type="invalid").run()


# =============================================================================
# EvaluateRegressionModel
# =============================================================================


class TestEvaluateRegressionModel:
    def test_output_keys(self, regression_data):
        X_train, y_train, X_test, y_test = regression_data
        reg = TrainRegressor(X_train, y_train, r_type="linear").run()

        metrics = EvaluateRegressionModel(reg, X_test, y_test).run()

        assert set(metrics) == {"R2", "MSE", "MAE"}

    def test_perfect_fit_gives_r2_of_one(self):
        X = pd.DataFrame({"x": np.arange(20, dtype=float)})
        y = pd.Series(2 * X["x"] + 1)
        reg = LinearRegression().fit(X, y)

        metrics = EvaluateRegressionModel(reg, X, y).run()

        assert metrics["R2"] == pytest.approx(1.0)
        assert metrics["MSE"] == pytest.approx(0.0, abs=1e-20)


# =============================================================================
# ChooseBestModel
# =============================================================================


class TestChooseBestModel:
    def test_model_1_wins_on_higher_r2(self):
        X = pd.DataFrame({"x": np.linspace(0, 10, 30)})
        y = pd.Series(2 * X["x"] + 1)

        good = {"model": LinearRegression().fit(X, y)}
        bad = {"model": LinearRegression().fit(X.iloc[:2], [0, 0])}

        best, results = ChooseBestModel(good, bad, X, y).run()

        assert best is good
        assert set(results) == {"model_1", "model_2"}

    def test_model_2_wins_on_higher_r2(self):
        X = pd.DataFrame({"x": np.linspace(0, 10, 30)})
        y = pd.Series(2 * X["x"] + 1)

        bad = {"model": LinearRegression().fit(X.iloc[:2], [0, 0])}
        good = {"model": LinearRegression().fit(X, y)}

        best, results = ChooseBestModel(bad, good, X, y).run()

        assert best is good

    def test_exact_tie_falls_back_to_model_2(self):
        """Identical models tie on both R2 and RMSE; ties resolve to model_2."""
        X = pd.DataFrame({"x": np.linspace(0, 10, 30)})
        y = pd.Series(2 * X["x"] + 1)
        model = {"model": LinearRegression().fit(X, y)}

        best, _ = ChooseBestModel(model, model, X, y).run()

        assert best is model

    def test_tie_break_prefers_lower_rmse(self):
        """
        Exercise the `rmse_1 < rmse_2` tie-break branch directly.

        For a fixed y_validation, equal R2 mathematically implies equal RMSE,
        so this branch can't be reached through real predictions - it's
        forced here by mocking the metric functions.
        """
        X = pd.DataFrame({"x": np.linspace(0, 10, 30)})
        y = pd.Series(2 * X["x"] + 1)
        model_1 = {"model": LinearRegression().fit(X, y)}
        model_2 = {"model": LinearRegression().fit(X, y)}

        with patch(
            "pyiron_nodes.machine_learning.pipeline.r2_score", side_effect=[0.9, 0.9]
        ), patch(
            "pyiron_nodes.machine_learning.pipeline.mean_squared_error",
            side_effect=[1.0, 4.0],
        ):
            best, _ = ChooseBestModel(model_1, model_2, X, y).run()

        assert best is model_1
