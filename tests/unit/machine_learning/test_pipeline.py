"""
Unit tests for machine_learning/pipeline.py
"""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from pyiron_nodes.machine_learning.pipeline import (
    ChooseBestModel,
    EvaluateRegressionModel,
    MLDataSplitter,
    TrainRegressor,
)

# =============================================================================
# SYNTHETIC TEST DATA HELPERS
# =============================================================================


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


def regression_data():
    """(X_train, y_train, X_test, y_test) for a simple linear relationship."""
    np.random.seed(42)
    n_train, n_test, n_features = 50, 20, 3

    X_train = pd.DataFrame(
        np.random.randn(n_train, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    y_train = pd.Series(
        3 * X_train.iloc[:, 0]
        + 2 * X_train.iloc[:, 1]
        + np.random.randn(n_train) * 0.1,
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


class TestMLDataSplitter(unittest.TestCase):
    def test_default_split_shapes(self):
        df = splitter_df()
        X_train, X_val, X_test, y_train, y_val, y_test = (
            MLDataSplitter._original_func(df, "target")
        )

        self.assertEqual(len(X_train) + len(X_val) + len(X_test), len(df))
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_val), len(y_val))
        self.assertEqual(len(X_test), len(y_test))

    def test_non_numeric_columns_dropped(self):
        df = splitter_df()
        X_train, X_val, X_test, y_train, y_val, y_test = (
            MLDataSplitter._original_func(df, "target")
        )

        self.assertEqual(set(X_train.columns), {"feature_0", "feature_1"})

    def test_rows_with_missing_values_dropped(self):
        df = splitter_df()
        df_with_nan = df.copy()
        df_with_nan.loc[0, "feature_0"] = np.nan

        X_train, X_val, X_test, y_train, y_val, y_test = (
            MLDataSplitter._original_func(df_with_nan, "target")
        )

        self.assertEqual(len(X_train) + len(X_val) + len(X_test), len(df) - 1)

    def test_custom_fractions(self):
        df = splitter_df()
        X_train, X_val, X_test, y_train, y_val, y_test = (
            MLDataSplitter._original_func(
                df,
                "target",
                train_fraction=0.5,
                validation_fraction=0.3,
                test_fraction=0.2,
            )
        )

        n = len(df)
        self.assertAlmostEqual(len(X_train), 0.5 * n, delta=2)
        self.assertAlmostEqual(len(X_val), 0.3 * n, delta=2)
        self.assertAlmostEqual(len(X_test), 0.2 * n, delta=2)

    def test_invalid_fractions_raise(self):
        df = splitter_df()
        with self.assertRaises(ValueError):
            MLDataSplitter._original_func(
                df,
                "target",
                train_fraction=0.5,
                validation_fraction=0.3,
                test_fraction=0.3,
            )

    def test_reproducible_with_random_state(self):
        df = splitter_df()
        result_1 = MLDataSplitter._original_func(df, "target", random_state=1)
        result_2 = MLDataSplitter._original_func(df, "target", random_state=1)

        pd.testing.assert_frame_equal(result_1[0], result_2[0])


# =============================================================================
# TrainRegressor
# =============================================================================


class TestTrainRegressor(unittest.TestCase):
    def test_linear(self):
        X_train, y_train, _, _ = regression_data()
        reg = TrainRegressor._original_func(X_train, y_train, r_type="linear")

        self.assertIsInstance(reg, LinearRegression)

    def test_tree(self):
        X_train, y_train, _, _ = regression_data()
        reg = TrainRegressor._original_func(X_train, y_train, r_type="tree")

        self.assertIsInstance(reg, RandomForestRegressor)

    def test_default_r_type_raises(self):
        X_train, y_train, _, _ = regression_data()
        with self.assertRaises(ValueError):
            TrainRegressor._original_func(X_train, y_train)

    def test_invalid_r_type_raises(self):
        X_train, y_train, _, _ = regression_data()
        with self.assertRaises(ValueError):
            TrainRegressor._original_func(X_train, y_train, r_type="invalid")


# =============================================================================
# EvaluateRegressionModel
# =============================================================================


class TestEvaluateRegressionModel(unittest.TestCase):
    def test_output_keys(self):
        X_train, y_train, X_test, y_test = regression_data()
        reg = TrainRegressor._original_func(X_train, y_train, r_type="linear")

        metrics = EvaluateRegressionModel._original_func(reg, X_test, y_test)

        self.assertEqual(set(metrics), {"R2", "MSE", "MAE"})

    def test_perfect_fit_gives_r2_of_one(self):
        X = pd.DataFrame({"x": np.arange(20, dtype=float)})
        y = pd.Series(2 * X["x"] + 1)
        reg = LinearRegression().fit(X, y)

        metrics = EvaluateRegressionModel._original_func(reg, X, y)

        self.assertAlmostEqual(metrics["R2"], 1.0, places=6)
        self.assertAlmostEqual(metrics["MSE"], 0.0, delta=1e-10)


# =============================================================================
# ChooseBestModel
# =============================================================================


class TestChooseBestModel(unittest.TestCase):
    def test_model_1_wins_on_higher_r2(self):
        X = pd.DataFrame({"x": np.linspace(0, 10, 30)})
        y = pd.Series(2 * X["x"] + 1)

        good = {"model": LinearRegression().fit(X, y)}
        bad = {"model": LinearRegression().fit(X.iloc[:2], [0, 0])}

        best, results = ChooseBestModel._original_func(good, bad, X, y)

        self.assertIs(best, good)
        self.assertEqual(set(results), {"model_1", "model_2"})

    def test_model_2_wins_on_higher_r2(self):
        X = pd.DataFrame({"x": np.linspace(0, 10, 30)})
        y = pd.Series(2 * X["x"] + 1)

        bad = {"model": LinearRegression().fit(X.iloc[:2], [0, 0])}
        good = {"model": LinearRegression().fit(X, y)}

        best, results = ChooseBestModel._original_func(bad, good, X, y)

        self.assertIs(best, good)

    def test_exact_tie_falls_back_to_model_2(self):
        """Identical models tie on both R2 and RMSE; ties resolve to model_2."""
        X = pd.DataFrame({"x": np.linspace(0, 10, 30)})
        y = pd.Series(2 * X["x"] + 1)
        model = {"model": LinearRegression().fit(X, y)}

        best, _ = ChooseBestModel._original_func(model, model, X, y)

        self.assertIs(best, model)

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
            best, _ = ChooseBestModel._original_func(model_1, model_2, X, y)

        self.assertIs(best, model_1)


if __name__ == "__main__":
    unittest.main()
