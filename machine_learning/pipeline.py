"""
Elementary ML nodes.

This module contains nodes for for machine learning workflows using sk-learn models.
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from core import as_function_node


@as_function_node
def MLDataSplitter(
    df,
    y_name: str,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    random_state: int = 42,
):
    """
    Splits dataframe into train, validation, and test sets. This node prevents data leakage when connected correctly.
    using ONLY numeric feature columns.
    """

    # -----------------------------
    # Validate fractions
    # -----------------------------
    total = train_fraction + validation_fraction + test_fraction

    if not np.isclose(total, 1.0):
        raise ValueError("Fractions must sum to 1.0")

    # -----------------------------
    # Remove missing rows
    # -----------------------------
    df = df.dropna()

    # -----------------------------
    # Target column
    # -----------------------------
    y = df[y_name].copy()

    # -----------------------------
    # Feature columns
    # -----------------------------
    X_candidates = df.drop(columns=[y_name])

    # Keep ONLY numeric columns
    X_numeric = X_candidates.select_dtypes(include=["number"])

    # -----------------------------
    # FIRST SPLIT
    # Train vs Temp
    # -----------------------------
    temp_fraction = validation_fraction + test_fraction

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_numeric, y, test_size=temp_fraction, random_state=random_state
    )

    # -----------------------------
    # SECOND SPLIT
    # Validation vs Test
    # -----------------------------
    validation_size_adjusted = validation_fraction / temp_fraction

    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=(1 - validation_size_adjusted),
        random_state=random_state,
    )

    return X_train, X_validation, X_test, y_train, y_validation, y_test


@as_function_node
def TrainRegressor(X_train: pd.DataFrame, y_train: pd.DataFrame, r_type: str = None):
    """
    trains a regressor
    """
    if r_type == "linear":
        reg = LinearRegression().fit(X_train, y_train)
    elif r_type == "tree":
        reg = RandomForestRegressor().fit(X_train, y_train)
    else:
        raise ValueError(f"Unknown r_type: {r_type!r}. Expected 'linear' or 'tree'.")
    return reg


# =========================================================
# 2) MODEL EVALUATION FUNCTION
# =========================================================


@as_function_node
def EvaluateRegressionModel(model, X_test, y_test):
    """
    Evaluates a regression model.

    Returns:
        R2
        MSE
        MAE
    """

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    out = {"R2": r2, "MSE": mse, "MAE": mae}
    return out


# =========================================================
# 3) MODEL COMPARISON FUNCTION
# =========================================================


@as_function_node
def ChooseBestModel(model_1, model_2, X_validation, y_validation):
    """
    Compares two regression models on VALIDATION DATA.

    Selection Priority:
        1. Higher R2
        2. Lower RMSE

    Returns:
        best_model
        comparison_results
    """

    # -----------------------------
    # Predictions
    # -----------------------------
    pred_1 = model_1["model"].predict(X_validation)
    pred_2 = model_2["model"].predict(X_validation)

    # -----------------------------
    # Metrics for model 1
    # -----------------------------
    r2_1 = r2_score(y_validation, pred_1)
    rmse_1 = np.sqrt(mean_squared_error(y_validation, pred_1))

    # -----------------------------
    # Metrics for model 2
    # -----------------------------
    r2_2 = r2_score(y_validation, pred_2)
    rmse_2 = np.sqrt(mean_squared_error(y_validation, pred_2))

    # -----------------------------
    # Choose best model
    # -----------------------------
    if r2_1 > r2_2:
        best_model = model_1

    elif r2_2 > r2_1:
        best_model = model_2

    else:
        # If R2 tied -> lower RMSE wins
        if rmse_1 < rmse_2:
            best_model = model_1
        else:
            best_model = model_2

    results = {
        "model_1": {"R2": r2_1, "RMSE": rmse_1},
        "model_2": {"R2": r2_2, "RMSE": rmse_2},
    }

    return best_model, results
