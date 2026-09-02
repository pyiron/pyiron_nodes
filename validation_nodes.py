"""
Model Validation and Selection Nodes for PyIron.

This module provides comprehensive validation and benchmarking for pre-trained
scikit-learn models. It evaluates multiple already-trained models on test data,
computes various performance metrics, and ranks them for comparison.

Features:
- Validate multiple pre-trained models on test data
- Automatic task type detection (regression vs classification)
- Multiple metrics support (R², RMSE, MAE, accuracy, F1, precision, recall)
- Results export to CSV
- Best model selection and ranking
- Detailed comparison reports
"""

from typing import Optional, Dict, Any, Tuple, List, Union
import pandas as pd
import numpy as np
import warnings
from pathlib import Path

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from core import as_function_node

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def infer_task_type(y: Union[pd.Series, np.ndarray]) -> str:
    """
    Automatically infer task type from target variable.

    Parameters:
        y: Target variable

    Returns:
        "classification" or "regression"
    """
    y_array = np.asarray(y)

    # Check if values are integers or categorical
    if y_array.dtype in [int, object, bool]:
        return "classification"

    # Check number of unique values
    n_unique = len(np.unique(y_array))
    if n_unique <= 20 and y_array.dtype in [int, np.int32, np.int64]:
        return "classification"

    return "regression"


def validate_data(
    X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Validate and convert input data to proper format.

    Parameters:
        X: Feature matrix
        y: Target variable

    Returns:
        Tuple of (X_validated, y_validated)
    """
    # Convert to pandas if necessary
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

    if isinstance(y, np.ndarray):
        y = pd.Series(y, name="target")

    # Check shapes
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y have different number of samples: {X.shape[0]} vs {y.shape[0]}"
        )

    # Check for NaN values
    if X.isna().any().any():
        warnings.warn("X contains NaN values, dropping rows with NaN")
        mask = ~X.isna().any(axis=1)
        X = X[mask].reset_index(drop=True)
        y = y[mask].reset_index(drop=True)

    if y.isna().any():
        warnings.warn("y contains NaN values, dropping rows with NaN")
        mask = ~y.isna()
        X = X[mask].reset_index(drop=True)
        y = y[mask].reset_index(drop=True)

    return X, y


def validate_models_dict(models: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate that all models in dictionary are fitted estimators.

    Parameters:
        models: Dictionary of {model_name: model_instance}

    Returns:
        Validated models dictionary
    """
    if not isinstance(models, dict):
        raise TypeError("models must be a dictionary of {name: model} pairs")

    if len(models) == 0:
        raise ValueError("models dictionary cannot be empty")

    for name, model in models.items():
        if not hasattr(model, "predict"):
            raise ValueError(
                f"Model '{name}' does not have a predict() method. "
                "Ensure it is a fitted estimator."
            )

    return models


# =============================================================================
# REGRESSION MODEL VALIDATION NODE
# =============================================================================


@as_function_node("validation_results")
def ValidateRegressionModels(
    models: Dict[str, Any],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    metric: str = "r2",
    output_file: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Validate and rank multiple pre-trained regression models.

    Evaluates multiple fitted regression models on test data using specified metrics.
    Computes performance metrics for each model and ranks them.

    Parameters:
        models: Dictionary of {model_name: fitted_model} pairs
        X_test: Test feature matrix (n_samples, n_features)
        y_test: Test target values (n_samples,)
        metric: Primary evaluation metric - 'r2', 'rmse', 'mae' (default: 'r2')
        output_file: Path to save results CSV (default: None)
        verbose: Print results during evaluation (default: True)

    Returns:
        Dictionary containing:
        - results_df: DataFrame with model rankings and metrics
        - best_model_name: Name of best-performing model
        - best_score: Score of best model
        - metric: Metric used for ranking
        - all_metrics_df: DataFrame with all computed metrics per model
    """

    # Validate inputs
    models = validate_models_dict(models)
    X_test, y_test = validate_data(X_test, y_test)

    if len(X_test) < 1:
        raise ValueError("X_test must contain at least 1 sample")

    if metric not in ["r2", "rmse", "mae"]:
        raise ValueError(f"Unsupported metric: {metric}. Use 'r2', 'rmse', or 'mae'")

    results_list = []

    if verbose:
        print(
            f"Validating {len(models)} regression models on test set ({len(X_test)} samples)..."
        )
        print(f"Primary metric: {metric}")
        print()

    # Evaluate each model
    for model_name, model in models.items():
        try:
            # Make predictions
            y_pred = model.predict(X_test)

            # Compute metrics
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

            results_list.append(
                {
                    "model_name": model_name,
                    "r2": r2,
                    "rmse": rmse,
                    "mae": mae,
                }
            )

            if verbose:
                print(
                    f"✓ {model_name:30s} | R²={r2:7.4f} | RMSE={rmse:7.4f} | MAE={mae:7.4f}"
                )

        except Exception as e:
            if verbose:
                print(f"✗ {model_name:30s} | Error: {str(e)}")
            continue

    if len(results_list) == 0:
        raise RuntimeError("No models could be evaluated successfully")

    # Create results DataFrame
    all_metrics_df = pd.DataFrame(results_list)

    # Sort by primary metric
    if metric == "r2":
        results_df = all_metrics_df.sort_values("r2", ascending=False).reset_index(
            drop=True
        )
    else:
        results_df = all_metrics_df.sort_values(metric, ascending=True).reset_index(
            drop=True
        )

    results_df["rank"] = range(1, len(results_df) + 1)

    # Get best model
    best_model_name = results_df.iloc[0]["model_name"]
    best_score = results_df.iloc[0][metric]

    if verbose:
        print()
        print(f"═" * 80)
        print(f"Best Model: {best_model_name:30s} {metric.upper()}={best_score:.6f}")
        print(f"═" * 80)

    # Save to CSV if requested
    if output_file:
        results_df.to_csv(output_file, index=False)
        if verbose:
            print(f"Results saved to {output_file}")

    return {
        "results_df": results_df[["rank", "model_name", "r2", "rmse", "mae"]],
        "best_model_name": best_model_name,
        "best_score": best_score,
        "metric": metric,
        "all_metrics_df": all_metrics_df,
        "n_samples": len(X_test),
        "n_features": X_test.shape[1],
    }


# =============================================================================
# CLASSIFICATION MODEL VALIDATION NODE
# =============================================================================


@as_function_node("validation_results")
def ValidateClassificationModels(
    models: Dict[str, Any],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    metric: str = "accuracy",
    output_file: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Validate and rank multiple pre-trained classification models.

    Evaluates multiple fitted classification models on test data using specified metrics.
    Computes performance metrics for each model and ranks them.

    Parameters:
        models: Dictionary of {model_name: fitted_model} pairs
        X_test: Test feature matrix (n_samples, n_features)
        y_test: Test target labels (n_samples,)
        metric: Primary evaluation metric - 'accuracy', 'f1', 'precision', 'recall' (default: 'accuracy')
        output_file: Path to save results CSV (default: None)
        verbose: Print results during evaluation (default: True)

    Returns:
        Dictionary containing:
        - results_df: DataFrame with model rankings and metrics
        - best_model_name: Name of best-performing model
        - best_score: Score of best model
        - metric: Metric used for ranking
        - all_metrics_df: DataFrame with all computed metrics per model
        - confusion_matrices: Dictionary of confusion matrices for each model
    """

    # Validate inputs
    models = validate_models_dict(models)
    X_test, y_test = validate_data(X_test, y_test)

    if len(X_test) < 1:
        raise ValueError("X_test must contain at least 1 sample")

    if metric not in ["accuracy", "f1", "precision", "recall"]:
        raise ValueError(f"Unsupported metric: {metric}")

    results_list = []
    confusion_matrices = {}

    if verbose:
        print(
            f"Validating {len(models)} classification models on test set ({len(X_test)} samples)..."
        )
        print(f"Primary metric: {metric}")
        print()

    # Evaluate each model
    for model_name, model in models.items():
        try:
            # Make predictions
            y_pred = model.predict(X_test)

            # Compute metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            precision = precision_score(
                y_test, y_pred, average="weighted", zero_division=0
            )
            recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)

            # Store confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            confusion_matrices[model_name] = cm

            results_list.append(
                {
                    "model_name": model_name,
                    "accuracy": accuracy,
                    "f1": f1,
                    "precision": precision,
                    "recall": recall,
                }
            )

            if verbose:
                print(
                    f"✓ {model_name:30s} | Acc={accuracy:6.4f} | F1={f1:6.4f} | Prec={precision:6.4f} | Rec={recall:6.4f}"
                )

        except Exception as e:
            if verbose:
                print(f"✗ {model_name:30s} | Error: {str(e)}")
            continue

    if len(results_list) == 0:
        raise RuntimeError("No models could be evaluated successfully")

    # Create results DataFrame
    all_metrics_df = pd.DataFrame(results_list)

    # Sort by primary metric (all higher is better for classification)
    results_df = all_metrics_df.sort_values(metric, ascending=False).reset_index(
        drop=True
    )
    results_df["rank"] = range(1, len(results_df) + 1)

    # Get best model
    best_model_name = results_df.iloc[0]["model_name"]
    best_score = results_df.iloc[0][metric]

    if verbose:
        print()
        print(f"═" * 80)
        print(f"Best Model: {best_model_name:30s} {metric.upper()}={best_score:.6f}")
        print(f"═" * 80)

    # Save to CSV if requested
    if output_file:
        results_df.to_csv(output_file, index=False)
        if verbose:
            print(f"Results saved to {output_file}")

    return {
        "results_df": results_df[
            ["rank", "model_name", "accuracy", "f1", "precision", "recall"]
        ],
        "best_model_name": best_model_name,
        "best_score": best_score,
        "metric": metric,
        "all_metrics_df": all_metrics_df,
        "confusion_matrices": confusion_matrices,
        "n_samples": len(X_test),
        "n_features": X_test.shape[1],
        "n_classes": len(np.unique(y_test)),
    }


# =============================================================================
# AUTO VALIDATION NODE (TASK TYPE DETECTION)
# =============================================================================


@as_function_node("validation_results")
def ValidateModelsAuto(
    models: Dict[str, Any],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    metric: Optional[str] = None,
    output_file: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Automatically detect task type and validate pre-trained models.

    This node infers whether the task is regression or classification based on
    the target variable, then applies the appropriate validation suite.

    Parameters:
        models: Dictionary of {model_name: fitted_model} pairs
        X_test: Test feature matrix
        y_test: Test target variable
        metric: Evaluation metric. If None, uses default for task type
        output_file: Path to save results CSV (default: None)
        verbose: Print results during evaluation (default: True)

    Returns:
        Dictionary containing validation results and rankings
    """

    # Infer task type
    task_type = infer_task_type(y_test)

    if verbose:
        print(f"Auto-detected task type: {task_type}")
        print()

    # Use default metric if not specified
    if metric is None:
        metric = "r2" if task_type == "regression" else "accuracy"

    # Delegate to appropriate validator
    if task_type == "regression":
        return ValidateRegressionModels(
            models,
            X_test,
            y_test,
            metric=metric,
            output_file=output_file,
            verbose=verbose,
        )
    else:
        return ValidateClassificationModels(
            models,
            X_test,
            y_test,
            metric=metric,
            output_file=output_file,
            verbose=verbose,
        )


# =============================================================================
# DETAILED VALIDATION REPORT NODE
# =============================================================================


@as_function_node("detailed_report")
def GenerateDetailedValidationReport(
    models: Dict[str, Any],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    output_file: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Generate a comprehensive validation report with all available metrics.

    Evaluates pre-trained models on test data and computes all relevant metrics
    for detailed comparison. Useful for understanding model behavior across
    multiple performance dimensions.

    Parameters:
        models: Dictionary of {model_name: fitted_model} pairs
        X_test: Test feature matrix
        y_test: Test target variable
        output_file: Path to save detailed report CSV (default: None)
        verbose: Print results during evaluation (default: True)

    Returns:
        Dictionary containing:
        - task_type: 'regression' or 'classification'
        - detailed_metrics_df: DataFrame with all metrics per model
        - summary_df: Summary of best model for each metric
        - best_overall_model: Top model by primary metric
    """

    # Validate inputs
    models = validate_models_dict(models)
    X_test, y_test = validate_data(X_test, y_test)

    # Infer task type
    task_type = infer_task_type(y_test)

    if verbose:
        print(f"Task Type: {task_type}")
        print(f"Generating detailed validation report...")
        print()

    detailed_results = []

    if task_type == "regression":
        # Regression: compute all regression metrics
        for model_name, model in models.items():
            try:
                y_pred = model.predict(X_test)

                detailed_results.append(
                    {
                        "model_name": model_name,
                        "r2": r2_score(y_test, y_pred),
                        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                        "mae": mean_absolute_error(y_test, y_pred),
                    }
                )

                if verbose:
                    print(f"✓ {model_name}")
            except Exception as e:
                if verbose:
                    print(f"✗ {model_name}: {str(e)}")

    else:  # Classification
        # Classification: compute all classification metrics
        for model_name, model in models.items():
            try:
                y_pred = model.predict(X_test)

                detailed_results.append(
                    {
                        "model_name": model_name,
                        "accuracy": accuracy_score(y_test, y_pred),
                        "f1": f1_score(
                            y_test, y_pred, average="weighted", zero_division=0
                        ),
                        "precision": precision_score(
                            y_test, y_pred, average="weighted", zero_division=0
                        ),
                        "recall": recall_score(
                            y_test, y_pred, average="weighted", zero_division=0
                        ),
                    }
                )

                if verbose:
                    print(f"✓ {model_name}")
            except Exception as e:
                if verbose:
                    print(f"✗ {model_name}: {str(e)}")

    if len(detailed_results) == 0:
        raise RuntimeError("No models could be evaluated successfully")

    # Create detailed DataFrame
    detailed_metrics_df = pd.DataFrame(detailed_results)

    # Generate summary (best model for each metric)
    metric_cols = [col for col in detailed_metrics_df.columns if col != "model_name"]
    summary_data = []

    for col in metric_cols:
        if col in ["rmse", "mae"]:  # Lower is better
            best_idx = detailed_metrics_df[col].idxmin()
        else:  # Higher is better
            best_idx = detailed_metrics_df[col].idxmax()

        summary_data.append(
            {
                "metric": col,
                "best_model": detailed_metrics_df.loc[best_idx, "model_name"],
                "score": detailed_metrics_df.loc[best_idx, col],
            }
        )

    summary_df = pd.DataFrame(summary_data)
    best_overall = detailed_metrics_df.iloc[0]["model_name"]

    if verbose:
        print()
        print("═" * 80)
        print("Summary of Best Models by Metric:")
        print(summary_df.to_string(index=False))
        print("═" * 80)

    # Save if requested
    if output_file:
        detailed_metrics_df.to_csv(f"{output_file}_detailed.csv", index=False)
        summary_df.to_csv(f"{output_file}_summary.csv", index=False)
        if verbose:
            print(
                f"Reports saved: {output_file}_detailed.csv, {output_file}_summary.csv"
            )

    return {
        "task_type": task_type,
        "detailed_metrics_df": detailed_metrics_df,
        "summary_df": summary_df,
        "best_overall_model": best_overall,
        "n_models": len(models),
        "n_samples": len(X_test),
        "n_features": X_test.shape[1],
    }


# =============================================================================
# PAIRWISE MODEL COMPARISON NODE
# =============================================================================


@as_function_node("comparison_results")
def CompareModelPair(
    model_1: Any,
    model_1_name: str,
    model_2: Any,
    model_2_name: str,
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Compare two pre-trained models side-by-side.

    Detailed comparison of two specific models with all relevant metrics
    and statistical analysis.

    Parameters:
        model_1: First fitted model
        model_1_name: Name of first model
        model_2: Second fitted model
        model_2_name: Name of second model
        X_test: Test feature matrix
        y_test: Test target variable
        verbose: Print comparison (default: True)

    Returns:
        Dictionary containing detailed comparison metrics and winner
    """

    # Validate inputs
    X_test, y_test = validate_data(X_test, y_test)

    # Infer task type
    task_type = infer_task_type(y_test)

    if not hasattr(model_1, "predict") or not hasattr(model_2, "predict"):
        raise ValueError("Both models must have a predict() method")

    if verbose:
        print(f"Comparing {model_1_name} vs {model_2_name}")
        print()

    y_pred_1 = model_1.predict(X_test)
    y_pred_2 = model_2.predict(X_test)

    comparison = {
        "model_1": model_1_name,
        "model_2": model_2_name,
        "task_type": task_type,
    }

    if task_type == "regression":
        # Regression metrics
        metrics_1 = {
            "r2": r2_score(y_test, y_pred_1),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred_1)),
            "mae": mean_absolute_error(y_test, y_pred_1),
        }
        metrics_2 = {
            "r2": r2_score(y_test, y_pred_2),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred_2)),
            "mae": mean_absolute_error(y_test, y_pred_2),
        }

        comparison[model_1_name] = metrics_1
        comparison[model_2_name] = metrics_2

        # Determine winner by R²
        winner = model_1_name if metrics_1["r2"] > metrics_2["r2"] else model_2_name
        comparison["winner"] = winner
        comparison["metric_used"] = "R²"

        if verbose:
            print(
                f"{model_1_name:30s}: R²={metrics_1['r2']:.6f}, RMSE={metrics_1['rmse']:.6f}, MAE={metrics_1['mae']:.6f}"
            )
            print(
                f"{model_2_name:30s}: R²={metrics_2['r2']:.6f}, RMSE={metrics_2['rmse']:.6f}, MAE={metrics_2['mae']:.6f}"
            )
            print()
            print(f"Winner: {winner}")

    else:  # Classification
        # Classification metrics
        metrics_1 = {
            "accuracy": accuracy_score(y_test, y_pred_1),
            "f1": f1_score(y_test, y_pred_1, average="weighted", zero_division=0),
            "precision": precision_score(
                y_test, y_pred_1, average="weighted", zero_division=0
            ),
            "recall": recall_score(
                y_test, y_pred_1, average="weighted", zero_division=0
            ),
        }
        metrics_2 = {
            "accuracy": accuracy_score(y_test, y_pred_2),
            "f1": f1_score(y_test, y_pred_2, average="weighted", zero_division=0),
            "precision": precision_score(
                y_test, y_pred_2, average="weighted", zero_division=0
            ),
            "recall": recall_score(
                y_test, y_pred_2, average="weighted", zero_division=0
            ),
        }

        comparison[model_1_name] = metrics_1
        comparison[model_2_name] = metrics_2

        # Determine winner by accuracy
        winner = (
            model_1_name
            if metrics_1["accuracy"] > metrics_2["accuracy"]
            else model_2_name
        )
        comparison["winner"] = winner
        comparison["metric_used"] = "Accuracy"

        if verbose:
            print(
                f"{model_1_name:30s}: Acc={metrics_1['accuracy']:.6f}, F1={metrics_1['f1']:.6f}, Prec={metrics_1['precision']:.6f}, Rec={metrics_1['recall']:.6f}"
            )
            print(
                f"{model_2_name:30s}: Acc={metrics_2['accuracy']:.6f}, F1={metrics_2['f1']:.6f}, Prec={metrics_2['precision']:.6f}, Rec={metrics_2['recall']:.6f}"
            )
            print()
            print(f"Winner: {winner}")

    return comparison
