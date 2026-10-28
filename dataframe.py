"""
Operations and views for pandas DataFrames in Pyiron Nodes.
This module provides functions to manipulate and analyze pandas DataFrames
"""

from typing import Literal

import pandas as pd

from core import Node, as_function_node


@as_function_node("df")
def ReadDataFrame(
    filename: str,
    file_format: Literal["csv", "pickle", "parquet", "json", "excel"] = "pickle",
):
    import pandas as pd

    if file_format == "csv":
        file = pd.read_csv(filename)
    elif file_format == "parquet":
        file = pd.read_parquet(filename)
    elif file_format == "json":
        file = pd.read_json(filename)
    elif file_format == "excel":
        file = pd.read_excel(filename)
    else:
        raise ValueError(
            f"Unsupported file format '{file_format}'. Choose one of: csv, pickle, parquet, json, excel."
        )

    return file


@as_function_node("df")
def ReadPickleDataFrame(filename: str, compression: str = "infer"):
    import pandas as pd

    file = pd.read_pickle(filename, compression=compression)
    return file


@as_function_node("csv")
def ReadCSV(
    filename: str,
    header: list = [0, 1],
    decimal: str = ",",
    delimiter: str = ";",
):
    """Read a delimited text file into a DataFrame.

    The defaults follow the European convention (';' separator, ',' decimal
    mark) and a two-row header, as produced by common lab instruments.
    """
    import pandas as pd

    return pd.read_csv(filename, delimiter=delimiter, header=header, decimal=decimal)


# get column from dataframe
@as_function_node
def GetColumnFromDataFrame(df, column_name: str, as_array: bool = False):
    import numpy as np

    column = df[column_name]
    if as_array:
        column = np.asarray(column.tolist())
    else:
        column = column.tolist()

    return column


# get rows from dataframe (from min to max index)
@as_function_node
def GetRowsFromDataFrame(df, min_index: int = 0, max_index: int = None):
    if max_index is None:  # return all rows from min_index to the end of the dataframe.
        max_index = len(df)

    rows = df.iloc[min_index:max_index]
    return rows


# display dataframe
@as_function_node
def DisplayDataFrame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Display the DataFrame.
    """
    return df


# merge two dataframes
@as_function_node
def MergeDataFrames(
    df1: pd.DataFrame, df2: pd.DataFrame, on: str = "index", how: str = "outer"
):
    """
    Merge two DataFrames on a specified column or index.
    """
    merged_df = pd.merge(df1, df2, on=on, how=how)
    return merged_df


def _bound_kwargs_excluding_first(func: Node) -> tuple[str, dict]:
    """The wrapped function's first parameter name, and a kwargs dict of its
    other declared parameters' current port values (only the ones that are
    ready — i.e. actually bound to something, not left at NotData).

    ``Node``/``FunctionNode`` no longer exposes ``.kwargs``/``._func`` (an
    older ``core`` API) — this reconstructs the same thing from
    ``func.func`` (the wrapped callable) and ``func.inputs``.
    """
    import inspect

    param_names = list(inspect.signature(func.func).parameters)
    first_arg = param_names[0]
    kwargs = {
        name: func.inputs.ports[name].value
        for name in param_names[1:]
        if name in func.inputs and func.inputs.ports[name].ready
    }
    return first_arg, kwargs


# Using apply to map the function on the Series
@as_function_node
def ApplyFunctionToSeries(series: pd.Series, func: Node, store: bool = False):
    """
    Apply a function to each element of a pandas Series.

    ``series`` also accepts a plain list/array — e.g. straight from
    ``GetColumnFromDataFrame``, which returns a list rather than a
    ``pd.Series`` — and is coerced internally.
    """
    import numpy as np

    if not isinstance(series, pd.Series):
        series = pd.Series(series)

    _, kwargs = _bound_kwargs_excluding_first(func)
    transformed_series = np.stack(series.apply(func.func, **kwargs))

    return transformed_series


# Using apply to map the function on the Series
@as_function_node
def ApplyFunctionToSeriesNew(series: pd.Series, function: Node, store: bool = False):
    """
    Apply a function to each element of a pandas Series.

    ``series`` also accepts a plain list/array — e.g. straight from
    ``GetColumnFromDataFrame``, which returns a list rather than a
    ``pd.Series`` — and is coerced internally.
    """
    import numpy as np

    if not isinstance(series, pd.Series):
        series = pd.Series(series)

    _, kwargs = _bound_kwargs_excluding_first(function)
    out = series.apply(function.func, **kwargs)
    print(f"apply function: {function.func}, out")
    transformed_series = np.stack(out)

    return transformed_series
