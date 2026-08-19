from core import as_function_node
import pandas as pd
import numpy as np


def _calc_rmse(array_1, array_2, rmse_in_milli: bool = True):
    """
    Calculates the RMSE value of two arrays

    Args:
    array_1: An array or list of energy or force values
    array_2: An array or list of energy or force values

    Returns:
    rmse_in_milli: (boolean, Default = True) Set False if you want the calculated RMSE value in decimals
    rmse: The calculated RMSE value
    """
    rmse = np.sqrt(np.mean((array_1 - array_2) ** 2))
    if rmse_in_milli:
        return rmse * 1000
    else:
        return rmse
