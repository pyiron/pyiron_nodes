"""
For mathematical operations.
"""

from __future__ import annotations

from typing import Optional
import numpy as np

from pyiron_workflow import as_function_node


@as_function_node("sin")
def Sin(x: Optional[list | np.ndarray] = None):
    return np.sin(x)


@as_function_node("cos")
def Cos(x: Optional[list | np.ndarray] = None):
    return np.cos(x)


@as_function_node("tan")
def Tan(x: Optional[list | np.ndarray] = None):
    return np.tan(x)


@as_function_node("arcsin")
def Arcsin(x: Optional[list | np.ndarray] = None):
    arcsin = np.arcsin(x)
    return arcsin


@as_function_node("arccos")
def Arccos(x: Optional[list | np.ndarray] = None):
    return np.arccos(x)


@as_function_node("arctan")
def Arctan(x: Optional[list | np.ndarray] = None):
    return np.arctan(x)


@as_function_node("arctan2")
def Arctan2(
    x: Optional[list | np.ndarray] = None, y: Optional[list | np.ndarray] = None
):
    return np.arctan2(y, x)
