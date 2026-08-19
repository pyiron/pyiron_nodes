from core import as_function_node
import numpy as np


@as_function_node("value")
def Index(values: list, index: int):
    return values[index]


@as_function_node("slice")
def Slice(values: list, start: int = 0, stop: int = -1, step: int = 1) -> list:
    return values[start:stop:step]


@as_function_node("list")
def Prepend(x, xs: list | None = None) -> list:
    if xs is None:
        xs = []
    return [x, *xs]


@as_function_node("list")
def Append(xs: list, x) -> list:
    if xs is None:
        xs = []
    return [*xs, x]


@as_function_node
def List5(x1, x2=None, x3=None, x4=None, x5=None) -> list:
    list_out = [x for x in (x1, x2, x3, x4, x5) if x is not None]
    return list_out


@as_function_node("item")
def GetItem(obj, index: int | str):
    return obj[index]
