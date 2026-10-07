"""Small list helpers.

For number sequences use ``pyiron_nodes.math_utils`` (``Linspace``, ``Arange``,
``Range``); for slicing use ``pyiron_nodes.controls.Slice``; for reading a
pickled dataframe use ``pyiron_nodes.dataframe.ReadDataFrame``.
"""

from core import as_function_node, PortList


@as_function_node("value")
def Index(values: list, index: int):
    return values[index]


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


@as_function_node("list")
def ListOf(items: PortList = PortList(["x1", "x2"])) -> list:
    """Collect a variable number of inputs into a list.

    Add, rename and remove inputs with the "+" and "x" buttons on the node.
    """
    list_out = [v for v in items.values() if v is not None]
    return list_out


@as_function_node("list")
def ListOfStrings(
    items: PortList = PortList[str](["x1", "x2"], required=False)
) -> list:
    """Collect a variable number of text inputs into a list.

    Add, rename and remove inputs with the "+" and "x" buttons on the node.
    Typed ports get an editable field in the GUI; blank ones are skipped.
    """
    list_out = [v for v in items.values() if v is not None]
    return list_out


@as_function_node("list")
def ListOfIntegers(
    items: PortList = PortList[int](["x1", "x2"], required=False)
) -> list:
    """Collect a variable number of integer inputs into a list.

    Add, rename and remove inputs with the "+" and "x" buttons on the node.
    Typed ports get an editable field in the GUI; blank ones are skipped.
    """
    list_out = [v for v in items.values() if v is not None]
    return list_out


@as_function_node("list")
def ListOfFloats(
    items: PortList = PortList[float](["x1", "x2"], required=False)
) -> list:
    """Collect a variable number of float inputs into a list.

    Add, rename and remove inputs with the "+" and "x" buttons on the node.
    Typed ports get an editable field in the GUI; blank ones are skipped.
    """
    list_out = [v for v in items.values() if v is not None]
    return list_out


@as_function_node("list")
def ListOfBooleans(
    items: PortList = PortList[bool](["x1", "x2"], required=False)
) -> list:
    """Collect a variable number of boolean inputs into a list.

    Add, rename and remove inputs with the "+" and "x" buttons on the node.
    Typed ports get a checkbox in the GUI; blank ones are skipped.
    """
    list_out = [v for v in items.values() if v is not None]
    return list_out


@as_function_node("list")
def List5(x1, x2=None, x3=None, x4=None, x5=None) -> list:
    """Deprecated: use ``ListOf``, which takes any number of inputs."""
    return [x for x in (x1, x2, x3, x4, x5) if x is not None]


@as_function_node("item")
def GetItem(obj, index: int | str):
    return obj[index]
