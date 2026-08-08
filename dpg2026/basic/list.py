from core import as_function_node, PortList


@as_function_node
def ListOf(items: PortList = PortList(["x1", "x2"])) -> list:
    """Collect a variable number of inputs into a list.

    Add, rename and remove inputs with the "+" and "x" buttons on the node.
    """
    list_out = [v for v in items.values() if v is not None]
    return list_out


@as_function_node
def ListOfStrings(
    items: PortList = PortList[str](["x1", "x2"], required=False)
) -> list:
    """Collect a variable number of text inputs into a list.

    Add, rename and remove inputs with the "+" and "x" buttons on the node.
    Typed ports get an editable field in the GUI; blank ones are skipped.
    """
    list_out = [v for v in items.values() if v is not None]
    return list_out


@as_function_node
def ListOfIntegers(
    items: PortList = PortList[int](["x1", "x2"], required=False)
) -> list:
    """Collect a variable number of integer inputs into a list.

    Add, rename and remove inputs with the "+" and "x" buttons on the node.
    Typed ports get an editable field in the GUI; blank ones are skipped.
    """
    list_out = [v for v in items.values() if v is not None]
    return list_out


@as_function_node
def ListOfFloats(
    items: PortList = PortList[float](["x1", "x2"], required=False)
) -> list:
    """Collect a variable number of float inputs into a list.

    Add, rename and remove inputs with the "+" and "x" buttons on the node.
    Typed ports get an editable field in the GUI; blank ones are skipped.
    """
    list_out = [v for v in items.values() if v is not None]
    return list_out


@as_function_node
def ListOfBooleans(
    items: PortList = PortList[bool](["x1", "x2"], required=False)
) -> list:
    """Collect a variable number of boolean inputs into a list.

    Add, rename and remove inputs with the "+" and "x" buttons on the node.
    Typed ports get a checkbox in the GUI; blank ones are skipped.
    """
    list_out = [v for v in items.values() if v is not None]
    return list_out


@as_function_node
def List5(x1, x2=None, x3=None, x4=None, x5=None) -> list:
    """Deprecated: use ``ListOf``, which takes any number of inputs."""
    list_out = [x for x in (x1, x2, x3, x4, x5) if x is not None]
    return list_out
