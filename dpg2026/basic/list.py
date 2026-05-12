from core import as_function_node


@as_function_node
def List5(x1, x2=None, x3=None, x4=None, x5=None) -> list:
    list_out = [x for x in (x1, x2, x3, x4, x5) if x is not None]
    return list_out
