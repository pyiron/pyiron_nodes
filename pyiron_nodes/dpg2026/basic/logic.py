from core import as_function_node


@as_function_node("or")
def Or(x, y):
    return x | y
