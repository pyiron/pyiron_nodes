from pyiron_workflow import as_function_node
from typing import Optional

@as_function_node("function_space")
def FunctionSpace(
    domain,
    el_type: Optional[str],
    el_order: Optional[int],
):
    from dolfinx import fem

    V = fem.functionspace(domain, (el_type, el_order))
    return V

@as_function_node("function_space")
def VectorFunctionSpace(
    domain,
    el_type: Optional[str],
    el_order: Optional[int],
):
    from dolfinx import fem

    V = fem.functionspace(domain, (el_type, el_order, (3,)))
    return V