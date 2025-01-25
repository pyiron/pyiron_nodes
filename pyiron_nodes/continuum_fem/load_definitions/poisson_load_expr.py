from pyiron_workflow import as_function_node
from typing import Optional


@as_function_node("load")
def BetaR0Function(
    domain,
    beta_val: Optional[float | int],
    R0_val: Optional[float | int],
    expression: Optional[str] = "ufl.exp(-beta**2 * (x[0]**2 + (x[1] - R0)**2))",
    factor: Optional[float | int] = 4,
):
    from dolfinx import fem, default_scalar_type
    import ufl
    import numpy as np

    x = ufl.SpatialCoordinate(domain)
    beta = fem.Constant(domain, default_scalar_type(beta_val))
    R0 = fem.Constant(domain, default_scalar_type(R0_val))
    p = factor * eval(expression)
    return p
