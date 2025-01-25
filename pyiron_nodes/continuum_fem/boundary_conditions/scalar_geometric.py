from pyiron_workflow import as_function_node
from typing import Optional
from typing import Callable
import numpy as np


@as_function_node("bc")
def ScalarDirichlet1D(function_space, bc_function: str, value: Optional[float | int]):
    from dolfinx import fem, default_scalar_type

    lazy_evaluation = lambda x: eval(bc_function)
    result = lazy_evaluation
    boundary_dofs = fem.locate_dofs_geometrical(function_space, result)
    bc = fem.dirichletbc(default_scalar_type(value), boundary_dofs, function_space)
    return bc


@as_function_node("bc")
def ScalarDirichlet3D(
    function_space,
    bc_function: str,
    value_x: Optional[float | int],
    value_y: Optional[float | int],
    value_z: Optional[float | int],
):

    from dolfinx import fem, default_scalar_type

    lazy_evaluation = lambda x: eval(bc_function)
    result = lazy_evaluation
    boundary_dofs = fem.locate_dofs_geometrical(function_space, result)
    u_D = np.array([value_x, value_y, value_z], dtype=default_scalar_type)
    bc = fem.dirichletbc(u_D, boundary_dofs, function_space)
    return bc


@as_function_node("bcs_array")
def CollectBcs(bc1=None, bc2=None, bc3=None, bc4=None, bc5=None, bc6=None):

    bcs = []
    if bc1 is not None:
        bcs.append(bc1)

    if bc2 is not None:
        bcs.append(bc2)

    if bc3 is not None:
        bcs.append(bc3)

    if bc4 is not None:
        bcs.append(bc4)

    if bc5 is not None:
        bcs.append(bc5)

    if bc6 is not None:
        bcs.append(bc6)

    return bcs
