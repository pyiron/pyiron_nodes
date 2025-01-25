from pyiron_workflow import as_function_node
from typing import Optional


@as_function_node("solution_vector")
def LinearPoissonSolver(function_space, load, bcs_array):

    from dolfinx.fem.petsc import LinearProblem
    import ufl
    import numpy as np

    u = ufl.TrialFunction(function_space)
    v = ufl.TestFunction(function_space)
    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = load * v * ufl.dx
    problem = LinearProblem(
        a, L, bcs=bcs_array, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    uh = problem.solve()
    return uh
