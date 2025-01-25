from pyiron_workflow import as_function_node, as_macro_node
from typing import Optional
from pyiron_nodes.continuum_fem.geometries.three_d_geometries import BarParameters
from pyiron_nodes.dev_tools import wf_data_class
from pyiron_workflow import as_dataclass_node
from dataclasses import field
import ufl


@as_function_node("traction_vector")
def TractionVector3D(
    domain,
    traction_x,
    traction_y,
    traction_z,
):

    from dolfinx import fem, default_scalar_type

    T = fem.Constant(domain, default_scalar_type((traction_x, traction_y, traction_z)))
    return T


@as_function_node("body_force_vector")
def BodyForceVectorBar(
    domain,
    body_force_x,
    body_force_y,
    body_force_z,
    gravity_factor,
    weight_params: Optional[BarParameters.dataclass] = BarParameters.dataclass(),
):

    from dolfinx import fem, default_scalar_type

    rho = weight_params.density
    g = (
        gravity_factor
        * weight_params.length
        * weight_params.width
        * weight_params.depth
    )
    f = fem.Constant(
        domain,
        default_scalar_type((body_force_x, body_force_y, body_force_z - (rho * g))),
    )
    return f


def epsilon(u):
    return ufl.sym(ufl.grad(u))


def three_d_strain_voigt(e):
    return ufl.as_vector(
        [e[0, 0], e[1, 1], e[2, 2], 2 * e[0, 1], 2 * e[0, 2], 2 * e[1, 2]]
    )


def three_d_voigt_stress(s):
    return ufl.as_tensor([[s[0], s[3], s[4]], [s[3], s[1], s[5]], [s[4], s[5], s[2]]])


def three_d_sigma(u, C):
    return three_d_voigt_stress(ufl.dot(C, three_d_strain_voigt(epsilon(u))))


@as_function_node("solution_vector")
def LinearElasticitySolver(
    function_space,
    domain,
    bcs_array,
    traction_vector,
    body_force_vector,
    elasticity_tensor,
):
    from dolfinx.fem.petsc import LinearProblem
    import ufl
    import numpy as np

    C = ufl.as_matrix(elasticity_tensor)
    ds = ufl.Measure("ds", domain=domain)
    T = traction_vector
    u = ufl.TrialFunction(function_space)
    v = ufl.TestFunction(function_space)
    f = body_force_vector
    a = ufl.inner(three_d_sigma(u, C), epsilon(v)) * ufl.dx
    L = ufl.dot(f, v) * ufl.dx + ufl.dot(T, v) * ds

    problem = LinearProblem(
        a, L, bcs=bcs_array, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    uh = problem.solve()
    return uh


@as_macro_node("solution_vector")
def LinearElasticity3D(
    self,
    domain,
    function_space,
    bcs_array,
    traction_x: Optional[float | int],
    traction_y: Optional[float | int],
    traction_z: Optional[float | int],
    body_force_x: Optional[float | int],
    body_force_y: Optional[float | int],
    body_force_z: Optional[float | int],
    gravity_factor: Optional[float | int],
    elasticity_tensor,
    parameters: Optional[BarParameters.dataclass] = BarParameters.dataclass(),
):

    self.T = TractionVector3D(
        domain=domain,
        traction_x=traction_x,
        traction_y=traction_y,
        traction_z=traction_z,
    )

    self.f = BodyForceVectorBar(
        domain=domain,
        body_force_x=body_force_x,
        body_force_y=body_force_y,
        body_force_z=body_force_z,
        gravity_factor=gravity_factor,
        weight_params=parameters,
    )

    self.uh = LinearElasticitySolver(
        domain=domain,
        function_space=function_space,
        bcs_array=bcs_array,
        traction_vector=self.T,
        body_force_vector=self.f,
        elasticity_tensor=elasticity_tensor,
    )

    return self.uh
