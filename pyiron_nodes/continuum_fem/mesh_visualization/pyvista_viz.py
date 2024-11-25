from pyiron_workflow import as_function_node
from typing import Optional


@as_function_node("plotter")
def PlotInitMeshObject(function_space):
    import pyvista
    from dolfinx.plot import vtk_mesh
    from dolfinx import mesh, fem, plot, io, default_scalar_type
    
    pyvista.start_xvfb()

    V = function_space
    plotter = pyvista.Plotter()
    topology, cell_types, geometry = plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    plotter.add_mesh(grid, show_edges=True)
    #return plotter.show(return_viewer=True)
    return plotter

@as_function_node("plotter")
def PlotDeformedMesh1DObject(function_space, solution_vector, warp_factor: Optional[float | int]):
    import pyvista
    from dolfinx.plot import vtk_mesh
    from dolfinx import fem, default_scalar_type
    
    pyvista.start_xvfb()

    topology, cell_types, x = vtk_mesh(function_space)
    grid = pyvista.UnstructuredGrid(topology, cell_types, x)
    grid.point_data["u"] = solution_vector.x.array
    warped = grid.warp_by_scalar("u", factor=warp_factor)
    plotter = pyvista.Plotter()
    plotter.add_mesh(warped, show_edges=True, show_scalar_bar=True, scalars="u")
    return plotter


def epsilon(u):
    import ufl
    return ufl.sym(ufl.grad(u))
    
def three_d_strain_voigt(e):
    import ufl
    return ufl.as_vector([e[0,0], e[1,1], e[2,2], 2*e[0,1], 2*e[0,2], 2*e[1,2]])
    
def three_d_voigt_stress(s):
    import ufl
    return ufl.as_tensor([[s[0], s[3], s[4]], [s[3], s[1], s[5]], [s[4], s[5], s[2]]])
    
def three_d_sigma(u, C):
    import ufl
    return three_d_voigt_stress(ufl.dot(C, three_d_strain_voigt(epsilon(u))))

@as_function_node("plotter")
def PlotVonMises3DObject(domain, function_space, solution_vector, elasticity_tensor, 
                                 warp_factor: Optional[float | int],
                                ):
    import pyvista
    from dolfinx.plot import vtk_mesh
    import ufl
    from dolfinx import fem, default_scalar_type, plot

    C = ufl.as_matrix(elasticity_tensor)

    p = pyvista.Plotter()
    topology, cell_types, geometry = plot.vtk_mesh(function_space)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    grid["u"] = solution_vector.x.array.reshape((geometry.shape[0], 3))
        
    warped = grid.warp_by_vector("u", factor=warp_factor)
    actor_1 = p.add_mesh(warped, show_edges=True)
    s = three_d_sigma(solution_vector, C) - 1. / 3 * ufl.tr(three_d_sigma(solution_vector, C)) * ufl.Identity(len(solution_vector))
    von_Mises = ufl.sqrt(3. / 2 * ufl.inner(s, s))
    V_von_mises = fem.functionspace(domain, ("DG", 0))
    stress_expr = fem.Expression(von_Mises, V_von_mises.element.interpolation_points())
    stresses = fem.Function(V_von_mises)
    stresses.interpolate(stress_expr)
    warped.cell_data["VonMises"] = stresses.vector.array
    warped.set_active_scalars("VonMises")
    p = pyvista.Plotter()
    p.add_mesh(warped, show_edges=True)
    return p


    