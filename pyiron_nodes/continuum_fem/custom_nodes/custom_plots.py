from pyiron_workflow import as_function_node
from typing import Optional

@as_function_node("load_deflection_plot")
def PoissonLoadDeflectionPlot(
    domain, 
    solution_vector, 
    load
):
    from dolfinx import fem
    import numpy as np
    import matplotlib.pyplot as plt
    from dolfinx import geometry

    Q = fem.functionspace(domain, ("Lagrange", 5))
    expr = fem.Expression(load, Q.element.interpolation_points())
    pressure = fem.Function(Q)
    pressure.interpolate(expr)
    tol = 0.001  # Avoid hitting the outside of the domain
    y = np.linspace(-1 + tol, 1 - tol, 101)
    points = np.zeros((3, 101))
    points[1] = y
    u_values = []
    p_values = []
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cells = []
    points_on_proc = []
    # Find cells whose bounding-box collide with the the points
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    # Choose one of the cells that contains the point
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    for i, point in enumerate(points.T):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
        
    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    u_values = solution_vector.eval(points_on_proc, cells)
    p_values = pressure.eval(points_on_proc, cells)
    fig = plt.figure()
    plt.plot(points_on_proc[:, 1], 50 * u_values, "k", linewidth=2, label="Deflection ($\\times 50$)")
    plt.plot(points_on_proc[:, 1], p_values, "b--", linewidth=2, label="Load")
    plt.grid(True)
    plt.xlabel("y")
    plt.legend()
    return plt.show()