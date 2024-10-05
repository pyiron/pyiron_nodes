from dataclasses import field
from pyiron_workflow import (
    as_function_node,
    as_macro_node,
    for_node,
    standard_nodes as standard,
)
from typing import Optional

from pyiron_nodes.dev_tools import wf_data_class
from pyiron_workflow import as_dataclass_node

@as_dataclass_node
# @wf_data_class()
class BarParameters:
    length: Optional[float|int] = 3.0
    width: Optional[float|int] = 0.4
    depth: Optional[float|int] = 0.4
    density: Optional[float|int] = 1.0


@as_function_node("domain")
def Bar(
    x0: Optional[float|int],
    y0: Optional[float|int],
    z0: Optional[float|int],
    min_mesh: Optional[float|int],
    max_mesh: Optional[float|int],
    parameters: Optional[BarParameters.dataclass] = BarParameters.dataclass(),
):
    
    from mpi4py import MPI
    from dolfinx.io import gmshio
    import gmsh

    gmsh.initialize()

    gmsh.clear()
    box=gmsh.model.occ.addBox(x0,y0,z0,parameters.length,parameters.width,parameters.depth)
    gmsh.model.occ.synchronize()
    gdim = 3
    gmsh.model.addPhysicalGroup(gdim, [box], 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min_mesh)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_mesh)
    #gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
    gmsh.model.mesh.generate(gdim)
    gmsh_model_rank = 0
    mesh_comm = MPI.COMM_WORLD
    domain, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=gdim)
    return domain