from pyiron_workflow import as_function_node

from typing import Optional
from ase import Atoms


@as_function_node("plot")
def plot3d(structure: Optional[Atoms], particle_size: int = 1):
    return structure.plot3d(particle_size=particle_size)


nodes = [plot3d]
