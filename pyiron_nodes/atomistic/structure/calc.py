from pyiron_workflow import as_function_node
from typing import Optional, Union

# Huge savings when replacing pyiron_atomistics atoms class with ase one!! (> 5s vs 40 ms)
# from pyiron_atomistics.atomistics.structure.atoms import Atoms
from ase import Atoms


@as_function_node("volume")
def volume(structure: Optional[Atoms] = None) -> float:
    return structure.get_volume()


@as_function_node("number_of_atoms")
def number_of_atoms(structure: Optional[Atoms] = None) -> int:
    return structure.get_number_of_atoms()


nodes = [volume, number_of_atoms]
