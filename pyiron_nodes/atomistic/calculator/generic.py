from typing import Optional
import numpy as np

from pyiron_nodes.atomistic.calculator.data import (
    InputCalcStatic,
    OutputSEFS,
)
from core import Node, as_function_node


@as_function_node("generic")
def Static(structure=None, engine=None):  # , keys_to_store=None):
    output = engine(
        structure=structure,
        calculator=InputCalcStatic(),  # keys_to_store=keys_to_store)
    )
    return output.generic


@as_function_node
def ApplyEngine(
    sefs_container: OutputSEFS = None,
    engine=None,
    energies: bool = True,
    forces: bool = False,
    stresses: bool = False,
    store: bool = True,
) -> OutputSEFS:
    for structure in sefs_container.structures:
        structure.calc = engine.calculator
        if energies:
            if sefs_container.energies is None:
                sefs_container.energies = []
            sefs_container.energies.append(structure.get_potential_energy())
        if forces:
            if sefs_container.forces is None:
                sefs_container.forces = []
            sefs_container.forces.append(structure.get_forces())
        if stresses:
            if sefs_container.stresses is None:
                sefs_container.stresses = []
            sefs_container.stresses.append(structure.get_stresses())

    return sefs_container


@as_function_node
def CreateSEFSContainer(
    structures: list | np.ndarray,
    energies: Optional[list] = None,
    forces: Optional[list] = None,
    stresses: Optional[list] = None,
) -> OutputSEFS:
    sefs_container = OutputSEFS.pure_dataclass()
    sefs_container.structures = structures
    if energies is not None:
        sefs_container.energies = energies
    if forces is not None:
        sefs_container.forces = forces
    if stresses is not None:
        sefs_container.stresses = stresses
    return sefs_container
