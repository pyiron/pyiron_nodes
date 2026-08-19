from ase.atoms import Atoms
import logging
from core import as_function_node


@as_function_node
def CombineStructures(
    spacegroups: list[Atoms] | None,
    volume_relax: list[Atoms] | None,
    full_relax: list[Atoms] | None,
    rattle: list[Atoms] | None,
    stretch: list[Atoms] | None,
    store: bool = True,
):
    """Combine individual structure sets into a full training set."""
    from functools import reduce

    structures = [spacegroups, volume_relax, full_relax, rattle, stretch]
    structures = reduce(list.__add__, (s or [] for s in structures), [])
    if len(structures) == 0:
        logging.warn(
            "Either no inputs given or all inputs are empty. "
            "Returning the empty list!"
        )
    return structures


@as_function_node
def generate_structures(structure, strain_lst):
    structure_lst = []
    for strain in strain_lst:
        structure_strain = structure.copy()
        structure_strain.set_cell(
            structure_strain.cell * strain ** (1 / 3), scale_atoms=True
        )
        structure_lst.append(structure_strain)
    return structure_lst
