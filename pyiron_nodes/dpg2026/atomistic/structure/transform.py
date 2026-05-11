from ase.atoms import Atoms
from enum import Enum
import numpy as np
from core import as_function_node


def rattle(structure: Atoms, sigma: float) -> Atoms:
    """Randomly displace positions with gaussian noise.

    Operates INPLACE."""
    structure.rattle(stdev=sigma)
    return structure


def stretch(structure: Atoms, hydro: float, shear: float) -> Atoms:
    """Randomly stretch cell with uniform noise.

    Operates INPLACE."""
    strain = shear * (2 * np.random.rand(3, 3) - 1)
    strain = 0.5 * (strain + strain.T)  # symmetrize
    np.fill_diagonal(strain, 1 + hydro * (2 * np.random.rand(3) - 1))
    structure.set_cell(structure.cell.array @ strain, scale_atoms=True)
    return structure


@as_function_node
def RattleAndStrech(structure: Atoms, sigma: float, samples: int) -> list[Atoms]:
    structures = []
    # no point in rattling single atoms
    if len(structure) > 1:
        for _ in range(samples):
            structures.append(
                    stretch(
                        rattle(structure.copy(), sigma),
                        hydro=0.05, shear=0.005
                    )
            )
    return structures


@as_function_node
def Rattle(structure, seed: int = 42, stdev: float = 0.1):
    """
    Randomly displace the atoms in the structure.

    Parameters:

    seed = 42:
    Random seed for the random number generator.
    amplitude = 0.1:
    The amplitude of the random displacement.
    """
    structure = structure.copy()
    structure.rattle(seed=seed, stdev=stdev)
    return structure


@as_function_node
def RattleLoop(
        structures: list[Atoms], sigma: float, samples: int
) -> list[Atoms]:
    rattled_structures = []
    for structure in structures:
        rattled_structures += RattleAndStrech(structure, sigma, samples).pull()
    return rattled_structures


@as_function_node
def Stretch(
        structure: Atoms, hydro: float, shear: float, samples: int,
        hydro_shear_ratio: float = 0.7
) -> list[Atoms]:
    structures = []
    for _ in range(samples):
        if np.random.rand() < hydro_shear_ratio:
            ihydro, ishear = hydro, 0.05
        else:
            ihydro, ishear = 0.05, shear
        structures.append(
                stretch(structure.copy(), hydro=ihydro, shear=ishear)
        )
    return structures


@as_function_node
def StretchLoop(
        structures: list[Atoms], hydro: float, shear: float, samples: int,
        hydro_shear_ratio: float = 0.7
) -> list[Atoms]:
    stretched_structures = []
    for structure in structures:
        stretched_structures += Stretch(
                structure, hydro, shear, samples, hydro_shear_ratio
        ).pull()
    return stretched_structures


@as_function_node("structure")
def Repeat(structure: Atoms, repeat_scalar: int = 1) -> Atoms:
    """
    Repeat a crystal structure periodically along all lattice vectors.

    Parameters
    ----------
    structure : Atoms
        The ASE ``Atoms`` object to be repeated.
    repeat_scalar : int, optional
        Number of repetitions along each lattice vector (default is ``1`` – no change).

    Returns
    -------
    Atoms
        A new ``Atoms`` object containing the repeated supercell.

    Task hint
    ----------
    Use this node when the workflow requires building a larger supercell
    from a primitive cell (e.g., “create a 2×2×2 bulk Al supercell”).
    """
    return structure.repeat(int(repeat_scalar))
