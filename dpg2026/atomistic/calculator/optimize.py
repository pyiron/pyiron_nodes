from dataclasses import field
from ase.atoms import Atoms
from ase.constraints import FixAtoms
from ase.filters import FrechetCellFilter
from enum import Enum
import numpy as np
from core import as_function_node, Node, as_inp_dataclass_node, as_out_dataclass_node
from core.data_fields import DataArray, EmptyArrayField
from typing import Literal, List, Optional
import pandas
from pyiron_nodes.atomistic.structure._atoms import (
    _ase_to_data,
    _data_to_ase,
    OutputAtoms,
)

"""Utility nodes for atomistic calculations using ASE within the pyiron workflow framework.

This module provides a concise yet comprehensive suite of helper nodes that
streamline common atomistic simulation tasks when building pyiron workflows.
"""


@as_inp_dataclass_node
class GenericOptimizerSettings:
    """Configuration parameters for generic optimization runs.

    Attributes
    ----------
    max_steps : int
        Maximum number of optimization steps.
    force_tolerance : float
        Convergence criterion for the maximum force (in eV/Å).
    """

    max_steps: int = 10
    force_tolerance: float = 1e-2


@as_out_dataclass_node
class OutputCalcOpt:
    """Result container for optimizer calculations.

    Holds the relaxed structure and associated physical quantities.
    """

    structure: Optional[List] = field(default=None)
    energy: float = 0
    forces: Optional[np.ndarray] = field(default=None)
    stress: Optional[np.ndarray] = field(default=None)


class RelaxMode(Enum):
    VOLUME = "volume"
    # CELL = "cell"
    INTERNAL = "internal"
    FULL = "full"

    def apply_filter_and_constraints(self, structure):
        match self:
            case RelaxMode.VOLUME:
                structure.set_constraint(FixAtoms(np.ones(len(structure), dtype=bool)))
                return FrechetCellFilter(
                    structure, constant_volume=True
                )  # hydrostatic_strain=True,
            case RelaxMode.INTERNAL:
                return structure
            case RelaxMode.FULL:
                return FrechetCellFilter(structure)
            case _:
                raise ValueError("Lazy Marvin")


@as_function_node
def MapCalculatorOnStructures(
    structures,
    calculator: Node,
    store: bool = False,
    t_forces: bool = False,
    t_stress: bool = False,
    t_volume: bool = False,
    store_structures: bool = False,
):
    """Map a calculator over a collection of structures.

    Parameters
    ----------
    structures : list, np.ndarray, or pandas.DataFrame
        Input structures to be processed.
    calculator : Node
        Calculator node that evaluates energies/forces/etc.
    store : bool
        If True, keep the original input structures.
    t_forces, t_stress, t_volume : bool
        Flags indicating whether to store forces, stress, and volume information.
    store_structures : bool
        If True, retain the relaxed structures in the returned DataFrame.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing energies and optionally forces, stresses, volumes.
    """
    from tqdm.auto import tqdm
    import pandas as pd

    if hasattr(structures, "structure"):
        atoms_list = structures.structure
    elif isinstance(structures, (list, np.ndarray)):
        atoms_list = structures
    else:
        raise ValueError("Unknown data type for structures")

    relaxed_structures, energies, forces, stresses, volumes = [], [], [], [], []
    for structure in atoms_list:
        if isinstance(structure, OutputAtoms):
            structure = _data_to_ase(structure)
        calculator.inputs.structure = structure
        out = calculator.pull()
        # print("out: ", out)
        relaxed_structures.append(_ase_to_data(out.structure))
        energies.append(out.energy)
        if t_forces:
            forces.append(out.forces)
        if t_stress:
            stresses.append(out.stress)

    if isinstance(structures, pd.DataFrame):
        df = structures.copy()
        df.structure = relaxed_structures
    else:
        df = pd.DataFrame({"structure": relaxed_structures, "energy": energies})

    df["energy"] = energies
    if t_forces:
        df["forces"] = forces
    if t_stress:
        df["stress"] = stresses
    if not store_structures:
        del df["structure"]

    return df


@as_function_node
def Relax(
    structure: Atoms,
    engine,
    opt_parameters: Optional[GenericOptimizerSettings] = None,
    opt_mode: Literal["volume", "full"] = "volume",
    store: bool = False,
) -> Atoms:
    """Relax a structure using the specified engine and optimizer settings.

    Parameters
    ----------
    structure : Atoms
        The atomic structure to be relaxed.
    engine : Any
        Engine providing a calculator for the relaxation.
    opt_parameters : GenericOptimizerSettings
        Optimizer settings such as ``max_steps`` and ``force_tolerance``.
    opt_mode : Literal["volume", "full"]
        Mode of relaxation; ``"volume"`` constrains volume, ``"full"`` relaxes cell and atoms.
    store : bool
        If True, cache node outputs using hash-based storage for faster re-execution.
        Highly recommended for expensive relaxation calculations.

    Returns
    -------
    OutputCalcOpt
        Dataclass containing the relaxed structure, energy, forces and stress.
    """
    from ase.optimize import LBFGS
    from pyiron_nodes.atomistic.structure._atoms import to_ase

    if opt_parameters is None:
        opt_parameters = GenericOptimizerSettings()
    print("opt_parameters: ", opt_parameters)
    # print("mode: ", opt_mode, engine, structure)
    mode = RelaxMode(opt_mode.lower())

    structure = to_ase(structure).copy()
    calculator = engine.calculator
    # print("calculator: ", calculator)

    structure.calc = calculator

    filtered_structure = mode.apply_filter_and_constraints(structure)
    lbfgs = LBFGS(filtered_structure, logfile="/dev/null")
    lbfgs.run(fmax=opt_parameters.force_tolerance, steps=opt_parameters.max_steps)
    calc = structure.calc

    # Convert Atoms to OutputAtoms for storage compatibility
    structure_data = _ase_to_data(structure)

    out = OutputCalcOpt.pure_dataclass(
        structure=structure_data,
        energy=calc.get_potential_energy(),
        forces=calc.get_forces(),
        stress=calc.get_stress(),
    )

    structure.constraints.clear()
    return out


@as_function_node
def Static(
    structure: Atoms,
    engine,
    store: bool = False,
) -> Atoms:
    """Perform a static (single‑point) calculation on a structure.

    This node evaluates a single‑point ASE calculation using the provided
    ``engine``. It does not modify the geometry; instead it returns the total
    energy, forces and stress tensor for the given ``structure``.

    Parameters
    ----------
    structure : Atoms
        The atomic structure on which to compute properties.
    engine : Any
        Engine providing a calculator for the calculation.
    store : bool
        If True, cache node outputs using hash-based storage for faster re-execution.
        Useful for expensive calculations that may be reused.

    Returns
    -------
    OutputCalcOpt
        Dataclass containing the structure, energy, forces and stress.
    """

    from pyiron_nodes.atomistic.structure._atoms import to_ase
    structure = to_ase(structure).copy()
    print("alat: ", structure.cell[0, 0])
    structure.calc = engine.calculator

    # Convert Atoms to OutputAtoms for storage compatibility
    structure_data = _ase_to_data(structure)

    out = OutputCalcOpt.pure_dataclass(
        structure=structure_data,
        energy=structure.get_potential_energy(),
        forces=structure.get_forces(),
        stress=structure.get_stress(),
    )

    return out
