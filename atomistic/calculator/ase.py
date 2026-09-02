"""Single-point and relaxation nodes built on ASE calculators."""

from dataclasses import field
from enum import Enum
from typing import List, Literal, Optional

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.filters import FrechetCellFilter

from pyiron_nodes.atomistic.engine.generic import OutputEngine
from pyiron_nodes.atomistic.structure._atoms import (
    OutputAtoms,
    _ase_to_data,
    _data_to_ase,
    _resolve_atoms,
)
from core import Node, as_function_node, as_inp_dataclass_node, as_out_dataclass_node


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
    """Result container for static and optimizer calculations.

    Holds the (relaxed) structure and associated physical quantities.
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
                raise ValueError(
                    f"Expected to match available enum: {RelaxMode.__members__}"
                )


@as_function_node
def Static(
    structure: Atoms,
    engine=None,
    store: bool = False,
) -> Atoms:
    """Perform a static (single-point) calculation on a structure.

    This node evaluates a single-point ASE calculation using the provided
    ``engine``. It does not modify the geometry; instead it returns the total
    energy, forces and stress tensor for the given ``structure``.

    Parameters
    ----------
    structure : Atoms
        The atomic structure on which to compute properties.
    engine : OutputEngine or None, optional
        Engine providing a calculator.  Defaults to EMT when ``None``.
    store : bool
        If True, cache node outputs using hash-based storage for faster re-execution.
        Useful for expensive calculations that may be reused.

    Returns
    -------
    OutputCalcOpt
        Dataclass containing the structure, energy, forces and stress.
    """
    atoms = _resolve_atoms(structure).copy()

    if engine is None:
        from ase.calculators.emt import EMT

        engine = OutputEngine(calculator=EMT())

    atoms.calc = engine.calculator

    # Convert Atoms to OutputAtoms for storage compatibility
    out = OutputCalcOpt.pure_dataclass(
        structure=_ase_to_data(atoms),
        energy=atoms.get_potential_energy(),
        forces=atoms.get_forces(),
        stress=atoms.get_stress(),
    )

    return out


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

    if opt_parameters is None:
        opt_parameters = GenericOptimizerSettings._original_dataclass()
    mode = RelaxMode(opt_mode.lower())

    structure = _resolve_atoms(structure).copy()
    structure.calc = engine.calculator

    filtered_structure = mode.apply_filter_and_constraints(structure)
    lbfgs = LBFGS(filtered_structure, logfile="/dev/null")
    lbfgs.run(fmax=opt_parameters.force_tolerance, steps=opt_parameters.max_steps)
    calc = structure.calc

    # Convert Atoms to OutputAtoms for storage compatibility
    out = OutputCalcOpt.pure_dataclass(
        structure=_ase_to_data(structure),
        energy=calc.get_potential_energy(),
        forces=calc.get_forces(),
        stress=calc.get_stress(),
    )

    structure.constraints.clear()
    return out


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
        If True, cache node outputs using hash-based storage.
    t_forces, t_stress, t_volume : bool
        Flags indicating whether to store forces, stress, and volume information.
    store_structures : bool
        If True, retain the relaxed structures in the returned DataFrame.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing energies and optionally forces, stresses, volumes.
    """
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
def StaticEnergy(
    structure,
    engine: OutputEngine,
):
    """
    Compute the potential energy of *structure*.

    Parameters
    ----------
    structure : Atoms or OutputAtoms
        The atomic structure to calculate.
    engine : OutputEngine
        Calculator engine.

    Returns
    -------
    float
        Potential energy in eV.
    """
    atoms = _resolve_atoms(structure)
    atoms.calc = engine.calculator
    energy = atoms.get_potential_energy()

    return energy


@as_function_node("out", isolate=True)
def Minimize(
    structure=None,
    engine=None,
    fmax: float = 0.005,
    log_file: str = "tmp.log",
):
    """
    Minimise *structure* using the BFGS algorithm.

    Parameters
    ----------
    structure : Atoms or OutputAtoms or None, optional
        The atomic structure to relax.
    engine : OutputEngine or None, optional
        Calculator engine.  Defaults to EMT when ``None``.
    fmax : float, optional
        Maximum force convergence criterion (eV/Å).  Default ``0.005``.
    log_file : str or None, optional
        Path to the BFGS log file.  Pass ``None`` to write to stdout.
        Default ``"tmp.log"``.

    Returns
    -------
    OutputCalcStaticList dataclass instance
    """
    from ase.io.trajectory import Trajectory
    from ase.optimize import BFGS
    from pyiron_nodes.atomistic.calculator.data import OutputCalcStaticList

    atoms = _resolve_atoms(structure)

    if engine is None:
        from ase.calculators.emt import EMT

        engine = OutputEngine(calculator=EMT())

    out = OutputCalcStaticList.pure_dataclass()
    out.energies_pot = []
    out.forces = []
    out.structures = []

    # Initial structure
    out.structures.append(atoms)
    initial = atoms.copy()
    initial.calc = engine.calculator
    out.energies_pot.append(float(initial.get_potential_energy()))
    out.forces.append(initial.get_forces())

    if log_file is None:
        log_file = "-"

    dyn = BFGS(initial, logfile=log_file, trajectory="minimize.traj")
    dyn.run(fmax=fmax)

    traj = Trajectory("minimize.traj")
    atoms_relaxed = traj[-1]  # has SinglePointCalculator with stored results

    forces_relaxed = atoms_relaxed.get_forces()
    out.forces.append(forces_relaxed)
    out.energies_pot.append(float(atoms_relaxed.get_potential_energy()))
    out.is_converged = dyn.converged(forces_relaxed.flatten())
    out.iter_steps = dyn.nsteps

    # ASE calculators are not picklable — detach before storing
    atoms_relaxed.calc = None
    out.structures.append(atoms_relaxed)

    return out
