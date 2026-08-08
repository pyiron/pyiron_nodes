from ase import Atoms

from pyiron_nodes.atomistic.engine.generic import OutputEngine
from pyiron_nodes.atomistic.structure._atoms import _resolve_atoms
from core import as_function_node


@as_function_node
def Static(
    structure,
    engine=None,
):
    """
    Run a single-point static calculation.

    Parameters
    ----------
    structure : Atoms or OutputAtoms
        The atomic structure to calculate.
    engine : OutputEngine or None, optional
        Calculator engine.  Defaults to EMT when ``None``.

    Returns
    -------
    OutputCalcStaticList dataclass instance
    """
    import numpy as np
    from pyiron_nodes.atomistic.calculator.data import OutputCalcStaticList

    atoms = _resolve_atoms(structure)

    if engine is None:
        from ase.calculators.emt import EMT

        engine = OutputEngine(calculator=EMT())

    atoms.calc = engine.calculator

    out = OutputCalcStaticList.pure_dataclass()
    out.energies_pot = np.array([float(atoms.get_potential_energy())])
    out.forces = np.array([atoms.get_forces()])

    return out


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


@as_function_node("out")
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
