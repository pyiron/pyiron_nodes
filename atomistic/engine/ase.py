from __future__ import annotations

from functools import lru_cache

from core import as_function_node


@as_function_node("engine")
def EMT():
    from ase.calculators.emt import EMT

    from pyiron_nodes.atomistic.engine.generic import OutputEngine

    out = OutputEngine(calculator=EMT())

    return out


# @as_function_node("engine")
# def M3GNet(model: str = "M3GNet-MP-2021.2.8-PES"):
#     """M3GNet: A universal neural network potential for atomistic simulations."""
#     import matgl

#     try:
#         # matgl >= 1.1 renamed M3GNetCalculator to the model-agnostic PESCalculator
#         from matgl.ext.ase import PESCalculator
#     except ImportError:  # pragma: no cover - older matgl
#         from matgl.ext.ase import M3GNetCalculator as PESCalculator

#     from pyiron_nodes.atomistic.engine.generic import OutputEngine

#     out = OutputEngine(calculator=PESCalculator(matgl.load_model(model)))
#     return out


@as_function_node("engine")
@lru_cache
def GRACE(model: str = "GRACE-FS-OAM", use_symmetry: bool = True):
    """Universal Graph Atomic Cluster Expansion models.

    Parameters
    ----------
    model : str, optional
        Model identifier, default ``"GRACE-FS-OAM"``.
    use_symmetry : bool, optional
        Whether to use symmetry optimizations.

    Returns
    -------
    engine
        ``OutputEngine`` holding the initialised GRACE calculator.  The
        ``lru_cache`` keeps the loaded model in memory across calls, which
        dominates the cost of using GRACE at all.
    """
    import hashlib
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    from tensorpotential.calculator import grace_fm

    from pyiron_nodes.atomistic.engine.generic import OutputEngine

    # A stable engine_id lets node_inputs_to_jsongroup contribute a reproducible
    # hash without pickling the calculator, which is neither cheap nor picklable.
    eid = int(
        hashlib.sha256(f"GRACE:{model}:{use_symmetry}".encode()).hexdigest()[:8], 16
    )
    out = OutputEngine(calculator=grace_fm(model), engine_id=eid)
    return out


@as_function_node("engine")
def LammpsEngine(potential: str, resource_path: str = None):
    """A LAMMPS potential from the interatomic-potentials repository, as an engine.

    Wraps ``ase.calculators.lammpslib.LAMMPSlib`` around the same catalogue
    entry the calphy and LAMMPS nodes take by name, so an ASE-driven node and a
    LAMMPS-driven one can be given *the same* potential.  Without this,
    ``EstimateCalphyTemperatureRange`` (which needs an ASE calculator) could
    only be paired with a different potential from the one
    ``SolidFreeEnergyWithTemp`` runs — and a temperature window is only valid
    for the potential it was measured with.

    Parameters
    ----------
    potential : str
        Catalogue name, e.g. ``"1999--Mishin-Y--Al--LAMMPS--ipr1"``.  Use
        ``ListPotentials`` to see what is available for a structure.
    resource_path : str, optional
        Override for the potential resource directory.

    Returns
    -------
    engine
        ``OutputEngine`` holding a ``LAMMPSlib`` calculator.  LAMMPS atom types
        are numbered in the order the catalogue lists ``Species``, which is the
        order its ``pair_coeff`` line uses.
    """
    import hashlib

    from ase.calculators.lammpslib import LAMMPSlib

    from pyiron_nodes.atomistic.engine.generic import OutputEngine
    from pyiron_nodes.atomistic.engine.lammps import get_usable_potential_by_name

    entry = get_usable_potential_by_name(
        potential_name=potential, resource_path=resource_path
    )
    species = list(entry["Species"])
    calc = LAMMPSlib(
        lmpcmds=list(entry["Config"]),
        atom_types={element: i + 1 for i, element in enumerate(species)},
        keep_alive=True,
        log_file=None,
    )
    # Stable across sessions, so cached results keyed on this engine stay valid.
    eid = int(hashlib.sha256(f"LAMMPSlib:{potential}".encode()).hexdigest()[:8], 16)
    out = OutputEngine(calculator=calc, engine_id=eid)
    return out


@as_function_node("engine")
def Ace(potential_file, use_symmetry: bool = True):
    """Atomic Cluster Expansion potential read from a ``.yaml``/``.yace`` file."""
    from logging import ERROR

    from pyace import PyACECalculator
    from pyiron_snippets.logger import logger

    from pyiron_nodes.atomistic.engine.generic import OutputEngine

    calc = PyACECalculator(potential_file)
    logger.setLevel(ERROR)
    out = OutputEngine(calculator=calc)
    return out
