from pyiron_workflow.function import as_function_node


@as_function_node()
def static(atoms=None, engine=None, _internal=None, keys_to_store=None):
    from node_library.atomistic.calculator.data import OutputCalcStatic

    if engine is None:
        from ase.calculators.emt import EMT
        from node_library.atomistic.engine.generic import OutputEngine

        engine = OutputEngine(calculator=EMT())

    atoms.calc = engine

    out = OutputCalcStatic()
    # out['structure'] = atoms # not needed since identical to input
    out.energy = atoms.get_potential_energy()
    out.forces = atoms.get_forces()

    if _internal is not None:
        out["iter_index"] = _internal[
            "iter_index"
        ]  # TODO: move _internal argument to decorator class
    return out.select(keys_to_store)


@as_function_node("out")
def minimize(atoms=None, engine=None, fmax=0.005, log_file="tmp.log"):
    from ase.optimize import BFGS
    from node_library.atomistic.calculator.data import OutputCalcMinimize

    # import numpy as np

    if engine is None:
        from ase.calculators.emt import EMT
        from node_library.atomistic.engine.generic import OutputEngine

        engine = OutputEngine(calculator=EMT())

    atoms.calc = engine.calculator

    if log_file is None:  # write to standard io
        log_file = "-"

    dyn = BFGS(atoms, logfile=log_file)
    dyn.run(fmax=fmax)

    # it appears that r0 is the structure of the second to last step (check)
    atoms_relaxed = atoms.copy()
    atoms_relaxed.calc = atoms.calc
    if dyn.r0 is not None:
        atoms_relaxed.positions = dyn.r0.reshape(-1, 3)

    out = OutputCalcMinimize()
    out.final.structure = atoms_relaxed
    # out["forces"] = dyn.f0.reshape(-1, 3)
    out.final.forces = atoms_relaxed.get_forces()
    out.final.energy = float(atoms_relaxed.get_potential_energy())
    out.initial.energy = float(atoms.get_potential_energy())
    print("energy: ", out.final.energy, out.initial.energy)
    # print("energy: ", out["energy"], "max_force: ", np.min(np.abs(out["forces"])))

    return out


nodes = [
    static,
    minimize,
]
