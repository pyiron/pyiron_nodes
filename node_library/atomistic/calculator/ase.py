from pyiron_workflow.function import as_function_node


@as_function_node()
def static(structure=None, engine=None, keys_to_store=None, job_name=None):  # , _internal=None
    import numpy as np
    from node_library.atomistic.calculator.data import OutputCalcStatic, OutputCalcStaticList

    if engine is None:
        from ase.calculators.emt import EMT
        from node_library.atomistic.engine.generic import OutputEngine

        engine = OutputEngine(calculator=EMT())

    # print ('engine: ', engine)
    # print ('engine (calculator): ', engine.calculator)
    import ase
    import node_library.atomistic.property.elastic as elastic
    if isinstance(structure, ase.atoms.Atoms):
        structure.calc = engine.calculator

        out = OutputCalcStatic()
        # out['structure'] = atoms # not needed since identical to input
        out.energy = np.array([float(structure.get_potential_energy())])  # TODO: originally of type np.float32 -> why??
        out.forces = np.array([structure.get_forces()])

        # print("energy: ", out.energy)
    elif isinstance(structure, np.ndarray):
        print('Implement list')
    elif isinstance(structure, elastic.DataStructureContainer):
        print('structures from DataContainer')
        structures = structure['structure']
        out = OutputCalcStaticList()
        out.energies = []
        for structure in structures:
            structure.calc = engine.calculator

            out.energies.append(np.array([float(structure.get_potential_energy())]))


    else:
        print('error (not implemented): ',type(structure))

    # if _internal is not None:
    #     out["iter_index"] = _internal[
    #         "iter_index"
    #     ]  # TODO: move _internal argument to decorator class

    return out  # .select(keys_to_store)


@as_function_node("out")
def minimize(structure=None, engine=None, fmax=0.005, log_file="tmp.log"):
    from ase.optimize import BFGS
    from node_library.atomistic.calculator.data import OutputCalcMinimize

    # import numpy as np

    if engine is None:
        from ase.calculators.emt import EMT
        from node_library.atomistic.engine.generic import OutputEngine

        engine = OutputEngine(calculator=EMT())

    structure.calc = engine.calculator

    if log_file is None:  # write to standard io
        log_file = "-"

    dyn = BFGS(structure, logfile=log_file)
    dyn.run(fmax=fmax)

    # it appears that r0 is the structure of the second to last step (check)
    atoms_relaxed = structure.copy()
    atoms_relaxed.calc = structure.calc
    if dyn.r0 is not None:
        atoms_relaxed.positions = dyn.r0.reshape(-1, 3)

    out = OutputCalcMinimize()
    out.final.structure = atoms_relaxed
    # out["forces"] = dyn.f0.reshape(-1, 3)
    out.final.forces = atoms_relaxed.get_forces()
    out.final.energy = float(atoms_relaxed.get_potential_energy())
    out.initial.energy = float(structure.get_potential_energy())
    # print("energy: ", out.final.energy, out.initial.energy)
    # print("energy: ", out["energy"], "max_force: ", np.min(np.abs(out["forces"])))

    return out


nodes = [
    static,
    minimize,
]
