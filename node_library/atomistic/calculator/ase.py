from pyiron_workflow import as_function_node


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
        print('error (not implemented): ', type(structure))

    # if _internal is not None:
    #     out["iter_index"] = _internal[
    #         "iter_index"
    #     ]  # TODO: move _internal argument to decorator class

    return out  # .select(keys_to_store)


@as_function_node("out")
def minimize(structure=None, engine=None, fmax=0.005, log_file="tmp.log"):
    from ase.optimize import BFGS
    from ase.io.trajectory import Trajectory
    from node_library.atomistic.calculator.data import OutputCalcMinimize

    # import numpy as np

    if engine is None:
        from ase.calculators.emt import EMT
        from node_library.atomistic.engine.generic import OutputEngine

        engine = OutputEngine(calculator=EMT())

    out = OutputCalcMinimize()

    initial_structure = structure.copy()
    initial_structure.calc = engine.calculator
    out.initial.energy = float(initial_structure.get_potential_energy())
    out.initial.forces = initial_structure.get_forces()

    if log_file is None:  # write to standard io
        log_file = "-"

    dyn = BFGS(initial_structure, logfile=log_file, trajectory='minimize.traj')
    out_dyn = dyn.run(fmax=fmax)

    traj = Trajectory('minimize.traj')
    atoms_relaxed = traj[-1]
    atoms_relaxed.calc = engine.calculator

    out.final.forces = atoms_relaxed.get_forces()
    out.final.energy = float(atoms_relaxed.get_potential_energy())
    atoms_relaxed.calc = None # ase calculator is not pickable!!
    out.final.structure = atoms_relaxed

    out.is_converged = dyn.converged()
    out.iter_steps = dyn.nsteps

    return out


nodes = [
    static,
    minimize,
]
