# from pyiron_workflow.function import as_function_node
from pyiron_workflow import as_macro_node
from pyiron_workflow import Workflow


# Workflow.register("node_library.atomistic", "atomistic")


@as_macro_node("chemical_potential")
def get_chemical_potential(
        wf, element: str, engine=None,
):

    import node_library.atomistic as atomistic

    wf.bulk = Workflow.create.atomistic.structure.build.bulk(name=element)  # con: no autocompletion
    wf.minimize = atomistic.calculator.ase.minimize(structure=wf.bulk, engine=engine)  # pressure = 0
    wf.n_atoms = atomistic.structure.calc.number_of_atoms(structure=wf.bulk)
    wf.energy = atomistic.calculator.output.get_energy_last(calculator=wf.minimize)

    return wf.energy / wf.n_atoms


nodes = [get_chemical_potential]
