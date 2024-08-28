from __future__ import annotations

from pyiron_workflow import as_macro_node, Workflow


@as_macro_node("chemical_potential")
def get_chemical_potential(
    wf,
    element: str,
    engine=None,
):

    import pyiron_nodes.atomistic as atomistic

    wf.bulk = atomistic.structure.build.Bulk(name=element)  # con: no autocompletion
    wf.minimize = atomistic.calculator.ase.minimize(
        structure=wf.bulk, engine=engine
    )  # pressure = 0
    wf.n_atoms = atomistic.structure.calc.number_of_atoms(structure=wf.bulk)
    wf.energy = atomistic.calculator.output.get_energy_last(calculator=wf.minimize)

    return wf.energy / wf.n_atoms
