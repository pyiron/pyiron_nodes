from __future__ import annotations

from core import group_node


@group_node("chemical_potential")
def GetChemicalPotential(
    element: str,
    engine=None,
):
    from core import Workflow
    from pyiron_nodes.atomistic.calculator.ase import Minimize
    from pyiron_nodes.atomistic.calculator.output import GetEnergyLast
    from pyiron_nodes.atomistic.structure.build import Bulk
    from pyiron_nodes.atomistic.structure.calc import NumberOfAtoms

    wf = Workflow("GetChemicalPotential")
    wf.bulk = Bulk(name=element)
    wf.minimize = Minimize(structure=wf.bulk, engine=engine)  # pressure = 0
    wf.n_atoms = NumberOfAtoms(structure=wf.bulk)
    wf.energy = GetEnergyLast(calculator=wf.minimize)

    return wf.energy / wf.n_atoms
