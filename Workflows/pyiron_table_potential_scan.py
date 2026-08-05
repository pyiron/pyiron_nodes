"""
Reproduces the "Data mining using pyiron tables" notebook
(potential_scan project: equilibrium lattice parameter / bulk modulus per
potential, from the Murnaghan jobs) using BuildTable from
pyiron_nodes.databases.pyiron_tables - no pyiron_base / pyiron_atomistics
dependency, reads the project directory and job HDF5 files directly.
"""

import sys

from core import Workflow
from pyiron_nodes.databases.pyiron_tables import (
    AddBulkModulus,
    AddLatticeParameter,
    AddPotential,
    BuildTable,
    DbFilterFunction,
)


def make_workflow(
    project_path: str = "/u/pchilaka/1_Work/1_My_Notebooks/1_Beginners_Guide/DONE/potential_scan",
) -> Workflow:
    wf = Workflow("pyiron_table_potential_scan")

    wf.DbFilterFunction = DbFilterFunction(hamilton="Murnaghan")

    # each node takes the previous one's output back in as `functions` and
    # grows the same dict, like AddPristine chains a StructureContainer - no
    # separate merge node needed. Each node wraps one named function
    # (get_bulk_modulus, get_potential, get_lattice_parameter) from
    # pyiron_tables.py.
    wf.BulkModulusFunction = AddBulkModulus()
    wf.PotentialFunction = AddPotential(functions=wf.BulkModulusFunction)
    wf.LatticeParameterFunction = AddLatticeParameter(functions=wf.PotentialFunction)

    wf.BuildTable = BuildTable(
        project_path=project_path,
        functions=wf.LatticeParameterFunction,
        status=["finished"],
        db_filter_function=wf.DbFilterFunction,
    )
    return wf


# module-level graph so UIs that exec() this file (e.g. pyironflow) can find it
wf = make_workflow()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        wf = make_workflow(project_path=sys.argv[1])
    out = wf.run()
    print(out)
