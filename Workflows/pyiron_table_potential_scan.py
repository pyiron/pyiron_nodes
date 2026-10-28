"""
Example of "data mining using pyiron tables": equilibrium lattice parameter,
bulk modulus and potential per Murnaghan job in a pyiron project, using
BuildTable from pyiron_nodes.databases.pyiron_tables - no pyiron_base /
pyiron_atomistics dependency, reads the project directory and job HDF5 files
directly.

Run it with the path of a project that contains Murnaghan jobs:

    python pyiron_table_potential_scan.py /path/to/project
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


def make_workflow(project_path: str = "") -> Workflow:
    """Table of lattice parameter, bulk modulus and potential for the finished
    Murnaghan jobs in *project_path*. The path has to be set, here or on the
    BuildTable node, before the workflow can run."""
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
    if len(sys.argv) != 2:
        sys.exit(f"usage: python {sys.argv[0]} <path to pyiron project>")
    wf = make_workflow(project_path=sys.argv[1])
    out = wf.run()
    print(out)
