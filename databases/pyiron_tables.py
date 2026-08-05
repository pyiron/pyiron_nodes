"""
Recreates pyiron_base's job-table / PyironTable data mining functionality
(pyiron_base.database.filetable.FileTable + pyiron_base.jobs.datamining.PyironTable/TableJob)
as plain functions, reading directly from the project directory and job HDF5 files.

No dependency on pyiron_base or pyiron_atomistics - only on the lightweight libraries
pyiron_base itself builds on: pyfileindex (file-system indexing) and h5io_browser
(HDF5 read access).
"""

from core import as_function_node

# ---------------------------------------------------------------------------
# low level HDF5 / file-system helpers
# ---------------------------------------------------------------------------


def _read_hdf_value(file_name: str, h5_path: str):
    from h5io_browser.base import _read_hdf

    return _read_hdf(hdf_filehandle=file_name, h5_path=h5_path)


def _parse_job_type(type_string: str) -> str:
    # type_string looks like "<class 'pyiron_atomistics.lammps.lammps.Lammps'>"
    return type_string.split(".")[-1].split("'")[0]


def get_job_status(file_name: str, job_name: str):
    import os

    if not os.path.exists(file_name):
        return None
    try:
        return _read_hdf_value(file_name, job_name + "/status")
    except (KeyError, OSError):
        return None


def get_job_type(file_name: str, job_name: str):
    try:
        return _parse_job_type(_read_hdf_value(file_name, job_name + "/TYPE"))
    except (KeyError, OSError):
        return None


def get_job_id(file_name: str, job_name: str):
    try:
        return _read_hdf_value(file_name, job_name + "/job_id")
    except (KeyError, OSError):
        return None


def read_value(file_name: str, job_name: str, path: str):
    """
    Read an arbitrary value out of a job's HDF5 file, analogous to `job["<path>"]`
    in pyiron_base, e.g. read_value(file_name, job_name, "output/generic/energy_pot")
    """
    return _read_hdf_value(file_name, job_name + "/" + path)


def get_bulk_modulus(file_name: str, job_name: str):
    return read_value(file_name, job_name, "output/equilibrium_bulk_modulus")


def get_potential(file_name: str, job_name: str):
    return read_value(file_name, job_name, "Al_ref/input/potential_inp/potential/Name")


def get_lattice_parameter(file_name: str, job_name: str):
    return read_value(file_name, job_name, "output/equilibrium_volume") ** (1 / 3)


# ---------------------------------------------------------------------------
# job table (replaces Project.job_table() / FileTable)
# ---------------------------------------------------------------------------


def index_project(project_path: str, recursive: bool = True):
    """
    Walk a pyiron project directory and build a job table purely from the job
    HDF5 files on disk - no SQL database involved.

    Mirrors pyiron_base.database.filetable.FileTable.init_table / get_extract.
    """
    import os

    import pandas as pd
    from pyfileindex import PyFileIndex

    def _is_h5(file_name: str) -> bool:
        return file_name.endswith(".h5")

    fileindex = PyFileIndex(path=project_path, filter_function=_is_h5)
    df_files = fileindex.dataframe
    df_files = df_files[~df_files.is_directory]

    if not recursive:
        project_path_abs = os.path.abspath(project_path)
        df_files = df_files[
            df_files.path.apply(lambda p: os.path.dirname(p) == project_path_abs)
        ]

    rows = []
    for path in df_files.path.values:
        job_name = os.path.splitext(os.path.basename(path))[0]
        status = get_job_status(path, job_name)
        if status is None:
            # not a pyiron job hdf5 file (e.g. no top-level "<job_name>/status" node)
            continue
        rows.append(
            {
                "id": get_job_id(path, job_name),
                "job": job_name,
                "project": os.path.dirname(path) + "/",
                "path": path,
                "status": status,
                "hamilton": get_job_type(path, job_name),
            }
        )

    return pd.DataFrame(
        rows, columns=["id", "job", "project", "path", "status", "hamilton"]
    )


def filter_job_table(job_table, status=("finished",), db_filter_function=None):
    """
    Args:
        job_table (pandas.DataFrame): as returned by index_project
        status (list/tuple of str): only keep jobs with one of these status values
        db_filter_function (callable/None): function(job_table) -> bool pandas.Series,
            same signature as pyiron_base's `TableJob.db_filter_function`
    """
    df = job_table[job_table.status.isin(status)]
    if db_filter_function is not None:
        df = df[db_filter_function(df)]
    return df


# ---------------------------------------------------------------------------
# applying user analysis functions to jobs (replaces PyironTable._iterate_over_job_lst)
# ---------------------------------------------------------------------------


def apply_functions_to_job(file_name: str, job_name: str, functions: dict):
    """
    Args:
        file_name (str): path to the job's hdf5 file
        job_name (str): name of the job (top level group in the hdf5 file)
        functions (dict): {label: callable(file_name, job_name) -> value}

    Returns:
        dict: {label: value}, functions that raise are set to None
    """
    result = {}
    for label, func in functions.items():
        try:
            result[label] = func(file_name, job_name)
        except Exception:
            result[label] = None
    return result


def build_table(
    project_path: str,
    functions: dict,
    status=("finished",),
    db_filter_function=None,
    recursive: bool = True,
):
    """
    End to end replacement for:

        table = pr.create.table("table")
        table.db_filter_function = db_filter_function
        table.add["label"] = func
        table.run()
        table.get_dataframe()

    Args:
        project_path (str): root directory of the pyiron project to scan
        functions (dict): {label: callable(file_name, job_name) -> value}
        status (list/tuple of str): job status values to include, default ("finished",)
        db_filter_function (callable/None): function(job_table) -> bool pandas.Series
        recursive (bool): include jobs in sub-projects

    Returns:
        pandas.DataFrame
    """
    import pandas as pd

    job_table = index_project(project_path=project_path, recursive=recursive)
    job_table = filter_job_table(
        job_table, status=status, db_filter_function=db_filter_function
    )

    rows = []
    for _, row in job_table.iterrows():
        values = {
            "job_id": row["id"],
            "job": row["job"],
            "hamilton": row["hamilton"],
        }
        values.update(
            apply_functions_to_job(
                file_name=row["path"], job_name=row["job"], functions=functions
            )
        )
        rows.append(values)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# pyiron_nodes wrappers
# ---------------------------------------------------------------------------


@as_function_node("job_table")
def IndexProject(project_path: str, recursive: bool = True):
    from pyiron_nodes.databases.pyiron_tables import index_project

    return index_project(project_path=project_path, recursive=recursive)


@as_function_node("table")
def BuildTable(
    project_path: str,
    functions: dict,
    status: list = ["finished"],
    db_filter_function=None,
    recursive: bool = True,
):
    from pyiron_nodes.databases.pyiron_tables import build_table

    return build_table(
        project_path=project_path,
        functions=functions,
        status=status,
        db_filter_function=db_filter_function,
        recursive=recursive,
    )


@as_function_node("db_filter_function")
def DbFilterFunction(hamilton=None, status=None, job_name_contains: str = None):
    """
    Build a `db_filter_function(job_table) -> bool Series`, pluggable directly into
    BuildTable(db_filter_function=...). Mirrors pyiron_base's JobFilters.job_type /
    JobFilters.job_name_contains, plus a status filter.

    Args:
        hamilton (str/list/None): keep only rows whose job type ("hamilton") matches
            one of these, e.g. "Murnaghan" or ["Murnaghan", "Lammps"]
        status (str/list/None): keep only rows whose status matches one of these
        job_name_contains (str/None): keep only rows whose job name contains this substring
    """

    def _db_filter_function(job_table):
        import pandas as pd

        mask = pd.Series(True, index=job_table.index)
        if status is not None:
            status_lst = status if isinstance(status, (list, tuple)) else [status]
            mask = mask & job_table.status.isin(status_lst)
        if hamilton is not None:
            hamilton_lst = (
                hamilton if isinstance(hamilton, (list, tuple)) else [hamilton]
            )
            mask = mask & job_table.hamilton.isin(hamilton_lst)
        if job_name_contains is not None:
            mask = mask & job_table.job.str.contains(job_name_contains)
        return mask

    return _db_filter_function


@as_function_node("functions")
def AddBulkModulus(functions: dict = None) -> dict:
    """
    Add {"bulk_modulus": get_bulk_modulus} to a functions dict and return it,
    pluggable directly into BuildTable(functions=...). Daisy-chains the same
    way AddPristine chains StructureContainer: pass a previous node's dict
    output back in as `functions` to keep growing the same dict, so no
    separate merge node is needed:

        wf.A = AddBulkModulus()
        wf.B = AddPotential(functions=wf.A)
        wf.C = AddLatticeParameter(functions=wf.B)
        wf.BuildTable = BuildTable(project_path=..., functions=wf.C)

    The dict key ("bulk_modulus") becomes the resulting table's column
    name - it is the property name, not the function name (get_bulk_modulus).
    """
    from pyiron_nodes.databases.pyiron_tables import get_bulk_modulus

    functions = dict(functions) if functions is not None else {}
    functions["bulk_modulus"] = get_bulk_modulus
    return functions


@as_function_node("functions")
def AddPotential(functions: dict = None) -> dict:
    """
    Add {"potential": get_potential} to a functions dict and return it,
    pluggable directly into BuildTable(functions=...). Daisy-chains the same
    way AddBulkModulus does.
    """
    from pyiron_nodes.databases.pyiron_tables import get_potential

    functions = dict(functions) if functions is not None else {}
    functions["potential"] = get_potential
    return functions


@as_function_node("functions")
def AddLatticeParameter(functions: dict = None) -> dict:
    """
    Add {"lattice_parameter": get_lattice_parameter} to a functions dict and
    return it, pluggable directly into BuildTable(functions=...).
    Daisy-chains the same way AddBulkModulus does.
    """
    from pyiron_nodes.databases.pyiron_tables import get_lattice_parameter

    functions = dict(functions) if functions is not None else {}
    functions["lattice_parameter"] = get_lattice_parameter
    return functions
