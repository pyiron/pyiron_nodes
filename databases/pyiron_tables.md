# pyiron_tables.py — how to add a new table property

This document explains the pattern used in
`pyiron_nodes/databases/pyiron_tables.py` for turning a single job property
into a `pyiron_nodes` node that can be daisy-chained into `BuildTable`. It is
written so another LLM can add a new property node without re-deriving the
design from scratch.

## What this module is

A reimplementation of pyiron_base's job-table / `PyironTable` data mining
(`pyiron_base.database.filetable.FileTable` +
`pyiron_base.jobs.datamining.PyironTable`/`TableJob`) as plain functions that
read directly from a project directory and each job's HDF5 file. **It has
zero dependency on pyiron_base or pyiron_atomistics** — no job objects, no
SQL database. Only `pyfileindex` (filesystem indexing) and `h5io_browser`
(HDF5 reads) are used. This constraint is intentional and must not be
reintroduced: never write a node/function here that takes a `job` object
with `__getitem__`; always take `(file_name, job_name)` strings.

## The calling convention every property function must follow

Every function that extracts one value from a job has this exact signature:

```python
def get_<property>(file_name: str, job_name: str):
    return read_value(file_name, job_name, "<hdf5/path/inside/the/job>")
```

- `file_name`: path to the job's `.h5` file on disk.
- `job_name`: the job's name — the top-level HDF5 group inside that file.
- `read_value(file_name, job_name, path)` (defined in this module) reads
  `<job_name>/<path>` out of the HDF5 file. `path` is exactly what you'd put
  inside `job["..."]` in real pyiron, e.g. `"output/generic/energy_pot"`.

Function names spell the property out in full — no abbreviations (`get_bm`
was renamed to `get_bulk_modulus` for exactly this reason). Existing examples
in this file:

```python
def get_bulk_modulus(file_name: str, job_name: str):
    return read_value(file_name, job_name, "output/equilibrium_bulk_modulus")


def get_potential(file_name: str, job_name: str):
    return read_value(file_name, job_name, "Al_ref/input/potential_inp/potential/Name")


def get_lattice_parameter(file_name: str, job_name: str):
    return read_value(file_name, job_name, "output/equilibrium_volume") ** (1 / 3)
```

`get_lattice_parameter` shows that any transform (unit conversion, `** power`,
etc.) is just plain Python applied to the value returned by `read_value` —
there is no separate "power" parameter or config system; the transform lives
directly in the function body.

If you don't know the exact HDF5 path for a property, inspect a real job
file directly, e.g.:

```python
import h5py
with h5py.File(file_name, "r") as f:
    print(list(f[job_name].keys()))        # top-level groups: input, output, server, ...
    print(list(f[job_name]["output"].keys()))
```

## The node that wraps each function: the "Add" pattern

Every property function gets exactly one matching `@as_function_node`
wrapper, named `Add<Property>`, with this exact shape:

```python
@as_function_node("functions")
def Add<Property>(functions: dict = None) -> dict:
    """
    Add {"<property>": get_<property>} to a functions dict and return it,
    pluggable directly into BuildTable(functions=...). Daisy-chains the same
    way AddBulkModulus does.
    """
    from pyiron_nodes.databases.pyiron_tables import get_<property>

    functions = dict(functions) if functions is not None else {}
    functions["<property>"] = get_<property>
    return functions
```

Rules, all load-bearing — do not deviate:

1. **One input**, `functions: dict = None`. `None` means "start a fresh
   dict"; a non-`None` dict means "grow this dict".
2. **Copy, don't mutate**: `functions = dict(functions) if functions is not
   None else {}` — the incoming dict is never mutated in place, a fresh copy
   is returned. This makes the node pure/side-effect-free, matching every
   other accumulator node in this codebase (e.g. `AddPristine` in
   `pyiron_nodes/atomistic/structure/container_new.py`).
3. **No shared helper function.** Each node inlines its own two-line
   copy-and-insert. There is deliberately no `add_table_function(func,
   functions)` utility — this was tried and explicitly rejected in favor of
   every node being self-contained. Do not reintroduce a shared accumulator
   helper.
4. **The import is local, inside the function body**
   (`from pyiron_nodes.databases.pyiron_tables import get_<property>`), not
   a module-level import — this matches the existing style of every node in
   this file (`IndexProject`, `BuildTable`, `AddBulkModulus`, ...).
5. **Output dict key is the bare property name, never the function name.**
   The `get_` prefix must never appear in the table: use `functions["bulk_modulus"]
   = get_bulk_modulus`, not `functions[get_bulk_modulus.__name__] = ...` (which
   would put `"get_bulk_modulus"` in the table). The dict key becomes the
   resulting table's column name, so it must be exactly the clean property
   name — full word, no abbreviation (`"bulk_modulus"`, not `"bm"`) and no
   `get_` prefix. The function name and the label are kept in sync by
   naming the function `get_<same property name>` — e.g. property
   `"bulk_modulus"` → function `get_bulk_modulus`, property
   `"lattice_parameter"` → function `get_lattice_parameter` — so there's only
   one property name to invent per node, just written two ways.
6. There is **no generic parametrized node** (no `AddFunction(column_label,
   hdf_path, power=...)`) — that approach existed earlier and was removed in
   favor of one named function + one node per property, because it's more
   explicit and each property's HDF5 path/transform lives in readable,
   greppable Python rather than as a string argument at the call site.

## Wiring nodes into a workflow

Nodes only resolve their upstream inputs correctly when connected through a
`core.Workflow` graph — instantiating and chaining nodes directly as plain
Python objects outside a `Workflow` does **not** auto-pull upstream node
outputs (you'd get the raw `Node` object passed to `dict(...)` and a
`TypeError`). Always wire like this:

```python
from core import Workflow
from pyiron_nodes.databases.pyiron_tables import (
    AddBulkModulus, AddPotential, AddLatticeParameter, BuildTable, DbFilterFunction,
)

wf = Workflow("my_table")

wf.DbFilterFunction = DbFilterFunction(hamilton="Murnaghan")

wf.BulkModulusFunction = AddBulkModulus()
wf.PotentialFunction = AddPotential(functions=wf.BulkModulusFunction)
wf.LatticeParameterFunction = AddLatticeParameter(functions=wf.PotentialFunction)

wf.BuildTable = BuildTable(
    project_path="/path/to/pyiron/project",
    functions=wf.LatticeParameterFunction,
    status=["finished"],
    db_filter_function=wf.DbFilterFunction,
)

result = wf.run()   # or wf.BuildTable.pull() for just that node
```

Each `Add*` node takes the *previous* node's output back in as `functions`
— this is the same accumulator-chaining pattern `AddPristine` uses for
`StructureContainer` in `container_new.py`: no separate merge node is ever
needed, you just keep threading the growing dict through the chain. Order of
chaining doesn't matter for correctness (dict keys don't collide unless two
properties reuse the same function name), only for readability.

The fully worked reference implementation is
`pyiron_nodes/Workflows/pyiron_table_potential_scan.py`.

## What `BuildTable` does with the dict

`functions` ends up as a plain `dict` of `{"<property>": callable}`, e.g.
`{"bulk_modulus": get_bulk_modulus, "potential": get_potential,
"lattice_parameter": get_lattice_parameter}`.
`BuildTable`/`build_table` walks every job in the project (via
`index_project` + `filter_job_table`), and for each job calls every function
in the dict as `func(file_name, job_name)` (see `apply_functions_to_job`).
A function that raises for a given job has its value set to `None` for that
row rather than aborting the whole table build — write property functions
assuming this: no need for your own broad `try/except`, just read the value
and let a genuine failure produce `None` for that row.

Each output row is built as `{"job_id": ..., "job": ..., "hamilton": ...}`
first, then merged with the function results — so `job_id`, `job`,
`hamilton` are always the first three columns, followed by one column per
property in the order its `Add*` node was chained in.

`job_id` is the **real** pyiron database job ID, read directly out of the
job's own HDF5 file via `get_job_id` (`<job_name>/job_id` inside the file).
It is not a synthetic per-scan counter — do not reintroduce
`enumerate(..., start=1)` style numbering for it.

## Checklist for adding a new property

1. Find the HDF5 path for the value (inspect a real job file if unsure).
2. Pick a clean, full-word property name (no abbreviations) — this is both
   the eventual table column name and, prefixed with `get_`, the function
   name.
3. Add `def get_<property>(file_name, job_name): return read_value(file_name, job_name, "<path>")`
   (plus any transform) near the other `get_*` functions.
4. Add `Add<Property>(functions: dict = None) -> dict` immediately after it,
   copying the exact shape of `AddBulkModulus`/`AddPotential`/
   `AddLatticeParameter` — remember the dict key is `"<property>"`, not
   `get_<property>.__name__`.
5. Wire it into a workflow: `wf.X = Add<Property>(functions=wf.<previous>)`.
6. Verify against a real project directory — `wf.BuildTable.pull()` (or
   `wf.run()`) and check the new column appears with sane values, not
   all-`None` (which usually means the HDF5 path is wrong).
