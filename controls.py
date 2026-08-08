import logging
import pathlib
from concurrent.futures import as_completed
from copy import copy
from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd

from core import Node, as_function_node
from core.node import _node_wants_storage


# ---------------------------------------------------------------------------
# Per-iteration content-addressed storage for higher-order nodes
# ---------------------------------------------------------------------------
#
# Higher-order nodes (IterToDataFrame, iterate) execute a *template* node once
# per swept value.  When the template opts into storage (``store=True``) and a
# database is reachable, each executed instance is persisted as its own node —
# distinct input value -> distinct content hash -> separate stored row — with a
# ``master_hash`` link back to the iterating template (its "generator").  This
# recovers full per-iteration provenance on demand.
#
# All of this is gated behind ``_node_wants_storage(template)`` AND a reachable
# ``db``.  When ``store`` is False (the default) the code path is byte-for-byte
# the legacy ``node.run()`` — no db lookup, no storage, identical results.
#
# NOTE: storage is mirrored from the Graph path (db.read + restore_node_outputs
# + store_node_in_database), NOT from Node.run(db): the latter's restore branch
# passes a db where a storage path is expected and never restores.


def _storage_context(node):
    """Return (db, storage_path) reachable from a template node, or (None, None).

    ``Graph.copy()`` propagates ``_db``, so a template node copied into a
    higher-order node reaches the database via ``node._graph._db``.
    """
    graph = getattr(node, "_graph", None)
    db = getattr(graph, "_db", None)
    if db is None:
        return None, None
    storage_path = getattr(db, "storage_path", None)
    if storage_path is None:
        storage_path = getattr(graph, "_storage_path", None)
    return db, storage_path


def _restore_instance(node, db, storage_path) -> bool:
    """Restore an instance's outputs from store if present. Mirrors Graph path."""
    import pyiron_database as _pdb

    node_hash = _pdb.get_hash(node)
    record = db.read(node_hash) if db is not None else None
    if record is not None and getattr(record, "output_path", None) is not None:
        output_dir = pathlib.Path(record.output_path).parent
        if _pdb.restore_node_outputs(node, output_dir):
            return True
    if storage_path is not None and (
        pathlib.Path(storage_path) / f"{node_hash}.hdf5"
    ).exists():
        return _pdb.restore_node_outputs(node, storage_path)
    return False


def _store_instance(node, db, storage_path) -> None:
    """Persist an executed instance as its own database row + HDF5 output."""
    import pyiron_database as _pdb

    _pdb.store_node_in_database(
        db,
        node,
        store_outputs=True,
        store_input_nodes_recursively=False,
        storage_path=storage_path,
    )


def _run_instance(node, db, storage_path, parent_hash) -> Any:
    """Run one iteration instance, persisting it when the template opted in.

    When ``store`` is off or no db is reachable this is exactly ``node.run()``.
    """
    if db is None or not _node_wants_storage(node):
        return node.run()  # legacy path — unchanged behavior

    # Link this instance back to the iterating template (NodeData.master_hash).
    node._hash_parent = parent_hash

    try:
        if _restore_instance(node, db, storage_path):
            return node._collect_outputs()
    except Exception as exc:  # noqa: BLE001 — cache miss must never be fatal
        logging.warning(
            "per-iteration restore failed for '%s': %s",
            getattr(node, "label", None),
            exc,
        )

    out = node.run()

    try:
        _store_instance(node, db, storage_path)
    except Exception as exc:  # noqa: BLE001 — storage must never crash a sweep
        logging.warning(
            "per-iteration storage failed for '%s': %s",
            getattr(node, "label", None),
            exc,
        )
    return out


def _run_instance_closure(node, input_label, value, db, storage_path, parent_hash):
    """Module-level (picklable) worker for the parallel iteration path."""
    node.inputs.__setattr__(input_label, value)
    return _run_instance(node, db, storage_path, parent_hash)


@as_function_node
def recursive(x: int, stop_at: int = 10) -> tuple[int, bool]:
    """Toy example for a recursive function."""
    x_new = x + 1

    break_condition = False
    if x_new > stop_at:
        break_condition = True
    return x_new, break_condition


@as_function_node
def loop_until(recursive_function: Node, max_steps: int = 10):
    x = recursive_function.inputs.x.value
    for i in range(max_steps):
        x, break_condition = recursive_function(x)
        print("loop: ", i, x, break_condition)

        if break_condition:
            break

    return x


@as_function_node
def branch(condition: bool, then_node: Node, else_node: Node):
    """Lazy conditional: execute exactly one of two template nodes.

    A higher-order control-flow primitive. ``then_node`` and ``else_node`` are
    delivered as whole node objects (``Node``-typed ports, i.e. ``self``-edges),
    so neither is executed by the enclosing graph up front. Only the branch
    selected by ``condition`` is pulled — computing just that node and its
    upstream dependencies — so the unused branch (and any expensive computation
    behind it) never runs. The outer graph gains only two ``self``-edges and
    stays a valid DAG.

    Parameters
    ----------
    condition : bool
        Selects ``then_node`` when truthy, otherwise ``else_node``.
    then_node : Node
        Template computation returned when ``condition`` is truthy.
    else_node : Node
        Template computation returned otherwise.

    Returns
    -------
    Any
        The result of pulling the selected node (same shape its own
        ``run``/``pull`` would return).
    """
    selected = then_node if condition else else_node
    result = selected.pull()
    return result


def _iterate_node(
    node,
    input_label: str,
    values,
    copy_results=True,
    collect_input=False,
    collect_errors=False,
    debug=False,
    executor=None,
):
    out_lst = []
    inp_lst = [] if collect_input else None
    err_lst = [] if collect_errors else None

    # Resolve per-iteration storage context (no-op unless template.store=True
    # and a database is reachable through the template's graph).
    db, storage_path = _storage_context(node)
    parent_hash = None
    if db is not None and _node_wants_storage(node):
        try:
            import pyiron_database as _pdb

            parent_hash = _pdb.get_hash(node)
        except Exception:  # noqa: BLE001 — provenance link is best-effort
            parent_hash = None

    if executor is None:
        # Sequential execution
        for value in values:
            node.inputs.__setattr__(input_label, value)
            try:
                out = _run_instance(node, db, storage_path, parent_hash)
                if copy_results:
                    out = copy(out)
                err = None
            except Exception as e:
                print("execution error: ", e)
                if collect_errors:
                    out = np.nan
                    err = f"{type(e).__name__}: {e}"
                else:
                    continue
            out_lst.append(out)
            if collect_input:
                inp_lst.append(value)
            if collect_errors:
                err_lst.append(err)
            if debug:
                print(f"iterating over {input_label} = {value}, out={out}")
                print("out list: ", [id(o) for o in out_lst])
    else:
        # Parallel execution
        futures = {
            executor.submit(
                _run_instance_closure,
                node,
                input_label,
                value,
                db,
                storage_path,
                parent_hash,
            ): (
                idx,
                value,
            )
            for idx, value in enumerate(values)
        }
        results = [None] * len(values)
        errors = [None] * len(values) if collect_errors else None
        for future in as_completed(futures):
            idx, val = futures[future]
            try:
                out = future.result()
                if copy_results:
                    out = copy(out)
                err = None
            except Exception as e:
                print("execution error: ", e)
                if collect_errors:
                    out = np.nan
                    err = f"{type(e).__name__}: {e}"
                else:
                    raise
            results[idx] = out
            if collect_errors:
                errors[idx] = err
            if debug:
                print(f"Parallel iter: {input_label}={val}, out={out}")
        out_lst = results
        if collect_input:
            inp_lst = list(values)
        if collect_errors:
            err_lst = errors

    if collect_errors:
        return out_lst, inp_lst, err_lst
    return (out_lst, inp_lst) if collect_input else out_lst


def _expand_df_columns(data_dict: dict):
    """
    If any output column in *data_dict* contains ``pd.DataFrame`` values,
    replace that column with the individual columns of the inner DataFrame.

    Single-row inner DataFrames
        Each inner column is spread directly into the outer dict.  Returns a
        plain ``dict`` so the caller can still pass it to ``pd.DataFrame()``.

    Multi-row inner DataFrames
        Inner rows are concatenated; scalar column values are repeated for
        every inner row.  Returns a fully merged ``pd.DataFrame`` so the
        caller can return it immediately.

    The input-label column (the first column, a scalar) is left in place and
    is always the leftmost column in the result.
    """
    # Identify which columns hold DataFrame values (skip plain scalars/NaN).
    df_cols: dict[str, tuple] = {}
    for col, vals in data_dict.items():
        first_df = next((v for v in vals if isinstance(v, pd.DataFrame)), None)
        if first_df is not None:
            df_cols[col] = (vals, list(first_df.columns))

    if not df_cols:
        return data_dict  # nothing to expand — fast path

    scalar_cols = {col: vals for col, vals in data_dict.items() if col not in df_cols}
    n = len(next(iter(data_dict.values())))

    has_multi = any(
        isinstance(v, pd.DataFrame) and len(v) > 1
        for vals, _ in df_cols.values()
        for v in vals
    )

    if not has_multi:
        # ── single-row case: spread inner columns into the dict ──────────
        new_dict: dict[str, list] = dict(scalar_cols)
        for col, (vals, inner_cols) in df_cols.items():
            for icol in inner_cols:
                new_dict[icol] = []
            for v in vals:
                if isinstance(v, pd.DataFrame) and len(v) >= 1:
                    for icol in inner_cols:
                        new_dict[icol].append(v[icol].iloc[0])
                else:
                    for icol in inner_cols:
                        new_dict[icol].append(np.nan)
        return new_dict

    else:
        # ── multi-row case: concat inner frames, repeat scalar values ────
        frames = []
        for i in range(n):
            inner_parts = []
            for col, (vals, inner_cols) in df_cols.items():
                v = vals[i]
                if isinstance(v, pd.DataFrame):
                    inner_parts.append(v.reset_index(drop=True))
                else:
                    inner_parts.append(pd.DataFrame({c: [np.nan] for c in inner_cols}))

            n_inner = max((len(df) for df in inner_parts), default=1)

            parts = []
            if scalar_cols:
                parts.append(pd.DataFrame(
                    {col: [vals[i]] * n_inner for col, vals in scalar_cols.items()}
                ))
            parts.extend(p.reset_index(drop=True) for p in inner_parts)
            frames.append(pd.concat(parts, axis=1))

        return pd.concat(frames, ignore_index=True)


# --- Node iteration to DataFrame ---
@as_function_node
def IterToDataFrame(
    node: Node,
    input_label: str,
    values: Union[list, np.ndarray],
    debug: bool = False,
    executor: type = None,
    store: bool = False,
) -> pd.DataFrame:
    """
    Iterate over ``values`` feeding each element into ``node`` under the name
    ``input_label`` and collect the results in a pandas DataFrame.

    New feature:
        – If the node returns a *dataclass* instance, each field of the
          dataclass becomes its own column in the DataFrame.

    Parameters
    ----------
    node : Node
        The node that will be executed for each value.
    input_label : str
        Name of the input attribute on ``node`` that receives each element of
        ``values``.
    values : list | np.ndarray
        Iterable of input values.
    debug : bool, optional
        Print debugging information.
    executor : concurrent.futures.Executor, optional
        If supplied, the iteration runs in parallel using the executor.
    store : bool, optional
        Used by decorator to implement hash storage (if set to True)

    Returns
    -------
    pd.DataFrame
        DataFrame where each column corresponds to an input or an output
        field.  When the node returns a dataclass, each field of the
        dataclass is a separate column.
    """
    from dataclasses import is_dataclass, fields

    # ------------------------------------------------------------------
    # 1️⃣ Run the node over all values
    # ------------------------------------------------------------------
    out_lst, inp_lst, err_lst = _iterate_node(
        node,
        input_label,
        values,
        copy_results=True,
        collect_input=True,
        collect_errors=True,
        debug=debug,
        executor=executor,
    )

    # ------------------------------------------------------------------
    # 2️⃣ Prepare a dict that will be fed to pd.DataFrame
    # ------------------------------------------------------------------
    data_dict: dict[str, List[Any]] = {}

    # ------------------------------------------------------------------
    # 2.1 Input column – avoid name clash with node outputs
    # ------------------------------------------------------------------
    output_labels = [label for label in node.outputs.keys() if label != "self"]
    if input_label in output_labels:
        data_dict[f"input_{input_label}"] = inp_lst
    else:
        data_dict[input_label] = inp_lst

    # ------------------------------------------------------------------
    # 3️⃣ Analyse the first output to decide how to unpack the rest
    # ------------------------------------------------------------------
    first_out = next(
        (o for o, e in zip(out_lst, err_lst) if e is None),
        None,
    )

    # Helper: is the result a dataclass instance?
    def _is_dataclass_instance(obj: Any) -> bool:
        return is_dataclass(obj) and not isinstance(obj, type)

    # ------------------------------------------------------------------
    # 3.1 Tuple / list / np.ndarray output (multiple scalar outputs)
    # ------------------------------------------------------------------
    multi_output = isinstance(first_out, (tuple, list, np.ndarray)) and len(
        first_out
    ) == len(output_labels)
    # print("multioutput: ", multi_output, len(output_labels), output_labels, _is_dataclass_instance(first_out))

    # ------------------------------------------------------------------
    # 3.2 Dataclass output – each field becomes a column
    # ------------------------------------------------------------------
    if _is_dataclass_instance(first_out):
        # Extract field names once – they will be the column names
        dc_fields = [f.name for f in fields(first_out)]

        # Initialise a list for each field
        for f_name in dc_fields:
            data_dict[f_name] = []

        # Fill the column lists
        for out in out_lst:
            # Defensive: if a particular iteration returned something else,
            # fall back to NaN for all fields.
            if _is_dataclass_instance(out):
                for f_name in dc_fields:
                    data_dict[f_name].append(getattr(out, f_name))
            else:
                for f_name in dc_fields:
                    data_dict[f_name].append(np.nan)

    # ------------------------------------------------------------------
    # 3.3 Regular scalar / single‑value output
    # ------------------------------------------------------------------
    elif multi_output:
        # Node returns a sequence that matches the declared output labels
        for idx, label in enumerate(output_labels):
            data_dict[label] = [
                out[idx] if e is None else np.nan
                for out, e in zip(out_lst, err_lst)
            ]

    else:
        # Node returns a single scalar (or a single object) per iteration
        if len(output_labels) == 1:
            # Simple case – one declared output
            data_dict[output_labels[0]] = out_lst
        else:
            # Ambiguous case – more declared outputs than we can unpack.
            # We store the whole object under each label (the original
            # behaviour) – this mirrors the previous implementation.
            for label in output_labels:
                data_dict[label] = out_lst

    # ------------------------------------------------------------------
    # 3.4 Error column — only added when at least one row errored
    # ------------------------------------------------------------------
    if any(e is not None for e in err_lst):
        data_dict["error"] = [e if e is not None else "" for e in err_lst]

    # ------------------------------------------------------------------
    # 3.5 Expand any output column whose values are DataFrames
    # ------------------------------------------------------------------
    expanded = _expand_df_columns(data_dict)
    if isinstance(expanded, pd.DataFrame):
        return expanded   # multi-row expansion already produced a DataFrame
    data_dict = expanded

    # ------------------------------------------------------------------
    # 4️⃣ Build the DataFrame (fallback to raw dict on error)
    # ------------------------------------------------------------------
    try:
        df = pd.DataFrame(data_dict)
    except Exception as e:
        print(f"Error creating DataFrame: {e}")
        df = pd.DataFrame.from_dict(data_dict, orient="columns")

    return df


# --- Simple iterator, parallel aware ---
@as_function_node
def iterate(
    node: Node,
    input_label: str,
    values: list | np.ndarray,
    debug: bool = False,
    executor: type = None,
):
    out_lst = _iterate_node(
        node,
        input_label,
        values,
        copy_results=True,
        collect_input=False,
        debug=debug,
        executor=executor,
    )
    # For compatibility: flatten if only one result
    if out_lst and isinstance(out_lst, list) and len(out_lst) == 1:
        out_lst = out_lst[0]
    return out_lst


@as_function_node
# pick a single element from a list
def pick_element(lst: list | np.ndarray, index: int) -> any:
    element = lst[index]
    return element


@as_function_node
def ExtractList(out_list: list, label: str, flatten: bool = True):
    import numpy as np

    collect = np.array([out.__getattribute__(label) for out in out_list])
    if flatten:
        collect = collect.flatten()
    return collect


@as_function_node
def InputVector(vec: str = ""):
    try:
        vector = eval(vec)
    except Exception:
        vector = None
    return vector


@as_function_node
def Slice(matrix, slice: str = "::"):
    try:
        result = eval(f"matrix[{slice}]")
    except Exception as e:
        result = None
        print("Slice failed: ", e)
    return result


@as_function_node
def Code(x, code: str = "x**2"):
    y = eval(code)
    return y


@as_function_node
def GetAttribute(obj, attr: str):
    """Get an attribute from an object."""
    try:
        value = obj.__getattribute__(attr)
    except AttributeError:
        value = None
    return value


@as_function_node
def SetAttribute(obj, attr: str, val: str) -> any:
    """Set an attribute on an object."""
    try:
        obj.__setattr__(attr, val)
    except AttributeError:
        print(f"Attribute {attr} not found in object {obj}")
    return obj


@as_function_node
def Print(x):
    """Print the input value."""
    print(f"Input value: {x}")
    return x


@as_function_node
def GetMask(x: np.ndarray, index: int = 0):
    mask = np.array(x) == index
    return mask


@as_function_node
def Filter(x: np.ndarray, index_vec: np.ndarray):
    result = x[:, index_vec]
    return result


@as_function_node
def Sleep(seconds: float = 1.0, a: float = 0) -> None:
    """Sleep for a specified number of seconds.

    Parameters
    ----------
    seconds : float, optional
        Number of seconds to sleep (default: 1.0)

    Returns
    -------
    None
    """
    import time

    print("sleep: ", a)
    time.sleep(seconds)
    status = None
    return status
