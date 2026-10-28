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
# ``master_hash`` link back to the sweep that ran it.  This recovers full
# per-iteration provenance on demand, and the Database panel lists a sweep's
# cells by that link.
#
# The database is the one of the graph *running the sweep*, published as
# ``core.remote_cache.current_run_node()`` — not ``template._graph``, which after
# ``Graph.copy()`` (the GUI always runs a copy) points at a graph without one.
# ``storage_enabled = False`` on the workflow turns all of this off.
#
# With an out-of-process executor (Slurm, Flux, process pools) the database
# never travels with the task — it cannot be pickled and carries credentials.
# Restore and storage then run on the client, around each submission; see
# ``_restore_on_client`` / ``_store_on_client``.  On a cluster executor every
# cell is also check-pointed and submitted under its own hash, whether or not
# the template stores, so a crashed client loses nothing: see ``_submit_cells``.
#
# NOTE: storage is mirrored from the Graph path (db.read + restore_node_outputs
# + store_node_in_database), NOT from Node.run(db): the latter's restore branch
# passes a db where a storage path is expected and never restores.


def _storage_context(node):
    """Return (db, storage_path) the sweep of *node* may store into, or (None, None).

    The graph executing the sweep publishes itself (``current_run_node``); that
    is where the database is.  ``node._graph`` is only a fallback for sweeps run
    outside a graph: nodes are shared between a graph and its copies, and the
    GUI runs a copy, so the template's back-reference points at a graph without
    the database.  ``storage_enabled = False`` on the workflow turns it off.
    """
    from core.remote_cache import current_run_node

    run = current_run_node()
    if run is not None and run.db is not None:
        if not run.recording:
            return None, None
        return run.db, run.storage_path

    graph = getattr(node, "_graph", None)
    db = getattr(graph, "_db", None)
    if db is None or not getattr(graph, "_recording_enabled", True):
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
    if (
        storage_path is not None
        and (pathlib.Path(storage_path) / f"{node_hash}.hdf5").exists()
    ):
        return _pdb.restore_node_outputs(node, storage_path)
    return False


def _store_instance(
    node, db, storage_path, outputs=True, node_hash=None, executor_hash=None
) -> None:
    """Persist an executed instance as its own database row (+ HDF5 output)."""
    import pyiron_database as _pdb

    _pdb.store_node_in_database(
        db,
        node,
        store_outputs=outputs,
        store_input_nodes_recursively=False,
        storage_path=storage_path,
        node_hash=node_hash,
        executor_hash=executor_hash,
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


def _sweep_labels(input_label: str) -> list[str]:
    """Split a comma-separated ``input_label`` into the swept port names.

    ``"x"`` sweeps one port; ``"task_index, model, rep"`` sweeps three at once,
    in which case every swept value is a tuple with one entry per label, or a
    dict keyed by label.  It stays a plain string so the GUI can render it as
    an editable text field — and so the canvas knows, without running anything,
    which ports of the template the sweep sets (it hides them).
    """
    return [label.strip() for label in str(input_label).split(",") if label.strip()]


def _check_sweep_values(labels: list[str], values) -> None:
    """Fail before the first iteration — and before anything is queued — when a
    multi-label sweep is handed values of the wrong shape."""
    if len(labels) < 2:
        return
    for idx, value in enumerate(values):
        if isinstance(value, dict):
            missing = [label for label in labels if label not in value]
            extra = [key for key in value if key not in labels]
            if missing or extra:
                raise ValueError(
                    f"values[{idx}] has keys {sorted(value)} but input_label is "
                    f"{', '.join(labels)!r} (missing {missing}, extra {extra})"
                )
        elif not isinstance(value, (tuple, list, np.ndarray)) or len(value) != len(
            labels
        ):
            raise ValueError(
                f"values[{idx}] = {value!r} does not match input_label "
                f"{', '.join(labels)!r}: expected a tuple of {len(labels)} entries "
                "or a dict keyed by label"
            )


def _sweep_items(labels: list[str], value) -> list:
    """The per-label entries of one multi-label sweep value, in label order."""
    if isinstance(value, dict):
        return [value[label] for label in labels]
    return list(value)


def _set_sweep_inputs(node, input_label: str, value) -> None:
    """Set the swept port(s) of *node* — one value per label in *input_label*."""
    labels = _sweep_labels(input_label)
    if len(labels) == 1:
        node.inputs.__setattr__(labels[0], value)
        return
    for label, item in zip(labels, _sweep_items(labels, value)):
        node.inputs.__setattr__(label, item)


def _warn_about_set_swept_ports(node, labels: list[str]) -> None:
    """Say so when a port the sweep overwrites was given a value on the template.

    The value never reaches a computation — every cell sets the port — so an
    edit there silently does nothing.  The canvas hides these ports; code that
    sets them anyway gets this warning.
    """
    from core.node import NotData, _differs_from_default

    for label in labels:
        port = node.inputs.ports.get(label)
        if port is None or port.connected or port._value is NotData:
            continue
        if _differs_from_default(port._value, port.default):
            logging.warning(
                "'%s.%s' = %r is ignored: the sweep sets it for every cell. "
                "Change it where the sweep values come from.",
                getattr(node, "label", node),
                label,
                port._value,
            )


def _run_instance_closure(node, input_label, value, db, storage_path, parent_hash):
    """Module-level (picklable) worker for the parallel iteration path."""
    _set_sweep_inputs(node, input_label, value)
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
def while_loop(
    step: Node,
    input_label: str = "x",
    max_steps: int = 100,
):
    """Repeat *step* until it reports that it is done.

    A generic ``while`` loop: it knows nothing about what *step* computes.
    *step* is typically a group node. Its output ports, in order, are the
    updated loop variable ``x`` and a boolean break condition. Each iteration
    the current ``x`` is written to the input port called *input_label*, the
    step runs, and the loop stops as soon as the break condition is true.

    ``x`` can be anything the step accepts and returns: a number, a dataclass,
    a structure, a dict. It is never converted.

    Parameters
    ----------
    step : Node
        Node with an input port *input_label* and outputs ``(x, break_condition)``.
        Its current value on *input_label* is the starting ``x``.
    input_label : str
        Label of the input port of *step* that receives the updated ``x``.
    max_steps : int
        Safety limit on the number of iterations. Hitting it raises an error
        instead of silently returning an unconverged ``x``.

    Returns
    -------
    x
        The value of ``x`` from the iteration that set the break condition.
    n_steps : int
        How many times *step* was run.
    """
    if input_label not in step.inputs:
        raise ValueError(
            f"'{input_label}' is not an input port of the step node; "
            f"available ports: {list(step.inputs.ports)}"
        )

    x = step.inputs[input_label].value

    for n_steps in range(1, max_steps + 1):
        step.inputs[input_label].value = x
        x, break_condition = step.run()

        if break_condition:
            return x, n_steps

    raise RuntimeError(f"while_loop did not finish within {max_steps} steps")


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


def _cell_copy(node):
    """A copy of *node* for one cell run concurrently with others in-process."""
    try:
        return node.copy()
    except Exception as exc:  # noqa: BLE001 — fall back to the old behaviour
        logging.warning(
            "Could not copy '%s' for a parallel cell (%s); cells share one "
            "node and may see each other's inputs.",
            getattr(node, "label", node),
            exc,
        )
        return node


def _runs_in_process(executor) -> bool:
    """Whether *executor* runs tasks in threads of this process.

    Only then may a task share objects that cannot be pickled — the database
    handle above all.  Every other executor (process pools, executorlib's
    Slurm and Flux executors) serialises what it is given.
    """
    from concurrent.futures import ThreadPoolExecutor

    return isinstance(executor, ThreadPoolExecutor)


def _check_picklable(node, values) -> None:
    """Raise now, on the canvas, if a sweep cannot be sent to its executor.

    executorlib serialises a task in a background thread; when that fails the
    thread dies without a trace and the sweep hangs.  One trial serialisation
    turns that into an error naming the template.
    """
    import cloudpickle

    try:
        cloudpickle.dumps((node, list(values)))
    except Exception as exc:
        raise TypeError(
            f"cannot send '{getattr(node, 'label', node)}' to the executor: one "
            f"of its inputs or swept values cannot be pickled ({exc})"
        ) from exc


def _restore_on_client(node, input_label, value, db, storage_path):
    """A completed future holding the stored outputs of this cell, or ``None``."""
    from concurrent.futures import Future

    _set_sweep_inputs(node, input_label, value)
    try:
        if not _restore_instance(node, db, storage_path):
            return None
    except Exception as exc:  # noqa: BLE001 — cache miss must never be fatal
        logging.warning(
            "per-iteration restore failed for '%s': %s",
            getattr(node, "label", None),
            exc,
        )
        return None
    future = Future()
    future.set_result(node._collect_outputs())
    return future


def _store_on_client(
    out,
    node,
    input_label,
    value,
    db,
    storage_path,
    parent_hash,
    outputs=True,
    node_hash=None,
    executor_hash=None,
):
    """Persist one remotely computed cell, as ``_run_instance`` does in-process.

    With ``outputs=False`` only the row is written — the completion of a cell
    that was check-pointed for recovery but whose template does not store.
    """
    from core.node import set_node_outputs

    try:
        _set_sweep_inputs(node, input_label, value)
        set_node_outputs(node, out)
        node._hash_parent = parent_hash
        _store_instance(
            node,
            db,
            storage_path,
            outputs=outputs,
            node_hash=node_hash,
            executor_hash=executor_hash,
        )
    except Exception as exc:  # noqa: BLE001 — storage must never crash a sweep
        logging.warning(
            "per-iteration storage failed for '%s': %s",
            getattr(node, "label", None),
            exc,
        )


def _parent_hash(node):
    """Hash of the sweep running *node* — what each cell row links to.

    Falls back to the template's own hash outside a graph run, where nothing
    names the sweep.
    """
    from core.remote_cache import current_run_node

    run = current_run_node()
    if run is not None and run.node is not node and run.node_hash:
        return run.node_hash
    try:
        import pyiron_database as _pdb

        return _pdb.get_hash(node)
    except Exception:  # noqa: BLE001 — provenance link is best-effort
        return None


def _executor_hash():
    """Hash of the executor node wired into the running sweep, or ``None``."""
    from core.remote_cache import current_run_node

    run = current_run_node()
    return run.executor_hash if run is not None else None


def _cells_are_recoverable(node, executor) -> bool:
    """Whether cells of *node* can be check-pointed and collected later.

    That needs an executorlib executor with a cache directory to come back to,
    and a function node — its function, called with its resolved inputs, is
    what goes to the queue, so the cache holds exactly what a node-level
    submission would and the Database panel can collect it the same way.
    """
    from core.remote_cache import executor_cache_directory, is_recoverable_executor

    return (
        is_recoverable_executor(executor)
        and bool(executor_cache_directory(executor))
        and callable(getattr(node, "func", None))
        and callable(getattr(node, "_call_inputs", None))
    )


def _submit_cells(
    node, input_label, values, executor, db, storage_path, parent_hash, stores, to_store
) -> dict:
    """Submit every cell of a sweep so that it survives the client.

    Per cell, keyed on the cell's own hash (the template with the swept inputs
    set — the key per-cell storage uses too):

    * outputs already stored, or a result already in the executor cache —
      use it, submit nothing;
    * the last job under this key died — report it for this cell instead of
      submitting a second doomed job (``retry=True`` clears the way);
    * otherwise check-point a ``Submitted`` row (linked to the sweep and to
      the executor node, so the Database panel can collect it) and submit
      under that key.  A job still out from an earlier session is re-attached
      by executorlib itself, which finds the key's queue id in the cache.

    A non-modal run (``SlurmExecutor(non_modal=True)``, ``wf.run(wait=False)``)
    does not wait: once every cell is either done or in the queue, the sweep
    is check-pointed and :class:`~core.remote_cache.NodeSubmitted` raised.  A
    later run picks up what has finished.

    Returns ``{future: (index, value)}``; completed rows are registered in
    *to_store* for the result loop.
    """
    from concurrent.futures import Future

    import pyiron_database as _pdb
    from core.cancellation import _submit
    from core.remote_cache import (
        NodeSubmitted,
        QueueJobFailed,
        _guard_dead_submission,
        _note_executor,
        collect_from_cache,
        current_run_node,
        executor_cache_directory,
        is_non_modal,
        queue_status_for,
        record_submission,
        run_is_non_modal,
        wait_for_queue_id,
    )

    run = current_run_node()
    executor_hash = run.executor_hash if run is not None else None
    cache_directory = executor_cache_directory(executor)
    non_modal = is_non_modal(executor) or run_is_non_modal()

    def done(result):
        future = Future()
        future.set_result(result)
        return future

    def failed(exc):
        future = Future()
        future.set_exception(exc)
        return future

    futures: dict = {}
    pending: list[str] = []
    submitted_now = False
    for idx, value in enumerate(values):
        _set_sweep_inputs(node, input_label, value)
        label = f"{getattr(node, 'label', 'cell')}[{idx}]"
        try:
            cell_hash = _pdb.get_hash(node)
        except Exception:  # noqa: BLE001 — unhashable: run it unrecoverably
            cell_hash = None
        if cell_hash is None:
            futures[
                executor.submit(
                    _run_instance_closure, node, input_label, value, None, None, None
                )
            ] = (idx, value)
            continue

        if stores:
            restored = _restore_on_client(node, input_label, value, db, storage_path)
            if restored is not None:
                futures[restored] = (idx, value)
                continue

        ready, result = collect_from_cache(cache_directory, cell_hash)
        if ready:
            future = done(result)
            futures[future] = (idx, value)
            to_store[future] = (stores, cell_hash)
            continue

        try:
            _guard_dead_submission(
                label, cache_directory, cell_hash, non_modal=non_modal
            )
        except QueueJobFailed as exc:
            futures[failed(exc)] = (idx, value)
            continue

        if non_modal:
            report = queue_status_for(cache_directory, cell_hash)
            if report["queue_id"] is not None and report["queue_status"] != "gone":
                pending.append(cell_hash)  # still out from an earlier run
                continue

        record_submission(
            node,
            db=db,
            node_hash=cell_hash,
            executor_hash=executor_hash,
            master_hash=parent_hash,
            show_status=False,
        )
        kwargs, _ = node._call_inputs()
        future = _submit(
            executor,
            node.func,
            kwargs,
            {"resource_dict": {"cache_key": cell_hash}},
        )
        submitted_now = True
        if non_modal:
            wait_for_queue_id(cache_directory, cell_hash)
            pending.append(cell_hash)
            continue
        futures[future] = (idx, value)
        to_store[future] = (stores, cell_hash)

    if pending:
        _note_executor(executor)
        sweep = run.node if run is not None else node
        sweep_hash = run.node_hash if run is not None else parent_hash
        if run is not None and sweep_hash:
            record_submission(sweep, db=db, node_hash=sweep_hash)
        raise NodeSubmitted(
            label=f"{getattr(sweep, 'label', 'sweep')} ({len(pending)} cell(s))",
            node_hash=sweep_hash or "",
            cache_directory=cache_directory,
            queue_id=None,
            resubmitted=submitted_now,
        )
    return futures


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
    """Run *node* once per value of *values*, setting the port(s) *input_label*.

    The swept ports are set on the template for each cell and put back
    afterwards, so the template ends the sweep exactly as it started.  Left
    holding the last cell's values, it would carry them into the sweep's hash
    — making the next run's cache key depend on run history — and, on the
    canvas, into ports the user cannot see.
    """
    labels = _sweep_labels(input_label)
    _check_sweep_values(labels, values)
    _warn_about_set_swept_ports(node, labels)
    saved = {
        label: (port._value, port.ready)
        for label in labels
        if (port := node.inputs.ports.get(label)) is not None
    }
    try:
        return _iterate_node_unguarded(
            node,
            input_label,
            values,
            copy_results=copy_results,
            collect_input=collect_input,
            collect_errors=collect_errors,
            debug=debug,
            executor=executor,
        )
    finally:
        for label, (value, ready) in saved.items():
            port = node.inputs.ports[label]
            port._value = value
            port.ready = ready


def _iterate_node_unguarded(
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

    # Resolve per-iteration storage context: the database of the run executing
    # the sweep, if recording is on.
    db, storage_path = _storage_context(node)
    parent_hash = _parent_hash(node) if db is not None else None

    # remotely computed cells to persist on this thread: future -> (outputs?, hash)
    to_store: dict = {}
    if executor is None:
        # Sequential execution
        for value in values:
            _set_sweep_inputs(node, input_label, value)
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
    elif _runs_in_process(executor):
        # Parallel execution in threads of this process: the db object is
        # shared in memory, so per-iteration storage happens in the worker.
        # Every cell gets its own copy of the template: with one shared node,
        # concurrent cells overwrite each other's inputs and outputs between
        # setting them and reading them back.  The copy shares the input
        # *values* — an engine stays one engine — only the ports are its own.
        futures = {
            executor.submit(
                _run_instance_closure,
                _cell_copy(node),
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
    else:
        # Parallel execution in other processes or batch jobs.  Only the node and
        # the swept value cross over: the database handle holds locks and
        # credentials, and an executor that cannot serialise an argument (such
        # as executorlib's background thread) dies silently, leaving the sweep
        # waiting forever on futures that never complete.  Storage runs here on
        # the client instead, around each submission.
        _check_picklable(node, values)
        stores = db is not None and _node_wants_storage(node)
        if db is not None and _cells_are_recoverable(node, executor):
            futures = _submit_cells(
                node,
                input_label,
                values,
                executor,
                db,
                storage_path,
                parent_hash,
                stores,
                to_store,
            )
            values_iter = ()
        else:
            futures = {}
            values_iter = values
        for idx, value in enumerate(values_iter):
            restored = (
                _restore_on_client(node, input_label, value, db, storage_path)
                if stores
                else None
            )
            if restored is not None:
                futures[restored] = (idx, value)
                continue
            future = executor.submit(
                _run_instance_closure, node, input_label, value, None, None, None
            )
            if stores:
                to_store[future] = (True, None)
            futures[future] = (idx, value)

    if executor is not None:
        results = [None] * len(values)
        errors = [None] * len(values) if collect_errors else None
        for future in as_completed(futures):
            idx, val = futures[future]
            try:
                out = future.result()
                if future in to_store:
                    outputs, cell_hash = to_store[future]
                    _store_on_client(
                        out,
                        node,
                        input_label,
                        val,
                        db,
                        storage_path,
                        parent_hash,
                        outputs=outputs,
                        node_hash=cell_hash,
                        executor_hash=_executor_hash(),
                    )
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
                parts.append(
                    pd.DataFrame(
                        {col: [vals[i]] * n_inner for col, vals in scalar_cols.items()}
                    )
                )
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
        – Several inputs can be swept together: give ``input_label`` as a
          comma-separated list (``"task_index,model,rep"``) and make every
          element of ``values`` a tuple with one entry per label, or a dict
          keyed by label (the keys must match).  Each label gets its own
          column.  For a full grid, wire a node that returns the cartesian
          product (e.g. ``benchmark.BenchmarkGrid``), or type the tuples into
          ``InputVector`` (``"[(0, 'a'), (1, 'b')]"``).
        – The swept ports belong to the sweep: they are set for every cell and
          restored afterwards, the canvas hides them on the template, and a
          value set on one in code is ignored with a warning.
        – ``store=True`` on the *template* stores every cell as its own row
          (linked to this sweep, restored on re-run).  On a cluster executor
          the cells are check-pointed regardless and can be collected after a
          crash; with ``non_modal`` the sweep submits them and returns.

    Parameters
    ----------
    node : Node
        The node that will be executed for each value.
    input_label : str
        Name of the input attribute on ``node`` that receives each element of
        ``values`` — or several names, comma-separated, to sweep them jointly.
    values : list | np.ndarray
        Iterable of input values; tuples of matching length when
        ``input_label`` names several ports.
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
    # 2.1 Input column(s) – one per swept label, avoiding name clashes
    #     with node outputs
    # ------------------------------------------------------------------
    output_labels = [label for label in node.outputs.keys() if label != "self"]
    labels = _sweep_labels(input_label)
    for idx, label in enumerate(labels):
        column = f"input_{label}" if label in output_labels else label
        data_dict[column] = (
            inp_lst
            if len(labels) == 1
            else [_sweep_items(labels, value)[idx] for value in inp_lst]
        )

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
                out[idx] if e is None else np.nan for out, e in zip(out_lst, err_lst)
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
        return expanded  # multi-row expansion already produced a DataFrame
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


@as_function_node("or")
def Or(x, y):
    """Element-wise / boolean ``x | y``, e.g. to combine two masks."""
    return x | y


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
