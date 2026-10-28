"""Per-iteration content-addressed storage and the lazy `branch` primitive.

Exercises the real ``controls.py`` code path with an in-memory database double,
so no Postgres/Neo4j backend is required.
"""

import pathlib
import tempfile

import pytest

from core import as_function_node
from core.graph import Graph
from pyiron_database.instance_database.InstanceDatabase import InstanceDatabase

from controls import branch, _iterate_node


class MemDB(InstanceDatabase):
    """Minimal in-memory InstanceDatabase for tests."""

    def __init__(self, storage_path):
        self.storage_path = pathlib.Path(storage_path)
        self.rows = {}

    def init(self):
        pass

    def drop(self):
        self.rows.clear()

    def create(self, node):
        self.rows[node.hash] = node
        return node.hash

    def read(self, hash):
        return self.rows.get(hash)

    def update(self, hash, **kwargs):
        for k, v in kwargs.items():
            setattr(self.rows[hash], k, v)

    def delete(self, hash):
        self.rows.pop(hash, None)


EXEC = {"n": 0}


@as_function_node
def Square(x: int = 0, store: bool = False):
    EXEC["n"] += 1
    y = x * x
    return y


@as_function_node("y")
def Boom(x: int = 0):
    raise RuntimeError("unused branch executed — laziness broken!")


@as_function_node
def Echo(x: int = 0):
    y = x
    return y


def _with_db(node, db):
    node._graph = Graph(db=db)
    return node


def test_store_false_is_legacy_path():
    """Default store=False: every value executes, DB is never touched."""
    EXEC["n"] = 0
    node = Square(store=False)
    out, _, _ = _iterate_node(
        node, "x", [1, 2, 3], collect_input=True, collect_errors=True
    )
    assert out == [1, 4, 9]
    assert EXEC["n"] == 3


def test_per_iteration_storage_and_restore():
    """store=True: each value → a separate row linked to the iter-node; re-run restores."""
    with tempfile.TemporaryDirectory() as tmp:
        db = MemDB(tmp)

        EXEC["n"] = 0
        node = _with_db(Square(store=True), db)
        out, _, _ = _iterate_node(
            node, "x", [1, 2, 3], collect_input=True, collect_errors=True
        )
        assert out == [1, 4, 9]
        assert EXEC["n"] == 3

        # three distinct instances, all sharing one master_hash (the template)
        assert len(db.rows) == 3
        parents = {r.master_hash for r in db.rows.values()}
        assert len(parents) == 1
        assert None not in parents

        # second pass over the same values restores everything (no execution)
        EXEC["n"] = 0
        node2 = _with_db(Square(store=True), db)
        out2, _, _ = _iterate_node(
            node2, "x", [1, 2, 3], collect_input=True, collect_errors=True
        )
        assert out2 == [1, 4, 9]
        assert EXEC["n"] == 0


def test_branch_runs_only_selected_node():
    assert branch(condition=True, then_node=Echo(x=42), else_node=Boom(x=0)).run() == 42
    assert branch(condition=False, then_node=Boom(x=0), else_node=Echo(x=7)).run() == 7


def test_branch_unselected_side_never_executes():
    # If the unselected branch ran, Boom would raise.
    with pytest.raises(RuntimeError):
        branch(condition=True, then_node=Boom(x=0), else_node=Echo(x=1)).run()


# ---------------------------------------------------------------------------
# Out-of-process executors (Slurm, Flux, process pools)
# ---------------------------------------------------------------------------


class LockedDB(MemDB):
    """Like a real database handle: holds a lock, so it cannot be pickled."""

    def __init__(self, storage_path):
        import threading

        super().__init__(storage_path)
        self._lock = threading.Lock()


class PicklingExecutor:
    """Serialises every task the way executorlib does, then runs it here."""

    def __init__(self):
        self.submitted = 0

    def submit(self, fn, *args, **kwargs):
        import cloudpickle
        from concurrent.futures import Future

        self.submitted += 1
        payload = cloudpickle.dumps((fn, args, kwargs))  # raises if unpicklable
        fn, args, kwargs = cloudpickle.loads(payload)
        future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:  # pragma: no cover - surfaced via the future
            future.set_exception(exc)
        return future


def test_remote_sweep_never_ships_the_database():
    """An attached db must not travel with the task — it hung GUI sweeps."""
    with tempfile.TemporaryDirectory() as tmp:
        db = LockedDB(tmp)
        node = _with_db(Square(), db)
        pool = PicklingExecutor()
        out = _iterate_node(node, "x", [1, 2, 3], executor=pool)
        assert out == [1, 4, 9]
        assert pool.submitted == 3
        assert db.rows == {}


def test_remote_sweep_stores_and_restores_on_the_client():
    with tempfile.TemporaryDirectory() as tmp:
        db = LockedDB(tmp)
        node = _with_db(Square(store=True), db)
        pool = PicklingExecutor()
        out = _iterate_node(node, "x", [1, 2, 3], executor=pool)
        assert out == [1, 4, 9]
        assert len(db.rows) == 3
        assert len({r.master_hash for r in db.rows.values()}) == 1

        # Second pass: every cell restored, nothing submitted.
        pool2 = PicklingExecutor()
        node2 = _with_db(Square(store=True), db)
        out2 = _iterate_node(node2, "x", [1, 2, 3], executor=pool2)
        assert out2 == [1, 4, 9]
        assert pool2.submitted == 0


def test_unpicklable_template_fails_before_anything_is_submitted():
    import threading

    pool = PicklingExecutor()
    node = Echo()
    node.inputs.x.value = threading.Lock()
    with pytest.raises(TypeError, match="cannot send 'Echo'"):
        _iterate_node(node, "x", [1], executor=pool)
    assert pool.submitted == 0


def test_remote_sweep_restores_the_swept_port():
    with tempfile.TemporaryDirectory() as tmp:
        node = _with_db(Square(store=True, x=7), LockedDB(tmp))
        _iterate_node(node, "x", [1, 2, 3], executor=PicklingExecutor())
        assert node.inputs.x.value == 7


# ---------------------------------------------------------------------------
# Inside a graph run, as the GUI does it: cells stored, linked, recoverable
# ---------------------------------------------------------------------------

import sys

import cloudpickle

# Jobs of the executorlib double below run in a fresh interpreter, which cannot
# import this test module; ship its functions by value instead.
cloudpickle.register_pickle_by_value(sys.modules[__name__])


@as_function_node("executor")
def Provide(executor=None):
    return executor


class CacheExecutor:
    """executorlib's file cache without a scheduler — recoverable, but local.

    Wraps ``SingleNodeExecutor`` and exposes ``info["executor_kwargs"]`` the way
    the cluster executors do, so ``core.remote_cache`` treats it as one.
    """

    def __init__(self, directory, on_submit=None):
        from executorlib import SingleNodeExecutor

        self._ex = SingleNodeExecutor(cache_directory=directory)
        self.info = {"executor_kwargs": {"cache_directory": directory}}
        self.future_queue = None
        self.keys = []
        self._on_submit = on_submit

    def submit(self, fn, *args, resource_dict=None, **kwargs):
        key = (resource_dict or {}).get("cache_key")
        self.keys.append(key)
        if self._on_submit is not None:
            self._on_submit(key)
        return self._ex.submit(fn, *args, resource_dict=resource_dict or {}, **kwargs)

    def shutdown(self, wait=True, **kwargs):
        self._ex.shutdown(wait=wait)


def _sweep_graph(template, executor=None, values=(1, 2, 3)):
    """Build like a workflow file, run like the GUI: a *copy* holds the db."""
    from core import Workflow

    from controls import IterToDataFrame

    wf = Workflow("cells")
    wf.T = template
    if executor is not None:
        wf.Ex = Provide(executor=executor)
        wf.Iter = IterToDataFrame(
            node=wf.T, input_label="x", values=list(values), executor=wf.Ex
        )
    else:
        wf.Iter = IterToDataFrame(node=wf.T, input_label="x", values=list(values))
    return wf._graph.copy()


def _cells(db):
    """Cell rows: the template's instances, each linked to the sweep.

    A non-modal sweep also check-points its upstream graph, the template
    included — that row has no master and is not a cell.
    """
    return [r for r in db.rows.values() if r.qualname == "Square" and r.master_hash]


@pytest.mark.parametrize("mode", ["sequential", "threads", "remote"])
def test_gui_like_run_stores_every_cell_linked_to_its_sweep(mode):
    from concurrent.futures import ThreadPoolExecutor

    import pyiron_database as pdb

    executor = {
        "sequential": None,
        "threads": ThreadPoolExecutor(max_workers=2),
        "remote": PicklingExecutor(),
    }[mode]
    with tempfile.TemporaryDirectory() as tmp:
        g = _sweep_graph(Square(store=True), executor)
        g._db = MemDB(tmp)
        g.pull_node("Iter")
        assert list(g.nodes["Iter"].outputs.df.value["y"]) == [1, 4, 9]
        cells = _cells(g._db)
        assert len(cells) == 3
        assert {r.master_hash for r in cells} == {pdb.get_hash(g.nodes["Iter"])}
        assert len(list(pathlib.Path(tmp).glob("*.hdf5"))) >= 3

        EXEC["n"] = 0
        g.pull_node("Iter")
        assert list(g.nodes["Iter"].outputs.df.value["y"]) == [1, 4, 9]
        if mode != "remote":  # remote runs in copies; no counter here
            assert EXEC["n"] == 0
    if mode == "threads":
        executor.shutdown()


def test_no_cells_are_stored_when_recording_is_off():
    with tempfile.TemporaryDirectory() as tmp:
        g = _sweep_graph(Square(store=True))
        g._db = MemDB(tmp)
        g._recording_enabled = False
        g.pull_node("Iter")
        assert _cells(g._db) == []


def test_queued_cells_are_checkpointed_keyed_on_their_hash_and_never_resubmitted():
    import pyiron_database as pdb

    with tempfile.TemporaryDirectory() as tmp:
        cache = str(pathlib.Path(tmp) / "cache")
        db = MemDB(tmp)
        seen = {}

        def at_submit(key):  # the row must exist *before* the job does
            row = db.read(key)
            seen[key] = (getattr(row, "status", None), row.executor, row.master_hash)

        ex = CacheExecutor(cache, on_submit=at_submit)
        g = _sweep_graph(Square(), ex)  # no store: check-pointed anyway
        g._db = db
        g.pull_node("Iter")
        ex.shutdown()
        assert list(g.nodes["Iter"].outputs.df.value["y"]) == [1, 4, 9]

        sweep_hash = pdb.get_hash(g.nodes["Iter"])
        executor_hash = pdb.get_hash(g.nodes["Ex"])
        assert len(ex.keys) == 3 and all(ex.keys)
        for key in ex.keys:
            assert seen[key] == ("Submitted", executor_hash, sweep_hash)
            assert getattr(db.read(key), "status", None) != "Submitted"  # completed
            assert (pathlib.Path(cache) / f"{key}_o.h5").exists()

        # A new session over the same cache: everything is collected, nothing
        # goes to the queue again.
        ex2 = CacheExecutor(cache)
        g2 = _sweep_graph(Square(), ex2)
        g2._db = db
        g2.pull_node("Iter")
        ex2.shutdown()
        assert ex2.keys == []
        assert list(g2.nodes["Iter"].outputs.df.value["y"]) == [1, 4, 9]


def test_non_modal_sweep_returns_at_once_and_is_collected_next_run(monkeypatch):
    import time

    import core.remote_cache as rc

    monkeypatch.setattr(rc, "wait_for_queue_id", lambda *a, **k: None)  # no scheduler
    with tempfile.TemporaryDirectory() as tmp:
        cache = str(pathlib.Path(tmp) / "cache")
        db = MemDB(tmp)
        ex = CacheExecutor(cache)
        rc.mark_non_modal(ex)
        g = _sweep_graph(Square(store=True), ex)
        g._db = db
        g.pull_node("Iter")
        assert len(ex.keys) == 3
        assert [s.label for s in g.last_submissions] == ["Iter (3 cell(s))"]
        assert all(getattr(r, "status", None) == "Submitted" for r in _cells(db))

        deadline = time.time() + 120
        while len(list(pathlib.Path(cache).glob("*_o.h5"))) < 3:
            assert time.time() < deadline, "local jobs did not finish"
            time.sleep(0.2)
        ex.shutdown()

        ex2 = CacheExecutor(cache)
        rc.mark_non_modal(ex2)
        g2 = _sweep_graph(Square(store=True), ex2)
        g2._db = db
        g2.pull_node("Iter")
        ex2.shutdown()
        assert ex2.keys == []
        assert list(g2.nodes["Iter"].outputs.df.value["y"]) == [1, 4, 9]
        assert all(r.output_path for r in _cells(db))
