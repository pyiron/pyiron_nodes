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
