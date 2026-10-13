"""Sweeping several input ports at once with ``IterToDataFrame``.

``input_label`` may name several ports, comma-separated; every swept value is
then a tuple with one entry per label.
"""

from concurrent.futures import ThreadPoolExecutor

import pytest

from core import as_function_node
from pyiron_nodes.controls import IterToDataFrame, _iterate_node


@as_function_node
def Affine(x: float = 0.0, a: float = 1.0, b: float = 0.0):
    y = a * x + b
    return y


def _sweep(input_label, values, executor=None):
    sweep = IterToDataFrame(
        node=Affine(), input_label=input_label, values=values, executor=executor
    )
    sweep.run()
    return sweep.outputs.df.value


CELLS = [(1.0, 2.0, 0.0), (2.0, 2.0, 1.0), (3.0, -1.0, 0.5)]


@pytest.mark.parametrize("parallel", [False, True])
def test_one_column_per_label_in_order(parallel):
    pool = ThreadPoolExecutor(max_workers=2) if parallel else None
    try:
        df = _sweep("x, a,b", CELLS, executor=pool)
    finally:
        if pool is not None:
            pool.shutdown()

    assert list(df.columns) == ["x", "a", "b", "y"]
    assert df["x"].tolist() == [1.0, 2.0, 3.0]
    assert df["y"].tolist() == [2.0, 5.0, -2.5]
    assert "error" not in df.columns


def test_single_label_is_unchanged():
    df = _sweep("x", [1.0, 2.0])
    assert list(df.columns) == ["x", "y"]
    assert df["y"].tolist() == [1.0, 2.0]


@as_function_node
def Echo(v=None):
    out = v
    return out


@as_function_node
def Shift(y: float = 0.0, d: float = 0.0):
    y = y + d
    return y


def test_a_single_label_still_takes_a_tuple_whole():
    # Only a comma in input_label unpacks; a tuple swept on one port arrives
    # as the tuple, as it always did.
    sweep = IterToDataFrame(node=Echo(), input_label="v", values=[(1, 2)])
    sweep.run()
    df = sweep.outputs.df.value
    assert df["v"].tolist() == [(1, 2)]
    assert df["out"].tolist() == [(1, 2)]


def test_a_label_that_clashes_with_an_output_gets_the_input_prefix():
    sweep = IterToDataFrame(node=Shift(), input_label="y,d", values=[(1.0, 0.5)])
    sweep.run()
    df = sweep.outputs.df.value
    assert list(df.columns) == ["input_y", "d", "y"]
    assert df.iloc[0].tolist() == [1.0, 0.5, 1.5]


@pytest.mark.parametrize("bad", [[(1.0, 2.0)], [1.0], ["xy"]])
def test_wrong_tuple_length_fails_before_the_first_iteration(bad):
    calls = []

    class Spy:
        def submit(self, *args, **kwargs):  # pragma: no cover - must not be reached
            calls.append(args)
            raise AssertionError("submitted despite a malformed sweep")

    with pytest.raises(ValueError, match="does not match input_label"):
        _iterate_node(Affine(), "x,a,b", bad, executor=Spy())
    assert calls == []


# ---------------------------------------------------------------------------
# Dict cells, and the swept ports belonging to the sweep
# ---------------------------------------------------------------------------


def test_dict_cells_fill_columns_in_label_order():
    cells = [{"b": 0.0, "x": 1.0, "a": 2.0}, {"a": -1.0, "b": 0.5, "x": 3.0}]
    df = _sweep("x,a,b", cells)
    assert list(df.columns) == ["x", "a", "b", "y"]
    assert df["y"].tolist() == [2.0, -2.5]


@pytest.mark.parametrize("cell", [{"x": 1.0, "a": 2.0}, {"x": 1.0, "a": 2.0, "b": 0.0, "c": 1}])
def test_dict_keys_must_match_input_label(cell):
    with pytest.raises(ValueError, match="missing|extra"):
        _iterate_node(Affine(), "x,a,b", [cell])


@pytest.mark.parametrize("parallel", [False, True])
def test_swept_ports_are_restored_after_the_sweep(parallel):
    template = Affine(a=5.0)  # `a` is swept below; the user's value comes back
    pool = ThreadPoolExecutor(max_workers=2) if parallel else None
    try:
        _iterate_node(template, "x,a", [(1.0, 2.0), (3.0, 4.0)], executor=pool)
    finally:
        if pool is not None:
            pool.shutdown()
    assert template.inputs.a.value == 5.0
    assert template.inputs.x.value == 0.0  # default, untouched by the last cell


def test_the_sweep_hash_does_not_depend_on_run_history():
    pdb = pytest.importorskip("pyiron_database")
    template = Affine(b=1.0)
    sweep = IterToDataFrame(node=template, input_label="x,a", values=[(1.0, 2.0), (3.0, 4.0)])
    before = pdb.get_hash(template)
    sweep.run()
    assert pdb.get_hash(template) == before


def test_a_set_swept_port_is_reported_as_ignored(caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        _iterate_node(Affine(a=5.0), "x,a", [(1.0, 2.0)])
    ignored = [r for r in caplog.records if "is ignored" in r.getMessage()]
    assert len(ignored) == 1
    assert "'Affine.a' = 5.0" in ignored[0].getMessage()


def test_unset_swept_ports_are_not_reported(caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        _iterate_node(Affine(b=1.0), "x,a", [(1.0, 2.0)])
    assert not [r for r in caplog.records if "is ignored" in r.getMessage()]
