from dataclasses import dataclass

import pytest

from core import Workflow, as_function_node, group_node
from pyiron_nodes.controls import while_loop


@dataclass
class Counter:
    value: int = 0


@as_function_node
def count_up(state: Counter, stop_at: int = 3) -> tuple[Counter, bool]:
    new_state = Counter(value=state.value + 1)
    break_condition = new_state.value >= stop_at

    return new_state, break_condition


@as_function_node
def halve(number, floor: float = 1.0) -> tuple[float, bool]:
    halved = number / 2
    break_condition = halved < floor

    return halved, break_condition


@as_function_node
def add_one(x: int) -> int:
    x_plus_one = x + 1

    return x_plus_one


@as_function_node
def at_least(value: int, threshold: int) -> bool:
    reached = value >= threshold

    return reached


@group_node("x", "break_condition")
def counting_step(x=0, threshold=3):
    wf = Workflow("counting_step")
    wf.add_one = add_one(x=x)
    wf.at_least = at_least(value=wf.add_one, threshold=threshold)

    return wf.add_one, wf.at_least


def test_dataclass_loop_variable_and_custom_input_label():
    step = count_up(state=Counter(0), stop_at=5)
    x, n_steps = while_loop(step=step, input_label="state").run()

    assert x == Counter(5)
    assert n_steps == 5


def test_float_loop_variable():
    x, n_steps = while_loop(step=halve(number=40.0), input_label="number").run()

    assert x == 0.625
    assert n_steps == 6


def test_unknown_input_label_raises():
    with pytest.raises(ValueError, match="not an input port"):
        while_loop(step=halve(number=1.0), input_label="nope").run()


def test_max_steps_raises_instead_of_returning():
    step = count_up(state=Counter(0), stop_at=100)
    with pytest.raises(RuntimeError, match="did not finish"):
        while_loop(step=step, input_label="state", max_steps=3).run()


def test_stops_after_a_single_step():
    x, n_steps = while_loop(step=halve(number=1.0), input_label="number").run()

    assert x == 0.5
    assert n_steps == 1


def test_starting_value_comes_from_the_step_input():
    step = count_up(state=Counter(2), stop_at=5)
    x, n_steps = while_loop(step=step, input_label="state").run()

    assert x == Counter(5)
    assert n_steps == 3


def test_prints_every_iteration(capsys):
    while_loop(step=halve(number=40.0), input_label="number").run()
    printed = capsys.readouterr().out.splitlines()
    loop_lines = [line for line in printed if line.startswith("loop:")]

    assert len(loop_lines) == 6
    assert loop_lines[0].split() == ["loop:", "1", "20.0", "False"]
    assert loop_lines[-1].split() == ["loop:", "6", "0.625", "True"]


def test_group_node_as_step():
    x, n_steps = while_loop(step=counting_step(x=0, threshold=4), input_label="x").run()

    assert x == 4
    assert n_steps == 4
