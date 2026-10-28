from dataclasses import dataclass

import pytest

from core import as_function_node
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
