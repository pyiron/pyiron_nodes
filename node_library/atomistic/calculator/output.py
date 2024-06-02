from pyiron_workflow.function import as_function_node
from typing import Optional, Union

from node_library.atomistic.calculator.data import OutputCalcMinimize, OutputCalcStatic


@as_function_node("energy_last")
def get_energy_last(calculator: Optional[OutputCalcMinimize | OutputCalcStatic] = None) -> float:
    if isinstance(calculator, OutputCalcMinimize):
        energy_last = calculator.final.energy
    elif isinstance(calculator, OutputCalcStatic):
        energy_last = calculator.energy
    return energy_last
    # print ('energy_last:', calculator.energy[-1], type(calculator.energy[-1]))
    # return calculator.energy[-1]


nodes = [get_energy_last]
