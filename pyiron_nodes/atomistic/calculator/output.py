from pyiron_nodes.atomistic.calculator.data import (
    OutputCalcMinimize,
    OutputCalcStaticList,
)
from core import as_function_node


@as_function_node("energy_last")
def GetEnergyLast(
    calculator=None,
    store: bool = False,
    _db=None,
) -> float:
    if isinstance(calculator, OutputCalcMinimize.pure_dataclass):
        energy_last = calculator.final.energy
    elif isinstance(calculator, OutputCalcStaticList.pure_dataclass):
        energy_last = calculator.energies_pot[-1]
    return energy_last
