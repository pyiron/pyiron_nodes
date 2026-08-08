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
    if hasattr(calculator, "final") and hasattr(calculator.final, "energy"):
        energy_last = calculator.final.energy
    elif hasattr(calculator, "energies_pot"):
        energy_last = calculator.energies_pot[-1]
    else:
        raise TypeError(f"Unrecognised calculator output type: {type(calculator)}")
    return energy_last
