from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.constraints import FixAtoms
from ase.filters import FrechetCellFilter

from core import (
    as_function_node,
    as_inp_dataclass_node,
)

GPA2EVA3 = 0.006_241_509_074


class AseCalculatorConfig(ABC):
    @abstractmethod
    def get_calculator(self) -> Calculator:
        pass


@dataclass
class PawDftInput:
    encut: int | float | None = 320.0
    kspacing: float | None = 0.5

    scf_energy_convergence: float = 1e-2


class RelaxMode(Enum):
    VOLUME = "volume"
    FULL = "full"

    def apply_filter_and_constraints(self, structure):
        match self:
            case RelaxMode.VOLUME:
                structure.set_constraint(FixAtoms(np.ones(len(structure), dtype=bool)))
                return FrechetCellFilter(structure, hydrostatic_strain=True)
            case RelaxMode.FULL:
                return FrechetCellFilter(structure)
            case _:
                raise ValueError(
                    f"Expected to match available enum: {RelaxMode.__members__}"
                )


def Relax(
    calculator: AseCalculatorConfig,
    opt: GenericOptimizerSettings,
    structure: Atoms,
    mode: str = "volume",
) -> Atoms:
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.optimize import LBFGS

    mode = RelaxMode(mode)

    structure = structure.copy()

    # FIXME: meh
    match mode:
        case RelaxMode.VOLUME:
            structure.calc = calculator.get_calculator(use_symmetry=True)
        case RelaxMode.FULL:
            structure.calc = calculator.get_calculator(use_symmetry=False)
        case _:
            raise ValueError(
                f"Expected to match available enum: {RelaxMode.__members__}"
            )

    filtered_structure = mode.apply_filter_and_constraints(structure)
    lbfgs = LBFGS(filtered_structure, logfile="/dev/null")
    lbfgs.run(fmax=opt.force_tolerance, steps=opt.max_steps)
    calc = structure.calc
    structure.calc = SinglePointCalculator(
        structure,
        **{
            "energy": calc.get_potential_energy(),
            "forces": calc.get_forces(),
            "stress": calc.get_stress(),
        },
    )
    # play catch with nodes
    relaxed_structure = structure
    relaxed_structure.constraints.clear()
    return relaxed_structure


@as_function_node
def RelaxLoop(
    calculator: AseCalculatorConfig,
    opt: GenericOptimizerSettings,
    structures: list[Atoms],
    mode: str = "volume",
) -> list[Atoms]:
    from tqdm.auto import tqdm

    mode = RelaxMode(mode)
    relaxed_structures = []
    for structure in tqdm(structures, desc=f"Relax {mode.value}"):
        relaxed_structures.append(Relax(calculator, opt, structure, mode))
    return relaxed_structures
