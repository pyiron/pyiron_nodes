from dataclasses import asdict, dataclass, field
from typing import Optional
from core import as_function_node
import numpy as np
import pandas as pd


@dataclass
class EmbeddingsALL:
    npot: str = "FinnisSinclairShiftedScaled"
    fs_parameters: list[int] = field(default_factory=lambda: [1, 1])
    ndensity: int = 1


@dataclass
class Embeddings:
    ALL: EmbeddingsALL = field(default_factory=EmbeddingsALL)


@dataclass
class BondsALL:
    radbase: str = "SBessel"
    radparameters: list[float] = field(default_factory=lambda: [5.25])
    rcut: float | int = 7.0
    dcut: float = 0.01


@dataclass
class Bonds:
    ALL: BondsALL = field(default_factory=BondsALL)


@dataclass
class FunctionsALL:
    nradmax_by_orders: list[int] = field(default_factory=lambda: [15, 3, 2, 1])
    lmax_by_orders: list[int] = field(default_factory=lambda: [0, 3, 2, 1])


@dataclass
class Functions:
    number_of_functions_per_element: Optional[int] = None
    ALL: FunctionsALL = field(default_factory=FunctionsALL)


@dataclass
class PotentialConfig:
    deltaSplineBins: float = 0.001
    elements: list[str] | None = None

    embeddings: Embeddings = field(default_factory=Embeddings)
    bonds: Bonds = field(default_factory=Bonds)
    functions: Functions = field(default_factory=Functions)

    def __post_init__(self):
        if not isinstance(self.embeddings, Embeddings):
            self.embeddings = Embeddings()
        if not isinstance(self.bonds, Bonds):
            self.bonds = Bonds()
        if not isinstance(self.functions, Functions):
            self.functions = Functions()

    def to_dict(self):
        def remove_none(d):
            """Recursively remove None values from dictionaries."""
            if isinstance(d, dict):
                return {k: remove_none(v) for k, v in d.items() if v is not None}
            elif isinstance(d, list):
                return [remove_none(v) for v in d if v is not None]
            else:
                return d

        return remove_none(asdict(self))


def _get_predicted_energies_forces(ace, structures):
    forces = []
    energies = []

    for s in structures:
        s.calc = ace
        energies.append(s.get_potential_energy())
        forces.append(s.get_forces())
        s.calc = None
    return energies, forces
