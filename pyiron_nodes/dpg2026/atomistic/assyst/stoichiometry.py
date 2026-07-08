from ase.atoms import Atoms
from dataclasses import dataclass
from itertools import product
from collections.abc import Sequence
import pandas as pd
from core import as_function_node
from assyst.crystals import pyxtal
from typing import Optional, Union
from pyiron_nodes.atomistic.structure._atoms import _ase_to_data


@dataclass(frozen=True)
class Stoichiometry(Sequence):
    stoichiometry: tuple[dict[str, int]]

    @property
    def elements(self) -> set[str]:
        """Set of elements present in stoichiometry."""
        e = set()
        for s in self.stoichiometry:
            s = e.union(s.keys())
        return s

    # FIXME: Self only availabe in >=3.11
    def __add__(self, other: "Stoichiometry") -> "Stoichiometry":
        """Extend underlying list of stoichiometries."""
        return Stoichiometry(self.stoichiometry + other.stoichiometry)

    def __or__(self, other: "Stoichiometry") -> "Stoichiometry":
        """Inner product of underlying stoichiometries.

        Must not share elements with other stoichiometry."""
        assert self.elements.isdisjoint(
            other.elements
        ), "Can only or stoichiometries of different elements!"
        s = ()
        for me, you in zip(self.stoichiometry, other.stoichiometry, strict=False):
            s += (me | you,)
        return Stoichiometry(s)

    def __mul__(self, other: "Stoichiometry") -> "Stoichiometry":
        """Outer product of underlying stoichiometries.

        Must not share elements with other stoichiometry."""
        assert self.elements.isdisjoint(
            other.elements
        ), "Can only multiply stoichiometries of different elements!"
        s = ()
        for me, you in product(self.stoichiometry, other.stoichiometry):
            s += (me | you,)
        return Stoichiometry(s)

    # Sequence Impl'
    def __getitem__(self, index: int) -> dict[str, int]:
        return self.stoichiometry[index]

    def __len__(self) -> int:
        return len(self.stoichiometry)


@as_function_node
def ElementInput(
    element: str,
    min_ion: int = 1,
    max_ion: int = 10,
    step_ion: int = 1,
) -> Stoichiometry:
    stoichiometry = Stoichiometry(
        tuple({element: i} for i in range(min_ion, max_ion + 1, step_ion))
    )
    return stoichiometry


@as_function_node("df")
def StoichiometryTable(stoichiometry: Stoichiometry) -> pd.DataFrame:
    return pd.DataFrame(stoichiometry.stoichiometry)


@as_function_node("filtered")
def FilterSize(
    elements: Stoichiometry,
    min: Optional[int] = 0,
    max: Optional[int] = None,
):
    """Filter an Elements object by size.

    Args:
        min (int): new object has at least this number of atoms
        max (int): new object has at most this number of atoms

    Returns:
        Elements: filtered object
    """
    import math

    if max is None:
        max = math.inf
    return Stoichiometry(tuple(s for s in elements if min <= sum(s.values()) <= max))


@as_function_node
def SpaceGroupSampling(
    elements: Stoichiometry,
    spacegroups: Optional[Union[list[int], tuple[int, ...]]] = None,
    max_atoms: int = 4,
    max_structures: int = 10,
    store: bool = False,
) -> list[Atoms]:
    """
    Create symmetric random structures.

    Args:
        elements (Elements): list of compositions per structure
        spacegroups (list of int): which space groups to generate
        max_atoms (int): do not generate structures larger than this
        max_structures (int): generate at most this many structures
    Returns:
        list of Atoms: generated structures
    """
    from warnings import catch_warnings
    from tqdm.auto import tqdm
    import math

    if spacegroups is None:
        spacegroups = list(range(1, 231))
    if max_structures is None:
        max_structures = math.inf

    structures = []
    with catch_warnings(category=UserWarning, action="ignore"):
        for stoich in (bar := tqdm(elements)):
            elements, num_atomss = zip(*stoich.items())
            stoich_str = "".join(f"{s}{n}" for s, n in zip(elements, num_atomss))
            bar.set_description(stoich_str)
            structures += [
                _ase_to_data(s["atoms"])
                for s in pyxtal(spacegroups, elements, num_atomss)
            ]
            if len(structures) > max_structures:
                structures = structures[:max_structures]
                break
    return structures
