from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from typing import Optional, Union

import pandas as pd
from ase import Atoms

from core import (
    as_function_node,
    PortList,
)
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
    """Filter a Stoichiometry by the number of atoms per structure.

    Args:
        min (int): keep only compositions with at least this many atoms
        max (int): keep only compositions with at most this many atoms

    Returns:
        Stoichiometry: filtered object
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
        elements (Stoichiometry): list of compositions, one per structure
        spacegroups (list of int): which space groups to generate; all 230 by default
        max_atoms (int): do not generate structures larger than this
        max_structures (int): generate at most this many structures
    Returns:
        list of Atoms: generated structures
    """
    import math
    from warnings import catch_warnings

    from assyst.crystals import pyxtal
    from tqdm.auto import tqdm

    if spacegroups is None:
        spacegroups = list(range(1, 231))
    if max_structures is None:
        max_structures = math.inf

    structures = []
    with catch_warnings(category=UserWarning, action="ignore"):
        for stoich in (bar := tqdm(elements)):
            symbols, num_ions = zip(*stoich.items(), strict=False)
            stoich_str = "".join(
                f"{s}{n}" for s, n in zip(symbols, num_ions, strict=False)
            )
            bar.set_description(stoich_str)
            structures += [
                _ase_to_data(s["atoms"]) for s in pyxtal(spacegroups, symbols, num_ions)
            ]
            if len(structures) > max_structures:
                structures = structures[:max_structures]
                break
        bar.close()
    return structures


@as_function_node
def CombineStructureSets(
    sets: PortList = PortList(
        ["spacegroups", "volume_relax", "full_relax", "rattle", "stretch"],
        required=False,
    ),
) -> list[Atoms]:
    """Combine any number of structure sets into a full training set.

    Add, rename and remove inputs with the "+" and "x" buttons on the node.
    The default port names follow the stages of the ASSYST pipeline.
    """
    import logging

    structures = [s for values in sets.values() if values for s in values]
    if len(structures) == 0:
        logging.warning(
            "Either no inputs given or all inputs are empty. "
            "Returning the empty list!"
        )
    return structures


@as_function_node
def CombineStructures(
    set1: list[Atoms],
    set2: list[Atoms],
    set3: list[Atoms] | None,
    set4: list[Atoms] | None,
    set5: list[Atoms] | None,
) -> list[Atoms]:
    """Deprecated: use ``CombineStructureSets``, which takes any number of sets."""
    set3 = set3 or []
    set4 = set4 or []
    set5 = set5 or []
    structures = set1 + set2 + set3 + set4 + set5
    return structures


@as_function_node("path")
def SaveStructures(structures: list[Atoms], filename: str):
    """Save list of structures into a pickled dataframe.

    Columns are:
        'name': a structure label
        'ase_atoms': the ASE object for the actual structure
        'number_of_atoms': the number of atoms inside the structure

    If `filename` does not end with 'pckl.gz', it is added.

    Args:
        structures (list of Atoms): structures to save
        filename (str): path where the dataframe is written to
    """
    import os.path

    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "name": s.info.get("label", f"structure_{i}"),
                "ase_atoms": s,
                "number_of_atoms": len(s),
            }
            for i, s in enumerate(structures)
        ]
    )
    if not filename.endswith("pckl.gz"):
        filename += ".pckl.gz"
    dirname = os.path.dirname(filename)
    if dirname:  # a bare filename has no directory part
        os.makedirs(dirname, exist_ok=True)
    df.to_pickle(filename)
    return filename
