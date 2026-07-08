from dataclasses import dataclass, field
from typing import Optional, Literal

import copy
import pandas as pd

from core import as_function_node
from pyiron_nodes.atomistic.structure._atoms import (
    OutputAtoms,
    _ase_to_data,
    _data_to_ase,
)

# ---------------------------------------------------------------------------
# Empty structure table
# ---------------------------------------------------------------------------


def _empty_structure_table() -> pd.DataFrame:
    """
    Return an empty DataFrame with the canonical columns and correct dtypes.

    Columns
    -------
    structure
        The structure stored as an :class:`OutputAtoms` dataclass instance
        (picklable).  Use :func:`_data_to_ase` to recover an ASE
        :class:`~ase.atoms.Atoms` object when needed.
    operation
        Human-readable string (``"pristine"`` for reference structures).
    stoichiometry
        Compact chemical formula, e.g. ``"Al107Mg1"``.
    is_pristine
        ``True`` for reference structures, ``False`` for defects.
    pristine_structure_index
        Absolute row index of the original pristine ancestor.  ``-1`` for
        pristine rows themselves.
    parent_index
        Absolute row index of the immediate parent this structure was
        derived from.  ``-1`` for pristine rows.
    """
    return pd.DataFrame(
        {
            "structure": pd.Series(dtype=object),
            "operation": pd.Series(dtype=str),
            "stoichiometry": pd.Series(dtype=str),
            "is_pristine": pd.Series(dtype=bool),
            "pristine_structure_index": pd.Series(dtype=int),
            "parent_index": pd.Series(dtype=int),
        }
    )


# ---------------------------------------------------------------------------
# Stoichiometry helper
# ---------------------------------------------------------------------------


def _get_stoichiometry(structure: OutputAtoms) -> str:
    """
    Derive a compact stoichiometry string from an :class:`OutputAtoms`
    instance.

    Parameters
    ----------
    structure : OutputAtoms

    Returns
    -------
    str
        E.g. ``"Al107Mg1"`` or ``"Cu4"``.
    """
    from collections import Counter

    counts = Counter(structure.symbols)
    return "".join(f"{el}{counts[el]}" for el in sorted(counts))


# ---------------------------------------------------------------------------
# StructureContainer
# ---------------------------------------------------------------------------


@dataclass
class StructureContainer:
    """
    Unified container for pristine reference structures and defect
    structures, including chained multi-defect and independent
    single-defect structures that share the same pristine ancestor.

    Structures are stored as :class:`OutputAtoms` dataclass instances so
    that the underlying DataFrame is fully picklable — raw ASE
    :class:`~ase.atoms.Atoms` objects are never stored directly.

    Use :func:`_data_to_ase` / :func:`_ase_to_data` (from
    ``pyiron_nodes.atomistic.structure._atoms``) to convert between
    :class:`OutputAtoms` and :class:`~ase.atoms.Atoms` at the call site.

    Attributes
    ----------
    table : pd.DataFrame
        One row per structure.  See :func:`_empty_structure_table` for
        column descriptions.
    """

    table: pd.DataFrame = field(default_factory=_empty_structure_table)

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _add_row(
        self,
        structure: OutputAtoms,
        operation: str,
        is_pristine: bool,
        pristine_structure_index: int,
        parent_index: int,
    ) -> int:
        new_row = pd.DataFrame(
            {
                "structure": pd.Series([structure], dtype=object),
                "operation": pd.Series([operation], dtype=str),
                "stoichiometry": pd.Series([_get_stoichiometry(structure)], dtype=str),
                "is_pristine": pd.Series([is_pristine], dtype=bool),
                "pristine_structure_index": pd.Series(
                    [pristine_structure_index], dtype=int
                ),
                "parent_index": pd.Series([parent_index], dtype=int),
            }
        )
        self.table = pd.concat([self.table, new_row], ignore_index=True)
        return int(self.table.index[-1])

    # ------------------------------------------------------------------ #
    #  Index translation                                                   #
    # ------------------------------------------------------------------ #

    def resolve_defect_row(self, relative_index: int) -> int:
        """
        Translate a *relative* defect index to an *absolute* table row
        index.

        Relative indices follow standard Python list semantics:

        * ``0``  → first defect ever added
        * ``1``  → second defect
        * ``-1`` → most recently added defect
        * ``-2`` → second-to-last defect

        Parameters
        ----------
        relative_index : int
            Position within the sub-sequence of defect rows only
            (``is_pristine == False``).

        Returns
        -------
        int
            Absolute row index in :attr:`table`.

        Raises
        ------
        ValueError
            If no defect rows exist yet.
        IndexError
            If *relative_index* is out of range.
        """
        # Get the subset of defect rows
        defect_subset = self.table[self.table["is_pristine"].eq(False)]
        defect_indices = defect_subset.index.tolist()

        if not defect_indices:
            raise ValueError(
                "No defect rows in the container yet. "
                "Create at least one defect before referencing by "
                "relative index."
            )

        # Validate relative_index is an integer
        if not isinstance(relative_index, int):
            raise TypeError(
                f"relative_index must be an integer, got {type(relative_index).__name__}"
            )

        try:
            return int(defect_indices[relative_index])
        except IndexError:
            raise IndexError(
                f"Relative defect index {relative_index} is out of "
                f"range; there are {len(defect_indices)} defect(s) in "
                f"the container."
            )

    def resolve_any_row(self, relative_index: int) -> int:
        """
        Translate a *relative* index over **all** rows (pristine + defect)
        to an absolute table row index.

        Parameters
        ----------
        relative_index : int

        Returns
        -------
        int
            Absolute row index in :attr:`table`.

        Raises
        ------
        ValueError
            If the container is empty.
        TypeError
            If *relative_index* is not an integer.
        IndexError
            If *relative_index* is out of range.
        """
        all_indices = self.table.index.tolist()

        if not all_indices:
            raise ValueError("The container is empty.")

        # Validate relative_index is an integer
        if not isinstance(relative_index, int):
            raise TypeError(
                f"relative_index must be an integer, got {type(relative_index).__name__}"
            )

        try:
            return int(all_indices[relative_index])
        except IndexError:
            raise IndexError(
                f"Relative index {relative_index} is out of range; "
                f"the container has {len(all_indices)} row(s)."
            )

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def add_pristine(self, atoms) -> int:
        """
        Register a pristine reference structure with deduplication.

        Accepts either an ASE :class:`~ase.atoms.Atoms` object or an
        :class:`OutputAtoms` instance.  The structure is always stored
        internally as :class:`OutputAtoms`.

        Two structures are considered identical when stoichiometry,
        positions, and cell all match within ``1e-6`` Å.

        Parameters
        ----------
        atoms : Atoms or OutputAtoms

        Returns
        -------
        int
            Absolute row index of the (possibly pre-existing) pristine
            entry.
        """
        import numpy as np
        from ase.atoms import Atoms as _Atoms

        structure = _ase_to_data(atoms) if isinstance(atoms, _Atoms) else atoms
        candidate_stoich = _get_stoichiometry(structure)

        pristine_rows = self.table[self.table["is_pristine"].eq(True)]
        for row_idx, stored in zip(pristine_rows.index, pristine_rows["structure"]):
            if (
                _get_stoichiometry(stored) == candidate_stoich
                and np.allclose(stored.positions, structure.positions, atol=1e-6)
                and np.allclose(stored.cell, structure.cell, atol=1e-6)
            ):
                return int(row_idx)

        return self._add_row(
            structure=structure,
            operation="pristine",
            is_pristine=True,
            pristine_structure_index=-1,
            parent_index=-1,
        )

    def add_defect(
        self,
        structure,
        operation: str,
        pristine_structure_index: int,
        parent_index: int,
    ) -> int:
        """
        Append a defect structure.

        Accepts either an ASE :class:`~ase.atoms.Atoms` object or an
        :class:`OutputAtoms` instance.  The structure is always stored
        internally as :class:`OutputAtoms`.

        Parameters
        ----------
        structure : Atoms or OutputAtoms
        operation : str
        pristine_structure_index : int
            Absolute row index of the original pristine ancestor.
        parent_index : int
            Absolute row index of the immediate parent.

        Returns
        -------
        int
            Absolute row index of the newly added entry.
        """
        from ase.atoms import Atoms as _Atoms

        if isinstance(structure, _Atoms):
            structure = _ase_to_data(structure)

        return self._add_row(
            structure=structure,
            operation=operation,
            is_pristine=False,
            pristine_structure_index=pristine_structure_index,
            parent_index=parent_index,
        )

    def latest_pristine_index(self) -> int:
        """
        Absolute row index of the most recently added pristine structure.

        Raises
        ------
        ValueError
            If no pristine has been registered yet.
        """
        pristine_rows = self.table[self.table["is_pristine"].eq(True)]
        if pristine_rows.empty:
            raise ValueError(
                "No pristine structure found in the container. "
                "Call AddPristine before creating defects."
            )
        return int(pristine_rows.index[-1])

    def row_index_of(self, structure) -> Optional[int]:
        """
        Return the absolute row index of *structure*, or ``None`` if
        absent.

        Accepts either an ASE :class:`~ase.atoms.Atoms` object or an
        :class:`OutputAtoms` instance.  Identity (``is``) is checked
        first; falls back to numerical equality.

        Parameters
        ----------
        structure : Atoms or OutputAtoms

        Returns
        -------
        int or None
        """
        import numpy as np
        from ase.atoms import Atoms as _Atoms

        if isinstance(structure, _Atoms):
            structure = _ase_to_data(structure)

        candidate_stoich = _get_stoichiometry(structure)

        for row_idx in self.table.index:
            stored = self.table.loc[row_idx, "structure"]
            if stored is structure:
                return int(row_idx)
            if (
                _get_stoichiometry(stored) == candidate_stoich
                and np.allclose(stored.positions, structure.positions, atol=1e-6)
                and np.allclose(stored.cell, structure.cell, atol=1e-6)
            ):
                return int(row_idx)
        return None

    def get_structure(self, absolute_row: int) -> OutputAtoms:
        """
        Return the :class:`OutputAtoms` at *absolute_row*.

        Use :func:`_data_to_ase` to convert to an ASE
        :class:`~ase.atoms.Atoms` object if needed.
        """
        return self.table.loc[absolute_row, "structure"]

    def get_atoms(self, absolute_row: int):
        """
        Return the structure at *absolute_row* as an ASE
        :class:`~ase.atoms.Atoms` object.
        """
        return _data_to_ase(self.get_structure(absolute_row))

    def get_pristine(self, absolute_row: int) -> OutputAtoms:
        """
        Return the pristine ancestor for a given defect row as
        :class:`OutputAtoms`.

        Raises
        ------
        ValueError
            If *absolute_row* is itself a pristine row.
        """
        pristine_idx = int(self.table.loc[absolute_row, "pristine_structure_index"])
        if pristine_idx == -1:
            raise ValueError(f"Row {absolute_row} is already a pristine structure.")
        return self.table.loc[pristine_idx, "structure"]

    def get_parent(self, absolute_row: int) -> OutputAtoms:
        """
        Return the immediate parent structure for a given defect row as
        :class:`OutputAtoms`.

        Raises
        ------
        ValueError
            If *absolute_row* is a pristine row.
        """
        parent_idx = int(self.table.loc[absolute_row, "parent_index"])
        if parent_idx == -1:
            raise ValueError(
                f"Row {absolute_row} is a pristine structure and has " f"no parent."
            )
        return self.table.loc[parent_idx, "structure"]

    def __repr__(self) -> str:  # pragma: no cover
        return f"StructureContainer(\n{self.table}\n)"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_structure(obj) -> OutputAtoms:
    """
    Unwrap *obj* to an :class:`OutputAtoms` dataclass instance.

    Accepts:

    * :class:`OutputAtoms` (plain dataclass instance — returned as-is)
    * ASE :class:`~ase.atoms.Atoms` (converted via :func:`_ase_to_data`)
    * Any object with an ``atoms`` attribute that is an ASE
      :class:`~ase.atoms.Atoms` instance (converted via
      :func:`_ase_to_data`)

    Raises
    ------
    TypeError
        If *obj* cannot be converted to an :class:`OutputAtoms` instance.
    """
    from ase.atoms import Atoms as _Atoms

    # OutputAtoms is a plain dataclass — check with isinstance directly.
    # Do NOT use OutputAtoms.dataclass_type; that attribute only exists
    # on Node classes produced by @as_out_dataclass_node, not on the
    # plain dataclass itself.
    if isinstance(obj, OutputAtoms):
        return obj
    if isinstance(obj, _Atoms):
        return _ase_to_data(obj)
    if hasattr(obj, "atoms") and isinstance(obj.atoms, _Atoms):
        return _ase_to_data(obj.atoms)
    raise TypeError(
        f"Cannot extract an OutputAtoms object from {type(obj)!r}. "
        f"Expected an OutputAtoms dataclass instance or an ASE Atoms "
        f"object."
    )


def _ensure_container(
    container: Optional[StructureContainer],
) -> StructureContainer:
    """Return *container* or a fresh :class:`StructureContainer`."""
    if container is None:
        return StructureContainer()
    if not isinstance(container, StructureContainer):
        raise TypeError(
            f"Expected a StructureContainer or None, " f"got {type(container)!r}."
        )
    return container


def _prepare(
    input_structure,
    structure_container: Optional[StructureContainer],
    parent_defect_index: Optional[int],
):
    """
    Resolve inputs and return everything a defect-creation node needs.

    All returned structures are :class:`OutputAtoms` instances.

    Resolution order
    ----------------
    1. *parent_defect_index* not ``None`` → translate relative defect
       index to absolute row; use that row as parent.
    2. *input_structure* given and already in container → chain on top.
    3. *input_structure* given but not in container → register as new
       pristine (with deduplication) and use as parent.
    4. Both ``None`` → use the latest pristine as parent.

    Parameters
    ----------
    input_structure : Atoms, OutputAtoms, or None
    structure_container : StructureContainer or None
    parent_defect_index : int or None

    Returns
    -------
    tuple[OutputAtoms, StructureContainer, int, int]
        ``(structure, container, pristine_index, parent_index)``
        where both index values are **absolute** row indices.
    """
    container = _ensure_container(structure_container)

    # Rule 1 — relative defect index supplied
    if parent_defect_index is not None:
        abs_row = container.resolve_defect_row(parent_defect_index)
        row_data = container.table.loc[abs_row]
        structure = row_data["structure"]
        is_prist = bool(row_data["is_pristine"])
        pristine_index = (
            abs_row if is_prist else int(row_data["pristine_structure_index"])
        )
        return structure, container, pristine_index, abs_row

    # Rule 2 & 3 — explicit input_structure
    if input_structure is not None:
        structure = _resolve_structure(input_structure)
        existing_row = container.row_index_of(structure)

        if existing_row is not None:
            # Rule 2: already in container → chain on top
            row_data = container.table.loc[existing_row]
            is_prist = bool(row_data["is_pristine"])
            pristine_index = (
                existing_row if is_prist else int(row_data["pristine_structure_index"])
            )
            return structure, container, pristine_index, existing_row

        # Rule 3: new → register as pristine (deep-copy for safety)
        pristine_copy = copy.deepcopy(structure)
        pristine_index = container.add_pristine(pristine_copy)
        structure = container.get_structure(pristine_index)
        return structure, container, pristine_index, pristine_index

    # Rule 4 — fall back to latest pristine
    pristine_index = container.latest_pristine_index()
    structure = container.get_structure(pristine_index)
    return structure, container, pristine_index, pristine_index


# ---------------------------------------------------------------------------
# Node: register a pristine structure
# ---------------------------------------------------------------------------


@as_function_node
def AddPristine(
    pristine_structure,
    structure_container: Optional[StructureContainer] = None,
) -> StructureContainer:
    """
    Register a pristine reference structure in the container.

    Parameters
    ----------
    pristine_structure : Atoms or OutputAtoms
    structure_container : StructureContainer or None, optional

    Returns
    -------
    StructureContainer
    """
    structure = _resolve_structure(pristine_structure)
    container = _ensure_container(structure_container)
    container.add_pristine(copy.deepcopy(structure))
    return container


# ---------------------------------------------------------------------------
# Pipeline nodes
# ---------------------------------------------------------------------------


@as_function_node
def CreateVacancy(
    index: int = 0,
    input_structure=None,
    structure_container: Optional[StructureContainer] = None,
    parent_defect_index: Optional[int] = None,
) -> StructureContainer:
    """
    Create a vacancy defect by removing one atom.

    Parent resolution
    -----------------
    * **Relative defect index** – set *parent_defect_index* to an integer
      relative to the defect-only rows: ``0`` = first defect,
      ``-1`` = latest defect.
    * **By object** – pass the parent structure via *input_structure*;
      it is looked up in the container.
    * **Single defect** – leave both ``None``; the latest pristine is
      used.

    Parameters
    ----------
    index : int, optional
        Zero-based index of the atom to remove.  Default ``0``.
    input_structure : Atoms, OutputAtoms, or None, optional
    structure_container : StructureContainer or None, optional
    parent_defect_index : int or None, optional

    Returns
    -------
    StructureContainer
    """
    structure, container, pristine_index, parent_index = _prepare(
        input_structure, structure_container, parent_defect_index
    )

    atoms = _data_to_ase(structure)
    n_atoms = len(atoms)
    if not (0 <= index < n_atoms):
        raise IndexError(
            f"Vacancy index {index} is out of range for a structure "
            f"with {n_atoms} atoms."
        )

    defect_atoms = copy.deepcopy(atoms)
    del defect_atoms[index]

    container.add_defect(
        structure=_ase_to_data(defect_atoms),
        operation=f"vacancy[{index}]",
        pristine_structure_index=pristine_index,
        parent_index=parent_index,
    )
    return container


@as_function_node
def CreateSubstitutional(
    element: str,
    index: int = 0,
    input_structure=None,
    structure_container: Optional[StructureContainer] = None,
    parent_defect_index: Optional[int] = None,
) -> StructureContainer:
    """
    Create a substitutional defect by replacing one atom with *element*.

    Parent resolution follows the same rules as :func:`CreateVacancy`.

    Parameters
    ----------
    element : str
        Chemical symbol of the substituting element (e.g. ``"Mg"``).
    index : int, optional
        Zero-based index of the atom to replace.  Default ``0``.
    input_structure : Atoms, OutputAtoms, or None, optional
    structure_container : StructureContainer or None, optional
    parent_defect_index : int or None, optional

    Returns
    -------
    StructureContainer
    """
    if not element:
        raise ValueError("'element' must be a non-empty chemical symbol string.")

    structure, container, pristine_index, parent_index = _prepare(
        input_structure, structure_container, parent_defect_index
    )

    atoms = _data_to_ase(structure)
    n_atoms = len(atoms)
    if not (0 <= index < n_atoms):
        raise IndexError(
            f"Substitutional index {index} is out of range for a "
            f"structure with {n_atoms} atoms."
        )

    defect_atoms = copy.deepcopy(atoms)
    defect_atoms[index].symbol = element

    container.add_defect(
        structure=_ase_to_data(defect_atoms),
        operation=f"substitutional[{index}->{element}]",
        pristine_structure_index=pristine_index,
        parent_index=parent_index,
    )
    return container


def _get_interstitial_sites_fcc(site_type: str) -> list:
    """
    Return fractional coordinates of interstitial sites in FCC structure.

    Parameters
    ----------
    site_type : str
        Either "octahedral" or "tetrahedral"

    Returns
    -------
    list of tuple
        Fractional coordinates (x, y, z) for each interstitial site
    """
    if site_type == "octahedral":
        # Octahedral sites in FCC: body center and edge centers
        return [
            (0.5, 0.5, 0.5),  # body center
            (0.5, 0.0, 0.0),  # edge centers
            (0.0, 0.5, 0.0),
            (0.0, 0.0, 0.5),
        ]
    elif site_type == "tetrahedral":
        # Tetrahedral sites in FCC
        return [
            (0.25, 0.25, 0.25),
            (0.75, 0.75, 0.25),
            (0.75, 0.25, 0.75),
            (0.25, 0.75, 0.75),
            (0.25, 0.75, 0.75),
            (0.75, 0.25, 0.75),
            (0.75, 0.75, 0.25),
            (0.25, 0.25, 0.75),
        ]
    else:
        raise ValueError(f"Unknown site_type: {site_type}")


def _get_interstitial_sites_bcc(site_type: str) -> list:
    """
    Return fractional coordinates of interstitial sites in BCC structure.

    Parameters
    ----------
    site_type : str
        Either "octahedral" or "tetrahedral"

    Returns
    -------
    list of tuple
        Fractional coordinates (x, y, z) for each interstitial site
    """
    if site_type == "octahedral":
        # Octahedral sites in BCC: edge centers
        return [
            (0.5, 0.0, 0.0),
            (0.0, 0.5, 0.0),
            (0.0, 0.0, 0.5),
        ]
    elif site_type == "tetrahedral":
        # Tetrahedral sites in BCC: face centers
        return [
            (0.5, 0.5, 0.0),
            (0.5, 0.0, 0.5),
            (0.0, 0.5, 0.5),
            (0.5, 0.5, 1.0),
            (0.5, 1.0, 0.5),
            (1.0, 0.5, 0.5),
        ]
    else:
        raise ValueError(f"Unknown site_type: {site_type}")


def _get_interstitial_sites_hcp(site_type: str, c_over_a: float = 1.633) -> list:
    """
    Return fractional coordinates of interstitial sites in HCP structure.

    Parameters
    ----------
    site_type : str
        Either "octahedral" or "tetrahedral"
    c_over_a : float, optional
        c/a ratio for the HCP structure. Default is ideal value 1.633.

    Returns
    -------
    list of tuple
        Fractional coordinates (x, y, z) for each interstitial site
    """
    if site_type == "octahedral":
        # Octahedral sites in HCP
        return [
            (0.0, 0.0, 0.25),
            (0.0, 0.0, 0.75),
            (2 / 3, 1 / 3, 0.25),
            (2 / 3, 1 / 3, 0.75),
        ]
    elif site_type == "tetrahedral":
        # Tetrahedral sites in HCP
        return [
            (0.0, 0.0, 0.375),
            (0.0, 0.0, 0.875),
            (2 / 3, 1 / 3, 0.375),
            (2 / 3, 1 / 3, 0.875),
            (1 / 3, 2 / 3, 0.125),
            (1 / 3, 2 / 3, 0.625),
        ]
    else:
        raise ValueError(f"Unknown site_type: {site_type}")


@as_function_node
def CreateInterstitial(
    element: str,
    crystal_type: Literal["fcc", "bcc", "hcp"],
    site_type: str = "octahedral",
    index: int = 0,
    input_structure=None,
    structure_container: Optional[StructureContainer] = None,
    parent_defect_index: Optional[int] = None,
) -> StructureContainer:
    """
    Create an interstitial defect by inserting a new atom at an interstitial site.

    An **interstitial** is a type of point defect where an atom occupies a
    position in the crystal lattice that is not normally occupied by atoms
    in the perfect crystal. These positions correspond to voids in the
    crystal structure:

    * **Octahedral interstitial** - surrounded by 6 nearest neighbors arranged
      as an octahedron
    * **Tetrahedral interstitial** - surrounded by 4 nearest neighbors arranged
      as a tetrahedron

    This function uses analytical positions based on the crystal structure type:

    * **fcc** (Face-Centered Cubic):
      - Octahedral: body center and edge centers
      - Tetrahedral: (1/4, 1/4, 1/4) type positions

    * **bcc** (Body-Centered Cubic):
      - Octahedral: edge centers
      - Tetrahedral: face centers

    * **hcp** (Hexagonal Close-Packed):
      - Octahedral: octahedral voids in hexagonal lattice
      - Tetrahedral: tetrahedral voids in hexagonal lattice

    Parent resolution follows the same rules as :func:`CreateVacancy`.

    Parameters
    ----------
    element : str
        Chemical symbol of the interstitial atom.
    crystal_type : Literal["fcc", "bcc", "hcp"]
        Type of crystal structure.
    site_type : str, optional
        Type of interstitial site, either ``"octahedral"`` or ``"tetrahedral"``.
        Default ``"octahedral"``.
    index : int, optional
        Zero-based index of which interstitial site to use when multiple
        sites of the requested type exist. Default ``0``.
    input_structure : Atoms, OutputAtoms, or None, optional
    structure_container : StructureContainer or None, optional
    parent_defect_index : int or None, optional

    Returns
    -------
    StructureContainer

    Raises
    ------
    ValueError
        If *site_type* or *crystal_type* is invalid.
    IndexError
        If *index* is out of range for the available sites.
    """
    import numpy as np
    from ase.atoms import Atoms as _Atoms

    # Validate element
    if not element:
        raise ValueError("'element' must be a non-empty chemical symbol string.")

    # Validate site_type
    if site_type not in ("octahedral", "tetrahedral"):
        raise ValueError(
            f"'site_type' must be 'octahedral' or 'tetrahedral', got '{site_type}'."
        )

    # Validate crystal_type
    crystal_type = crystal_type.lower()
    if crystal_type not in ("fcc", "bcc", "hcp"):
        raise ValueError(
            f"'crystal_type' must be 'fcc', 'bcc', or 'hcp', got '{crystal_type}'."
        )

    structure, container, pristine_index, parent_index = _prepare(
        input_structure, structure_container, parent_defect_index
    )

    atoms = _data_to_ase(structure)

    # Get analytical interstitial positions based on crystal structure
    if crystal_type == "fcc":
        fractional_positions = _get_interstitial_sites_fcc(site_type)
    elif crystal_type == "bcc":
        fractional_positions = _get_interstitial_sites_bcc(site_type)
    elif crystal_type == "hcp":
        fractional_positions = _get_interstitial_sites_hcp(site_type)
    else:
        raise ValueError(f"Unsupported crystal_type: {crystal_type}")

    # Validate index
    if not (0 <= index < len(fractional_positions)):
        raise IndexError(
            f"Interstitial index {index} is out of range for {crystal_type} "
            f"{site_type} sites. Available sites: {len(fractional_positions)}."
        )

    # Get the requested site
    frac_pos = fractional_positions[index]

    # Convert fractional to Cartesian coordinates
    cell = atoms.get_cell()
    cart_pos = np.array(frac_pos) @ cell.array

    # Apply minimum image convention to ensure position is in the unit cell
    from ase.geometry import find_mic

    cart_pos = find_mic(cart_pos.reshape(1, 3), cell, atoms.get_pbc())[0][0]

    # Create interstitial
    defect_atoms = copy.deepcopy(atoms)
    defect_atoms.append(_Atoms(element, positions=[cart_pos])[0])

    container.add_defect(
        structure=_ase_to_data(defect_atoms),
        operation=(
            f"interstitial[{element}@{crystal_type}_{site_type}_site#{index}"
            f"({cart_pos[0]:.3f},{cart_pos[1]:.3f},{cart_pos[2]:.3f})]"
        ),
        pristine_structure_index=pristine_index,
        parent_index=parent_index,
    )
    return container


# ---------------------------------------------------------------------------
# Utility nodes
# ---------------------------------------------------------------------------


@as_function_node
def GetStructureTable(
    structure_container: StructureContainer,
) -> pd.DataFrame:
    """
    Return a copy of the full structure table.

    The ``structure`` column contains :class:`OutputAtoms` instances
    (fully picklable).  Use :func:`_data_to_ase` to recover ASE
    :class:`~ase.atoms.Atoms` objects when needed.
    """
    if not isinstance(structure_container, StructureContainer):
        raise TypeError(
            f"Expected a StructureContainer, " f"got {type(structure_container)!r}."
        )
    table = structure_container.table.copy()
    return table


@as_function_node
def GetDefectTable(
    structure_container: StructureContainer,
) -> pd.DataFrame:
    """
    Return a filtered copy containing only defect rows
    (``is_pristine == False``).
    """
    if not isinstance(structure_container, StructureContainer):
        raise TypeError(
            f"Expected a StructureContainer, " f"got {type(structure_container)!r}."
        )
    mask = structure_container.table["is_pristine"].eq(False)
    table = structure_container.table[mask].copy()
    return table


@as_function_node
def GetPristineTable(
    structure_container: StructureContainer,
) -> pd.DataFrame:
    """
    Return a filtered copy containing only pristine rows
    (``is_pristine == True``).
    """
    if not isinstance(structure_container, StructureContainer):
        raise TypeError(
            f"Expected a StructureContainer, " f"got {type(structure_container)!r}."
        )
    mask = structure_container.table["is_pristine"].eq(True)
    table = structure_container.table[mask].copy()
    return table
