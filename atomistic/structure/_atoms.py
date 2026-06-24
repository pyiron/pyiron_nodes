"""
pyiron_nodes/atomistic/structure/_atoms.py
──────────────────────────────────────────
Lightweight conversion helpers between ASE :class:`~ase.atoms.Atoms`
objects and the picklable :class:`OutputAtoms` dataclass.

Motivation
----------
Raw ASE :class:`~ase.atoms.Atoms` objects are not picklable when they
carry an attached calculator, and they cannot be stored in pandas
DataFrames that need to round-trip through the hash-based storage layer.
:class:`OutputAtoms` is a plain Python dataclass (symbols, positions,
cell, pbc) that IS picklable and works transparently with
``pyiron_database``.

Public API
----------
OutputAtoms
    Picklable dataclass mirroring the four geometry arrays of an ASE
    :class:`~ase.atoms.Atoms` object.

to_ase(obj)
    Convert *obj* → :class:`~ase.atoms.Atoms`.
    Accepts :class:`OutputAtoms`, :class:`~ase.atoms.Atoms`, or any
    object that exposes an ``.atoms`` attribute.

to_data_atoms(obj)
    Convert *obj* → :class:`OutputAtoms`.
    Accepts :class:`~ase.atoms.Atoms`, :class:`OutputAtoms`, or any
    object that exposes an ``.atoms`` attribute.

Private helpers (used internally by pyiron_nodes)
-------------------------------------------------
_data_to_ase(atoms)   — :class:`OutputAtoms` → :class:`~ase.atoms.Atoms`
_ase_to_data(atoms)   — :class:`~ase.atoms.Atoms` → :class:`OutputAtoms`
_resolve_atoms(obj)   — any supported type → :class:`~ase.atoms.Atoms`
_resolve_structure(obj) — any supported type → :class:`OutputAtoms`

Usage
-----
::

    from pyiron_nodes.atomistic.structure._atoms import (
        OutputAtoms,
        to_ase,
        to_data_atoms,
    )

    # In a node function that must accept both types:
    def my_node(structure):
        atoms = to_ase(structure)          # always get ASE Atoms
        ...
        return to_data_atoms(atoms)      # always return picklable form
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from ase.atoms import Atoms
from core.data_fields import DataArray, EmptyArrayField

# ---------------------------------------------------------------------------
# OutputAtoms dataclass
# ---------------------------------------------------------------------------


@dataclass
class OutputAtoms:
    """
    Picklable representation of an ASE :class:`~ase.atoms.Atoms` object.

    Stores the four arrays that fully define a periodic atomic structure:
    chemical symbols, atomic positions, unit cell, and periodic boundary
    conditions.  All fields are plain Python lists or numpy arrays and
    are therefore fully picklable — unlike a raw
    :class:`~ase.atoms.Atoms` object that carries an attached calculator.

    Fields
    ------
    symbols : list[str]
        Chemical symbols, one per atom (e.g. ``["Al", "Al", "Mg"]``).
    positions : array-like, shape (N, 3)
        Cartesian atomic positions in Å.
    cell : list[list[float]], shape (3, 3)
        Unit cell vectors in Å.
    pbc : list[bool], length 3
        Periodic boundary conditions along each cell vector.

    Conversion
    ----------
    Use :func:`to_ase` to recover a live ASE
    :class:`~ase.atoms.Atoms` object, and :func:`to_data_atoms` to
    convert in the opposite direction.
    """

    symbols: DataArray = EmptyArrayField()
    positions: DataArray = EmptyArrayField()
    cell: DataArray = EmptyArrayField()
    pbc: DataArray = EmptyArrayField()


# ---------------------------------------------------------------------------
# Type alias for annotating function parameters
# ---------------------------------------------------------------------------

#: Type alias accepted by all public conversion functions.
#: Use this in function signatures to signal that both raw ASE
#: :class:`~ase.atoms.Atoms` and picklable :class:`OutputAtoms`
#: instances are valid inputs.
AtomsLike = Union[Atoms, OutputAtoms]


# ---------------------------------------------------------------------------
# Private low-level converters
# ---------------------------------------------------------------------------


def _data_to_ase(atoms: OutputAtoms) -> Atoms:
    """
    Convert an :class:`OutputAtoms` dataclass to an ASE
    :class:`~ase.atoms.Atoms` object.

    This is the inverse of :func:`_ase_to_data`.  No calculator is
    attached to the returned object.

    Parameters
    ----------
    atoms : OutputAtoms

    Returns
    -------
    Atoms
    """
    return Atoms(
        symbols=atoms.symbols,
        positions=atoms.positions,
        cell=atoms.cell,
        pbc=atoms.pbc,
    )


def _ase_to_data(atoms: Atoms) -> OutputAtoms:
    """
    Convert an ASE :class:`~ase.atoms.Atoms` object to an
    :class:`OutputAtoms` dataclass.

    The calculator (if any) is deliberately not copied — the returned
    :class:`OutputAtoms` contains only the geometry arrays and is
    therefore always picklable.

    This is the inverse of :func:`_data_to_ase`.

    Parameters
    ----------
    atoms : Atoms

    Returns
    -------
    OutputAtoms
    """
    if isinstance(atoms, OutputAtoms):
        return atoms
    return OutputAtoms(
        symbols=list(atoms.symbols),
        positions=atoms.positions,
        cell=atoms.cell.tolist(),
        pbc=atoms.pbc.tolist(),
    )


def _resolve_atoms(obj: AtomsLike) -> Atoms:
    """
    Resolve *obj* to a plain ASE :class:`~ase.atoms.Atoms` instance.

    Accepted input types
    --------------------
    * :class:`~ase.atoms.Atoms`   — returned as-is (no copy)
    * :class:`OutputAtoms`        — converted via :func:`_data_to_ase`
    * Any object with an ``.atoms`` attribute that is an
      :class:`~ase.atoms.Atoms` instance — the ``.atoms`` attribute is
      returned directly

    Parameters
    ----------
    obj : Atoms or OutputAtoms or object with .atoms

    Returns
    -------
    Atoms

    Raises
    ------
    TypeError
        If *obj* is not one of the accepted types.

    See Also
    --------
    to_ase : public alias with the same behaviour
    """
    if isinstance(obj, Atoms):
        return obj
    if isinstance(obj, OutputAtoms):
        return _data_to_ase(obj)
    if hasattr(obj, "atoms") and isinstance(obj.atoms, Atoms):
        return obj.atoms
    raise TypeError(
        f"Cannot extract an ASE Atoms object from {type(obj)!r}.\n"
        f"  Expected one of:\n"
        f"    - ase.atoms.Atoms\n"
        f"    - OutputAtoms (picklable dataclass)\n"
        f"    - an object with a .atoms attribute (ase.atoms.Atoms)"
    )


def _resolve_structure(obj: AtomsLike) -> OutputAtoms:
    """
    Resolve *obj* to an :class:`OutputAtoms` dataclass instance.

    Accepted input types
    --------------------
    * :class:`OutputAtoms`        — returned as-is (no copy)
    * :class:`~ase.atoms.Atoms`   — converted via :func:`_ase_to_data`
    * Any object with an ``.atoms`` attribute that is an
      :class:`~ase.atoms.Atoms` instance — converted via
      :func:`_ase_to_data`

    Parameters
    ----------
    obj : Atoms or OutputAtoms or object with .atoms

    Returns
    -------
    OutputAtoms

    Raises
    ------
    TypeError
        If *obj* is not one of the accepted types.

    See Also
    --------
    to_data_atoms : public alias with the same behaviour
    """
    if isinstance(obj, OutputAtoms):
        return obj
    if isinstance(obj, Atoms):
        return _ase_to_data(obj)
    if hasattr(obj, "atoms") and isinstance(obj.atoms, Atoms):
        return _ase_to_data(obj.atoms)
    raise TypeError(
        f"Cannot extract an OutputAtoms object from {type(obj)!r}.\n"
        f"  Expected one of:\n"
        f"    - OutputAtoms (picklable dataclass)\n"
        f"    - ase.atoms.Atoms\n"
        f"    - an object with a .atoms attribute (ase.atoms.Atoms)"
    )


# ---------------------------------------------------------------------------
# Public API — friendly names for the private resolvers
# ---------------------------------------------------------------------------


def to_ase(obj: AtomsLike) -> Atoms:
    """
    Convert *obj* to an ASE :class:`~ase.atoms.Atoms` object.

    This is the recommended public entry point for node functions that
    need to perform ASE calculations on a structure argument that may
    arrive as either a raw :class:`~ase.atoms.Atoms` or a picklable
    :class:`OutputAtoms`.

    Example
    -------
    ::

        from pyiron_nodes.atomistic.structure._atoms import to_ase

        @as_function_node
        def StaticEnergy(structure, engine: OutputEngine):
            atoms      = to_ase(structure)   # works for both types
            atoms.calc = engine.calculator
            energy     = atoms.get_potential_energy()
            return energy

    Parameters
    ----------
    obj : Atoms or OutputAtoms or object with .atoms

    Returns
    -------
    Atoms

    Raises
    ------
    TypeError
        If *obj* cannot be converted.
    """
    return _resolve_atoms(obj)


def to_data_atoms(obj: AtomsLike) -> OutputAtoms:
    """
    Convert *obj* to a picklable :class:`OutputAtoms` dataclass.

    This is the recommended public entry point for node functions that
    need to return or store a structure in a form that is compatible
    with the hash-based storage layer and pandas DataFrames.

    The calculator (if any) is NOT included — only the four geometry
    arrays (symbols, positions, cell, pbc) are preserved.

    Example
    -------
    ::

        from pyiron_nodes.atomistic.structure._atoms import (
            to_ase,
            to_data_atoms,
        )

        @as_function_node
        def Relax(structure, engine):
            atoms = to_ase(structure)
            atoms.calc = engine.calculator
            # ... relaxation ...
            atoms.calc = None               # detach before returning
            structure = to_data_atoms(atoms)
            return structure

    Parameters
    ----------
    obj : Atoms or OutputAtoms or object with .atoms

    Returns
    -------
    OutputAtoms

    Raises
    ------
    TypeError
        If *obj* cannot be converted.
    """
    return _resolve_structure(obj)
