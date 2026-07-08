from __future__ import annotations

from typing import Optional, Union, List, Iterable, Set

from ase import Atoms

from core import as_function_node
from pyiron_nodes.atomistic.structure._atoms import OutputAtoms, _data_to_ase


@as_function_node("structure")
def Repeat(structure: Atoms, repeat_scalar: int = 1) -> Atoms:
    """
    Repeat a crystal structure periodically along all lattice vectors.

    Parameters
    ----------
    structure : Atoms
        The ASE ``Atoms`` object to be repeated.
    repeat_scalar : int, optional
        Number of repetitions along each lattice vector (default is ``1`` – no change).

    Returns
    -------
    Atoms
        A new ``Atoms`` object containing the repeated supercell.

    Task hint
    ----------
    Use this node when the workflow requires building a larger supercell
    from a primitive cell (e.g., "create a 2×2×2 bulk supercell"). Create a supercell
    by repeating the input structure. Expand to a nxnxn supercell.
    """
    # Convert structure to ASE Atoms if necessary
    structure = _data_to_ase(structure)
    return structure.repeat(int(repeat_scalar))


@as_function_node("structure")
def RepeatXYZ(
    structure: Atoms,
    repeat_x: int = 1,
    repeat_y: int = 1,
    repeat_z: int = 1,
) -> Atoms:
    """
    Repeat a crystal structure with independent repetition counts along each
    lattice vector.

    Parameters
    ----------
    structure : Atoms
        The ASE ``Atoms`` object to be repeated.
    repeat_x : int, optional
        Repetitions along the first lattice vector (default ``1``).
    repeat_y : int, optional
        Repetitions along the second lattice vector (default ``1``).
    repeat_z : int, optional
        Repetitions along the third lattice vector (default ``1``).

    Returns
    -------
    Atoms
        A new ``Atoms`` object containing the repeated supercell.
    """
    # Convert structure to ASE Atoms if necessary
    structure = _data_to_ase(structure)
    return structure.repeat([int(repeat_x), int(repeat_y), int(repeat_z)])


@as_function_node("structure")
def ApplyStrain(structure: Optional[Atoms] = None, strain: Union[float] = 0) -> Atoms:
    """
    Apply a homogeneous strain to a structure.

    Parameters
    ----------
    structure : Atoms, optional
        The input structure. If ``None`` the node will raise an error.
    strain : float, optional
        Strain magnitude (default ``0`` – no deformation). Positive values
        expand the lattice, negative values compress it.

    Returns
    -------
    Atoms
        A copy of the input structure with the strain applied.

    Task hint
    ----------
    Suitable for tasks such as "apply 5 % tensile strain to a bulk cell"
    or "compress a slab by 2 % before relaxation".
    """
    # Convert structure to ASE Atoms if necessary
    structure = _data_to_ase(structure)
    struct = structure.copy()
    struct.apply_strain(strain)
    return struct


@as_function_node
def CreateVacancy(structure, index: Optional[int] = None) -> Atoms:
    """
    Remove a single atom from a structure, creating a vacancy.

    Parameters
    ----------
    structure : Atoms
        The input structure from which the atom will be removed.
    index : int, optional
        Index of the atom to delete. If ``None`` the node does nothing
        (useful as a placeholder).

    Returns
    -------
    Atoms
        A copy of the original structure with the specified atom removed.

    Task hint
    ----------
    Use when the scientific goal is "introduce a vacancy at site 5"
    or "generate a defect structure for defect formation energy calculations".
    """
    # Convert OutputAtomsstructure to ASE Atoms if necessary
    structure = _data_to_ase(structure)
    structure = structure.copy()
    if index is not None:
        del structure[int(index)]

    return structure


@as_function_node("structure")
def RotateAxisAngle(
    structure: Atoms,
    angle: float | int = 0,
    axis: list | str | tuple = (0, 0, 1),
    center=(0, 0, 0),
    rotate_cell: bool = False,
) -> Atoms:
    """
    Rotate a structure around a given axis by a specified angle.

    Parameters
    ----------
    structure : Atoms
        The structure to rotate.
    angle : float or int, optional
        Rotation angle in degrees (default ``0`` – no rotation).
    axis : list of three floats, optional
        Rotation axis vector (default ``[0, 0, 1]`` – the z‑axis).
    center : tuple of three floats, optional
        Point about which the rotation is performed (default origin).
    rotate_cell : bool, optional
        If ``True`` also rotate the simulation cell (default ``False``).

    Returns
    -------
    Atoms
        A new ``Atoms`` object with the rotated coordinates (and optionally cell).

    Task hint
    ----------
    Ideal for "orient a surface normal to the x‑axis",
    "apply a 45° tilt to a grain boundary", or any situation where a
    specific crystallographic orientation is required.
    """
    # Convert structure to ASE Atoms if necessary
    structure = _data_to_ase(structure)

    structure_rotated = structure.copy()
    structure_rotated.rotate(a=angle, v=axis, center=center, rotate_cell=rotate_cell)
    return structure_rotated


@as_function_node
def FixAtoms(
    structure: Atoms,
    fix_xyz: str = "1 1 1",
    fixed_species: Optional[str] = None,
    fix_z_coordinate: Optional[float] = None,
    fix_z_tolerance: Optional[float] = 0.5,
    fix_atom_indices: Optional[str] = None,
) -> Atoms:
    """
    Return a copy of *structure* with constraints applied to the requested
    atoms (by species, z-coordinate, or atom indices) and degrees of freedom.

    Parameters
    ----------
    structure : ase.Atoms
        Atomic configuration to be copied and (optionally) constrained.

    fixed_species : None or str, optional
        * ``None``        – no atoms are fixed by species.
        * ``"Cu"``        – all copper atoms are fixed.
        * ``'["O", "H"]'``– all oxygen and hydrogen atoms are fixed.

    fix_xyz : str, optional
        Which Cartesian degrees of freedom to fix, given as three
        space-separated 1/0 flags for x, y, z respectively.
        Default is ``"1 1 1"`` (fix all directions).

        Examples
        --------
        * ``"1 1 1"`` – fix x, y, and z  (default)
        * ``"1 1 0"`` – fix x and y,  free z
        * ``"0 0 1"`` – fix z only,   free x and y

    fix_z_coordinate : float or None, optional
        Fix all atoms within ``fix_z_tolerance`` Å of this z-coordinate.
        * ``None`` – no atoms are fixed by z-coordinate (default).
        * ``5.0``  – fix all atoms with z in [4.5, 5.5] Å.

    fix_z_tolerance : float, optional
        Tolerance in Å around ``fix_z_coordinate``. Default is 0.5 Å.
        Only used when ``fix_z_coordinate`` is not None.

    fix_atom_indices : None or list[int], optional
        Fix specific atoms by index.
        * ``None``        – no atoms fixed by index (default).
        * ``"0"``         – fix atom 0.
        * ``'[0, 1, 2]'`` – fix atoms 0, 1, and 2.

    Returns
    -------
    ase.Atoms
        A copy of *structure* with the appropriate constraints attached.
    """
    import ast
    from ase.constraints import FixAtoms, FixCartesian

    # ------------------------------------------------------------------
    #     Parse ``fix_xyz`` → boolean mask [fix_x, fix_y, fix_z]
    #     Accepts "1 1 1", "1 0 0", "0 0 1", etc.
    # ------------------------------------------------------------------
    try:
        flags = [int(v) for v in fix_xyz.strip().split()]
    except ValueError:
        raise ValueError(
            f"fix_xyz must be three space-separated 1/0 values, e.g. '1 1 0'. Got: '{fix_xyz}'"
        )

    if len(flags) != 3:
        raise ValueError(
            f"fix_xyz must contain exactly three values (x y z). Got {len(flags)}: '{fix_xyz}'"
        )

    if not all(f in (0, 1) for f in flags):
        raise ValueError(f"fix_xyz values must be 0 or 1. Got: '{fix_xyz}'")

    dof_mask = [bool(f) for f in flags]  # [fix_x, fix_y, fix_z]

    # Use FixAtoms (simpler) when all three directions are fixed
    use_fix_atoms = all(dof_mask)

    # No directions fixed at all → nothing to constrain
    if not any(dof_mask):
        return structure.copy()

    # ------------------------------------------------------------------
    #   Normalise ``fixed_species`` → set of element symbols
    # ------------------------------------------------------------------
    species_set: Set[str] = set()

    if fixed_species is not None:
        try:
            parsed_species = ast.literal_eval(fixed_species)
        except (SyntaxError, ValueError):
            parsed_species = fixed_species

        if isinstance(parsed_species, str):
            species_set = set(item.strip() for item in parsed_species.split(","))
        elif isinstance(parsed_species, (list, tuple, set)):
            if not all(isinstance(item, str) for item in parsed_species):
                raise ValueError("All entries in the element list must be strings.")
            species_set = set(parsed_species)
        else:
            raise ValueError(
                "fixed_species must be None, a single element symbol, "
                "or a string representation of a list/tuple of symbols."
            )

    # ------------------------------------------------------------------
    #   Parse ``fix_atom_indices`` → set of integer indices
    # ------------------------------------------------------------------
    index_set: Set[int] = set()

    if fix_atom_indices is not None:
        try:
            parsed_indices = ast.literal_eval(fix_atom_indices)
        except (SyntaxError, ValueError):
            parsed_indices = fix_atom_indices

        if isinstance(parsed_indices, int):
            index_set = {parsed_indices}
        elif isinstance(parsed_indices, (list, tuple, set)):
            if not all(isinstance(item, int) for item in parsed_indices):
                raise ValueError("All entries in the index list must be integers.")
            index_set = set(parsed_indices)

    # ------------------------------------------------------------------
    #     Build the boolean mask combining all three selection methods.
    #     An atom is fixed if it satisfies ANY of the criteria.
    # ------------------------------------------------------------------
    mask: List[bool] = []

    for i, atom in enumerate(structure):
        fix_by_species = atom.symbol in species_set
        fix_by_index = i in index_set
        fix_by_z = (
            fix_z_coordinate is not None
            and fix_z_coordinate != ""
            and abs(atom.position[2] - fix_z_coordinate) <= fix_z_tolerance
        )
        mask.append(fix_by_species or fix_by_index or fix_by_z)

    # ------------------------------------------------------------------
    #   Create a copy and attach the constraint (if any atom is fixed).
    # ------------------------------------------------------------------
    new_structure = structure.copy()

    if any(mask):
        fixed_indices = [i for i, fixed in enumerate(mask) if fixed]

        if use_fix_atoms:
            # All three directions fixed → FixAtoms (simpler, more efficient)
            constraint = FixAtoms(mask=mask)
        else:
            # Partial directions fixed → FixCartesian
            constraint = FixCartesian(
                a=fixed_indices,
                mask=dof_mask,  # [fix_x, fix_y, fix_z]
            )

        new_structure.set_constraint(constraint)

    return new_structure


@as_function_node
def FixSpecies(
    structure: Atoms,
    fixed_species: Optional[str] = None,
) -> Atoms:
    """
    Return a copy of *structure* with a ``FixAtoms`` constraint applied to the
    requested chemical species.

    Parameters
    ----------
    structure : ase.Atoms
        Atomic configuration to be copied and (optionally) constrained.
    fixed_species : None or str, optional
        * ``None`` – no atoms are fixed.
        * ``"Cu"`` – all copper atoms are fixed.
        * ``'["O", "H"]'`` – a string that represents a list/tuple of symbols;
          all oxygen **and** hydrogen atoms are fixed.

    Returns
    -------
    ase.Atoms
        A copy of *structure* with the appropriate ``FixAtoms`` constraint
        attached (or the unchanged copy if *fixed_species* is ``None``).
    """
    import ast
    from ase.constraints import FixAtoms

    # ------------------------------------------------------------------
    # 1️⃣ Normalise ``fixed_species`` to a *set* of element symbols.
    # ------------------------------------------------------------------
    species_set: Set[str] = set()  # default → nothing to fix

    if fixed_species is not None:
        # ``fixed_species`` is a string.  It may be a plain symbol
        # (e.g. "Cu") or a string that looks like a Python container
        # (e.g. '["O","H"]' or '("C","N")').
        try:
            parsed = ast.literal_eval(fixed_species)
        except (SyntaxError, ValueError):
            # Not a container literal → treat the whole string as a single symbol.
            parsed = fixed_species

        if isinstance(parsed, str):
            species_set = {parsed}
        elif isinstance(parsed, (list, tuple, set)):
            # Ensure every entry is a string; otherwise raise a clear error.
            if not all(isinstance(item, str) for item in parsed):
                raise ValueError("All entries in the element list must be strings.")
            species_set = set(parsed)
        else:
            raise ValueError(
                "fixed_species must be None, a single element symbol, "
                "or a string representation of a list/tuple of symbols."
            )

    # ------------------------------------------------------------------
    # 2️⃣ Build the boolean mask required by ``FixAtoms``.
    #    If ``species_set`` is empty the mask will be all ``False``.
    # ------------------------------------------------------------------
    mask: List[bool] = [atom.symbol in species_set for atom in structure]

    # ------------------------------------------------------------------
    # 3️⃣ Create a copy of the original structure and (optionally) attach the
    #    constraint.  ``FixAtoms`` tolerates a mask that is all ``False``,
    #    but we skip adding the constraint for a cleaner object.
    # ------------------------------------------------------------------
    new_structure = structure.copy()
    if any(mask):  # at least one atom should be fixed
        new_structure.set_constraint(FixAtoms(mask=mask))

    # ------------------------------------------------------------------
    # 4️⃣ Single return statement – the function's result.
    # ------------------------------------------------------------------
    return new_structure


@as_function_node("structure")
def LayerShift(
    structure: Atoms,
    shift_fraction: float = 0.5,
    shift_x: float = 0.0,
    shift_y: float = 0.0,
) -> Atoms:
    """
    Shift upper layers of a structure by an in-plane translation vector.

    This is used for creating stacking faults and computing gamma surfaces.
    The shift is applied to the top fraction of layers in fractional coordinates.

    Parameters
    ----------
    structure : Atoms
        The input structure (typically a slab).
    shift_fraction : float, optional
        Fraction of top layers to shift (default 0.5 for half the layers).
        For example, 0.5 shifts the upper half, 0.33 shifts the upper third.
    shift_x : float, optional
        First component of shift vector in fractional coordinates (default 0.0).
        Applied to the first cell vector (cell[0]).
    shift_y : float, optional
        Second component of shift vector in fractional coordinates (default 0.0).
        Applied to the second cell vector (cell[1]).

    Returns
    -------
    Atoms
        A copy of the structure with the upper layers shifted in the xy-plane.

    Scientific Purpose
    ------------------
    * Create stacking faults for gamma surface energy calculations
    * Generate different stacking configurations for defect studies
    * Probe energy landscapes for planar defects in crystals

    Typical Use Cases
    -----------------
    * Compute gamma surface for fcc {111} plane: shift_x=1/3, shift_y=1/3
    * Create intrinsic stacking fault: shift by Burgers vector direction
    * Sample energy landscape: vary shift_x, shift_y across [0,1]×[0,1]

    Example
    -------
    >>> # Shift top half by [1/3, 1/3] in fractional coords (intrinsic stacking fault)
    >>> shifted = LayerShift(slab, shift_fraction=0.5, shift_x=0.333, shift_y=0.333)
    >>>
    >>> # Shift top layer by half of first cell vector
    >>> shifted = LayerShift(slab, shift_fraction=0.5, shift_x=0.5, shift_y=0.0)
    """
    import numpy as np

    # Step 1: Copy the structure (never modify input in-place)
    new_structure = structure.copy()

    # Step 2: Sort atoms by z-coordinate
    z_positions = new_structure.positions[:, 2]
    sorted_indices = np.argsort(z_positions)

    # Step 3: Calculate cutoff position
    z_min = np.min(z_positions)
    z_max = np.max(z_positions)
    z_cutoff = z_min + shift_fraction * (z_max - z_min)

    # Step 4: Identify atoms in the upper layers (z >= cutoff)
    upper_mask = z_positions >= z_cutoff
    upper_indices = np.where(upper_mask)[0]

    # Step 5: Apply shift to upper atoms
    # Calculate the total shift vector in Cartesian coordinates
    cell = new_structure.cell
    shift_vector = shift_x * cell[0] + shift_y * cell[1]

    # Apply shift to upper layer atoms
    new_structure.positions[upper_indices] += shift_vector

    # Step 6: Return modified structure
    # ASE's PBC handling automatically wraps atoms that move beyond cell boundaries
    return new_structure


@as_function_node("structure")
def Stack(
    bottom: Atoms,
    top: Atoms,
    axis: int = 2,
    distance: Optional[float] = None,
    reorder: bool = False,
) -> Atoms:
    """
    Stack two structures on top of each other along a lattice axis.

    Parameters
    ----------
    bottom : Atoms
        The lower grain (placed first along the stacking axis).
    top : Atoms
        The upper grain (placed on top of ``bottom``).
    axis : int, optional
        Lattice axis along which to stack (0, 1, or 2; default ``2`` = z).
    distance : float or None, optional
        Gap between the two grains in Å. ``None`` uses ASE's default
        (interface distance inferred from the cell).
    reorder : bool, optional
        If ``True``, reorder atoms so that species are grouped together.

    Returns
    -------
    Atoms
        The combined bicrystal structure.
    """
    from ase.build import stack as ase_stack

    return ase_stack(bottom, top, axis=axis, distance=distance, reorder=reorder)
