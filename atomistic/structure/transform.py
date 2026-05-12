from __future__ import annotations

from typing import Optional, Union, List, Iterable, Set

from ase import Atoms

from core import as_function_node


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
    return structure.repeat(int(repeat_scalar))


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

    structure_rotated = structure.copy()
    structure_rotated.rotate(a=angle, v=axis, center=center, rotate_cell=rotate_cell)
    return structure_rotated


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
    species_set: Set[str] = set()          # default → nothing to fix

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
    if any(mask):                                   # at least one atom should be fixed
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