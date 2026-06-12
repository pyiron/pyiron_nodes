from __future__ import annotations

from typing import Optional, Union, List, Iterable, Set

from ase import Atoms

from core import as_function_node

import numpy as np
from ase.neighborlist import NeighborList
import copy


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
def FixAtoms(
    structure: Atoms,
    xyz_constraint: str = "1 1 1",
    fix_species: Optional[str] = None,
    fix_at_z: Optional[float] = None,
    fix_at_z_tol: Optional[float] = 0.5,
    fix_atom_indices: Optional[str] = None,
) -> Atoms:
    """
    Return a copy of *structure* with constraints applied to the requested
    atoms (by species, z-coordinate, or atom indices) and degrees of freedom.

    Parameters
    ----------
    structure : ase.Atoms
        Atomic configuration to be copied and (optionally) constrained.

    xyz_constraint : str
        Which Cartesian degrees of freedom to fix, given as three
        space-separated 1/0 flags for x, y, z respectively.
        Default is ``"1 1 1"`` (fix all directions).

    fix_species : None or str, optional
        * ``None``        – no atoms are fixed by species.
        * ``"Cu"``        – all copper atoms are fixed.
        * ``'["O", "H"]'``– all oxygen and hydrogen atoms are fixed.

    fix_at_z : float or None, optional
        Fix all atoms within ``fix_at_z_tol`` Å of this z-coordinate.
        * ``None`` – no atoms are fixed by z-coordinate (default).
        * ``5.0``  – fix all atoms with z in [4.5, 5.5] Å.

    fix_at_z_tol : float, optional
        Tolerance in Å around ``fix_at_z``. Default is 0.5 Å.
        Only used when ``fix_at_z`` is not None.

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
    #     Parse ``xyz_constraint`` → boolean mask [fix_x, fix_y, fix_z]
    # ------------------------------------------------------------------
    try:
        flags = [int(v) for v in xyz_constraint.strip().split()]
    except ValueError:
        raise ValueError(
            f"xyz_constraint must be three space-separated 1/0 values, e.g. '1 1 0'. Got: '{xyz_constraint}'"
        )

    if len(flags) != 3:
        raise ValueError(
            f"xyz_constraint must contain exactly three values (x y z). Got {len(flags)}: '{xyz_constraint}'"
        )

    if not all(f in (0, 1) for f in flags):
        raise ValueError(
            f"xyz_constraint values must be 0 or 1. Got: '{xyz_constraint}'"
        )

    dof_mask = [bool(f) for f in flags]   # [fix_x, fix_y, fix_z]

    # Use FixAtoms (simpler) when all three directions are fixed
    use_fix_atoms = all(dof_mask)

    # No directions fixed at all → nothing to constrain
    if not any(dof_mask):
        return structure.copy()

    # ------------------------------------------------------------------
    #   Normalise ``fix_species`` → set of element symbols
    # ------------------------------------------------------------------
    species_set: Set[str] = set()

    if fix_species is not None:
        try:
            parsed_species = ast.literal_eval(fix_species)
        except (SyntaxError, ValueError):
            parsed_species = fix_species

        if isinstance(parsed_species, str):
            species_set = set(item.strip() for item in parsed_species.split(','))
        elif isinstance(parsed_species, (list, tuple, set)):
            if not all(isinstance(item, str) for item in parsed_species):
                raise ValueError("All entries in the element list must be strings.")
            species_set = set(parsed_species)
        else:
            raise ValueError(
                "fix_species must be None, a single element symbol, "
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
    # ------------------------------------------------------------------
    mask: List[bool] = []

    for i, atom in enumerate(structure):
        fix_by_species = atom.symbol in species_set
        fix_by_index   = i in index_set
        fix_by_z       = (
            fix_at_z is not None and fix_at_z != ''
            and abs(atom.position[2] - fix_at_z) <= fix_at_z_tol
        )
        mask.append(fix_by_species or fix_by_index or fix_by_z)

    # ------------------------------------------------------------------
    #   Attach the constraint (if any atom is fixed).
    # ------------------------------------------------------------------

    if any(mask):
        fixed_indices = [i for i, fixed in enumerate(mask) if fixed]

        if use_fix_atoms:
            constraint = FixAtoms(mask=mask)
        else:
            constraint = FixCartesian(
                a=fixed_indices,
                mask=dof_mask,   # [fix_x, fix_y, fix_z]
            )

        structure.set_constraint(constraint)

    return structure

@as_function_node
def GenerateHEAStructures(
    structure: Atoms,
    elements: str,
    shares: str,
    n_structures: int = 1,
    fixed_index: Optional[int] = None,
    r_cutoff: Optional[float] = None,
    seed: Optional[int] = None
) -> list[Atoms]:
    """
    Generate random High Entropy Alloy (HEA) structures from an input ASE structure.
    
    Parameters
    ----------
    structure : Atoms
        Input ASE Atoms object to use as template
    elements : str
        Space-separated string of element symbols (e.g., "Ru Pt Ni Cu Fe")
    shares : str
        Space-separated string of atomic fractions for each element (e.g., "0.2 0.2 0.2 0.2 0.2").
        Shares will be renormalized if they don't sum to 1.0.
        Fixed atoms are excluded from the count before applying shares.
    n_structures : int
        Number of random structures to generate (default: 1)
    fixed_index : int, optional
        Index of an atom whose element should remain unchanged.
        All atoms within r_cutoff of this atom are also fixed if r_cutoff is provided.
    r_cutoff : float, optional
        Cutoff radius in Angstroms. All atoms within this distance from
        fixed_index will keep their original element type.
        Only used if fixed_index is provided.
    seed : int, optional
        Random seed for reproducibility. If n_structures > 1 and seed is provided,
        each structure gets a different but deterministic seed (seed + i).
        
    Returns
    -------
    list[Atoms]
        List of ASE Atoms objects with randomized element assignments
        
    Raises
    ------
    ValueError
        If elements and shares have different lengths, shares are invalid,
        or fixed_index is out of range.
        
    """
    
    from ase.io.trajectory import Trajectory

    # -------------------------------------------------------------------------
    # Parse and validate inputs
    # -------------------------------------------------------------------------
    element_list = elements.strip().split()
    share_list = [float(s) for s in shares.strip().split()]
    
    if len(element_list) != len(share_list):
        raise ValueError(
            f"Number of elements ({len(element_list)}) must match "
            f"number of shares ({len(share_list)})"
        )
    
    if any(s < 0 for s in share_list):
        raise ValueError("All shares must be non-negative.")
    
    total_share = sum(share_list)
    if total_share == 0:
        raise ValueError("Shares must not all be zero.")
    
    # Renormalize shares just in case they don't sum to exactly 1
    share_array = np.array(share_list) / total_share
    
    n_atoms = len(structure)
    
    if fixed_index is not None:
        if not (0 <= fixed_index < n_atoms):
            raise ValueError(
                f"fixed_index {fixed_index} is out of range for structure "
                f"with {n_atoms} atoms."
            )
    
    # -------------------------------------------------------------------------
    # Determine which atom indices are "free" (can be reassigned)
    # -------------------------------------------------------------------------
    fixed_indices = set()
    
    if fixed_index is not None:
        fixed_indices.add(fixed_index)
        
        if r_cutoff is not None:
            # Use ASE NeighborList to find all atoms within r_cutoff
            # We set cutoffs for each atom: only fixed_index needs a real cutoff
            cutoffs = [0.0] * n_atoms
            cutoffs[fixed_index] = r_cutoff
            
            nl = NeighborList(
                cutoffs,
                skin=0.0,
                self_interaction=False,
                bothways=True
            )
            nl.update(structure)
            
            neighbors, _ = nl.get_neighbors(fixed_index)
            for neighbor_idx in neighbors:
                fixed_indices.add(int(neighbor_idx))
    
    free_indices = [i for i in range(n_atoms) if i not in fixed_indices]
    n_free = len(free_indices)
    
    if n_free == 0:
        raise ValueError(
            "No free atoms to randomize. All atoms are fixed by "
            "fixed_index and/or r_cutoff constraints."
        )
    
    # -------------------------------------------------------------------------
    # Calculate how many atoms of each element to assign among free sites
    # -------------------------------------------------------------------------
    # We use a floor + remainder approach to respect shares as closely as possible
    def compute_counts(n_sites: int, fractions: np.ndarray) -> np.ndarray:
        """
        Distribute n_sites atoms among elements according to fractions.
        Uses floor allocation + assigns remainders to the largest fractional parts.
        """
        exact_counts = fractions * n_sites
        floor_counts = np.floor(exact_counts).astype(int)
        remainder = n_sites - floor_counts.sum()
        
        # Distribute remaining slots by largest fractional parts
        fractional_parts = exact_counts - floor_counts
        indices_sorted = np.argsort(-fractional_parts)  # descending
        for i in range(remainder):
            floor_counts[indices_sorted[i]] += 1
            
        return floor_counts
    
    element_counts = compute_counts(n_free, share_array)
    
    # Verbose summary
    print(f"Structure has {n_atoms} total atoms, {len(fixed_indices)} fixed, {n_free} free.")
    print(f"Element distribution among free sites:")
    for el, cnt, frac in zip(element_list, element_counts, share_array):
        print(f"  {el}: {cnt} atoms ({cnt/n_free*100:.1f}%,  target: {frac*100:.1f}%)")
    
    # -------------------------------------------------------------------------
    # Generate n_structures random structures
    # -------------------------------------------------------------------------
    results = []
    
    for i in range(n_structures):
        
        # Handle seeding
        if seed is not None:
            # Each structure gets its own deterministic seed
            rng = np.random.default_rng(seed + i)
        else:
            rng = np.random.default_rng()
        
        # Create a shuffled assignment of elements to free sites
        element_assignment = []
        for el, cnt in zip(element_list, element_counts):
            element_assignment.extend([el] * cnt)
        
        element_assignment = np.array(element_assignment)
        rng.shuffle(element_assignment)
        
        # Build new structure as a copy
        new_structure = copy.deepcopy(structure)
        symbols = list(new_structure.get_chemical_symbols())
        
        # Assign shuffled elements to free indices
        for free_pos, atom_idx in enumerate(free_indices):
            symbols[atom_idx] = element_assignment[free_pos]
        
        new_structure.set_chemical_symbols(symbols)
        results.append(new_structure)

    return results




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