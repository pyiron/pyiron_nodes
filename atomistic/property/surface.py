"""
pyiron_nodes/atomistic/property/surface.py
──────────────────────────────────────────
High-level nodes for binary-compound surface-phase-diagram workflows.

Typical pipeline
----------------
::

    wf.slab_configs = BinarySlabConfigurations(
        cation="Ga", anion="N", a=3.19, c_over_a=1.627,
        n_layers=4, repeat=2, vacuum=12.0,
    )
    wf.slab_df = RelaxSlabsDataFrame(
        engine=wf.grace_engine,
        structures=wf.slab_configs.outputs.structures,
        names=wf.slab_configs.outputs.names,
    )
    # → pass wf.slab_df to AddElementCountColumns, etc.
"""

from __future__ import annotations

from core import as_function_node


@as_function_node
def BinarySlabConfigurations(
    cation: str,
    anion: str,
    a: float = 3.19,
    c_over_a: float = 1.627,
    n_layers: int = 4,
    repeat: int = 2,
    vacuum: float = 12.0,
    fix_bottom_fraction: float = 0.5,
    adatom_height: float = 2.0,
):
    """
    Build five (0001) surface configurations for a binary wurtzite compound.

    Starting from the pristine slab, four defect variants are constructed:

    ============  ============================================================
    Name          Description
    ============  ============================================================
    pristine      Clean (0001) slab — the reference configuration.
    {cation}_adatom  +1 cation adatom placed at a T4-like hollow site above
                  the top surface layer.
    {cation}_bilayer +2 cations (bilayer reconstruction seed): two adatoms at
                  offset positions to start a Ga/Al bilayer.
    {cation}_vacancy −1 cation: topmost cation atom removed from the surface.
    {anion}_adatom   +1 anion adatom placed above the top surface.
    ============  ============================================================

    Configuration names are constructed from the ``cation`` and ``anion``
    parameters — no element string is hardcoded in the node body.

    **Required inputs**
    - ``cation``, ``anion``:    Element symbols (e.g. ``"Ga"``, ``"N"``).
    - ``a``, ``c_over_a``:      Lattice constants (Å, dimensionless).
    - ``n_layers``:              Number of wurtzite bilayers (default 4).
    - ``repeat``:                In-plane supercell repeat (default 2 → 2×2).
    - ``vacuum``:                Vacuum thickness above slab (Å; default 12).
    - ``fix_bottom_fraction``:   Fraction of slab height to freeze (default 0.5).
    - ``adatom_height``:         Height of first adatom above top layer (Å; default 2).

    **Typical use-cases**
    * Generate all input structures for a GaN or AlN surface phase diagram in
      one node, then pass to :func:`RelaxSlabsDataFrame`.
    * Extend the configuration list by calling :func:`AddSurfaceAdatom` or
      :func:`RemoveTopSurfaceAtom` on individual structures in the lists.

    Returns
    -------
    structures : list of ase.atoms.Atoms
        Five slab configurations.
    names : list of str
        Corresponding configuration labels.
    """
    import numpy as np
    from ase import Atom
    from ase.build import bulk as ase_bulk
    from ase.build import surface as ase_surface
    from ase.constraints import FixAtoms

    # ── build pristine slab ────────────────────────────────────────────────
    bulk = ase_bulk(cation + anion, crystalstructure="wurtzite", a=a, c=a * c_over_a)
    pristine = ase_surface(bulk, (0, 0, 1), n_layers, vacuum=vacuum)
    pristine.center(vacuum=vacuum, axis=2)
    if repeat > 1:
        pristine = pristine.repeat([repeat, repeat, 1])
    if fix_bottom_fraction > 0:
        z_min = pristine.positions[:, 2].min()
        z_max_p = pristine.positions[:, 2].max()
        z_cut = z_min + fix_bottom_fraction * (z_max_p - z_min)
        pristine.set_constraint(FixAtoms(mask=pristine.positions[:, 2] <= z_cut))

    structures = [pristine.copy()]
    names = ["pristine"]

    pos = np.array(pristine.get_positions())
    sym = np.array(pristine.get_chemical_symbols())
    cell = pristine.get_cell()
    z_max = pos[:, 2].max()

    # centroid of topmost-layer atoms (any species)
    top_mask = pos[:, 2] > z_max - 0.5
    x0 = float(pos[top_mask, 0].mean()) if top_mask.any() else float(cell[0, 0] / 3)
    y0 = float(pos[top_mask, 1].mean()) if top_mask.any() else float(cell[1, 1] / 3)

    # +1 cation adatom at T4-like hollow
    slab_p1 = pristine.copy()
    slab_p1.append(
        Atom(cation, position=(x0 + cell[0, 0] * 0.33, y0 + cell[1, 1] * 0.33, z_max + adatom_height))
    )
    structures.append(slab_p1)
    names.append(f"{cation}_adatom")

    # +2 cation bilayer seed
    slab_p2 = pristine.copy()
    slab_p2.append(Atom(cation, position=(x0, y0, z_max + adatom_height)))
    slab_p2.append(Atom(cation, position=(x0 + cell[0, 0] * 0.5, y0 + cell[1, 1] * 0.5, z_max + adatom_height + 0.6)))
    structures.append(slab_p2)
    names.append(f"{cation}_bilayer")

    # -1 cation: remove topmost cation
    cation_mask = sym == cation
    if cation_mask.any():
        z_top_cation = pos[cation_mask, 2].max()
        top_cation_idx = np.where(cation_mask & (pos[:, 2] > z_top_cation - 0.5))[0]
        if len(top_cation_idx) > 0:
            slab_m1 = pristine.copy()
            del slab_m1[int(top_cation_idx[0])]
            structures.append(slab_m1)
            names.append(f"{cation}_vacancy")

    # +1 anion adatom
    slab_n1 = pristine.copy()
    slab_n1.append(Atom(anion, position=(x0, y0, z_max + adatom_height)))
    structures.append(slab_n1)
    names.append(f"{anion}_adatom")

    return structures, names


@as_function_node("df")
def RelaxSlabsDataFrame(
    engine,
    structures: list,
    names: list,
    fmax: float = 0.05,
    max_steps: int = 300,
):
    """
    BFGS-relax a list of slab structures and return the results as a DataFrame.

    Each structure is relaxed independently with the provided engine's
    calculator.  BFGS output is suppressed.  The returned DataFrame has
    columns ``structure``, ``energy``, and ``name``, suitable for direct
    input to :func:`~pyiron_nodes.atomistic.thermodynamics.defect_phases.AddElementCountColumns`.

    This node intentionally bypasses per-structure aiflow caching because
    slab configurations have heterogeneous numbers of atoms and cannot be
    iterated with ``IterToDataFrame``.

    **Required inputs**
    - ``engine``:     GRACE or LAMMPS engine node (must expose ``.calculator``).
    - ``structures``: List of :class:`ase.atoms.Atoms` objects to relax.
    - ``names``:      Corresponding list of configuration labels.
    - ``fmax``:       Force convergence threshold (eV/Å; default 0.05).
    - ``max_steps``:  Maximum BFGS steps (default 300).

    **Typical use-cases**
    * Relax all slab configurations returned by :func:`BinarySlabConfigurations`.
    * Collect energies for the defect-formation-energy analysis.

    Returns
    -------
    pd.DataFrame
        Columns: ``structure`` (ase.Atoms), ``energy`` (float, eV), ``name`` (str).
    """
    import io

    import pandas as pd
    from ase.optimize import BFGS

    calc = engine.calculator
    rows = []
    for structure, name in zip(structures, names):
        atoms = structure.copy()
        atoms.calc = calc
        BFGS(atoms, logfile=io.StringIO()).run(fmax=fmax, steps=max_steps)
        energy = float(atoms.get_potential_energy())
        final = atoms.copy()
        final.calc = None
        rows.append({"structure": final, "energy": energy, "name": name})
    return pd.DataFrame(rows)
