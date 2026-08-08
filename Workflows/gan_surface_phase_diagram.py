from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Literal, Optional
from core import Workflow, as_function_node, group_node

from pyiron_nodes.atomistic.thermodynamics.defect_phases import (
    SelectStableStructures,
    AddElementCountColumns,
    AddDefectConcentrationColumns,
    ComputeDefectFormationEnergy,
)
from pyiron_nodes.dpg2026.atomistic.calculator.optimize import GenericOptimizerSettings, Relax
from pyiron_nodes.dpg2026.atomistic.engine.grace import Grace


@as_function_node("structure")
def BulkGa(a: float = 4.05, crystalstructure: str = 'fcc'):
    """Build Ga bulk reference structure (FCC approximation for chemical potential)."""
    from ase.build import bulk
    return bulk('Ga', crystalstructure, a=a)


@as_function_node("structure")
def BulkGaN(a: float = 3.189, c: float = 5.185):
    """Build GaN wurtzite bulk (4-atom primitive cell)."""
    from ase.build import bulk
    return bulk('GaN', 'wurtzite', a=a, c=c)


@as_function_node("structure")
def BuildGaNSurface(
    a: float = 3.189,
    c: float = 5.185,
    n_layers: int = 4,
    repeat: int = 2,
    vacuum: float = 10.0,
    orthogonal: bool = True,
    ga_polar: bool = False,
):
    """Build GaN slab. orthogonal=True → rectangular cell; ga_polar=True → (0001) Ga at top."""
    from ase.build import bulk, surface
    import numpy as np
    bulk_cell = bulk('GaN', 'wurtzite', a=a, c=c, orthorhombic=orthogonal)
    slab = surface(bulk_cell, (0, 0, 1), n_layers, vacuum=vacuum, periodic=True)
    if repeat > 1:
        slab = slab.repeat((repeat, repeat, 1))
    if ga_polar:
        # ASE default is N-polar (N at top). Mirror z to expose Ga-polar face.
        pos = slab.get_positions()
        cell_z = slab.get_cell()[2, 2]
        pos[:, 2] = cell_z - pos[:, 2]   # mirror
        pos[:, 2] -= pos[:, 2].min()     # shift: vacuum back to top, min z → 0
        slab.set_positions(pos)
    return slab


@as_function_node("df")
def BuildSurfaceDefects(
    surface_structure,
    n_seeds: int = 1,
    adatom_height: float = 2.0,
    top_layer_depth: float = 3.0,
) -> pd.DataFrame:
    """Build pristine + defect configurations: V_Ga, V_N, Ga adatom, N adatom."""
    import numpy as np
    import pandas as pd
    from pyiron_nodes.atomistic.structure.build_point_defects import (
        make_pristine_reference, make_config_row, op_vacancy, op_interstitial,
    )
    from pyiron_nodes.atomistic.structure._atoms import to_ase, _ase_to_data

    ase_slab = to_ase(surface_structure)
    atoms0, pristine_pos = make_pristine_reference._original_func(ase_slab)

    base_row = make_config_row(
        atoms=atoms0, structure_id="surface_pristine", events=[], seed=0,
        pristine_n_sites=len(atoms0), pristine_positions=pristine_pos,
    )

    positions = atoms0.get_positions()
    symbols = np.array(atoms0.get_chemical_symbols())
    uids = atoms0.arrays['uid'].astype(int)
    max_z = positions[:, 2].max()

    top_mask = positions[:, 2] > (max_z - top_layer_depth)
    interior_uids = uids[~top_mask].tolist()

    top_ga_idx = np.where(top_mask & (symbols == 'Ga'))[0]
    top_n_idx  = np.where(top_mask & (symbols == 'N'))[0]

    # T1 adatom sites: directly above each top-layer atom
    ga_adatom_positions = positions[top_ga_idx] + np.array([[0.0, 0.0, adatom_height]])
    n_adatom_positions  = positions[top_n_idx]  + np.array([[0.0, 0.0, adatom_height]])

    rows = [base_row]

    # Surface vacancies (restricted to top bilayer)
    for seed in range(n_seeds):
        if len(top_ga_idx) > 0:
            rows.append(op_vacancy(base_row, vacancy_element='Ga', n=1,
                                   seed=seed, forbid_uids=interior_uids))
        if len(top_n_idx) > 0:
            rows.append(op_vacancy(base_row, vacancy_element='N', n=1,
                                   seed=seed, forbid_uids=interior_uids))

    # Surface adatoms (T1 sites above top bilayer)
    for seed in range(n_seeds):
        if len(ga_adatom_positions) > 0:
            pos = [ga_adatom_positions[seed % len(ga_adatom_positions)].tolist()]
            rows.append(op_interstitial(base_row, element='Ga', positions=pos, n=1, seed=seed))
        if len(n_adatom_positions) > 0:
            pos = [n_adatom_positions[seed % len(n_adatom_positions)].tolist()]
            rows.append(op_interstitial(base_row, element='N', positions=pos, n=1, seed=seed))

    combined = pd.DataFrame(rows)
    combined["structure"] = combined["atoms"].apply(_ase_to_data)

    def _name(row):
        events = row.get("events") or []
        if not events:
            return "pristine"
        ev = events[-1]
        s = row.get("seed", 0) or 0
        t = ev.get("type", "")
        if t == "vacancy":
            return f"V_{ev.get('removed_element', '?')} (seed {s})"
        if t == "interstitial":
            return f"{ev.get('element', '?')}_ad (seed {s})"
        return f"defect (seed {s})"

    combined["name"] = [_name(r) for _, r in combined.iterrows()]
    return combined[["structure", "name"]]


@as_function_node("df")
def RelaxStructuresDataFrame(
    df: pd.DataFrame,
    engine,
    opt_parameters=None,
    opt_mode: str = 'full',
    store: bool = True,
) -> pd.DataFrame:
    from pyiron_nodes.atomistic.structure._atoms import to_ase
    from pyiron_nodes.dpg2026.atomistic.calculator.optimize import Relax, GenericOptimizerSettings
    if opt_parameters is None:
        opt_parameters = GenericOptimizerSettings()
    relaxed, energies = [], []
    for s in df["structure"]:
        out = Relax._original_func(structure=to_ase(s), engine=engine,
                                   opt_parameters=opt_parameters, opt_mode=opt_mode)
        relaxed.append(out.structure)
        energies.append(out.energy)
    result = df.copy()
    result["structure"] = relaxed
    result["energy"] = energies
    return result


@as_function_node("mu_sweep")
def ChemicalPotentialSweep(mu_ref: float, delta_mu: float = -2.5, num_points: int = 200):
    # linspace(mu_ref, mu_ref + delta_mu): Ga-rich (right) → N-rich (left) after set_xlim(min,max)
    return np.linspace(mu_ref, mu_ref + delta_mu, int(num_points))


@as_function_node("fig")
def PlotDefectPhase(
    formation_energies: dict,
    mu_label: str = 'μ_Ga  (eV)',
    ef_label: str = 'Formation energy (eV)',
    title: str = 'GaN(0001) surface phase diagram',
    exclude_keys: list = None,
) -> object:
    import numpy as np
    import matplotlib.pyplot as plt

    mu_arr = np.atleast_1d(np.asarray(formation_energies["mu_values"], dtype=float))
    reserved = {"mu_values"}
    if exclude_keys:
        reserved.update(exclude_keys)
    keys = [k for k in formation_energies if k not in reserved]
    if not keys:
        raise ValueError("No plottable entries in formation_energies.")
    ef_matrix = np.array([np.atleast_1d(np.asarray(formation_energies[k], dtype=float))
                          for k in keys])
    hull_ef = np.min(ef_matrix, axis=0)
    hull_idx = np.argmin(ef_matrix, axis=0)
    cmap = plt.colormaps["tab10"]
    colors = [cmap(i % 10) for i in range(len(keys))]
    fig, ax = plt.subplots(figsize=(9, 5))
    y_min = hull_ef.min()
    y_range = hull_ef.max() - y_min
    y_bottom = y_min - 0.08 * max(y_range, 1e-6)
    for i, color in enumerate(colors):
        mask = hull_idx == i
        if mask.any():
            ax.fill_between(mu_arr, hull_ef, y_bottom,
                            where=mask, color=color, alpha=0.20, linewidth=0)
    for i, (key, color) in enumerate(zip(keys, colors)):
        ef = np.atleast_1d(np.asarray(formation_energies[key], dtype=float))
        if key == "pristine":
            ax.plot(mu_arr, ef, color=color, lw=1.2, ls='--', alpha=0.7, label=key)
        else:
            stable = hull_idx == i
            ax.plot(mu_arr, ef, color=color, lw=1.0, alpha=0.45, zorder=2)
            if stable.any():
                ax.plot(mu_arr[stable], ef[stable], color=color, lw=2.5,
                        alpha=1.0, zorder=3, label=key)
            else:
                ax.plot([], [], color=color, lw=2.5, label=key)
    transitions = np.where(np.diff(hull_idx))[0]
    for t in transitions:
        ax.axvline(0.5 * (mu_arr[t] + mu_arr[t + 1]),
                   color='crimson', lw=1.2, ls='--', alpha=0.85, zorder=4)
    ax.set_xlim(mu_arr.min(), mu_arr.max())   # N-rich (left) → Ga-rich (right)
    ax.set_ylim(bottom=y_bottom)
    ax.set_xlabel(mu_label, fontsize=13)
    ax.set_ylabel(ef_label, fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=9, framealpha=0.9, ncol=2)
    ax.grid(True, linestyle='--', alpha=0.3)
    fig.tight_layout()
    return fig


def _draw_bonds_2d(atoms, ax, rotation_str, bond_color='#505050'):
    """Draw bonds over a matplotlib axis (call AFTER plot_atoms; default zorder=2 is on top)."""
    import numpy as np
    import re
    from math import sin, cos, radians
    from ase.neighborlist import NeighborList, natural_cutoffs

    R = np.eye(3)
    for m in re.finditer(r'([+-]?\d*\.?\d+)\s*([xyz])', rotation_str):
        a = radians(float(m.group(1)))
        ca, sa = cos(a), sin(a)
        axis = m.group(2)
        if axis == 'x':
            Ri = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
        elif axis == 'y':
            Ri = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])
        else:
            Ri = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
        R = Ri @ R

    pos_rot = atoms.get_positions() @ R.T
    cell_rot = atoms.get_cell() @ R.T

    # bothways=True gives both directions per bond; dedup via canonical key
    cutoffs = natural_cutoffs(atoms, mult=1.1)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)

    drawn = set()
    for i in range(len(atoms)):
        neighbors, offsets = nl.get_neighbors(i)
        for j, off in zip(neighbors, offsets):
            S = tuple(int(x) for x in off)
            key = (i, j, S) if i < j else (j, i, tuple(-x for x in S))
            if key in drawn:
                continue
            drawn.add(key)
            pj = pos_rot[j] + np.array(off, float) @ cell_rot
            dx = pj[0] - pos_rot[i, 0]
            dy = pj[1] - pos_rot[i, 1]
            if dx * dx + dy * dy > 16.0:   # skip projected length > 4 Å
                continue
            # No explicit zorder → Line2D default (2) sits above patches (1)
            ax.plot([pos_rot[i, 0], pj[0]], [pos_rot[i, 1], pj[1]],
                    '-', color=bond_color, lw=1.5, alpha=0.7, solid_capstyle='round')


@as_function_node("fig")
def PlotStableDecoratedStructures(
    formation_energies: dict,
    relaxed_df: pd.DataFrame,
    rotation: str = '-90x',
    columns: int = 2,
    figure_size: float = 4.0,
    render_style: Literal["spacefill", "ball+stick"] = "spacefill",
) -> object:
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    from ase.visualize.plot import plot_atoms
    from pyiron_nodes.atomistic.structure._atoms import to_ase

    _GA_COLOR   = '#ff8c00'  # orange for bulk Ga
    _N_COLOR    = '#4fc3f7'  # sky blue for N
    _GA_AD_COLOR = '#d32f2f' # red for Ga adatom
    _N_AD_COLOR  = '#0288d1' # deep blue for N adatom

    reserved = {"mu_values"}
    keys = [k for k in formation_energies if k not in reserved]
    if not keys:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No plottable data', ha='center', va='center', transform=ax.transAxes)
        return fig
    ef_matrix = np.array([np.atleast_1d(np.asarray(formation_energies[k], dtype=float))
                          for k in keys])
    hull_idx = np.argmin(ef_matrix, axis=0)
    stable_names = [keys[i] for i in sorted(set(hull_idx.tolist()))]
    name_to_struct = {row['name']: row['structure'] for _, row in relaxed_df.iterrows()}
    stable = [(name, to_ase(name_to_struct[name]))
              for name in stable_names if name in name_to_struct]
    if not stable:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No stable structures found', ha='center', va='center',
                transform=ax.transAxes)
        return fig

    n = len(stable)
    cols = min(n, columns)
    rows_n = math.ceil(n / cols)
    fig, axes = plt.subplots(rows_n, cols,
                              figsize=(figure_size * cols, figure_size * rows_n), squeeze=False)

    for idx, (name, atoms) in enumerate(stable):
        row_i, col_i = divmod(idx, cols)
        ax = axes[row_i][col_i]
        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()

        is_ga_ad = 'Ga_ad' in name
        is_n_ad  = 'N_ad'  in name

        def _atom_color(i, sym):
            if sym == 'Ga':
                if is_ga_ad:
                    ga_zs = [positions[j, 2] for j, s in enumerate(symbols) if s == 'Ga']
                    if positions[i, 2] >= max(ga_zs) - 0.1:
                        return _GA_AD_COLOR
                return _GA_COLOR
            elif sym == 'N':
                if is_n_ad:
                    n_zs = [positions[j, 2] for j, s in enumerate(symbols) if s == 'N']
                    if positions[i, 2] >= max(n_zs) - 0.1:
                        return _N_AD_COLOR
                return _N_COLOR
            return '#c0c0c0'

        colors = [_atom_color(i, sym) for i, sym in enumerate(symbols)]
        radii = 0.35 if render_style == "ball+stick" else 0.89
        plot_atoms(atoms, ax=ax, rotation=rotation, colors=colors, show_unit_cell=1, radii=radii)
        if render_style == "ball+stick":
            _draw_bonds_2d(atoms, ax, rotation)   # drawn after atoms so lines (zorder=2) sit on top
        ax.set_title(name, fontsize=10)
        ax.axis('off')

    for idx in range(n, rows_n * cols):
        row_i, col_i = divmod(idx, cols)
        axes[row_i][col_i].axis('off')

    fig.suptitle('Stable GaN(0001) surface structures\n(Ga=orange, N=blue, adatoms highlighted)',
                 fontsize=12)
    fig.tight_layout()
    return fig


# ── From-scratch 3D structure renderer (independent of structuretoolkit) ────────
_PLANE_TO_NORMAL = {"xy": [0, 0, 1], "xz": [0, 1, 0], "yz": [1, 0, 0]}


def _element_color_hex(symbol: str) -> str:
    """Per-element hex colour: jmol default, with Ga/N overridden for this workflow."""
    from ase.data import atomic_numbers
    from ase.data.colors import jmol_colors
    overrides = {"Ga": "#ff8c00", "N": "#4fc3f7"}
    if symbol in overrides:
        return overrides[symbol]
    rgb = jmol_colors[atomic_numbers[symbol]]
    return "#%02x%02x%02x" % tuple(int(round(c * 255)) for c in rgb)


def _orientation_matrix(view_plane):
    """3×3 camera-orientation tensor from a view-plane normal (Gram–Schmidt)."""
    import numpy as np
    vp = np.array(view_plane, dtype=float).reshape(-1, 3)
    R = np.roll(np.eye(3), -1, axis=0)
    R[:len(vp)] = vp
    R /= np.linalg.norm(R, axis=-1)[:, np.newaxis]
    R[1] -= np.dot(R[0], R[1]) * R[0]              # orthogonalise horizontal
    R[2] = np.cross(R[0], R[1])                    # vertical = h × depth
    if np.isclose(np.linalg.det(R), 0):
        return np.eye(3)                           # default NGL view
    return np.roll(R / np.linalg.norm(R, axis=-1)[:, np.newaxis], 2, axis=0).T


def _flattened_orientation(view_plane, distance_from_camera):
    """4×4 orientation flattened to the 16-element list expected by view.control.orient."""
    import numpy as np
    if distance_from_camera <= 0:
        raise ValueError("distance_from_camera must be positive")
    m = np.eye(4)
    m[:3, :3] = _orientation_matrix(view_plane)
    return (distance_from_camera * m).ravel().tolist()


def _build_view3d(atoms, colors_hex, render_style="ball+stick", particle_size=1.0,
                  camera="orthographic", background="white",
                  view_plane=(0, 1, 0), distance_from_camera=1.0, show_cell=True):
    """Render an ASE Atoms object with nglview — no structuretoolkit dependency.

    Atoms are grouped by colour and drawn with ONE representation per distinct colour
    (so N atoms → a handful of representations, not N), which keeps the build O(colours)
    instead of O(atoms²).  Bonds are inferred by NGL from interatomic distances via a
    single ``licorice`` representation — no per-bond shapes.

    spacefill   : vdW spheres (radiusScale 0.55), coloured per group.
    ball+stick  : small vdW spheres (radiusScale 0.18) + grey ``licorice`` bond sticks.
    """
    import numpy as np
    from collections import OrderedDict
    import nglview as nv

    view = nv.show_ase(atoms)
    view.clear_representations()

    # Group atom indices by colour → one representation per colour (fast + GUI-replayable)
    groups = OrderedDict()
    for i, c in enumerate(colors_hex):
        groups.setdefault(c, []).append(i)

    ball_scale = 0.55 if render_style == "spacefill" else 0.18
    for color, idx in groups.items():
        view.add_representation("spacefill", selection=idx, color=color,
                                radiusType="vdw", radiusScale=ball_scale * particle_size)

    if render_style == "ball+stick":
        # NGL auto-bonds by distance; one grey stick representation covers all bonds
        view.add_representation("licorice", selection="all", color="#808080",
                                radius=0.12 * particle_size)

    if show_cell and atoms.cell is not None and np.any(np.abs(np.array(atoms.cell)) > 1e-2):
        view.add_unitcell()

    view.camera = camera
    view.background = background
    view.control.orient(_flattened_orientation(view_plane, distance_from_camera * 14))
    view.center()
    return view


@as_function_node("view")
def Plot3D(
    structure,
    render_style: Literal["spacefill", "ball+stick"] = "ball+stick",
    particle_size: float = 1.0,
    perspective: Literal["xy", "xz", "yz"] = "xz",
    camera: Literal["orthographic", "perspective"] = "orthographic",
    background: Literal["white", "black"] = "white",
    distance_from_camera: float = 1.0,
    show_cell: bool = True,
):
    """Self-contained 3D structure viewer (ASE Atoms or OutputAtoms) built on nglview.

    Colours atoms by element (Ga=orange, N=blue, others jmol).  Supports spacefill and
    ball+stick rendering, adjustable particle size, camera/orientation and background.
    """
    from pyiron_nodes.atomistic.structure._atoms import to_ase
    atoms = to_ase(structure)
    colors = [_element_color_hex(s) for s in atoms.get_chemical_symbols()]
    return _build_view3d(
        atoms, colors, render_style=render_style, particle_size=particle_size,
        camera=camera, background=background,
        view_plane=_PLANE_TO_NORMAL[perspective],
        distance_from_camera=distance_from_camera, show_cell=show_cell)


@as_function_node("view")
def View3DStructure(
    relaxed_df: pd.DataFrame,
    index: int = 0,
    coloring: Literal["element", "cna"] = "element",
    perspective: Literal["xy", "xz", "yz"] = "xz",
    render_style: Literal["spacefill", "ball+stick"] = "spacefill",
    stable_only: bool = False,
    formation_energies: Optional[dict] = None,
    camera: str = "orthographic",
    particle_size: float = 1.0,
    background: Literal["white", "black"] = "white",
    distance_from_camera: float = 1.0,
):
    import numpy as np
    from pyiron_nodes.atomistic.structure._atoms import to_ase

    _CNA_COLOR = {0: "#c0c0c0", 1: "#39a600", 2: "#ff2800", 3: "#0053d6", 4: "#ffdc00"}

    if stable_only and formation_energies is not None:
        reserved = {"mu_values"}
        keys = [k for k in formation_energies if k not in reserved]
        ef_matrix = np.array([np.atleast_1d(np.asarray(formation_energies[k], dtype=float))
                               for k in keys])
        hull_idx = np.argmin(ef_matrix, axis=0)
        stable_names = [keys[i] for i in sorted(set(hull_idx.tolist()))]
        name_to_struct = {row["name"]: row["structure"] for _, row in relaxed_df.iterrows()}
        stable_structs = [name_to_struct[n] for n in stable_names if n in name_to_struct]
        if not stable_structs:
            raise ValueError("No stable structures found in relaxed_df.")
        atoms = to_ase(stable_structs[index % len(stable_structs)])
    else:
        atoms = to_ase(relaxed_df.iloc[index]["structure"])

    symbols = atoms.get_chemical_symbols()

    if coloring == "element":
        colors = [_element_color_hex(s) for s in symbols]
    else:
        import structuretoolkit as stk
        labels = stk.analyse_cna_adaptive(atoms, mode="numeric")
        colors = [
            _element_color_hex('Ga') if s == 'Ga' else _CNA_COLOR.get(int(lbl), "#c0c0c0")
            for s, lbl in zip(symbols, labels)
        ]

    return _build_view3d(
        atoms, colors, render_style=render_style, particle_size=particle_size,
        camera=camera, background=background,
        view_plane=_PLANE_TO_NORMAL[perspective],
        distance_from_camera=distance_from_camera, show_cell=True)


@group_node("mu")
def compute_reference_chemical_potential(structure, engine, optimizer_settings):
    from core import Workflow, as_function_node
    from pyiron_nodes.dpg2026.atomistic.calculator.optimize import Relax

    @as_function_node("mu")
    def _energy_per_atom(calc_result):
        from pyiron_nodes.atomistic.structure._atoms import to_ase
        s = to_ase(calc_result.structure)
        return calc_result.energy / len(s)

    inner = Workflow("compute_reference_chemical_potential")
    inner.relaxed = Relax(structure=structure, engine=engine,
                          opt_parameters=optimizer_settings, opt_mode='full')
    inner.relaxed.inputs.add("store", port_type=bool, default=False, value=True,
                             has_explicit_default=True)
    inner.mu = _energy_per_atom(calc_result=inner.relaxed)
    return inner.mu


@group_node("mu_gan_fu")
def compute_gan_bulk_energy_per_fu(structure, engine, optimizer_settings):
    """Relax GaN bulk and return total energy per formula unit (GaN pair)."""
    from core import Workflow, as_function_node
    from pyiron_nodes.dpg2026.atomistic.calculator.optimize import Relax

    @as_function_node("mu_gan_fu")
    def _energy_per_fu(calc_result):
        from pyiron_nodes.atomistic.structure._atoms import to_ase
        s = to_ase(calc_result.structure)
        # Wurtzite GaN: equal Ga and N atoms, so n_fu = n_atoms / 2
        return calc_result.energy / (len(s) / 2)

    inner = Workflow("compute_gan_bulk_energy_per_fu")
    inner.relaxed = Relax(structure=structure, engine=engine,
                          opt_parameters=optimizer_settings, opt_mode='full')
    inner.relaxed.inputs.add("store", port_type=bool, default=False, value=True,
                             has_explicit_default=True)
    inner.mu_gan_fu = _energy_per_fu(calc_result=inner.relaxed)
    return inner.mu_gan_fu


@group_node("formation_energies")
def compute_surface_formation_energies(relaxed_df, mu_ga_sweep, mu_gan_fu):
    """Formation energies with GaN stoichiometric constraint: μ_N = μ_GaN_fu - μ_Ga."""
    from core import Workflow, as_function_node
    from pyiron_nodes.atomistic.thermodynamics.defect_phases import (
        AddElementCountColumns, AddDefectConcentrationColumns, ComputeDefectFormationEnergy,
    )

    @as_function_node("chemical_potentials")
    def _build_chem_pots(mu_ga_sweep, mu_gan_fu: float):
        import numpy as np
        mu_ga = np.atleast_1d(np.asarray(mu_ga_sweep, dtype=float))
        mu_n = float(mu_gan_fu) - mu_ga
        return {"Ga": mu_ga, "N": mu_n}

    inner = Workflow("compute_surface_formation_energies")
    inner.with_counts = AddElementCountColumns(df=relaxed_df)
    inner.with_deltas = AddDefectConcentrationColumns(df=inner.with_counts, pristine_row=0)
    inner.chem_pots = _build_chem_pots(mu_ga_sweep=mu_ga_sweep, mu_gan_fu=mu_gan_fu)
    inner.formation_energies = ComputeDefectFormationEnergy(
        df=inner.with_deltas, chemical_potentials=inner.chem_pots, pristine_row=0)
    return inner.formation_energies


# ── Workflow ──────────────────────────────────────────────────────────────────
wf = Workflow("gan_surface_phase_diagram")

wf.grace_engine = Grace(model='GRACE-2L-OAM')
wf.optimizer_settings = GenericOptimizerSettings(max_steps=500, force_tolerance=0.001)

wf.bulk_ga  = BulkGa()
wf.bulk_gan = BulkGaN()

# Reference chemical potentials
wf.mu_ga = compute_reference_chemical_potential(
    structure=wf.bulk_ga, engine=wf.grace_engine, optimizer_settings=wf.optimizer_settings)

wf.mu_gan_fu = compute_gan_bulk_energy_per_fu(
    structure=wf.bulk_gan, engine=wf.grace_engine, optimizer_settings=wf.optimizer_settings)

# GaN(0001) slab and defect enumeration
wf.surface = BuildGaNSurface(a=3.189, c=5.185, n_layers=4, repeat=2, vacuum=10.0)

# From-scratch 3D view of the freshly built slab
wf.view_surface = Plot3D(structure=wf.surface, render_style="ball+stick", particle_size=1.0)

wf.surface_defects = BuildSurfaceDefects(
    surface_structure=wf.surface, n_seeds=1, adatom_height=2.0, top_layer_depth=3.0)

# Structure relaxation
wf.relaxed_surface_df = RelaxStructuresDataFrame(
    df=wf.surface_defects, engine=wf.grace_engine,
    opt_parameters=wf.optimizer_settings, opt_mode='full')

# μ_Ga sweep: from μ_Ga_ref (Ga-rich, right) down to μ_Ga_ref - 2.5 eV (N-rich, left)
wf.mu_ga_sweep = ChemicalPotentialSweep(mu_ref=wf.mu_ga, delta_mu=-2.5, num_points=200)

# Formation energies with GaN stoichiometric constraint
wf.formation_energies = compute_surface_formation_energies(
    relaxed_df=wf.relaxed_surface_df, mu_ga_sweep=wf.mu_ga_sweep, mu_gan_fu=wf.mu_gan_fu)

# Phase diagram plot
wf.plot_phases = PlotDefectPhase(
    formation_energies=wf.formation_energies,
    mu_label='μ_Ga  (eV)',
    ef_label='Formation energy  (eV)',
    title='GaN(0001) surface phase diagram — GRACE-ML')

# Summary table of stable phases
wf.stable_phases = SelectStableStructures(formation_energies=wf.formation_energies)

# Schematic matplotlib views of stable structures
wf.view_stable = PlotStableDecoratedStructures(
    formation_energies=wf.formation_energies, relaxed_df=wf.relaxed_surface_df)

# Interactive 3D viewer
wf.view_3d = View3DStructure(
    relaxed_df=wf.relaxed_surface_df, formation_energies=wf.formation_energies,
    index=0, coloring="element", stable_only=False)
