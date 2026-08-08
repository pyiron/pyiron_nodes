from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Literal, Optional
from core import Workflow, as_function_node, group_node

from pyiron_nodes.atomistic.structure.build import Bulk
from pyiron_nodes.atomistic.structure.build_gb import GrainBoundaryOptions, BuildGrainBoundary
from pyiron_nodes.atomistic.structure.view import PlotCNA
from pyiron_nodes.atomistic.thermodynamics.defect_phases import SelectStableStructures
from pyiron_nodes.dpg2026.atomistic.calculator.optimize import GenericOptimizerSettings, Relax
from pyiron_nodes.dpg2026.atomistic.engine.grace import Grace


# ── Local node definitions ──────────────────────────────────────────────────

@as_function_node("gamma_meV_per_A2")
def GrainBoundaryEnergy(
    relaxed_df: pd.DataFrame = None,
    bulk_energy_per_atom: float = 0.0,
    gb_structure_original=None,
):
    """
    γ = (E_GB - N·e_bulk) / (2·A)  [eV/Å²] × 1000  →  meV/Å²

    relaxed_df : DataFrame from RelaxStructuresDataFrame; pristine GB is row 0.
    gb_structure_original : unrelaxed OutputAtoms from BuildGrainBoundary (used for area).
    """
    from pyiron_nodes.atomistic.structure._atoms import to_ase

    gb_total_energy = relaxed_df.iloc[0]['energy']
    atoms = to_ase(gb_structure_original)
    n_gb = len(atoms)
    cell = np.array(atoms.cell)
    area = np.linalg.norm(np.cross(cell[0], cell[1]))
    excess_eV = gb_total_energy - n_gb * bulk_energy_per_atom
    gamma_meV_per_A2 = (excess_eV / (2.0 * area)) * 1000.0
    return gamma_meV_per_A2


@as_function_node("df")
def BuildDecoratedStructures(
    host_structure,
    solute: str = "Nb",
    host: str = "Ni",
    max_solute_atoms: int = 4,
    n_seeds: int = 3,
) -> pd.DataFrame:
    """
    Return a DataFrame with the pristine GB structure plus random
    realisations of 1..max_solute_atoms solute-for-host substitutions.

    host_structure may be OutputAtoms or OutputCalcOpt (structure extracted automatically).
    Columns: structure (OutputAtoms), name (str).
    """
    from pyiron_nodes.atomistic.structure.build_point_defects import (
        make_pristine_reference, make_config_row, op_substitute, expand_configs,
    )
    from pyiron_nodes.atomistic.structure._atoms import to_ase, _ase_to_data

    if hasattr(host_structure, 'structure'):
        ase_input = to_ase(host_structure.structure)
    else:
        ase_input = to_ase(host_structure)

    atoms0, pristine_pos = make_pristine_reference._original_func(ase_input)
    base = make_config_row(
        atoms=atoms0,
        structure_id="gb_pristine",
        events=[],
        seed=0,
        pristine_n_sites=len(atoms0),
        pristine_positions=pristine_pos,
    )
    base_df = pd.DataFrame([base])

    rows = [base_df]
    for n_solute in range(1, max_solute_atoms + 1):
        kwargs_list = [
            {"from_element": host, "to_element": solute, "n": n_solute, "seed": s}
            for s in range(n_seeds)
        ]
        rows.append(
            expand_configs._original_func(
                base_df, op_substitute._original_func, kwargs_list, keep_input=False
            )
        )

    combined = pd.concat(rows, ignore_index=True)
    combined["structure"] = combined["atoms"].apply(_ase_to_data)

    def _name(row):
        events = row.get("events") or []
        n = sum(
            1 for e in events
            if e.get("type") == "substitution" and e.get("to") == solute
        )
        if n == 0:
            return "pristine"
        seed = row.get("seed", 0) or 0
        return f"{solute}_{n} (seed {seed})"

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
    """
    Relax every structure in df['structure'].

    Calls Relax._original_func directly — no template node required.
    store=True enables hash-based caching (recommended for production runs).
    Returns df extended with 'structure' (OutputAtoms) and 'energy' (eV).
    """
    from pyiron_nodes.atomistic.structure._atoms import to_ase
    from pyiron_nodes.dpg2026.atomistic.calculator.optimize import Relax, GenericOptimizerSettings

    if opt_parameters is None:
        opt_parameters = GenericOptimizerSettings()

    relaxed, energies = [], []
    for s in df["structure"]:
        out = Relax._original_func(
            structure=to_ase(s),
            engine=engine,
            opt_parameters=opt_parameters,
            opt_mode=opt_mode,
        )
        relaxed.append(out.structure)
        energies.append(out.energy)

    result = df.copy()
    result["structure"] = relaxed
    result["energy"] = energies
    return result


@as_function_node("mu_sweep")
def ChemicalPotentialSweep(mu_ref: float, delta_mu: float = -1.5, num_points: int = 200):
    """Linearly spaced values from mu_ref to mu_ref + delta_mu."""
    return np.linspace(mu_ref, mu_ref + delta_mu, int(num_points))


@as_function_node("fig")
def PlotDefectPhase(
    formation_energies: dict,
    mu_label: str = 'μ (eV)',
    ef_label: str = 'Formation energy (eV)',
    title: str = 'Defect phase diagram',
    exclude_keys: list = None,
) -> object:
    """
    Plot defect formation energies with the stable-phase region filled.

    Each phase is drawn as a colored line. The area below the convex-hull
    envelope is shaded with the color of the locally stable phase, making
    phase transitions immediately visible as color changes. Transition
    boundaries are marked with vertical dashed lines.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    mu_arr = np.atleast_1d(np.asarray(formation_energies["mu_values"], dtype=float))
    reserved = {"mu_values"}
    if exclude_keys:
        reserved.update(exclude_keys)

    keys = [k for k in formation_energies if k not in reserved]
    if not keys:
        raise ValueError("No plottable entries in formation_energies.")

    ef_matrix = np.array([
        np.atleast_1d(np.asarray(formation_energies[k], dtype=float))
        for k in keys
    ])  # (n_phases, n_mu)

    hull_ef = np.min(ef_matrix, axis=0)
    hull_idx = np.argmin(ef_matrix, axis=0)

    cmap = plt.colormaps["tab10"]
    colors = [cmap(i % 10) for i in range(len(keys))]

    fig, ax = plt.subplots(figsize=(9, 5))

    # Shaded fill: for each phase, fill its stable μ-region from hull down to plot bottom
    y_min = hull_ef.min()
    y_range = hull_ef.max() - y_min
    y_bottom = y_min - 0.08 * max(y_range, 1e-6)

    for i, color in enumerate(colors):
        mask = hull_idx == i
        if mask.any():
            ax.fill_between(
                mu_arr, hull_ef, y_bottom,
                where=mask, color=color, alpha=0.20, linewidth=0,
            )

    # Individual formation energy lines
    for i, (key, color) in enumerate(zip(keys, colors)):
        ef = np.atleast_1d(np.asarray(formation_energies[key], dtype=float))
        if key == "pristine":
            ax.plot(mu_arr, ef, color=color, lw=1.2, ls='--', alpha=0.7, label=key)
        else:
            # Emphasise the line where it is the stable (lowest-energy) phase
            stable = hull_idx == i
            ax.plot(mu_arr, ef, color=color, lw=1.0, alpha=0.45, zorder=2)
            if stable.any():
                ax.plot(
                    mu_arr[stable], ef[stable],
                    color=color, lw=2.5, alpha=1.0, zorder=3,
                    label=key,
                )
            else:
                ax.plot([], [], color=color, lw=2.5, label=key)

    # Phase-transition boundaries
    transitions = np.where(np.diff(hull_idx))[0]
    for t in transitions:
        ax.axvline(
            0.5 * (mu_arr[t] + mu_arr[t + 1]),
            color='crimson', lw=1.2, ls='--', alpha=0.85, zorder=4,
        )

    ax.set_xlim(mu_arr.min(), mu_arr.max())
    ax.set_ylim(bottom=y_bottom)
    ax.set_xlabel(mu_label, fontsize=13)
    ax.set_ylabel(ef_label, fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=9, framealpha=0.9, ncol=2)
    ax.grid(True, linestyle='--', alpha=0.3)
    fig.tight_layout()
    return fig


@as_function_node("fig")
def PlotStableDecoratedStructures(
    formation_energies: dict,
    relaxed_df: pd.DataFrame,
    rotation: str = '-90x',
    columns: int = 2,
    figure_size: float = 4.0,
) -> object:
    """
    Render schematic matplotlib pictures of each distinct stable decorated GB structure.

    Stable phases are those that minimise the formation energy for at least one
    value of μ_Nb.  Ni atoms are colored by CNA type (FCC=green, other=grey);
    Nb substitutional atoms are shown in orange.

    Parameters
    ----------
    rotation : str
        ASE rotation string (default '-90x' gives a cross-section view along y,
        showing the stacking direction z vertically with the GB interface in the
        middle of the cell).
    """
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    from ase.visualize.plot import plot_atoms
    from pyiron_nodes.atomistic.structure._atoms import to_ase
    import structuretoolkit as stk

    # ── Find stable phase names ──────────────────────────────────────────────
    reserved = {"mu_values"}
    keys = [k for k in formation_energies if k not in reserved]
    ef_matrix = np.array([
        np.atleast_1d(np.asarray(formation_energies[k], dtype=float))
        for k in keys
    ])
    hull_idx = np.argmin(ef_matrix, axis=0)
    stable_names = [keys[i] for i in sorted(set(hull_idx.tolist()))]

    # ── Look up structures from relaxed_df ──────────────────────────────────
    name_to_struct = {row['name']: row['structure']
                      for _, row in relaxed_df.iterrows()}
    stable = [(name, to_ase(name_to_struct[name]))
              for name in stable_names if name in name_to_struct]

    if not stable:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No stable structures found', ha='center', va='center',
                transform=ax.transAxes)
        return fig

    # ── CNA + element coloring ───────────────────────────────────────────────
    _CNA_COLOR = {
        0: '#c0c0c0',   # unknown / GB interface → grey
        1: '#39a600',   # FCC → green
        2: '#ff2800',   # HCP → red
        3: '#0053d6',   # BCC → blue
        4: '#ffdc00',   # icosahedral → yellow
    }
    _NB_COLOR = '#ff8c00'   # Nb substitutional → orange

    # ── Layout ───────────────────────────────────────────────────────────────
    n = len(stable)
    cols = min(n, columns)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(figure_size * cols, figure_size * rows),
        squeeze=False,
    )

    for idx, (name, atoms) in enumerate(stable):
        row_i, col_i = divmod(idx, cols)
        ax = axes[row_i][col_i]

        symbols = atoms.get_chemical_symbols()
        cna = stk.analyse_cna_adaptive(atoms, mode='numeric')
        colors = [
            _NB_COLOR if sym == 'Nb' else _CNA_COLOR.get(int(c), '#c0c0c0')
            for sym, c in zip(symbols, cna)
        ]

        plot_atoms(atoms, ax=ax, rotation=rotation, colors=colors, show_unit_cell=1)
        ax.set_title(name, fontsize=10)
        ax.axis('off')

    # Hide unused axes
    for idx in range(n, rows * cols):
        row_i, col_i = divmod(idx, cols)
        axes[row_i][col_i].axis('off')

    fig.suptitle('Stable decorated Ni Σ5 GB structures (Nb = orange)', fontsize=12)
    fig.tight_layout()
    return fig


@as_function_node("view")
def View3DStructure(
    relaxed_df: pd.DataFrame,
    index: int = 0,
    coloring: Literal["cna", "element"] = "cna",
    perspective: Literal["xy", "xz", "yz"] = "xz",
    stable_only: bool = False,
    formation_energies: Optional[dict] = None,
    camera: str = "orthographic",
    particle_size: float = 1.0,
    background: Literal["white", "black"] = "white",
    distance_from_camera: float = 1.0,
):
    """
    Interactive nglview of a structure from relaxed_gb_df, fitted to the widget.

    Parameters
    ----------
    index : int
        When stable_only=False: row index into all of relaxed_df.
        When stable_only=True: index into the list of stable phases only
        (sorted by their first appearance on the convex hull).
    stable_only : bool
        If True, restrict the index space to structures that are thermodynamically
        stable (i.e. minimise E_f) at some μ_Nb value.  Requires formation_energies.
    formation_energies : dict, optional
        Output of compute_defect_formation_energies.  Needed when stable_only=True.
    coloring : {"cna", "element"}
        "cna"     — CNA colors: FCC=green, HCP=red, BCC=blue, interface=grey;
                    Nb substitutional atoms=orange.
        "element" — standard Jmol element colors (Ni=silver, Nb=teal).
    perspective : {"xy", "xz", "yz"}
        "xy" looks along z (top view, GB plane);
        "xz" looks along y (cross-section, stacking direction vertical);
        "yz" looks along x (side view).
    """
    import numpy as np
    import structuretoolkit as stk
    from pyiron_nodes.atomistic.structure._atoms import to_ase

    _PLANE_TO_NORMAL = {"xy": [0, 0, 1], "xz": [0, 1, 0], "yz": [1, 0, 0]}
    view_plane = _PLANE_TO_NORMAL[perspective]

    # ── Select structure ─────────────────────────────────────────────────────
    if stable_only and formation_energies is not None:
        reserved = {"mu_values"}
        keys = [k for k in formation_energies if k not in reserved]
        ef_matrix = np.array([
            np.atleast_1d(np.asarray(formation_energies[k], dtype=float))
            for k in keys
        ])
        hull_idx = np.argmin(ef_matrix, axis=0)
        stable_names = [keys[i] for i in sorted(set(hull_idx.tolist()))]
        name_to_struct = {row["name"]: row["structure"]
                          for _, row in relaxed_df.iterrows()}
        stable_structs = [name_to_struct[n] for n in stable_names if n in name_to_struct]
        if not stable_structs:
            raise ValueError("No stable structures found in relaxed_df.")
        atoms = to_ase(stable_structs[index % len(stable_structs)])
    else:
        atoms = to_ase(relaxed_df.iloc[index]["structure"])

    # ── Coloring ─────────────────────────────────────────────────────────────
    if coloring == "cna":
        _CNA_COLOR = {
            0: "#c0c0c0",
            1: "#39a600",
            2: "#ff2800",
            3: "#0053d6",
            4: "#ffdc00",
        }
        labels = stk.analyse_cna_adaptive(atoms, mode="numeric")
        symbols = atoms.get_chemical_symbols()
        colors = np.array([
            "#ff8c00" if sym == "Nb" else _CNA_COLOR.get(int(lbl), "#c0c0c0")
            for sym, lbl in zip(symbols, labels)
        ])
        view = stk.plot3d(
            atoms,
            camera=camera,
            particle_size=particle_size,
            background=background,
            colors=colors,
            view_plane=view_plane,
            distance_from_camera=distance_from_camera,
        )
    else:
        view = stk.plot3d(
            atoms,
            camera=camera,
            particle_size=particle_size,
            background=background,
            view_plane=view_plane,
            distance_from_camera=distance_from_camera,
        )

    view.center()
    return view


# ── Group node factories ────────────────────────────────────────────────────

@group_node("mu")
def compute_reference_chemical_potential(structure, engine, optimizer_settings):
    """Relax a bulk reference and return its energy per atom (eV/atom)."""
    from core import Workflow, as_function_node
    from pyiron_nodes.dpg2026.atomistic.calculator.optimize import Relax

    @as_function_node("mu")
    def _energy_per_atom(calc_result):
        from pyiron_nodes.atomistic.structure._atoms import to_ase
        s = to_ase(calc_result.structure)
        return calc_result.energy / len(s)

    inner = Workflow("compute_reference_chemical_potential")
    inner.relaxed = Relax(
        structure=structure, engine=engine,
        opt_parameters=optimizer_settings, opt_mode='full',
    )
    inner.relaxed.inputs.add(
        "store", port_type=bool, default=False, value=True, has_explicit_default=True
    )
    inner.mu = _energy_per_atom(calc_result=inner.relaxed)
    return inner.mu


@group_node("formation_energies")
def compute_defect_formation_energies(
    relaxed_df, mu_host, mu_solute_sweep, host='Ni', solute='Nb',
):
    """
    Compute absolute defect formation energies:
        E_f = E_defect - n_host · μ_host - n_solute · μ_solute

    Energies are relative to the elemental bulk references, not to the
    undecorated GB.  This gives physically meaningful absolute values
    that reflect the thermodynamic cost of Nb incorporation.
    """
    from core import Workflow, as_function_node
    from pyiron_nodes.atomistic.thermodynamics.defect_phases import AddElementCountColumns

    @as_function_node
    def _absolute_formation_energy(df, mu_host, mu_solute_sweep, host='Ni', solute='Nb'):
        import numpy as np
        mu_arr = np.atleast_1d(np.asarray(mu_solute_sweep, dtype=float))
        result = {"mu_values": mu_arr}
        for _, row in df.iterrows():
            n_host = int(row.get(f'n_{host}', 0))
            n_solute = int(row.get(f'n_{solute}', 0))
            e_defect = float(row['energy'])
            name = str(row.get('name', 'unknown'))
            # E_f = E - n_Ni·μ_Ni - n_Nb·μ_Nb  (broadcast over μ_Nb sweep)
            ef = e_defect - n_host * float(mu_host) - n_solute * mu_arr
            result[name] = ef
        return result

    inner = Workflow("compute_defect_formation_energies")
    inner.with_counts = AddElementCountColumns(df=relaxed_df)
    inner.formation_energies = _absolute_formation_energy(
        df=inner.with_counts,
        mu_host=mu_host,
        mu_solute_sweep=mu_solute_sweep,
        host=host,
        solute=solute,
    )
    return inner.formation_energies


# ── Workflow ────────────────────────────────────────────────────────────────

wf = Workflow("ni_nb_gb_defect_phase_diagram")

# Engine and optimizer
wf.grace_engine = Grace(model='GRACE-2L-OAM')
wf.optimizer_settings = GenericOptimizerSettings(max_steps=500, force_tolerance=0.001)

# ── Grain boundary structure (Ni Σ5 FCC) ────────────────────────────────────
wf.gb_options = GrainBoundaryOptions(
    sigma=5,
    crystalstructure='fcc',
    a=3.52,          # Ni experimental lattice parameter (Å)
)
wf.gb_structure = BuildGrainBoundary(
    options=wf.gb_options.outputs.options,
    index=0,
    symbol='Ni',
    min_slab_thickness=15.0,
    vacuum=0.0,
    merge_tol=0.5,
)

# Visualise the as-built GB (CNA: grey atoms = disordered interface)
wf.view_gb = PlotCNA(structure=wf.gb_structure.outputs.structure)

# ── Bulk references ──────────────────────────────────────────────────────────
wf.bulk_ni = Bulk(name='Ni', crystalstructure='fcc', a=3.52)
wf.bulk_nb = Bulk(name='Nb', crystalstructure='bcc', a=3.30)

wf.mu_ni = compute_reference_chemical_potential(
    structure=wf.bulk_ni,
    engine=wf.grace_engine,
    optimizer_settings=wf.optimizer_settings,
)
wf.mu_nb_ref = compute_reference_chemical_potential(
    structure=wf.bulk_nb,
    engine=wf.grace_engine,
    optimizer_settings=wf.optimizer_settings,
)

# ── Decorated GB structures (pristine + Nb substitutions) ───────────────────
# Pristine GB is always row 0; Nb-decorated configurations follow.
wf.decorated_gb_structures = BuildDecoratedStructures(
    host_structure=wf.gb_structure.outputs.structure,
    solute='Nb',
    host='Ni',
    max_solute_atoms=4,
    n_seeds=3,
)

# ── Relax all structures (store=True for hash-based caching) ────────────────
wf.relaxed_gb_df = RelaxStructuresDataFrame(
    df=wf.decorated_gb_structures,
    engine=wf.grace_engine,
    opt_parameters=wf.optimizer_settings,
    opt_mode='full',
)

# ── Grain boundary energy γ  [meV/Å²] ───────────────────────────────────────
wf.gb_energy = GrainBoundaryEnergy(
    relaxed_df=wf.relaxed_gb_df,
    bulk_energy_per_atom=wf.mu_ni,
    gb_structure_original=wf.gb_structure.outputs.structure,
)

# ── Chemical potential sweep for Nb ─────────────────────────────────────────
wf.mu_nb_sweep = ChemicalPotentialSweep(
    mu_ref=wf.mu_nb_ref,
    delta_mu=-1.5,
    num_points=200,
)

# ── Absolute defect formation energies  E_f = E - n_Ni·μ_Ni - n_Nb·μ_Nb ────
wf.formation_energies = compute_defect_formation_energies(
    relaxed_df=wf.relaxed_gb_df,
    mu_host=wf.mu_ni,
    mu_solute_sweep=wf.mu_nb_sweep,
    host='Ni',
    solute='Nb',
)

# ── Phase diagram: colored lines + stable-region fills ──────────────────────
wf.plot_phases = PlotDefectPhase(
    formation_energies=wf.formation_energies,
    mu_label='μ_Nb  (eV)',
    ef_label='Formation energy  (eV)',
    title='Defect phase diagram: Nb in Ni Σ5 GB — GRACE-ML',
)

# ── Stable structures at each μ_Nb ──────────────────────────────────────────
wf.stable_structures = SelectStableStructures(
    formation_energies=wf.formation_energies,
)

# ── Schematic pictures of stable decorated GB structures ─────────────────────
# Ni colored by CNA (green=FCC, grey=interface); Nb shown in orange.
# rotation='-90x' gives a cross-section: tilt axis horizontal, stacking vertical.
wf.view_stable = PlotStableDecoratedStructures(
    formation_energies=wf.formation_energies,
    relaxed_df=wf.relaxed_gb_df,
)

# ── Interactive 3D nglview of any individual structure ───────────────────────
# index=0 → first structure (pristine when stable_only=False, first stable phase
#            when stable_only=True).
# Toggle 'coloring' and 'perspective' via the Literal ports; flip 'stable_only'
# to restrict the index space to thermodynamically stable phases only.
wf.view_3d = View3DStructure(
    relaxed_df=wf.relaxed_gb_df,
    formation_energies=wf.formation_energies,
    index=0,
    coloring="cna",
    stable_only=False,
)
