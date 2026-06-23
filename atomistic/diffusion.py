"""Reusable nodes for H-diffusion workflows (MD and NEB)."""

from __future__ import annotations

from typing import Literal

from core import as_function_node, as_out_dataclass_node
from core.data_fields import DataArray, EmptyArrayField

# ── Structure helpers ────────────────────────────────────────────────────────


@as_function_node
def InterstitialHPositions(path_type: Literal["O-T-O", "T-O-T"] = "T-O-T"):
    """Return fractional initial/final positions for a nearest-neighbour H hop.

    path_type='O-T-O': endpoints at octahedral sites; tetrahedral site is the
        transition-state saddle (use for potentials where O is the stable site).
    path_type='T-O-T': endpoints at tetrahedral sites; the body-centre octahedral
        site lies exactly at the path midpoint (use for potentials where T is stable).

    Positions are given in fractional coordinates of the FCC conventional unit cell
    and should be passed with repeat_scalar matching the supercell build.
    """
    _sites = {
        # O1=[0.5,0,0] and O2=[0,0,0.5]: nearest-neighbour O–O pair in FCC,
        # connected via a T-site saddle at ~[0.25,0.25,0.25].
        "O-T-O": ([0.5, 0.0, 0.0], [0.0, 0.0, 0.5]),
        # T1=[0.25,0.25,0.25] and T2=[0.75,0.75,0.75]: the body-centre O at
        # [0.5,0.5,0.5] lies exactly at the midpoint of this T–T vector.
        "T-O-T": ([0.25, 0.25, 0.25], [0.25, 0.25, 0.75]),
    }
    if path_type not in _sites:
        raise ValueError(f"path_type must be 'O-T-O' or 'T-O-T', got {path_type!r}")
    initial_pos, final_pos = _sites[path_type]
    return initial_pos, final_pos


@as_function_node
def AddInterstitialH(structure, frac_pos: list = None, repeat_scalar: int = 1):
    """Append one H atom at a fractional interstitial position in *structure*.

    frac_pos is given relative to the *unit cell*, not the supercell.
    repeat_scalar must match the value used when building the supercell so that
    the unit cell matrix can be recovered: unit_cell = supercell_cell / repeat_scalar.
    With repeat_scalar=1 frac_pos is interpreted relative to the supercell directly.
    """
    import numpy as np

    if frac_pos is None:
        frac_pos = [0.25, 0.0, 0.0]
    new_atoms = structure.copy()
    unit_cell = new_atoms.cell / repeat_scalar
    cart_pos = np.dot(frac_pos, unit_cell)
    new_atoms.append("H")
    new_atoms.positions[-1] = cart_pos
    new_structure = new_atoms
    return new_structure


# ── MD analysis ──────────────────────────────────────────────────────────────


@as_function_node
def ComputeMSD(md_output, species_symbol: str = "H"):
    """Compute mean-square displacement of *species_symbol* over the MD trajectory.

    Returns MSD in Å² as a 1-D array with one value per printed frame.
    Uses unwrapped positions so PBC discontinuities do not affect the result.
    """
    import numpy as np

    species = np.array(md_output.species)
    h_indices = np.where(species == species_symbol)[0]
    unwrapped_pos = np.array(md_output.unwrapped_positions)  # (n_frames, n_atoms, 3)
    h_pos = unwrapped_pos[:, h_indices, :]  # (n_frames, n_h, 3)
    ref = h_pos[0]
    disp = h_pos - ref[np.newaxis, :, :]
    msd = np.mean(np.sum(disp**2, axis=-1), axis=-1)  # Å², shape (n_frames,)
    return msd


@as_function_node
def DiffusionConstant(msd, md_input, fit_start_fraction: float = 0.2):
    """Compute the diffusion constant from MSD via the Einstein relation.

    D = slope(MSD vs t) / 6   [3-D diffusion]

    The linear fit is restricted to frames beyond *fit_start_fraction* of the
    total trajectory to avoid the ballistic short-time regime.

    Returns
    -------
    diffusion_constant : float
        Diffusion constant D in m²/s.
    times : np.ndarray
        Time axis in ps (one value per printed frame).
    """
    import numpy as np

    n_frames = len(msd)
    time_step_ps = md_input.time_step * 1e-3  # fs → ps
    times = np.arange(n_frames) * md_input.n_print * time_step_ps  # ps
    start = int(fit_start_fraction * n_frames)
    slope = np.polyfit(times[start:], msd[start:], 1)[0]  # Å²/ps
    # 1 Å²/ps = 1e-20 m² / 1e-12 s = 1e-8 m²/s
    diffusion_constant = slope / 6.0 * 1e-8  # m²/s
    return diffusion_constant, times


@as_function_node
def PlotHPositions(md_output, md_input, species_symbol: str = "H"):
    """Plot x, y, z positions of *species_symbol* vs time on three shared-x subplots."""
    import numpy as np
    import matplotlib.pyplot as plt

    species = np.array(md_output.species)
    h_indices = np.where(species == species_symbol)[0]
    unwrapped_pos = np.array(md_output.unwrapped_positions)  # (n_frames, n_atoms, 3)
    h_pos = unwrapped_pos[:, h_indices, :]  # (n_frames, n_h, 3)

    n_frames = unwrapped_pos.shape[0]
    time_step_ps = md_input.time_step * 1e-3
    times = np.arange(n_frames) * md_input.n_print * time_step_ps  # ps

    x_pos = h_pos[:, :, 0].mean(axis=1)  # Å
    y_pos = h_pos[:, :, 1].mean(axis=1)
    z_pos = h_pos[:, :, 2].mean(axis=1)

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 6))
    for ax, pos, label in zip(axes, [x_pos, y_pos, z_pos], ["x", "y", "z"]):
        ax.plot(times, pos)
        ax.set_ylabel(f"{label} (Å)")
    axes[-1].set_xlabel("time (ps)")
    fig.suptitle(f"{species_symbol} position vs time")
    fig.tight_layout()

    figure = fig
    return figure


# ── Free energy surface ──────────────────────────────────────────────────────


@as_function_node
def FoldPositionsToUnitCell(md_output, al_bulk, species_symbol: str = "H"):
    """Fold H Cartesian positions back into the unit cell as fractional coords in [0, 1)."""
    import numpy as np

    species = np.array(md_output.species)
    h_indices = np.where(species == species_symbol)[0]
    unwrapped_pos = np.array(md_output.unwrapped_positions)  # (n_frames, n_atoms, 3)
    h_pos = unwrapped_pos[:, h_indices, :].reshape(-1, 3)  # (N, 3) Cartesian Å

    unit_cell = np.array(al_bulk.cell)  # (3,3), rows = lattice vectors
    inv_cell = np.linalg.inv(unit_cell)
    frac = h_pos @ inv_cell  # fractional coords (row-vector convention)
    folded_positions = frac % 1.0  # fold into [0, 1)
    return folded_positions


@as_function_node
def AugmentWithSymmetry(folded_positions, al_bulk):
    """Multiply H positions by all crystallographic symmetry operations of the host lattice.

    Uses spglib to obtain the full space-group operations; falls back to the 48
    point-group operations of the cubic group Oh when spglib is unavailable.
    """
    import numpy as np
    import itertools

    try:
        import spglib

        cell = (al_bulk.cell, al_bulk.get_scaled_positions(), al_bulk.numbers)
        sym = spglib.get_symmetry(cell, symprec=1e-5)
        rotations = sym["rotations"]  # (n_ops, 3, 3) int; col-vec: x' = R @ x
        translations = sym["translations"]
    except Exception:
        # 48 elements of Oh: all permutations of axes × all sign combinations
        ops = []
        for perm in itertools.permutations([0, 1, 2]):
            for signs in itertools.product([-1, 1], repeat=3):
                R = np.zeros((3, 3), dtype=int)
                for i, (p, s) in enumerate(zip(perm, signs)):
                    R[i, p] = s
                ops.append(R)
        rotations = np.array(ops)
        translations = np.zeros((len(rotations), 3))

    all_pos = [folded_positions]
    for R, t in zip(rotations, translations):
        # row-vector equivalent of x' = R @ x + t is x' = x @ R^T + t
        new_pos = (folded_positions @ R.T + t) % 1.0
        all_pos.append(new_pos)

    augmented_positions = np.vstack(all_pos)
    return augmented_positions


@as_function_node
def ComputeFreeEnergySurface(augmented_positions, md_input, n_bins: int = 30):
    """Boltzmann inversion of the 3-D H position histogram.

    F(x,y,z) = -kT ln P(x,y,z), shifted so that F_min = 0.
    Returns the 3-D free energy grid (eV) and the bin-centre coordinates (fractional).
    """
    import numpy as np

    kB = 8.617333e-5  # eV/K
    kT = kB * md_input.temperature

    hist, edges = np.histogramdd(
        augmented_positions,
        bins=n_bins,
        range=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        F = np.where(hist > 0, -kT * np.log(hist / hist.max()), np.nan)

    F -= np.nanmin(F)
    grid_centers = [(e[:-1] + e[1:]) / 2.0 for e in edges]
    free_energy = F  # eV, shape (n_bins, n_bins, n_bins)
    return free_energy, grid_centers


@as_function_node
def ExtractMigrationBarrier(free_energy, grid_centers):
    """Sample F along T-site → O-site paths and return the minimum barrier in eV.

    The T-site (global minimum) is located automatically.  Four distinct O-site
    candidates reachable without crossing the unit-cell boundary are probed; the
    minimum peak along those paths is the barrier estimate.
    """
    import numpy as np
    from scipy.interpolate import RegularGridInterpolator

    cx, cy, cz = grid_centers
    F_filled = np.where(
        np.isnan(free_energy), np.nanmax(free_energy) * 2.0, free_energy
    )
    interp = RegularGridInterpolator(
        (cx, cy, cz), F_filled, method="linear", bounds_error=False, fill_value=None
    )

    # Locate T-site as the global minimum
    F_min_idx = np.unravel_index(np.nanargmin(free_energy), free_energy.shape)
    t_site = np.array([cx[F_min_idx[0]], cy[F_min_idx[1]], cz[F_min_idx[2]]])

    # Octahedral interstitials in FCC (fractional coords of conventional cell)
    o_candidates = np.array(
        [
            [0.5, 0.5, 0.5],  # body centre
            [0.5, 0.5, 0.0],  # face centre (and periodic images)
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ]
    )

    n_sample = 200
    barriers = []
    for o in o_candidates:
        delta = o - t_site
        delta -= np.round(delta)  # minimum-image convention
        path = t_site + np.outer(np.linspace(0.0, 1.0, n_sample), delta)
        path = path % 1.0
        F_path = interp(path)
        barriers.append(float(np.max(F_path)))  # F = 0 at T-site by construction

    barrier_ev = float(min(barriers))
    return barrier_ev


@as_function_node
def PlotMigrationPath(
    free_energy,
    grid_centers,
    augmented_positions,
    al_bulk,
    md_input,
    n_bins_1d: int = 100,
    tube_fraction: float = 0.35,
):
    """1D free energy T → O → T' profile with cosine-model extrapolation.

    The x-axis is normalised to [0, 1]: 0 = T-site (initial config.),
    1 = T'-site (final config.), 0.5 = O-site (saddle point).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d
    from scipy.optimize import curve_fit

    unit_cell = np.array(al_bulk.cell)
    kB = 8.617333e-5
    kT = kB * md_input.temperature
    n_bins_grid = len(grid_centers[0])

    # ── T-site: bin with maximum visit count (mode of augmented histogram) ────
    hist3d, edges3d = np.histogramdd(
        augmented_positions,
        bins=n_bins_grid,
        range=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
    )
    mode_idx = np.unravel_index(np.argmax(hist3d), hist3d.shape)

    def _centers(e):
        return (e[:-1] + e[1:]) / 2.0

    t_site = np.array([_centers(edges3d[i])[mode_idx[i]] for i in range(3)])
    t_cart = t_site @ unit_cell

    # Pre-compute Cartesian positions for all augmented points
    cart_pos = augmented_positions @ unit_cell  # (N, 3) Å

    # O-site candidates (octahedral interstitials in FCC conventional cell)
    o_candidates = np.array(
        [
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ]
    )

    best_barrier = np.inf
    best_result = None

    for o in o_candidates:
        delta = o - t_site
        delta -= np.floor(delta + 0.5)  # robust min-image (avoids banker's rounding)
        delta_cart = delta @ unit_cell
        d_to = np.linalg.norm(delta_cart)
        if d_to < 0.5:  # skip if T and O overlap (misidentified T-site)
            continue
        delta_hat = delta_cart / d_to
        r_tube = tube_fraction * d_to

        # Project all augmented positions onto the T→O axis
        diff = cart_pos - t_cart  # (N, 3)
        along = diff @ delta_hat  # (N,) signed distance along path in Å
        perp_sq = np.sum((diff - np.outer(along, delta_hat)) ** 2, axis=1)

        in_tube = (perp_sq < r_tube**2) & (along >= -0.2 * d_to) & (along <= 2.2 * d_to)
        rc_tube = along[in_tube]
        if rc_tube.size < 50:
            continue

        hist1d, edges1d = np.histogram(
            rc_tube, bins=n_bins_1d, range=(-0.1 * d_to, 2.1 * d_to)
        )
        rc_c = (edges1d[:-1] + edges1d[1:]) / 2.0

        with np.errstate(divide="ignore", invalid="ignore"):
            F1d = np.where(hist1d > 0, -kT * np.log(hist1d / hist1d.max()), np.nan)
        F1d -= np.nanmin(F1d)

        valid = ~np.isnan(F1d)
        if valid.sum() < 20:
            continue
        F_sm = F1d.copy()
        F_sm[valid] = gaussian_filter1d(F1d[valid], sigma=2.0)
        barrier = float(np.nanmax(F_sm[valid]))

        if barrier < best_barrier:
            best_barrier = barrier
            best_result = (rc_c, F_sm, valid, d_to)

    if best_result is None:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(
            0.5,
            0.5,
            "Insufficient sampling for 1D path profile.\n"
            "Try a longer trajectory or reduce tube_fraction.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
        )
        ax.set_axis_off()
        figure = fig
        return figure

    rc_c, F_sm, valid, d_to = best_result

    # ── Cosine fit to the sampled rising edge ─────────────────────────────────
    # F(rc) = (A/2)(1 − cos(π·rc/L))  →  F(0)=0, F(L)=A, F(2L)=0
    def cosine_model(rc, A, L):
        return (A / 2.0) * (1.0 - np.cos(np.pi * rc / L))

    rc_valid = rc_c[valid]
    F_valid = F_sm[valid]
    fit_mask = rc_valid > 0.0  # rising edge only

    A_fit, L_fit = best_barrier * 1.5, d_to  # defaults if fit fails
    if fit_mask.sum() > 5:
        try:
            popt, _ = curve_fit(
                cosine_model,
                rc_valid[fit_mask],
                F_valid[fit_mask],
                p0=[best_barrier * 1.5, d_to],
                bounds=(
                    [0.0, 0.5 * d_to],
                    [max(1.0, 50.0 * best_barrier), 3.0 * d_to],
                ),
                maxfev=10000,
            )
            A_fit, L_fit = float(popt[0]), float(popt[1])
        except Exception:
            pass  # keep defaults

    # Full cosine profile — only within [0, 2L] where the model is physical
    rc_full = np.linspace(-0.15 * d_to, 2.15 * d_to, 600)
    F_cos = cosine_model(rc_full, A_fit, L_fit)
    cos_mask = (rc_full >= 0.0) & (rc_full <= 2.0 * L_fit)
    F_cos[~cos_mask] = np.nan

    # ── Normalise x-axis: 0 = T-site (initial), 1 = T′-site (final) ─────────
    norm = 2.0 * L_fit
    rc_full_n = rc_full / norm
    rc_data_n = rc_c[valid] / norm

    # ── Plot ──────────────────────────────────────────────────────────────────
    rc_fill_n = rc_full_n[cos_mask]
    F_fill = F_cos[cos_mask]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.fill_between(rc_fill_n, F_fill, alpha=0.10, color="steelblue")
    ax.plot(
        rc_full_n,
        F_cos,
        lw=1.5,
        ls="--",
        color="tomato",
        alpha=0.85,
        label=f"cosine fit  ΔF = {A_fit:.3f} eV,  L = {L_fit:.2f} Å",
    )
    ax.plot(
        rc_data_n, F_sm[valid], lw=2.5, color="steelblue", label="MD data (smoothed)"
    )
    ax.axvline(0.5, color="gray", ls=":", lw=1.2)
    ax.text(0.5 * 1.02, A_fit * 0.05, "O-site", color="gray", fontsize=9, va="bottom")
    ax.set_xlabel("Reaction coordinate  (0 = T-site, 1 = T′-site)")
    ax.set_ylabel("Free energy (eV)")
    ax.set_title("1D free energy profile: T → O → T′ (minimum energy path)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    figure = fig
    return figure


@as_function_node
def PlotFreeEnergySurface(free_energy, grid_centers, al_bulk):
    """2-D contour plots of F(x, y) at three z-slices (T-site, O-site, z = 0.75)."""
    import numpy as np
    import matplotlib.pyplot as plt

    cx, cy, cz = grid_centers
    a = float(np.linalg.norm(al_bulk.cell[0]))  # lattice parameter in Å
    cx_ang = np.array(cx) * a
    cy_ang = np.array(cy) * a

    z_fracs = [0.25, 0.50, 0.75]
    z_idx = [np.argmin(np.abs(np.array(cz) - z)) for z in z_fracs]
    z_labels = ["z ≈ 0.25 (T-site plane)", "z ≈ 0.50 (O-site plane)", "z ≈ 0.75"]

    F_max = np.nanpercentile(free_energy, 95)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, iz, label in zip(axes, z_idx, z_labels):
        F_slice = free_energy[:, :, iz].T  # (n_y, n_x) for contourf
        im = ax.contourf(cx_ang, cy_ang, F_slice, levels=20, cmap="viridis", vmax=F_max)
        ax.contour(
            cx_ang, cy_ang, F_slice, levels=10, colors="k", linewidths=0.3, alpha=0.5
        )
        plt.colorbar(im, ax=ax, label="F (eV)")
        ax.set_title(label)
        ax.set_xlabel("x (Å)")
        ax.set_ylabel("y (Å)")
        ax.set_aspect("equal")
    fig.suptitle(
        "Free energy surface F(x, y) at fixed z (symmetry-augmented trajectory)"
    )
    fig.tight_layout()
    figure = fig
    return figure


# ── ASE MD ───────────────────────────────────────────────────────────────────


@as_function_node
def RunASEMD(structure, engine, md_input, store: bool = False):
    """Run NVT MD with a Langevin thermostat via ASE.

    Parameters mirror InputCalcMD: temperature (K), n_ionic_steps, n_print,
    time_step (fs), temperature_damping_timescale (fs).

    Returns an OutputCalcMD-compatible dataclass with positions,
    unwrapped_positions, cells, energies_pot, temperatures, steps, and species.
    Unwrapped positions are integrated step-by-step using the minimum-image
    convention so MSD and free-energy analysis work correctly across PBC.
    """
    import numpy as np
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from pyiron_nodes.atomistic.structure._atoms import to_ase
    from pyiron_nodes.atomistic.calculator.data import OutputCalcMD

    atoms = to_ase(structure)
    atoms.calc = engine.calculator

    T = md_input.temperature
    dt_fs = md_input.time_step
    n_steps = md_input.n_ionic_steps
    n_print = md_input.n_print
    tau_fs = md_input.temperature_damping_timescale or 100.0

    MaxwellBoltzmannDistribution(atoms, temperature_K=T, rng=None)

    dyn = Langevin(
        atoms,
        timestep=dt_fs * units.fs,
        temperature_K=T,
        friction=1.0 / (tau_fs * units.fs),
    )

    cell_mat = atoms.get_cell()[:]
    inv_cell = np.linalg.inv(cell_mat)

    buf_pos, buf_uw, buf_cells = [], [], []
    buf_epot, buf_temp, buf_steps = [], [], []
    species = atoms.get_chemical_symbols()
    n_atoms = len(atoms)

    state = {
        "prev": atoms.get_positions().copy(),
        "uw": atoms.get_positions().copy(),
        "step": 0,
    }

    def _record():
        cur = atoms.get_positions()
        delta = cur - state["prev"]
        frac_d = delta @ inv_cell
        frac_d -= np.round(frac_d)
        state["uw"] = state["uw"] + frac_d @ cell_mat
        state["prev"] = cur.copy()

        buf_pos.append(cur.copy())
        buf_uw.append(state["uw"].copy())
        buf_cells.append(atoms.get_cell()[:].copy())
        buf_epot.append(atoms.get_potential_energy())
        buf_temp.append(atoms.get_temperature())
        buf_steps.append(state["step"])
        state["step"] += n_print

    dyn.attach(_record, interval=n_print)
    dyn.run(n_steps)

    md_output = OutputCalcMD.pure_dataclass(
        species=species,
        positions=np.array(buf_pos),
        unwrapped_positions=np.array(buf_uw),
        cells=np.array(buf_cells),
        energies_pot=np.array(buf_epot),
        temperatures=np.array(buf_temp),
        steps=np.array(buf_steps),
        natoms=np.full(len(buf_steps), n_atoms),
    )
    return md_output


# ── NEB ──────────────────────────────────────────────────────────────────────


@as_out_dataclass_node
class NEBTrajectory:
    species: DataArray = EmptyArrayField()
    positions: DataArray = EmptyArrayField()
    cells: DataArray = EmptyArrayField()


@as_function_node
def LammpsAseEngine(potential, resource_path=None, cores: int = 1):
    """Wrap a pyiron LAMMPS potential as an ASE LAMMPS calculator (OutputEngine)."""
    import os
    from lammpsparser.compatibility.file import _get_potential
    from ase.calculators.lammpsrun import LAMMPS
    from pyiron_nodes.atomistic.engine.generic import OutputEngine

    pot_lines, pot_replace, species = _get_potential(
        potential=potential, resource_path=resource_path
    )

    pair_style = None
    pair_coeff = []
    for line in pot_lines:
        line = line.strip()
        if line.startswith("pair_style"):
            pair_style = " ".join(line.split()[1:])
        elif line.startswith("pair_coeff"):
            pair_coeff.append(" ".join(line.split()[1:]))

    if pair_style is None:
        raise ValueError("Could not find pair_style in potential lines.")
    if not pair_coeff:
        raise ValueError("Could not find pair_coeff in potential lines.")

    lmp_cmd = os.getenv(
        "LAMMPS_COMMAND",
        os.getenv(
            "ASE_LAMMPSRUN_COMMAND", f"mpiexec -n {cores} --oversubscribe lmp_mpi"
        ),
    )
    os.environ["ASE_LAMMPSRUN_COMMAND"] = lmp_cmd

    calc = LAMMPS(
        specorder=species,
        pair_style=pair_style,
        pair_coeff=pair_coeff,
        keep_tmp_files=False,
        keep_alive=True,
    )

    engine = OutputEngine(calculator=calc, engine_id=0)
    return engine


@as_function_node
def RunNEB(
    initial_state,
    final_state,
    engine,
    n_images: int = 7,
    fmax: float = 0.05,
    max_steps: int = 200,
    store: bool = False,
):
    """Run a fixed-endpoint NEB between two relaxed end states.

    Returns
    -------
    path_energies : np.ndarray
        Image energies relative to initial state (n_images + 2 values), eV.
    barrier : float
        Forward activation barrier (max of path_energies), eV.
    trajectory : object
        Trajectory-like object (species, positions, cells) compatible with the
        ``Animate`` node — one frame per NEB image including the two fixed
        endpoints.
    """
    from ase.mep.neb import SingleCalculatorNEB
    from ase.optimize import LBFGS
    from pyiron_nodes.atomistic.structure._atoms import to_ase
    import numpy as np

    initial_atoms = to_ase(initial_state.structure)
    final_atoms = to_ase(final_state.structure)

    images = (
        [initial_atoms.copy()]
        + [initial_atoms.copy() for _ in range(n_images)]
        + [final_atoms.copy()]
    )

    calc = engine.calculator
    for img in images:
        img.calc = calc

    neb = SingleCalculatorNEB(images)
    neb.interpolate("idpp")

    opt = LBFGS(neb, logfile="/dev/null")
    opt.run(fmax=fmax, steps=max_steps)

    # Re-evaluate all images with the same calculator so energies are on a
    # consistent scale (avoids Relax vs NEB reference mismatch).
    image_energies = [img.get_potential_energy() for img in images]

    path_energies = np.array(image_energies) - image_energies[0]
    barrier = float(np.max(path_energies))

    trajectory = NEBTrajectory.pure_dataclass(
        species=images[0].get_chemical_symbols(),
        positions=np.array([img.positions for img in images]),
        cells=np.array([img.get_cell()[:] for img in images]),
    )
    return path_energies, barrier, trajectory


@as_function_node
def PlotNEBPath(path_energies, barrier):
    """Plot the NEB path energies with the reaction coordinate normalised to [0, 1]."""
    import numpy as np
    import matplotlib.pyplot as plt

    n = len(path_energies)
    image_idx = np.arange(n) / (n - 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        image_idx,
        path_energies,
        "o-",
        color="steelblue",
        lw=2.0,
        markersize=7,
        markerfacecolor="white",
        markeredgewidth=2,
    )
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Reaction coordinate  (0 = initial, 1 = final)")
    ax.set_ylabel("Energy relative to initial state (eV)")
    ax.set_title(f"NEB path — barrier = {barrier:.3f} eV")
    ax.set_xticks(image_idx)
    fig.tight_layout()
    figure = fig
    return figure
