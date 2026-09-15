from __future__ import annotations
from typing import Literal

from core import as_function_node
from matplotlib.figure import Figure


@as_function_node("Ion_density")
def ElementDensityFromTrajectory(trajectory, metal: str = "Al", initial_step: int = 0):
    import numpy as np

    species_array = np.asarray(trajectory.species)
    ind_metal = np.where(species_array == metal)[0]
    ind_Ne = np.where(species_array == "Ne")[0]
    ind_Na = np.where(species_array == "Na")[0]
    ind_F = np.where(species_array == "F")[0]
    ind_O = np.where(species_array == "O")[0]
    ind_H = np.where(species_array == "H")[0]
    positions = trajectory.positions[initial_step:]
    first_frame = positions[0]
    slab_bot = np.max(first_frame[ind_metal, 2])
    slab_top = np.max(first_frame[ind_Ne, 2])
    # extend bins to cover the full electrode thickness (below slab_bot) and a
    # small vacuum margin above the Ne layer so neither electrode is clipped
    electrode_bottom = np.min(first_frame[ind_metal, 2])
    deltares = 0.2
    margin = 3 * deltares
    z_min = electrode_bottom - slab_bot - margin
    z_max = slab_top - slab_bot + margin
    Data = {}
    elements = ["Na", "F", "O", "H", metal, "Ne"]
    indices = [ind_Na, ind_F, ind_O, ind_H, ind_metal, ind_Ne]
    for element, ind_el in zip(elements, indices):
        z_el = np.array([snapshot[ind_el, 2] for snapshot in positions])
        z_d = z_el - slab_bot
        binedges = np.arange(z_min, z_max, deltares)
        hist, bin_edges = np.histogram(z_d, bins=binedges)
        Data[element] = bin_edges[:-1], hist / np.shape(positions)[0]
    return Data


@as_function_node
def BuildDensityContext(density_data: dict, initial_structure, charges: dict):
    """Bundle density data, structure, and charges into a single context dict.

    Downstream plot nodes each accept this single object instead of three
    separate input ports, keeping the workflow graph free of long-range edges.
    Keys: 'density_data', 'initial_structure', 'charges'.
    """
    context = {
        "density_data": density_data,
        "initial_structure": initial_structure,
        "charges": charges,
    }
    return context


@as_function_node
def PlotWaterDensity(
    context: dict,
    xlabel: str = "z (Å)",
    ylabel: str = "Water density (g cm⁻³)",
) -> Figure:
    import numpy as np
    import matplotlib.pyplot as plt

    Data = context["density_data"]
    initial_structure = context["initial_structure"]

    z = np.array(Data["O"][0])
    counts = np.array(Data["O"][1])

    # Bin width and xy cross-section area (both in Å)
    dz = z[1] - z[0] if len(z) > 1 else 0.2
    A_ang2 = initial_structure.cell[0, 0] * initial_structure.cell[1, 1]

    # density [g/cm³]:  counts × M_water / (N_A × V_bin)
    # V_bin [cm³] = A_ang2 [Å²] × dz [Å] × 1e-24  (1 Å³ = 1e-24 cm³)
    M_water = 18.015  # g/mol
    N_A = 6.02214076e23  # mol⁻¹
    density = counts * M_water / (N_A * A_ang2 * dz * 1e-24)

    fig, ax = plt.subplots()

    # Water density — first categorical hue, 2 px line
    ax.plot(z, density, color="#3B82F6", linewidth=2, label="water")

    # Experimental reference — neutral, dashed, recessive
    ax.axhline(
        1.0, color="#6B7280", linewidth=1.2, linestyle="--", label="bulk (1 g cm⁻³)"
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)

    # Recessive grid
    ax.yaxis.grid(True, color="#E5E7EB", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    return fig


@as_function_node
def PlotChargeDensity(
    context: dict,
    plot_electrode: bool = False,
    plot_water: bool = False,
    xlabel: str = "z (Å)",
    ylabel: str = "Charge density (e/Å)",
) -> Figure:
    import numpy as np
    import matplotlib.pyplot as plt

    Data = context["density_data"]
    charges = context["charges"]
    metal = charges["metal"]

    ion_map = {
        charges["cation"]: charges["cation_charge"],
        charges["anion"]: charges["anion_charge"],
    }
    electrode_map = {
        metal: charges["metal_charge"],
        "Ne": charges["neon_charge"],
    }
    water_map = {
        "O": charges["O_charge"],
        "H": charges["H_charge"],
    }

    fig, ax = plt.subplots()

    def _plot_species(species_map, label_prefix=""):
        for el, q in species_map.items():
            if el not in Data:
                continue
            z, density = Data[el]
            ax.plot(z, q * np.array(density), label=f"{label_prefix}{el} (q={q:+.3f})")

    _plot_species(ion_map)
    if plot_electrode:
        _plot_species(electrode_map)
    if plot_water:
        _plot_species(water_map)

    # total charge density from all plotted groups
    all_map = dict(ion_map)
    if plot_electrode:
        all_map.update(electrode_map)
    if plot_water:
        all_map.update(water_map)

    if len(all_map) > 1:
        z_ref = np.array(Data[next(iter(Data))][0])
        total = sum(
            q * np.array(Data[el][1]) for el, q in all_map.items() if el in Data
        )
        ax.plot(
            z_ref, total, color="black", linewidth=1.5, linestyle="--", label="total"
        )

    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    return fig


@as_function_node
def PlotIonTrajectory(
    trajectory,
    component: Literal["x", "y", "z"] = "z",
    ion_species: str = "Na F",
    xlabel: str = "MD step",
) -> Figure:
    import numpy as np
    import matplotlib.pyplot as plt

    comp_idx = {"x": 0, "y": 1, "z": 2}[component]
    species_array = np.asarray(trajectory.species)
    positions = np.asarray(trajectory.positions)  # (n_frames, n_atoms, 3)
    steps = np.arange(positions.shape[0])

    colors = ["#3B82F6", "#EF4444", "#10B981", "#F59E0B", "#8B5CF6", "#EC4899"]
    fig, ax = plt.subplots()

    for color, sp in zip(colors, ion_species.split()):
        idx = np.where(species_array == sp)[0]
        if len(idx) == 0:
            continue
        ion_pos = positions[:, idx, comp_idx]  # (n_frames, n_ions)
        for i in range(ion_pos.shape[1]):
            ax.plot(
                steps,
                ion_pos[:, i],
                linewidth=0.6,
                linestyle="--",
                color=color,
                alpha=0.35,
            )
        mean_pos = ion_pos.mean(axis=1)
        ax.plot(steps, mean_pos, linewidth=1.8, color=color, label=sp)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"{component} position (Å)")
    ax.legend(frameon=False)
    ax.yaxis.grid(True, color="#E5E7EB", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    return fig


@as_function_node
def PlotEnergyConvergence(
    trajectory,
    first_step: int = 0,
    smooth: bool = False,
    plot_total: bool = True,
    plot_potential: bool = True,
    xlabel: str = "MD step",
    ylabel: str = "Energy (eV)",
) -> Figure:
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import uniform_filter1d

    steps = np.asarray(trajectory.steps)[first_step:]

    def _prepare(arr):
        a = np.asarray(arr)[first_step:]
        if smooth:
            window = max(3, len(a) // 20)
            a = uniform_filter1d(a, size=window)
        return a

    fig, ax = plt.subplots()

    if plot_total:
        ax.plot(
            steps,
            _prepare(trajectory.energies_tot),
            color="#3B82F6",
            linewidth=1.5,
            label="total",
        )
    if plot_potential:
        ax.plot(
            steps,
            _prepare(trajectory.energies_pot),
            color="#EF4444",
            linewidth=1.5,
            label="potential",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if plot_total or plot_potential:
        ax.legend(frameon=False)
    ax.yaxis.grid(True, color="#E5E7EB", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    return fig


@as_function_node
def PlotElectrostaticPotential(
    context: dict,
    dipole_correction: bool = False,
    xlabel: str = "z (Å)",
    ylabel: str = "Electrostatic potential (V)",
) -> Figure:
    import numpy as np
    import matplotlib.pyplot as plt
    from ase.units import Bohr

    Data = context["density_data"]
    initial_structure = context["initial_structure"]
    charges = context["charges"]

    z = np.array(Data["Na"][0])

    metal = charges["metal"]
    charge_map = {
        charges["cation"]: charges["cation_charge"],
        charges["anion"]: charges["anion_charge"],
        "O": charges["O_charge"],
        "H": charges["H_charge"],
        metal: charges["metal_charge"],
        "Ne": charges["neon_charge"],
    }
    rho_e = sum(
        charge * np.array(Data[el][1])
        for el, charge in charge_map.items()
        if el in Data
    )

    delta_z = np.gradient(z)[0]
    vol_bohr = (
        initial_structure.cell[0, 0]
        * initial_structure.cell[1, 1]
        * delta_z
        * (1 / Bohr) ** 3
    )
    rho_e = rho_e / vol_bohr

    epsilon_0 = 8.854187817e-12
    e_charge = 1.602176634e-19
    angstrom_to_meter = 1e-10
    rho_c = -rho_e * e_charge * (1 / Bohr / angstrom_to_meter) ** 3
    dz_m = np.gradient(z * angstrom_to_meter)
    E = (1 / epsilon_0) * np.cumsum(rho_c * dz_m)
    V = np.cumsum(E * dz_m)

    if dipole_correction:
        n = max(len(z) // 10, 1)
        V_left = np.mean(V[:n])
        V_right = np.mean(V[-n:])
        ramp = V_left + (V_right - V_left) * np.arange(len(z)) / (len(z) - 1)
        V = V - ramp

    fig, ax = plt.subplots()
    ax.plot(z, V)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig


@as_function_node
def PlotBulkElectricField(
    trajectory,
    context: dict,
    bulk_fraction_lo: float = 0.35,
    bulk_fraction_hi: float = 0.65,
    n_bins: int = 200,
    initial_step: int = 0,
    smooth: bool = True,
    xlabel: str = "MD step",
    ylabel: str = "Mean electric field (V/Å)",
) -> Figure:
    """Compute and plot the time evolution of the mean electric field in the
    bulk-like electrolyte region.

    Concept
    -------
    In the planar (quasi-2D) geometry of an electrochemical cell the
    electrostatic potential V(z) obeys the 1-D Poisson equation:

        d²V/dz² = −ρ_free(z) / ε₀

    Integrating once from the bottom electrode surface gives the z-component
    of the electric field:

        E(z) = −dV/dz = (1/ε₀) ∫_{z_ref}^{z} ρ_free(z') dz'

    At every stored MD frame this function:

      1. Bins all charged species (ions, water, electrodes) along z into
         *n_bins* uniform slabs and computes the volumetric charge density
         ρ(z, t) = Σ_i q_i · n_i(z, t) / (A_xy · dz).
      2. Integrates ρ numerically via a cumulative sum (Gauss / Poisson) to
         obtain E(z, t).
      3. Averages E(z, t) over the *bulk electrolyte region*, defined as the
         relative span [bulk_fraction_lo, bulk_fraction_hi] of the distance
         between the Al electrode surface (z = z_bot) and the Ne pseudo-
         electrode layer (z = z_top).

    The bulk window is chosen because the double-layer contribution vanishes
    there at equilibrium.  A non-zero or drifting E_bulk therefore signals
    incomplete equilibration, residual charge imbalance, or growing ion
    concentration gradients — all physically meaningful indicators that can
    be read off this time series.

    Assumptions
    -----------
    * Orthogonal simulation box; only the (0,0) and (1,1) cell entries are
      used for the cross-section area A_xy.
    * Electrode species are `charges['metal']` (e.g. Al, bottom) and Ne
      (top); their outermost z-positions in frame 0 fix the electrolyte
      window that is kept constant across all frames.
    * Point-charge model: each atom contributes its partial charge q_i to
      the bin it occupies.
    * Periodic boundary conditions along z are *not* applied; the Poisson
      integration starts at z_bot and runs toward z_top.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    initial_structure = context["initial_structure"]
    charges = context["charges"]

    species_array = np.asarray(trajectory.species)
    positions = np.asarray(trajectory.positions)[initial_step:]   # (n_frames, n_atoms, 3)
    steps = np.asarray(trajectory.steps)[initial_step:]
    n_frames = positions.shape[0]

    A_xy = initial_structure.cell[0, 0] * initial_structure.cell[1, 1]  # Å²

    ind_metal = np.where(species_array == charges["metal"])[0]
    ind_Ne = np.where(species_array == "Ne")[0]

    first = positions[0]
    z_bot = float(np.max(first[ind_metal, 2]))
    z_top = float(np.max(first[ind_Ne, 2]))

    charge_map = {
        charges["cation"]: charges["cation_charge"],
        charges["anion"]: charges["anion_charge"],
        "O": charges["O_charge"],
        "H": charges["H_charge"],
        charges["metal"]: charges["metal_charge"],
        "Ne": charges["neon_charge"],
    }

    bin_edges = np.linspace(z_bot, z_top, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    dz = float(bin_edges[1] - bin_edges[0])
    V_bin = A_xy * dz  # Å³

    z_extent = z_top - z_bot
    z_lo = z_bot + bulk_fraction_lo * z_extent
    z_hi = z_bot + bulk_fraction_hi * z_extent
    bulk_mask = (bin_centers >= z_lo) & (bin_centers <= z_hi)

    epsilon_0 = 8.854187817e-12   # F/m
    e_charge = 1.602176634e-19    # C
    ang_to_m = 1e-10              # Å → m

    # Accumulate charge density: rho_all[t, z_bin] in e/Å³
    rho_all = np.zeros((n_frames, n_bins))
    for el, q in charge_map.items():
        idx = np.where(species_array == el)[0]
        if len(idx) == 0 or q == 0.0:
            continue
        z_el = positions[:, idx, 2]                                   # (n_frames, n_el)
        bin_idx = np.clip(
            np.searchsorted(bin_edges[1:], z_el), 0, n_bins - 1
        )                                                             # (n_frames, n_el)
        frame_idx = np.broadcast_to(
            np.arange(n_frames)[:, None], z_el.shape
        )
        np.add.at(rho_all.ravel(), (frame_idx * n_bins + bin_idx).ravel(), q / V_bin)

    # Poisson integration: E(z, t) in V/Å
    rho_c = rho_all * e_charge / ang_to_m**3                          # C/m³
    dz_m = dz * ang_to_m
    E_z = np.cumsum(rho_c * dz_m, axis=1) / epsilon_0                # V/m, shape (n_frames, n_bins)
    E_z_per_ang = E_z * ang_to_m                                      # V/Å

    E_bulk_series = np.mean(E_z_per_ang[:, bulk_mask], axis=1)        # (n_frames,)

    if smooth and n_frames > 20:
        from scipy.ndimage import uniform_filter1d
        window = max(3, n_frames // 20)
        E_smooth = uniform_filter1d(E_bulk_series, size=window)
    else:
        E_smooth = None

    fig, ax = plt.subplots()
    ax.plot(steps, E_bulk_series, color="#9CA3AF", linewidth=0.8, alpha=0.5, label="raw")
    if E_smooth is not None:
        ax.plot(steps, E_smooth, color="#3B82F6", linewidth=2.0, label="smoothed")
    ax.axhline(0.0, color="#6B7280", linewidth=1.0, linestyle="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    ax.yaxis.grid(True, color="#E5E7EB", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    return fig


@as_function_node
def DoublLayerCapacitance(context: dict):
    import numpy as np
    from ase.units import Bohr

    Data = context["density_data"]
    initial_structure = context["initial_structure"]
    charges = context["charges"]
    metal = charges["metal"]
    z = np.array(Data[metal][0])
    e_charge = 1.602176634e-19
    angstrom_to_meter = 1e-10
    charge_map = {
        charges["cation"]: charges["cation_charge"],
        charges["anion"]: charges["anion_charge"],
        "O": charges["O_charge"],
        "H": charges["H_charge"],
        metal: charges["metal_charge"],
        "Ne": charges["neon_charge"],
    }
    rho_e = sum(
        charge * np.array(Data[el][1])
        for el, charge in charge_map.items()
        if el in Data
    )
    delta_z = np.gradient(z)[0]
    A_ang2 = initial_structure.cell[0, 0] * initial_structure.cell[1, 1]
    vol_bohr = A_ang2 * delta_z * (1 / Bohr) ** 3
    rho_e_vol = rho_e / vol_bohr
    epsilon_0 = 8.854187817e-12
    rho_c = -rho_e_vol * e_charge * (1 / Bohr / angstrom_to_meter) ** 3
    dz_m = np.gradient(z * angstrom_to_meter)
    V = np.cumsum((1 / epsilon_0) * np.cumsum(rho_c * dz_m) * dz_m)
    A_m2 = A_ang2 * angstrom_to_meter**2
    metal_mask = np.array(Data[metal][1]) > 0.01 * np.max(Data[metal][1])
    ne_mask = np.array(Data["Ne"][1]) > 0.01 * np.max(Data["Ne"][1])
    n = len(z)
    bulk_mask = np.zeros(n, dtype=bool)
    bulk_mask[3 * n // 10 : 7 * n // 10] = True
    bulk_mask &= ~metal_mask & ~ne_mask
    V_bulk = np.mean(V[bulk_mask]) if np.any(bulk_mask) else 0.0

    def _dlc(electrode_mask, species, charge_per_atom):
        if not np.any(electrode_mask) or not np.any(bulk_mask):
            return float("nan")
        dV = np.mean(V[electrode_mask]) - V_bulk
        if abs(dV) < 1e-10:
            return float("nan")
        n_atoms = np.sum(Data[species][1])
        sigma = n_atoms * charge_per_atom * e_charge / A_m2
        return float(abs(sigma / dV) * 100.0)  # F/m² → μF/cm²

    C_metal = _dlc(metal_mask, metal, charges["metal_charge"])
    C_Ne = _dlc(ne_mask, "Ne", charges["neon_charge"])
    return C_metal, C_Ne
