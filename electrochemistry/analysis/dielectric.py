"""Dielectric response and capacitance of the confined electrolyte.

Two routes to the perpendicular dielectric profile ε⊥(z) of the water film:

* :class:`LocalDielectricProfile` — the ratio D(z)/(ε₀E(z)) of the averaged
  density profile.  Cheap, reads only ``density_ctx``, but needs a charged
  electrode to have a field to respond to.
* :class:`DielectricFromPolarizationFluctuations` — the Ballenegger–Hansen
  fluctuation formula.  Works at zero applied field, but needs the full
  trajectory and is noisier.

and the differential capacitance from the fluctuations of the potential drop,
which is the constant-charge-ensemble counterpart of the integral capacitance
that ``plots.DoublLayerCapacitance`` reports.

The private helpers at the top are shared with
``plots.PlotBulkElectricField``; they are plain functions, not nodes, so the
nodes can import them into their own namespace.
"""

from __future__ import annotations

from core import as_function_node
from matplotlib.figure import Figure

# Physical constants, SI
EPSILON_0 = 8.854187817e-12  # F/m
E_CHARGE = 1.602176634e-19  # C
K_BOLTZMANN = 1.380649e-23  # J/K
ANG_TO_M = 1e-10


def _charge_map(charges: dict) -> dict:
    """Species -> partial charge, for every species the cell can contain."""
    metal = charges["metal"]
    return {
        charges["cation"]: charges["cation_charge"],
        charges["anion"]: charges["anion_charge"],
        "O": charges["O_charge"],
        "H": charges["H_charge"],
        metal: charges["metal_charge"],
        "Ne": charges["neon_charge"],
    }


def _free_bound_maps(charges: dict) -> tuple[dict, dict]:
    """Split the charge map into free (electrodes, ions) and bound (water).

    The split is what makes a dielectric constant meaningful here: the water is
    the dielectric medium, so its charge is the bound charge that screens the
    free charge sitting on the electrodes and the dissolved ions.
    """
    metal = charges["metal"]
    free = {
        metal: charges["metal_charge"],
        "Ne": charges["neon_charge"],
        charges["cation"]: charges["cation_charge"],
        charges["anion"]: charges["anion_charge"],
    }
    bound = {"O": charges["O_charge"], "H": charges["H_charge"]}
    return free, bound


def _frame_charge_density(
    positions,
    species_array,
    charge_map: dict,
    z_min: float,
    z_max: float,
    n_bins: int,
    area_ang2: float,
):
    """Bin the per-frame charge density along z.

    Parameters
    ----------
    positions:
        ``(n_frames, n_atoms, 3)`` array, already sliced to the frames wanted.
    species_array:
        ``(n_atoms,)`` array of chemical symbols.
    charge_map:
        Species -> partial charge; species absent from the cell are skipped.

    Returns
    -------
    bin_centers:
        ``(n_bins,)`` bin centres in Å.
    rho:
        ``(n_frames, n_bins)`` charge density in e/Å³.
    dz:
        Bin width in Å.
    """
    import numpy as np

    positions = np.asarray(positions)
    n_frames = positions.shape[0]

    bin_edges = np.linspace(z_min, z_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    dz = float(bin_edges[1] - bin_edges[0])
    v_bin = area_ang2 * dz  # Å³

    rho = np.zeros((n_frames, n_bins))
    for el, q in charge_map.items():
        idx = np.where(species_array == el)[0]
        if len(idx) == 0 or q == 0.0:
            continue
        z_el = positions[:, idx, 2]  # (n_frames, n_el)
        bin_idx = np.clip(np.searchsorted(bin_edges[1:], z_el), 0, n_bins - 1)
        frame_idx = np.broadcast_to(np.arange(n_frames)[:, None], z_el.shape)
        np.add.at(rho.ravel(), (frame_idx * n_bins + bin_idx).ravel(), q / v_bin)

    return bin_centers, rho, dz


def _poisson_potential(rho, dz_ang: float):
    """Integrate Gauss's law twice along the last axis.

    ``rho`` is a charge density in e/Å³ (any leading shape).  Returns the field
    ``E`` in V/Å and the potential ``psi`` in V, using the physical convention

        E(z)  = (1/ε₀) ∫ ρ dz'
        Ψ(z)  = −∫ E dz'
    """
    import numpy as np

    rho_c = np.asarray(rho) * E_CHARGE / ANG_TO_M**3  # C/m³
    dz_m = dz_ang * ANG_TO_M
    e_field = np.cumsum(rho_c * dz_m, axis=-1) / EPSILON_0  # V/m
    psi = -np.cumsum(e_field * dz_m, axis=-1)  # V
    return e_field * ANG_TO_M, psi  # V/Å, V


def _electrode_span(density_data: dict, metal: str, threshold: float = 0.01):
    """Locate the electrolyte-facing planes of the two electrodes.

    Returns ``(z_metal_surface, z_neon)`` on the z grid of ``density_data``:
    the topmost metal bin still carrying density, and the peak of the Ne layer.
    """
    import numpy as np

    z = np.asarray(density_data[metal][0])
    metal_density = np.asarray(density_data[metal][1])
    ne_density = np.asarray(density_data["Ne"][1])

    metal_bins = np.where(metal_density > threshold * metal_density.max())[0]
    z_metal = float(z[metal_bins[-1]])
    z_neon = float(z[int(np.argmax(ne_density))])
    return z_metal, z_neon


def _window_mean(values, window) -> float:
    """Mean over a boolean window, NaN when it holds nothing usable.

    ``eps_clip`` can mask out every bin in the window, and ``np.nanmean`` warns
    on an all-NaN slice rather than simply returning NaN.
    """
    import numpy as np

    if not np.any(window):
        return float("nan")
    selected = np.asarray(values)[window]
    if not np.any(np.isfinite(selected)):
        return float("nan")
    return float(np.nanmean(selected))


def _resolve_initial_step(context: dict, initial_step) -> int:
    """``initial_step=None`` means 'whatever the density context used'."""
    if initial_step is not None:
        return int(initial_step)
    return int(context.get("initial_step", 0))


@as_function_node
def LocalDielectricProfile(
    context: dict,
    water_fraction_lo: float = 0.35,
    water_fraction_hi: float = 0.65,
    eps_clip: float = 1000.0,
):
    """Perpendicular dielectric profile ε⊥(z) from the averaged density profile.

    Splitting the charge into free (electrodes, ions) and bound (water), Gauss's
    law gives the displacement field from the free charge alone and the true
    field from all of it:

        D(z)   = ∫ ρ_free dz'
        E(z)   = (1/ε₀) ∫ (ρ_free + ρ_bound) dz'
        ε⊥(z)  = D(z) / (ε₀ E(z))

    which collapses to a ratio of the two cumulative integrals, so the unit
    conversions cancel.

    This is a linear-response result: it needs a field to respond to, hence a
    non-zero electrode charge.  For an uncharged cell use
    :class:`DielectricFromPolarizationFluctuations`.

    Parameters
    ----------
    context:
        Bundle from ``BuildDensityContext`` / ``EnrichDensityContext``.
    water_fraction_lo, water_fraction_hi:
        Bulk-water window, as fractions of the metal-surface-to-Ne span.  The
        reported mean is taken over this window, away from the interfacial
        layers where ε⊥ is strongly structured.
    eps_clip:
        Values whose magnitude exceeds this are set to NaN.  ε⊥ has a genuine
        pole wherever E(z) crosses zero, which is a property of the inverse
        profile and not a numerical failure, but it ruins a plot's y-range.

    Returns
    -------
    z:
        Bin centres in Å, measured from the metal surface.
    eps_profile:
        ε⊥(z), with poles masked to NaN.
    eps_water:
        Mean of ε⊥ over the bulk-water window.
    """
    import numpy as np

    Data = context["density_data"]
    initial_structure = context["initial_structure"]
    charges = context["charges"]
    metal = charges["metal"]

    if charges["metal_charge"] == 0.0 and charges["neon_charge"] == 0.0:
        raise ValueError(
            "LocalDielectricProfile needs a charged electrode to produce a "
            "field; both metal_charge and neon_charge are 0.  Use "
            "DielectricFromPolarizationFluctuations for an uncharged cell."
        )

    free_map, bound_map = _free_bound_maps(charges)

    z = np.asarray(Data[metal][0], dtype=float)
    dz = float(z[1] - z[0]) if len(z) > 1 else 0.2
    area_ang2 = float(initial_structure.cell[0, 0] * initial_structure.cell[1, 1])
    v_bin = area_ang2 * dz

    def _density(species_map):
        total = np.zeros_like(z)
        for el, q in species_map.items():
            if el in Data and q != 0.0:
                total = total + q * np.asarray(Data[el][1], dtype=float)
        return total / v_bin  # e/Å³

    rho_free = _density(free_map)
    rho_total = rho_free + _density(bound_map)

    # Both integrals carry the same prefactor, so the ratio is already ε.
    d_field = np.cumsum(rho_free * dz)
    e_field = np.cumsum(rho_total * dz)

    with np.errstate(divide="ignore", invalid="ignore"):
        eps_profile = np.where(e_field != 0.0, d_field / e_field, np.nan)
    eps_profile = np.where(np.abs(eps_profile) > eps_clip, np.nan, eps_profile)

    z_metal, z_neon = _electrode_span(Data, metal)
    span = z_neon - z_metal
    window = (z >= z_metal + water_fraction_lo * span) & (
        z <= z_metal + water_fraction_hi * span
    )
    eps_water = _window_mean(eps_profile, window)

    return z, eps_profile, eps_water


@as_function_node
def DielectricFromPolarizationFluctuations(
    trajectory,
    context: dict,
    n_bins: int = 200,
    initial_step: int = None,
    temperature: float = 300.0,
    water_fraction_lo: float = 0.35,
    water_fraction_hi: float = 0.65,
    eps_clip: float = 1000.0,
):
    """ε⊥(z) from water polarization fluctuations (Ballenegger–Hansen).

    Linear response of the local polarization to an applied field, evaluated as
    an equilibrium fluctuation instead of by applying the field:

        P_z(z, t) = −∫ ρ_water(z', t) dz'      (polarization, C/m²)
        M_z(t)    = Σ_i q_i z_i   over water   (cell dipole, C·m)
        1/ε⊥(z)   = 1 − ⟨δP_z(z) δM_z⟩ / (ε₀ k_B T)

    Unlike :class:`LocalDielectricProfile` this needs no applied field, so it
    also works for an uncharged cell — but it converges slowly, and a short run
    will give a noisy profile.  Running both and comparing is the cheapest
    available check on either.

    Parameters
    ----------
    trajectory:
        ``OutputCalcMD`` from ``ParseLammpsOutput``.
    context:
        Bundle from ``EnrichDensityContext``; supplies the structure, the
        charges, and the default ``initial_step``.
    initial_step:
        First frame to use.  ``None`` takes the value the density context was
        built with, so the two averages cover the same frames.
    temperature:
        Temperature of the run in K, used for the k_BT prefactor.

    Returns
    -------
    z:
        Bin centres in Å, measured from the metal surface.
    eps_profile:
        ε⊥(z), with poles masked to NaN.
    eps_water:
        Mean of ε⊥ over the bulk-water window.
    """
    import numpy as np

    initial_structure = context["initial_structure"]
    charges = context["charges"]
    metal = charges["metal"]
    start = _resolve_initial_step(context, initial_step)

    species_array = np.asarray(trajectory.species)
    positions = np.asarray(trajectory.positions)[start:]
    if positions.shape[0] < 2:
        raise ValueError(
            f"need at least 2 frames for a fluctuation average, got "
            f"{positions.shape[0]} after dropping {start} frames"
        )

    area_ang2 = float(initial_structure.cell[0, 0] * initial_structure.cell[1, 1])
    _, bound_map = _free_bound_maps(charges)

    ind_metal = np.where(species_array == metal)[0]
    ind_ne = np.where(species_array == "Ne")[0]
    first = positions[0]
    z_metal = float(np.max(first[ind_metal, 2]))
    z_neon = float(np.max(first[ind_ne, 2]))

    bin_centers, rho_water, dz = _frame_charge_density(
        positions=positions,
        species_array=species_array,
        charge_map=bound_map,
        z_min=z_metal,
        z_max=z_neon,
        n_bins=n_bins,
        area_ang2=area_ang2,
    )

    # Polarization: P_z(z) = -∫ rho_water dz', in e/Å² then C/m².
    polarization = -np.cumsum(rho_water * dz, axis=1) * E_CHARGE / ANG_TO_M**2

    # Water dipole of each frame, in e·Å then C·m.
    water_charges = np.zeros(len(species_array))
    for el, q in bound_map.items():
        water_charges[species_array == el] = q
    dipole = positions[:, :, 2] @ water_charges * E_CHARGE * ANG_TO_M

    d_polarization = polarization - polarization.mean(axis=0)
    d_dipole = dipole - dipole.mean()
    covariance = (d_polarization * d_dipole[:, None]).mean(axis=0)

    inv_eps = 1.0 - covariance / (EPSILON_0 * K_BOLTZMANN * temperature)
    with np.errstate(divide="ignore", invalid="ignore"):
        eps_profile = np.where(inv_eps != 0.0, 1.0 / inv_eps, np.nan)
    eps_profile = np.where(np.abs(eps_profile) > eps_clip, np.nan, eps_profile)

    z = bin_centers - z_metal
    span = z_neon - z_metal
    window = (z >= water_fraction_lo * span) & (z <= water_fraction_hi * span)
    eps_water = _window_mean(eps_profile, window)

    return z, eps_profile, eps_water


@as_function_node
def DifferentialCapacitance(
    trajectory,
    context: dict,
    n_bins: int = 200,
    initial_step: int = None,
    temperature: float = 300.0,
    bulk_fraction_lo: float = 0.35,
    bulk_fraction_hi: float = 0.65,
):
    """Differential capacitance of both electrodes from potential fluctuations.

    An MD run at fixed electrode charge samples the conjugate ensemble to a
    fixed-potential one, so the differential capacitance follows from how much
    the potential drop fluctuates rather than from a derivative taken over a
    series of runs:

        C_diff = k_B T / (A · Var(ΔΨ))

    (Limmer, Merlet, Salanne, Chandler, Madden, Van Roij, Rotenberg, PRL 111,
    106102.)  ``plots.DoublLayerCapacitance`` reports the *integral*
    capacitance σ/ΔV from the same cell; the two agree only where C is
    independent of potential.

    Each frame is binned, Poisson-integrated to Ψ(z), and the drop from each
    electrode plane to the bulk electrolyte is recorded.  The electrode planes
    are tracked per frame rather than taken from frame 0, because under a piston
    barostat the Ne electrode moves.

    Parameters
    ----------
    trajectory:
        ``OutputCalcMD`` from ``ParseLammpsOutput``.
    context:
        Bundle from ``EnrichDensityContext``.
    initial_step:
        First frame to use.  ``None`` takes the density context's value.
    temperature:
        Temperature of the run in K.
    bulk_fraction_lo, bulk_fraction_hi:
        Bulk window used as the potential reference, as fractions of the
        electrode-to-electrode span.

    Returns
    -------
    C_metal:
        Differential capacitance of the metal electrode in µF/cm².
    C_Ne:
        Differential capacitance of the Ne electrode in µF/cm².
    potential_drops:
        ``{"steps": ..., metal: ΔΨ(t), "Ne": ΔΨ(t)}`` in V, so the distribution
        behind the variance can be inspected.
    """
    import numpy as np

    initial_structure = context["initial_structure"]
    charges = context["charges"]
    metal = charges["metal"]
    start = _resolve_initial_step(context, initial_step)

    species_array = np.asarray(trajectory.species)
    positions = np.asarray(trajectory.positions)[start:]
    steps = np.asarray(trajectory.steps)[start:]
    n_frames = positions.shape[0]
    if n_frames < 2:
        raise ValueError(
            f"need at least 2 frames for a fluctuation average, got {n_frames} "
            f"after dropping {start} frames"
        )

    area_ang2 = float(initial_structure.cell[0, 0] * initial_structure.cell[1, 1])
    area_m2 = area_ang2 * ANG_TO_M**2

    ind_metal = np.where(species_array == metal)[0]
    ind_ne = np.where(species_array == "Ne")[0]

    # Per-frame electrode planes: the electrolyte-facing metal surface and the
    # Ne layer, which the piston moves.
    z_metal_t = positions[:, ind_metal, 2].max(axis=1)
    z_neon_t = positions[:, ind_ne, 2].max(axis=1)

    # One fixed grid spanning every frame's electrode positions, so a moving
    # piston never falls outside it.
    z_min = float(z_metal_t.min())
    z_max = float(z_neon_t.max())
    bin_centers, rho, dz = _frame_charge_density(
        positions=positions,
        species_array=species_array,
        charge_map=_charge_map(charges),
        z_min=z_min,
        z_max=z_max,
        n_bins=n_bins,
        area_ang2=area_ang2,
    )
    _, psi = _poisson_potential(rho, dz)  # (n_frames, n_bins), V

    frames = np.arange(n_frames)
    metal_bin = np.clip(np.searchsorted(bin_centers, z_metal_t), 0, n_bins - 1)
    neon_bin = np.clip(np.searchsorted(bin_centers, z_neon_t), 0, n_bins - 1)

    span = z_neon_t - z_metal_t
    bulk_lo = z_metal_t + bulk_fraction_lo * span
    bulk_hi = z_metal_t + bulk_fraction_hi * span
    bulk_mask = (bin_centers[None, :] >= bulk_lo[:, None]) & (
        bin_centers[None, :] <= bulk_hi[:, None]
    )
    if not bulk_mask.any():
        raise ValueError(
            "the bulk window contains no bins; widen bulk_fraction_lo/hi or "
            "raise n_bins"
        )
    psi_bulk = np.where(bulk_mask, psi, np.nan)
    psi_bulk = np.nanmean(psi_bulk, axis=1)

    drop_metal = psi[frames, metal_bin] - psi_bulk
    drop_neon = psi[frames, neon_bin] - psi_bulk

    kT = K_BOLTZMANN * temperature

    def _capacitance(drop):
        variance = float(np.var(drop))
        # A frozen or single-configuration run has no fluctuation to measure.
        # Compare against the float resolution of ΔΨ itself: cancelling two
        # large potentials leaves rounding noise, not a physical variance.
        noise_floor = np.finfo(float).eps * max(float(np.mean(drop)) ** 2, 1.0)
        if not np.isfinite(variance) or variance <= noise_floor:
            return float("nan")
        # J / (m² V²) = F/m²;  1 F/m² = 100 µF/cm²
        return float(kT / (area_m2 * variance) * 100.0)

    C_metal = _capacitance(drop_metal)
    C_Ne = _capacitance(drop_neon)
    potential_drops = {"steps": steps, metal: drop_metal, "Ne": drop_neon}

    return C_metal, C_Ne, potential_drops


@as_function_node
def PlotDielectricProfile(
    z,
    eps_profile,
    eps_water: float = float("nan"),
    bulk_reference: float = 97.0,
    xlabel: str = "z (Å)",
    ylabel: str = "ε⊥(z)",
) -> Figure:
    """Plot the perpendicular dielectric profile.

    ``bulk_reference`` defaults to 97, the bulk dielectric constant of TIP3P
    water — the model's own value, which is well below the experimental 78 for
    SPC/E-like comparisons and worth having on the axes.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    z = np.asarray(z)
    eps_profile = np.asarray(eps_profile)

    fig, ax = plt.subplots()
    ax.plot(z, eps_profile, color="#3B82F6", linewidth=2, label="ε⊥(z)")

    if bulk_reference:
        ax.axhline(
            bulk_reference,
            color="#6B7280",
            linewidth=1.2,
            linestyle="--",
            label=f"bulk TIP3P ({bulk_reference:g})",
        )
    if np.isfinite(eps_water):
        ax.axhline(
            eps_water,
            color="#EF4444",
            linewidth=1.2,
            linestyle=":",
            label=f"water layer mean ({eps_water:.1f})",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    ax.yaxis.grid(True, color="#E5E7EB", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    return fig


@as_function_node
def PlotPotentialDropHistogram(
    potential_drops: dict,
    n_bins: int = 40,
    xlabel: str = "ΔΨ (V)",
    ylabel: str = "probability density",
) -> Figure:
    """Plot the ΔΨ distributions whose variance sets the differential capacitance.

    The fluctuation formula assumes Gaussian statistics, so a visibly skewed or
    bimodal histogram means ``DifferentialCapacitance`` is not applicable to
    that run yet — usually a sign of too little sampling.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    colors = ["#3B82F6", "#EF4444", "#10B981", "#F59E0B"]
    fig, ax = plt.subplots()

    series = {k: v for k, v in potential_drops.items() if k != "steps"}
    for color, (label, drop) in zip(colors, series.items()):
        drop = np.asarray(drop)
        ax.hist(
            drop,
            bins=n_bins,
            density=True,
            histtype="stepfilled",
            alpha=0.35,
            color=color,
        )
        # Gaussian with the same mean and variance — the formula's assumption.
        grid = np.linspace(drop.min(), drop.max(), 200)
        sigma = drop.std()
        if sigma > 0:
            gaussian = np.exp(-0.5 * ((grid - drop.mean()) / sigma) ** 2) / (
                sigma * np.sqrt(2 * np.pi)
            )
            ax.plot(grid, gaussian, color=color, linewidth=2, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    ax.yaxis.grid(True, color="#E5E7EB", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    return fig
