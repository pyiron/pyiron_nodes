"""
TRIP vs. TWIP deformation-mode assessment for an Fe(1-x)X(x) austenitic steel.

The deformation mechanism of a high-manganese austenitic steel is selected by
the stacking fault energy (SFE) of the fcc (gamma) matrix:

* SFE below ~20 mJ/m^2 -- the intrinsic stacking fault is an embryo of hcp
  epsilon-martensite, the fault ribbons grow and the steel deforms by
  transformation induced plasticity (**TRIP**).
* SFE between ~20 and ~45 mJ/m^2 -- partials stay dissociated but the faulted
  region is no longer an epsilon embryo, so deformation twinning takes over
  (**TWIP**).
* SFE above ~45 mJ/m^2 -- partials recombine and ordinary dislocation glide
  with cross-slip dominates (no TRIP, no TWIP).

The SFE is obtained from the classical Olson-Cohen / Allain thermodynamic model

    SFE = 2 * rho * dG(gamma->epsilon) + 2 * sigma(gamma/epsilon)

where ``rho`` is the molar planar atom density of the {111} plane (computed
from the Vegard lattice parameter of the alloy), ``dG(gamma->epsilon)`` is the
molar Gibbs energy difference between the fcc and hcp phases from a regular
solution model, and ``sigma`` is the gamma/epsilon interfacial energy.

Because the chain is evaluated on a composition grid, the same graph answers
two questions at once: the verdict for the requested composition, and the
complete TRIP/TWIP map including the transition compositions.

## Graph layout
The alloy definition, the mechanism thresholds and the composition grid are
merged once into a **scan context** which is then threaded through the physical
steps, each of which enriches it with the quantity it computes (thermodynamic
parameters, lattice parameter, planar density, driving force, SFE, Md, mode).
Every edge therefore connects neighbouring steps, and the three answers hang off
the final context as separate single-output sinks.

## Inputs
- `alloy_definition` -- base element, solute X, solute fraction x, temperature
- `composition_grid` -- the x-range over which the map is drawn
- `mode_criteria` -- the two SFE thresholds separating TRIP / TWIP / slip
- `stacking_fault_energy` -- the gamma/epsilon interfacial energy (calibration)
- `martensite_stability` -- the mechanical driving force at yield

## Outputs
- `alloy_verdict` -- markdown report answering TRIP or TWIP for the given steel
- `mode_map_view` -- SFE, driving force, Md and mode for every composition
- `mode_map_plot` -- SFE(x) with the TRIP / TWIP / slip bands shaded

## Model validity
The regular-solution parameters are calibrated for Fe-Mn austenite
(SFE ~ 15 mJ/m^2 at 22 at.% Mn, ~ 25 mJ/m^2 at 35 at.% Mn, 300 K).  For the
other tabulated solutes the parameters are *effective* values reproducing the
reported dSFE/dx slope only; `GammaEpsilonParameters` records this in the
context, and it is carried into the verdict.  The magnetic (Neel) contribution
to dG is not resolved explicitly -- it is absorbed into the linear A + B*T
coefficients and into sigma -- so the experimental SFE minimum near 12 at.% Mn
is not reproduced.
"""

from pyiron_nodes.dataframe import DisplayDataFrame
from pyiron_nodes.math_utils import Linspace
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from core import Workflow
from core import group_node
from core import as_function_node

# ── Local node definitions ──────────────────────


@as_function_node("alloy")
def AlloyDefinition(
    base_element: str = "Fe",
    solute_element: str = "Mn",
    x_solute: float = 0.22,
    temperature: float = 300.0,
):
    """Single source of truth for the alloy under investigation.

    Parameters
    ----------
    base_element : str
        Matrix element of the austenite, ``Fe`` for a steel.
    solute_element : str
        The alloying element ``X`` in Fe(1-x)X(x).
    x_solute : float
        Atomic fraction ``x`` of the solute (not weight percent).
    temperature : float
        Deformation temperature in K.  The SFE of austenite rises by roughly
        0.05-0.1 mJ/m^2 per K, so this is not a cosmetic parameter.

    Returns
    -------
    alloy : dict
        The alloy specification, consumed by ``CompositionScan``.
    """
    if not 0.0 <= x_solute <= 1.0:
        raise ValueError(
            f"x_solute is an atomic fraction and must lie in [0, 1]; got {x_solute}."
        )
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive (K); got {temperature}.")
    alloy = {
        "base_element": base_element,
        "solute_element": solute_element,
        "x_solute": float(x_solute),
        "temperature": float(temperature),
    }
    return alloy


@as_function_node("criteria")
def DeformationModeCriteria(
    trip_twip_boundary_mJ_per_m2: float = 20.0,
    twip_slip_boundary_mJ_per_m2: float = 45.0,
):
    """The two SFE thresholds that separate the deformation mechanisms.

    Kept as a node of its own so that the classifier, the transition-composition
    search and the plot all read the same numbers.  The values below are the
    ones commonly quoted for austenitic Fe-Mn(-C) steels; different authors
    place the TRIP/TWIP boundary anywhere between 15 and 20 mJ/m^2.
    """
    if not 0.0 < trip_twip_boundary_mJ_per_m2 < twip_slip_boundary_mJ_per_m2:
        raise ValueError(
            "Need 0 < trip_twip_boundary < twip_slip_boundary; got "
            f"{trip_twip_boundary_mJ_per_m2} and {twip_slip_boundary_mJ_per_m2}."
        )
    criteria = {
        "trip_twip": float(trip_twip_boundary_mJ_per_m2),
        "twip_slip": float(twip_slip_boundary_mJ_per_m2),
    }
    return criteria


@as_function_node("scan")
def CompositionScan(
    alloy: dict = None,
    criteria: dict = None,
    x_values: list | np.ndarray = None,
):
    """Open the scan context: alloy, mechanism thresholds and composition grid.

    The whole workflow is evaluated on a composition grid so that the map and
    the single-alloy answer come from one and the same calculation.  This node
    guarantees that the requested composition is a grid point and stores its
    position, so the downstream steps can stay purely array-valued and the
    verdict can be read off the same arrays.
    """
    import numpy as np

    grid = np.sort(np.asarray(x_values, dtype=float).ravel())
    x_target = float(alloy["x_solute"])
    hit = np.isclose(grid, x_target, atol=1e-6)
    if hit.any():
        compositions = grid
        target_index = int(np.argmax(hit))
    else:
        compositions = np.sort(np.append(grid, x_target))
        target_index = int(np.argmin(np.abs(compositions - x_target)))

    scan = dict(alloy)
    scan["criteria"] = criteria
    scan["compositions"] = compositions
    scan["target_index"] = target_index
    return scan


@as_function_node("verdict")
def AlloyVerdict(assessment: dict = None):
    """The answer for the requested alloy: TRIP or TWIP, and why.

    Returns
    -------
    verdict : str
        Markdown report giving the mechanism, the SFE, the driving force, the
        Md temperature and the distance to the nearest mechanism boundary.
    """
    import numpy as np

    i = int(assessment["target_index"])
    base_element = assessment["base_element"]
    solute_element = assessment["solute_element"]
    temperature = float(assessment["temperature"])
    x = float(np.asarray(assessment["compositions"], dtype=float)[i])
    sfe = float(np.asarray(assessment["sfe_mJ_per_m2"], dtype=float)[i])
    dg = float(np.asarray(assessment["dg_chem_J_per_mol"], dtype=float)[i])
    md = float(np.asarray(assessment["md_epsilon_K"], dtype=float)[i])
    mode = str(np.asarray(assessment["modes"], dtype=object)[i])
    x_trip_twip = float(assessment["x_trip_twip"])
    x_twip_slip = float(assessment["x_twip_slip"])

    headline = {
        "TRIP": "TRIP-dominated (strain-induced gamma -> epsilon martensite)",
        "TWIP": "TWIP-dominated (mechanical twinning)",
        "slip": "neither TRIP nor TWIP -- planar/wavy dislocation glide",
    }[mode]

    lines = [
        f"# {base_element}(1-x){solute_element}(x), x = {x:.3f} "
        f"({100.0 * x:.1f} at.% {solute_element}) at {temperature:.0f} K",
        "",
        f"**Verdict: {headline}**",
        "",
        f"- Stacking fault energy: **{sfe:.1f} mJ/m^2**",
        f"- Driving force dG(gamma->epsilon): {dg:+.0f} J/mol "
        f"({'epsilon stable' if dg < 0 else 'gamma stable'})",
        f"- Md(epsilon): {md:.0f} K -- deformation temperature is "
        f"{'below' if temperature <= md else 'above'} it, so the transformation "
        f"{'can' if temperature <= md else 'cannot'} be triggered mechanically",
    ]
    if np.isfinite(x_trip_twip):
        lines.append(
            f"- TRIP/TWIP boundary of this system: x = {x_trip_twip:.3f} "
            f"({100.0 * x_trip_twip:.1f} at.% {solute_element}); the alloy sits "
            f"{100.0 * (x - x_trip_twip):+.1f} at.% from it"
        )
    else:
        lines.append(
            "- The TRIP/TWIP threshold is not crossed on the scanned "
            "composition range"
        )
    if np.isfinite(x_twip_slip):
        lines.append(
            f"- TWIP/slip boundary: x = {x_twip_slip:.3f} "
            f"({100.0 * x_twip_slip:.1f} at.% {solute_element})"
        )
    note = assessment["calibration_note"]
    if note:
        lines += ["", f"> **Caution:** {note}"]

    verdict = "\n".join(lines)
    return verdict


@as_function_node("fig")
def PlotDeformationMap(assessment: dict = None):
    """SFE against solute content with the TRIP / TWIP / slip bands shaded."""
    import matplotlib.pyplot as plt
    import numpy as np

    solute_element = assessment["solute_element"]
    base_element = assessment["base_element"]
    temperature = float(assessment["temperature"])
    criteria = assessment["criteria"]
    lo, hi = criteria["trip_twip"], criteria["twip_slip"]

    x = 100.0 * np.asarray(assessment["compositions"], dtype=float)
    sfe = np.asarray(assessment["sfe_mJ_per_m2"], dtype=float)
    i = int(assessment["target_index"])
    mode = str(np.asarray(assessment["modes"], dtype=object)[i])

    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=140)
    y_lo = min(float(sfe.min()) - 5.0, -5.0)
    y_hi = max(float(sfe.max()) + 5.0, hi + 10.0)
    ax.axhspan(y_lo, lo, color="tab:red", alpha=0.12)
    ax.axhspan(lo, hi, color="tab:green", alpha=0.12)
    ax.axhspan(hi, y_hi, color="tab:blue", alpha=0.10)
    for y, name in (
        (0.5 * (y_lo + lo), "TRIP"),
        (0.5 * (lo + hi), "TWIP"),
        (0.5 * (hi + y_hi), "slip"),
    ):
        ax.text(
            0.985,
            y,
            name,
            ha="right",
            va="center",
            fontsize=11,
            alpha=0.55,
            transform=ax.get_yaxis_transform(),
        )

    ax.plot(x, sfe, "-", color="k", lw=2.0, label=f"SFE at {temperature:.0f} K")
    ax.axhline(lo, color="0.4", ls="--", lw=1.0)
    ax.axhline(hi, color="0.4", ls="--", lw=1.0)

    ax.plot(
        x[i],
        sfe[i],
        "o",
        ms=9,
        color="tab:orange",
        zorder=5,
        label=f"{x[i]:.1f} at.% {solute_element}: {mode}",
    )
    ax.annotate(
        f"{sfe[i]:.1f} mJ/m$^2$",
        xy=(x[i], sfe[i]),
        xytext=(8, 10),
        textcoords="offset points",
        fontsize=10,
    )

    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(f"{solute_element} content (at.%)")
    ax.set_ylabel("stacking fault energy (mJ/m$^2$)")
    ax.set_title(
        f"Deformation-mode map of {base_element}(1-x){solute_element}(x) austenite"
    )
    ax.tick_params(labelsize=10)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    return fig


@as_function_node("df")
def DeformationMapTable(assessment: dict = None):
    """Collect the whole composition scan into one table."""
    import numpy as np
    import pandas as pd

    x = np.asarray(assessment["compositions"], dtype=float)
    df = pd.DataFrame(
        {
            "x_solute": x,
            "solute_at_pct": 100.0 * x,
            "a_fcc_A": np.asarray(assessment["a_fcc_angstrom"], dtype=float),
            "dG_gamma_eps_J_per_mol": np.asarray(
                assessment["dg_chem_J_per_mol"], dtype=float
            ),
            "sfe_mJ_per_m2": np.asarray(assessment["sfe_mJ_per_m2"], dtype=float),
            "Md_epsilon_K": np.asarray(assessment["md_epsilon_K"], dtype=float),
            "mode": np.asarray(assessment["modes"], dtype=object),
        }
    )
    return df


# ── Group node factories ─────────────────────────────


@group_node("assessment")
def compute(scan=None):
    from core import Workflow
    from core import as_function_node

    @as_function_node("thermo")
    def GammaEpsilonParameters(scan: dict = None):
        """Regular-solution parameters for the fcc -> hcp (gamma -> epsilon) transformation.

        The molar Gibbs energy difference of a binary A(1-x)B(x) austenite is

            dG(x, T) = (1-x) * dG_A(T) + x * dG_B(T) + x * (1-x) * Omega(x)

        with ``dG_i(T) = A_i + B_i * T`` and ``Omega(x) = Omega_0 + Omega_1 * x``.

        The Fe and Mn coefficients follow the assessment used throughout the
        TWIP-steel literature (Allain et al., Mater. Sci. Eng. A 387-389 (2004) 158)
        and are calibrated against measured Fe-Mn stacking fault energies.  The
        remaining solutes carry *effective* coefficients chosen to reproduce the
        reported dSFE/dx slope at dilution; they give the correct sign and order of
        magnitude but should be replaced by a proper CALPHAD assessment before the
        number is used quantitatively.  This is flagged in the context entry
        ``calibration_note``.
        """
        base_element = scan["base_element"]
        solute_element = scan["solute_element"]

        # A [J/mol], B [J/(mol K)] of dG(gamma->epsilon) = A + B*T for the pure element
        base_table = {
            "Fe": (-2243.38, 4.309),
        }
        # solute: (A, B, Omega_0, Omega_1, calibrated)
        solute_table = {
            "Mn": (-1000.0, 1.123, 2873.0, -717.0, True),
            "C": (0.0, 0.0, 6650.0, 0.0, False),
            "N": (0.0, 0.0, 5300.0, 0.0, False),
            "Al": (2800.0, 0.0, 3320.0, 0.0, False),
            "Si": (-560.0, 0.0, -3770.0, 0.0, False),
            "Cr": (1370.0, -0.163, -4805.0, 0.0, False),
            "Ni": (-1000.0, -1.86, 2970.0, 0.0, False),
            "Cu": (600.0, 0.0, 1830.0, 0.0, False),
        }

        if base_element not in base_table:
            raise ValueError(
                f"No gamma->epsilon assessment for base element {base_element!r}. "
                f"Available: {sorted(base_table)}."
            )
        if solute_element not in solute_table:
            raise ValueError(
                f"No gamma->epsilon assessment for solute {solute_element!r}. "
                f"Available: {sorted(solute_table)}."
            )

        a_base, b_base = base_table[base_element]
        a_sol, b_sol, omega_0, omega_1, calibrated = solute_table[solute_element]

        thermo = dict(scan)
        thermo["thermo_params"] = {
            "a_base": a_base,
            "b_base": b_base,
            "a_solute": a_sol,
            "b_solute": b_sol,
            "omega_0": omega_0,
            "omega_1": omega_1,
            "calibrated": calibrated,
        }
        thermo["calibration_note"] = (
            ""
            if calibrated
            else (
                f"The {base_element}-{solute_element} interaction parameters are "
                f"effective values reproducing the reported dSFE/dx slope only; the "
                f"absolute SFE is calibrated for {base_element}-Mn."
            )
        )
        return thermo

    @as_function_node("lattice")
    def FccLatticeParameter(thermo: dict = None):
        """Lattice parameter of the austenite from Vegard's law.

        ``a(x) = a_base + x * (da/dx)_solute``.  The slopes are the difference
        between the fcc lattice parameter of the pure solute (extrapolated where
        the fcc phase is not stable) and that of the base, except for the
        interstitials C and N where the tabulated slope is the measured dilation of
        austenite per unit interstitial fraction.
        """
        import numpy as np

        base_element = thermo["base_element"]
        solute_element = thermo["solute_element"]

        a_base_table = {"Fe": 3.575, "Ni": 3.524, "Co": 3.545}
        slope_table = {
            "Mn": 0.285,
            "C": 0.550,
            "N": 0.480,
            "Al": 0.475,
            "Si": -0.100,
            "Cr": 0.045,
            "Ni": -0.051,
            "Cu": 0.055,
        }
        if base_element not in a_base_table:
            raise ValueError(
                f"No fcc lattice parameter for {base_element!r}; "
                f"available: {sorted(a_base_table)}."
            )
        if solute_element not in slope_table:
            raise ValueError(
                f"No Vegard slope for solute {solute_element!r}; "
                f"available: {sorted(slope_table)}."
            )

        x = np.asarray(thermo["compositions"], dtype=float)
        lattice = dict(thermo)
        lattice["a_fcc_angstrom"] = (
            a_base_table[base_element] + x * slope_table[solute_element]
        )
        return lattice

    @as_function_node("density")
    def PlanarPackingDensity111(lattice: dict = None):
        """Molar atom density of the close-packed {111} plane.

        The {111} plane of an fcc lattice with cubic lattice parameter ``a`` carries
        ``4 / (sqrt(3) a^2)`` atoms per unit area; dividing by Avogadro's number
        turns the molar Gibbs energy difference into an energy per area.  For
        austenite this is close to 2.9e-5 mol/m^2.
        """
        import numpy as np

        n_avogadro = 6.02214076e23
        a_m = np.asarray(lattice["a_fcc_angstrom"], dtype=float) * 1.0e-10

        density = dict(lattice)
        density["rho_mol_per_m2"] = 4.0 / (np.sqrt(3.0) * a_m**2 * n_avogadro)
        return density

    @as_function_node("driving_force")
    def ChemicalDrivingForceGammaEpsilon(density: dict = None):
        """Molar Gibbs energy difference dG(gamma->epsilon) of the austenite.

        Evaluates the regular solution model of ``GammaEpsilonParameters`` and
        stores, besides the value at the deformation temperature, the effective
        linear coefficients of ``dG = A_eff + B_eff * T``.  A negative dG means the
        hcp epsilon phase is the stable one and the stacking fault is an epsilon
        embryo -- the thermodynamic precondition for TRIP.
        """
        import numpy as np

        params = density["thermo_params"]
        x = np.asarray(density["compositions"], dtype=float)
        omega = params["omega_0"] + params["omega_1"] * x

        a_eff = (
            (1.0 - x) * params["a_base"]
            + x * params["a_solute"]
            + x * (1.0 - x) * omega
        )
        b_eff = (1.0 - x) * params["b_base"] + x * params["b_solute"]

        driving_force = dict(density)
        driving_force["dg_a_eff_J_per_mol"] = a_eff
        driving_force["dg_b_eff_J_per_mol_K"] = b_eff
        driving_force["dg_chem_J_per_mol"] = a_eff + b_eff * density["temperature"]
        return driving_force

    @as_function_node("sfe")
    def StackingFaultEnergy(
        driving_force: dict = None,
        sigma_interface_mJ_per_m2: float = 20.0,
    ):
        """Stacking fault energy of the austenite, ``SFE = 2 rho dG + 2 sigma``.

        An intrinsic stacking fault is two {111} layers of hcp stacking, hence the
        factor two on both the volume and the interface term.

        ``sigma_interface_mJ_per_m2`` is the gamma/epsilon interfacial energy and is
        the single calibration knob of the model; published assessments use values
        between 8 and 27 mJ/m^2 depending on how much of the magnetic contribution
        to dG is resolved explicitly.  The default of 20 mJ/m^2 reproduces the
        measured Fe-Mn austenite SFE with the linear (non-magnetic) dG coefficients
        used here.
        """
        import numpy as np

        rho = np.asarray(driving_force["rho_mol_per_m2"], dtype=float)
        dg = np.asarray(driving_force["dg_chem_J_per_mol"], dtype=float)

        sfe = dict(driving_force)
        sfe["sigma_interface_mJ_per_m2"] = float(sigma_interface_mJ_per_m2)
        # rho [mol/m^2] * dG [J/mol] = J/m^2 -> 1e3 mJ/m^2
        sfe["sfe_mJ_per_m2"] = 2.0e3 * rho * dg + 2.0 * float(sigma_interface_mJ_per_m2)
        return sfe

    @as_function_node("stability")
    def EpsilonMartensiteAccessibility(
        sfe: dict = None,
        mechanical_driving_force_J_per_mol: float = 100.0,
    ):
        """Second, independent TRIP criterion: is epsilon-martensite reachable at T?

        A low SFE only says the fault is an epsilon embryo; the transformation must
        also be driven.  With ``dG = A_eff + B_eff * T`` (``B_eff > 0``, so the fcc
        phase becomes more stable on heating) the two characteristic temperatures
        are

            T0  = -A_eff / B_eff                                  (dG = 0)
            Md  = (U_mech - A_eff) / B_eff                        (dG = U_mech)

        where ``U_mech`` is the mechanical driving force supplied by the applied
        stress (of order 100 J/mol at yield).  Above Md no amount of straining
        forms epsilon-martensite and TRIP is suppressed however low the SFE is.
        """
        import numpy as np

        a_eff = np.asarray(sfe["dg_a_eff_J_per_mol"], dtype=float)
        b_eff = np.asarray(sfe["dg_b_eff_J_per_mol_K"], dtype=float)
        u_mech = float(mechanical_driving_force_J_per_mol)

        with np.errstate(divide="ignore", invalid="ignore"):
            t_zero = np.where(b_eff != 0.0, -a_eff / b_eff, np.nan)
            md = np.where(b_eff != 0.0, (u_mech - a_eff) / b_eff, np.nan)

        stability = dict(sfe)
        stability["mechanical_driving_force_J_per_mol"] = u_mech
        stability["t_zero_K"] = t_zero
        stability["md_epsilon_K"] = md
        stability["martensite_accessible"] = sfe["temperature"] <= md
        return stability

    @as_function_node("modes")
    def ClassifyDeformationModes(stability: dict = None):
        """Assign TRIP / TWIP / slip to every composition.

        The SFE window sets the mechanism; the driving-force check can only veto
        TRIP.  A composition with a TRIP-level SFE but no accessible epsilon
        martensite (deformation above Md) is reported as ``TWIP`` -- the fault
        ribbons are wide but cannot complete the transformation, so twinning takes
        over.
        """
        import numpy as np

        criteria = stability["criteria"]
        sfe = np.asarray(stability["sfe_mJ_per_m2"], dtype=float)
        accessible = np.broadcast_to(
            np.asarray(stability["martensite_accessible"]), sfe.shape
        )

        mode_array = np.where(
            sfe >= criteria["twip_slip"],
            "slip",
            np.where(sfe >= criteria["trip_twip"], "TWIP", "TRIP"),
        )
        # above Md the gamma->epsilon transformation cannot be triggered mechanically
        mode_array = np.where((mode_array == "TRIP") & ~accessible, "TWIP", mode_array)

        modes = dict(stability)
        modes["modes"] = mode_array.astype(object)
        return modes

    @as_function_node("assessment")
    def ModeTransitionCompositions(modes: dict = None):
        """Solute contents at which the SFE crosses the two mechanism thresholds.

        These are the numbers an alloy designer wants: below ``x_trip_twip`` the
        steel is a TRIP steel, above it a TWIP steel, and above ``x_twip_slip`` the
        austenite deforms by plain dislocation glide.  ``NaN`` means the threshold
        is not crossed anywhere on the scanned range.
        """
        import numpy as np

        x = np.asarray(modes["compositions"], dtype=float)
        sfe = np.asarray(modes["sfe_mJ_per_m2"], dtype=float)
        criteria = modes["criteria"]

        def first_crossing(level):
            sign_change = np.where(np.diff(np.sign(sfe - level)) != 0)[0]
            if sign_change.size == 0:
                return float("nan")
            i = int(sign_change[0])
            dy = sfe[i + 1] - sfe[i]
            if dy == 0.0:
                return float(x[i])
            return float(x[i] + (level - sfe[i]) * (x[i + 1] - x[i]) / dy)

        assessment = dict(modes)
        assessment["x_trip_twip"] = first_crossing(criteria["trip_twip"])
        assessment["x_twip_slip"] = first_crossing(criteria["twip_slip"])
        return assessment

    inner_wf = Workflow("compute")
    inner_wf.thermo_parameters = GammaEpsilonParameters(scan=scan)
    inner_wf.lattice_parameter = FccLatticeParameter(thermo=inner_wf.thermo_parameters)
    inner_wf.planar_density = PlanarPackingDensity111(
        lattice=inner_wf.lattice_parameter
    )
    inner_wf.driving_force = ChemicalDrivingForceGammaEpsilon(
        density=inner_wf.planar_density
    )
    inner_wf.stacking_fault_energy = StackingFaultEnergy(
        driving_force=inner_wf.driving_force
    )
    inner_wf.martensite_stability = EpsilonMartensiteAccessibility(
        sfe=inner_wf.stacking_fault_energy
    )
    inner_wf.deformation_modes = ClassifyDeformationModes(
        stability=inner_wf.martensite_stability
    )
    inner_wf.mode_transitions = ModeTransitionCompositions(
        modes=inner_wf.deformation_modes
    )
    return inner_wf.mode_transitions.outputs.assessment


wf = Workflow("Steel_TWIP_TRIP")

wf.alloy_definition = AlloyDefinition()

wf.composition_grid = Linspace(x_max=0.45, num_points=91)

wf.mode_criteria = DeformationModeCriteria()

wf.composition_scan = CompositionScan(
    alloy=wf.alloy_definition, criteria=wf.mode_criteria, x_values=wf.composition_grid
)

wf.compute = compute(scan=wf.composition_scan)

wf.alloy_verdict = AlloyVerdict(assessment=wf.compute)

wf.mode_map_plot = PlotDeformationMap(assessment=wf.compute)

wf.mode_map_table = DeformationMapTable(assessment=wf.compute)

wf.mode_map_view = DisplayDataFrame(df=wf.mode_map_table)
