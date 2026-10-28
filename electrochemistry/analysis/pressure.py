from core import as_function_node


@as_function_node
def ElectrodePressure(
    initial_structure, electrode_forces: dict, electrode: str = "Al", n_skip: int = 0
):
    """Convert the z-force on one electrode species (eV/Å) to pressure in Pa.

    Parameters
    ----------
    electrode_forces:
        Dict returned by ParseElectrodeForce, keyed by chemical symbol.
    initial_structure:
        ASE Atoms object; provides the xy cross-section area.
    electrode:
        Chemical symbol of the electrode to compute pressure for.
    n_skip:
        Number of initial frames to discard as equilibration.

    Returns
    -------
    pressure:
        Time-series pressure in Pa (1-D array).
    mean_pressure:
        Mean pressure over the production run (Pa).
    std_pressure:
        Standard deviation of the pressure (Pa).
    """
    import numpy as np

    A_ang2 = initial_structure.cell[0, 0] * initial_structure.cell[1, 1]

    # 1 eV/Å³ = 1.602176634e-19 J / (1e-10 m)³ = 1.602176634e11 Pa
    eV_per_ang3_to_Pa = 1.602176634e11

    fz = np.asarray(electrode_forces[electrode]["fz"])[n_skip:]
    pressure = (fz / A_ang2) * eV_per_ang3_to_Pa

    mean_pressure = float(np.mean(pressure))
    std_pressure = float(np.std(pressure))

    return pressure, mean_pressure, std_pressure


@as_function_node
def PistonPressureConvergence(
    initial_structure,
    electrode_forces: dict,
    target_pressure: float = 1.0,
    electrodes: str = '["Al", "Ne"]',
    n_skip: int = 0,
    tolerance: float = 0.1,
    charges: dict = None,
):
    """Check that a piston barostat reached its target pressure.

    With one movable electrode and one frozen reference electrode, mechanical
    equilibrium requires the normal pressure on both to balance the forces on
    the piston.  Two readings that agree with each other but not with the
    expected value mean the piston has not finished moving; two readings that
    disagree with each other mean the run is not equilibrated at all.

    The electrostatic attraction between the plates
    ----------------------------------------------
    ``ParseElectrodeForce`` measures the force the *electrolyte* exerts on an
    electrode, because ``compute group/group`` is evaluated against the mobile
    group only.  The piston, however, also feels the counter-electrode: two
    plates carrying ±σ attract with the Maxwell stress

        P_Maxwell = σ² / (2 ε₀)

    so at equilibrium the electrolyte has to push *harder* than the applied
    load, by exactly that amount:

        P_electrolyte = P_applied + P_Maxwell

    Pass ``charges`` (the dict carried by ``SimSetupBundle``) and that term is
    computed and added to the expected pressure, which is what makes
    ``converged`` meaningful for a charged cell.  Leave it ``None`` for an
    uncharged cell, where the term vanishes anyway.

    Parameters
    ----------
    initial_structure:
        ASE Atoms object; provides the xy cross-section area and the electrode
        positions.
    electrode_forces:
        Dict returned by ``ParseElectrodeForce``, keyed by chemical symbol.
    target_pressure:
        Applied piston load in bar, i.e. ``CreateLammpsMDInput.piston_pressure``.
    electrodes:
        Symbols to report on, as a string holding a list.
    n_skip:
        Number of initial frames to discard as equilibration.
    tolerance:
        Relative deviation from the expected pressure still counted as
        converged.  Applied to ``max(abs(expected), 1.0)`` so an expected
        pressure of 0 bar does not make every run fail.
    charges:
        Per-species charge dict from ``SimSetupBundle``.  Used only for the
        Maxwell-stress correction described above.

    Returns
    -------
    pressures:
        Dict of ``{symbol: {"mean": bar, "std": bar}}``, as *compressive*
        pressures: positive means the electrolyte pushes the electrode outward.
        Unlike ``ElectrodePressure``, which reports the signed z-force, the sign
        is flipped for the lower electrode so that the two are comparable.
    expected_pressure:
        ``target_pressure`` plus the Maxwell stress, in bar — the value the
        electrolyte pressure is actually compared against.
    imbalance:
        Largest pairwise difference between the electrode means, in bar.
    converged:
        Whether every electrode mean is within ``tolerance`` of
        ``expected_pressure``.
    """
    import ast
    import numpy as np

    try:
        parsed = ast.literal_eval(electrodes)
    except (SyntaxError, ValueError):
        parsed = electrodes
    electrode_list = [parsed] if isinstance(parsed, str) else list(parsed)

    A_ang2 = initial_structure.cell[0, 0] * initial_structure.cell[1, 1]
    # 1 eV/Å³ = 1.602176634e11 Pa; 1 bar = 1e5 Pa
    eV_per_ang3_to_bar = 1.602176634e11 / 1e5

    # Which side of the electrolyte each electrode sits on fixes the sign of its
    # outward normal: the electrolyte pushes the lower electrode toward -z and
    # the upper one toward +z.
    symbols = np.asarray(initial_structure.get_chemical_symbols())
    z = initial_structure.positions[:, 2]
    electrode_z = {el: float(np.mean(z[symbols == el])) for el in electrode_list}
    midpoint = 0.5 * (min(electrode_z.values()) + max(electrode_z.values()))

    pressures = {}
    for electrode in electrode_list:
        if electrode not in electrode_forces:
            raise KeyError(
                f"no electrode force data for {electrode!r}; "
                f"available: {sorted(electrode_forces)}"
            )
        outward = 1.0 if electrode_z[electrode] >= midpoint else -1.0
        fz = np.asarray(electrode_forces[electrode]["fz"])[n_skip:]
        p = outward * (fz / A_ang2) * eV_per_ang3_to_bar
        pressures[electrode] = {"mean": float(np.mean(p)), "std": float(np.std(p))}

    means = [v["mean"] for v in pressures.values()]
    imbalance = float(np.max(means) - np.min(means)) if len(means) > 1 else 0.0

    # Maxwell stress between the two charged plates, P = sigma^2 / (2 eps0).
    maxwell_bar = 0.0
    if charges is not None:
        metal = charges["metal"]
        plate_charge = {
            metal: charges["metal_charge"],
            "Ne": charges["neon_charge"],
        }
        # Use the plate whose total charge is known on both sides; they are
        # equal and opposite by construction of the cell.
        sigma_e_per_ang2 = [
            abs(q) * int(np.count_nonzero(symbols == el)) / A_ang2
            for el, q in plate_charge.items()
            if q != 0.0
        ]
        if sigma_e_per_ang2:
            e_charge = 1.602176634e-19  # C
            epsilon_0 = 8.854187817e-12  # F/m
            sigma = min(sigma_e_per_ang2) * e_charge / 1e-20  # C/m²
            maxwell_bar = sigma**2 / (2 * epsilon_0) / 1e5

    expected_pressure = float(target_pressure + maxwell_bar)

    scale = max(abs(expected_pressure), 1.0)
    converged = bool(
        all(abs(m - expected_pressure) <= tolerance * scale for m in means)
    )

    return pressures, expected_pressure, imbalance, converged
