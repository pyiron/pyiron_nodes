from core import as_function_node


@as_function_node
def ElectrodePressure(initial_structure, electrode_forces: dict, electrode: str = "Al", n_skip: int = 0):
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
