from dataclasses import dataclass, asdict
from core import as_inp_dataclass_node, as_function_node
import random
import string
from typing import Optional, Tuple
from ase.atoms import Atoms
import numpy as np
import os
import pandas as pd


@as_inp_dataclass_node
@dataclass
class MD:
    """
    Molecular dynamics parameters.

    Attributes:
    -----------
    timestep: float
        https://calphy.org/en/latest/inputfile.html#timestep
    n_small_steps: int
        https://calphy.org/en/latest/inputfile.html#n-small-steps
    n_every_steps: int
        https://calphy.org/en/latest/inputfile.html#n-every-steps
    n_repeat_steps: int
        https://calphy.org/en/latest/inputfile.html#n-repeat-steps
    n_cycles: int
        https://calphy.org/en/latest/inputfile.html#n-cycles
    thermostat_damping: float
        https://calphy.org/en/latest/inputfile.html#thermostat-damping
    barostat_damping: float
        https://calphy.org/en/latest/inputfile.html#barostat-damping
    """

    timestep: float = 0.001
    n_small_steps: int = 10000
    n_every_steps: int = 10
    n_repeat_steps: int = 10
    n_cycles: int = 100
    thermostat_damping: float = 0.5
    barostat_damping: float = 0.1


@as_inp_dataclass_node
@dataclass
class NoseHoover:
    """
    Nose-Hoover parameters.

    Attributes:
    -----------
    thermostat_damping: float
        https://calphy.org/en/latest/inputfile.html#nose-hoover-thermostat-damping
    barostat_damping: float
        https://calphy.org/en/latest/inputfile.html#nose-hoover-barostat-damping
    """

    thermostat_damping: float = 0.1
    barostat_damping: float = 0.1


@as_inp_dataclass_node
@dataclass
class Berendsen:
    """
    Berendsen parameters.

    Attributes:
    -----------
    thermostat_damping: float
        https://calphy.org/en/latest/inputfile.html#berendsen-thermostat-damping
    barostat_damping: float
        https://calphy.org/en/latest/inputfile.html#berendsen-barostat-damping
    """

    thermostat_damping: float = 100.0
    barostat_damping: float = 100.0


@as_inp_dataclass_node
@dataclass
class Tolerance:
    """
    Tolerance parameters.

    Attributes:
    -----------
    spring_constant: float
        https://calphy.org/en/latest/inputfile.html#tol-spring-constant
    solid_fraction: float
        https://calphy.org/en/latest/inputfile.html#tol-solid-fraction
    liquid_fraction: float
        https://calphy.org/en/latest/inputfile.html#tol-liquid-fraction
    pressure: float
        https://calphy.org/en/latest/inputfile.html#tol-pressure
    """

    spring_constant: float = 0.01
    solid_fraction: float = 0.7
    liquid_fraction: float = 0.05
    pressure: float = 1.0


@as_inp_dataclass_node
@dataclass
class InputClass:
    """
    Input parameters for calphy calculations.

    Attributes:
    -----------
    md: MD
        Molecular dynamics parameters.
    tolerance: Tolerance
        Tolerance parameters.
    nose_hoover: NoseHoover
        Nose-Hoover parameters.
    berendsen: Berendsen
        Berendsen parameters.
    queue: Queue
        Queue parameters.
    pressure: int
        https://calphy.org/en/latest/inputfile.html#pressure
    temperature: int
        https://calphy.org/en/latest/inputfile.html#temperature
    npt: bool
        https://calphy.org/en/latest/inputfile.html#npt
    n_equilibration_steps: int
        https://calphy.org/en/latest/inputfile.html#n-equilibration-steps
    n_switching_steps: int
        https://calphy.org/en/latest/inputfile.html#n-switching-steps
    n_print_steps: int
        https://calphy.org/en/latest/inputfile.html#n-print-steps
    n_iterations: int
        https://calphy.org/en/latest/inputfile.html#n-iterations
    equilibration_control: str
        https://calphy.org/en/latest/inputfile.html#equilibration-control
    melting_cycle: bool
        https://calphy.org/en/latest/inputfile.html#melting-cycle
    spring_constants: Optional[float]
        https://calphy.org/en/latest/inputfile.html#spring-constants
    """

    md: Optional[MD] = None
    tolerance: Optional[Tolerance] = None
    nose_hoover: Optional[NoseHoover] = None
    berendsen: Optional[Berendsen] = None
    pressure: int = 0
    temperature: int = 300
    temperature_stop: int = 600
    npt: bool = True
    n_equilibration_steps: int = 2500
    n_switching_steps: int = 2500
    n_print_steps: int = 1000
    n_iterations: int = 1
    equilibration_control: str = "nose-hoover"
    melting_cycle: bool = False
    cores: Optional[int] = 1


def _generate_random_string(length: str) -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))


def _prepare_potential_and_structure(potential, structure):
    import os
    import shutil
    from ase.data import atomic_masses, atomic_numbers
    from pyiron_lammps.potential import get_potential_by_name
    from pyiron_lammps.structure import (
        LammpsStructure,
    )

    potential = get_potential_by_name(potential_name=potential)

    pair_style = []
    pair_coeff = []

    pair_style.append(" ".join(potential["Config"][0].strip().split()[1:]))
    pair_coeff.append(" ".join(potential["Config"][1].strip().split()[1:]))

    # now prepare the list of elements
    elements = list(potential["Species"])
    elements_from_pot = list(potential["Species"])

    lmp_structure = LammpsStructure()
    lmp_structure.potential = potential
    lmp_structure.atom_type = "atomic"
    lmp_structure.el_eam_lst = list(potential["Species"])
    lmp_structure.structure = structure

    # elements_object_lst = structure.get_species_objects()
    elements_struct_lst = structure.get_chemical_symbols()

    masses = []
    for element_name in elements_from_pot:
        if element_name in elements_struct_lst:
            index = list(elements_struct_lst).index(element_name)
            masses.append(atomic_masses[atomic_numbers[element_name]])
        else:
            masses.append(1.0)

    file_name = os.path.join(os.getcwd(), _generate_random_string(7) + ".dat")
    lmp_structure.write_file(file_name=file_name)
    return pair_style, pair_coeff, elements, masses, file_name


def _prepare_input(inp, potential, structure, mode="fe", reference_phase="solid"):
    from calphy.input import Calculation
    import os

    pair_style, pair_coeff, elements, masses, file_name = (
        _prepare_potential_and_structure(potential, structure)
    )

    inpdict = asdict(inp)
    inpdict["pair_style"] = pair_style
    inpdict["pair_coeff"] = pair_coeff
    inpdict["element"] = elements
    inpdict["mass"] = masses
    inpdict["mode"] = mode
    inpdict["reference_phase"] = reference_phase
    inpdict["lattice"] = file_name
    inpdict["queue"] = {
        "cores": inpdict["cores"],
    }
    del inpdict["cores"]

    if inpdict["md"] is None:
        inpdict["md"] = {
            "timestep": 0.001,
            "n_small_steps": 10000,
            "n_every_steps": 10,
            "n_repeat_steps": 10,
            "n_cycles": 100,
            "thermostat_damping": 0.5,
            "barostat_damping": 0.1,
        }
    if inpdict["tolerance"] is None:
        inpdict["tolerance"] = {
            "spring_constant": 0.01,
            "solid_fraction": 0.7,
            "liquid_fraction": 0.05,
            "pressure": 1.0,
        }
    if inpdict["nose_hoover"] is None:
        inpdict["nose_hoover"] = {
            "thermostat_damping": 0.1,
            "barostat_damping": 0.1,
        }
    if inpdict["berendsen"] is None:
        inpdict["berendsen"] = {
            "thermostat_damping": 100.0,
            "barostat_damping": 100.0,
        }
    if mode == "ts":
        inpdict["temperature"] = [inpdict["temperature"], inpdict["temperature_stop"]]
        del inpdict["temperature_stop"]

    calc = Calculation(**inpdict)
    return calc


def _run_cleanup(simfolder, lattice, delete_folder=False):
    import shutil
    import os

    os.remove(lattice)
    if delete_folder:
        shutil.rmtree(simfolder)


@as_function_node
def SolidFreeEnergy(inp, structure: Atoms, potential: str, store: bool = True) -> float:
    """
    Calculate the free energy of a solid phase.

    Parameters:
    -----------
    inp: InputClass
        Input parameters for calphy calculations.
    structure: Atoms
        Atomic structure.
    potential: str
        Potential name.

    Returns:
    --------
    float
        Free energy in eV/atom
    """
    from calphy.solid import Solid
    from calphy.routines import routine_fe
    import os

    calc = _prepare_input(inp, potential, structure, mode="fe", reference_phase="solid")
    # os.chdir()
    simfolder = calc.create_folders()
    job = Solid(calculation=calc, simfolder=simfolder)
    job = routine_fe(job)
    _run_cleanup(simfolder, calc.lattice)
    free_energy = job.report["results"]["free_energy"].tolist()
    return free_energy


@as_function_node
def LiquidFreeEnergy(
    inp, structure: Atoms, potential: str, store: bool = True
) -> float:
    """
    Calculate the free energy of a liquid phase.

    Parameters:
    -----------
    inp: InputClass
        Input parameters for calphy calculations.
    structure: Atoms
        Atomic structure.
    potential: str
        Potential name.

    Returns:
    --------
    float
        Free energy in eV/atom
    """
    from calphy.liquid import Liquid
    from calphy.routines import routine_fe

    calc = _prepare_input(
        inp, potential, structure, mode="fe", reference_phase="liquid"
    )
    simfolder = calc.create_folders()
    job = Liquid(calculation=calc, simfolder=simfolder)
    job = routine_fe(job)
    # run calculation
    _run_cleanup(simfolder, calc.lattice)
    free_energy = job.report["results"]["free_energy"].tolist()
    return free_energy


@as_function_node
def SolidFreeEnergyWithTemp(inp, structure: Atoms, potential: str, store: bool = True):
    """
    Calculate the free energy of a solid phase as a function of temperature.

    Parameters:
    -----------
    inp: InputClass
        Input parameters for calphy calculations.
    structure: Atoms
        Atomic structure.
    potential: str
        Potential name.

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Temperature and free energy in K and eV/atom, respectively.
    """
    from calphy.solid import Solid
    from calphy.routines import routine_ts

    calc = _prepare_input(inp, potential, structure, mode="ts", reference_phase="solid")
    simfolder = calc.create_folders()
    job = Solid(calculation=calc, simfolder=simfolder)
    job = routine_ts(job)
    # run calculation

    # grab the results
    datafile = os.path.join(os.getcwd(), simfolder, "temperature_sweep.dat")
    temperature_array, free_energy_array = np.loadtxt(
        datafile, unpack=True, usecols=(0, 1)
    )
    temperature = temperature_array.tolist()
    free_energy = free_energy_array.tolist()

    _run_cleanup(simfolder, calc.lattice)
    return free_energy, temperature


@as_function_node
def LiquidFreeEnergyWithTemp(inp, structure: Atoms, potential: str, store: bool = True):
    """
    Calculate the free energy of a liquid phase as a function of temperature.

    Parameters:
    -----------
    inp: InputClass
        Input parameters for calphy calculations.
    structure: Atoms
        Atomic structure.
    potential: str
        Potential name.

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Temperature and free energy in K and eV/atom, respectively.
    """
    from calphy.liquid import Liquid
    from calphy.routines import routine_ts

    calc = _prepare_input(
        inp, potential, structure, mode="ts", reference_phase="liquid"
    )
    simfolder = calc.create_folders()
    job = Liquid(calculation=calc, simfolder=simfolder)
    job = routine_ts(job)

    # grab the results
    datafile = os.path.join(os.getcwd(), simfolder, "temperature_sweep.dat")
    temperature_array, free_energy_array = np.loadtxt(
        datafile, unpack=True, usecols=(0, 1)
    )
    temperature = temperature_array.tolist()
    free_energy = free_energy_array.tolist()

    _run_cleanup(simfolder, calc.lattice)
    return free_energy, temperature


@as_function_node("fig")
def PlotFreeEnergy(temperature: np.ndarray, free_energy: np.ndarray):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(temperature, free_energy, label="free energy")
    ax.set_ylabel("Free energy (eV/atom)")
    ax.set_xlabel("Temperature (K)")
    plt.legend(frameon=False)
    return fig


@as_function_node
def CalcPhaseTransformationTemp(
    temp_A: np.ndarray,
    fe_A: np.ndarray,
    temp_B: np.ndarray,
    fe_B: np.ndarray,
    fit_order: int = 4,
):
    """
    Calculate the phase transformation temperature from free energy data.

    Parameters:
    -----------
    temp_A: np.ndarray
        Temperature array for phase 1.
    fe_A: np.ndarray
        Free energy array for phase 1.
    temp_B: np.ndarray
        Temperature array for phase 2.
    fe_B: np.ndarray
        Free energy array for phase 2.
    fit_order: int
        Order of the polynomial fit.

    Returns:
    --------
    float
        Phase transformation temperature
    """
    import matplotlib.pyplot as plt
    import warnings

    # do some fitting to determine temps
    t1min = np.min(temp_A)
    t2min = np.min(temp_B)
    t1max = np.max(temp_A)
    t2max = np.max(temp_B)

    tmin = np.min([t1min, t2min])
    tmax = np.max([t1max, t2max])

    # warn about extrapolation
    if not t1min == t2min:
        warnings.warn(f"free energy is being extrapolated!")
    if not t1max == t2max:
        warnings.warn(f"free energy is being extrapolated!")

    # now fit
    f1fit = np.polyfit(temp_A, fe_A, fit_order)
    f2fit = np.polyfit(temp_B, fe_B, fit_order)

    # reevaluate over the new range
    fit_t = np.arange(tmin, tmax + 1, 1)
    fit_f1 = np.polyval(f1fit, fit_t)
    fit_f2 = np.polyval(f2fit, fit_t)

    # now evaluate the intersection temp
    arg = np.argsort(np.abs(fit_f1 - fit_f2))[0]
    phase_transition_temperature = fit_t[arg]

    # warn if the temperature is shady
    if np.abs(phase_transition_temperature - tmin) < 1e-3:
        warnings.warn("It is likely there is no intersection of free energies")
    elif np.abs(phase_transition_temperature - tmax) < 1e-3:
        warnings.warn("It is likely there is no intersection of free energies")

    # plot
    c1lo = "#ef9a9a"
    c1hi = "#b71c1c"
    c2lo = "#90caf9"
    c2hi = "#0d47a1"

    fig, ax = plt.subplots()
    ax.plot(fit_t, fit_f1, color=c1lo, label=f"phase A fit")
    ax.plot(fit_t, fit_f2, color=c2lo, label=f"phase B fit")
    ax.plot(temp_A, fe_A, color=c1hi, label="phase A", ls="dashed")
    ax.plot(temp_B, fe_B, color=c2hi, label="phase B", ls="dashed")
    ax.axvline(phase_transition_temperature, ls="dashed", c="#37474f")
    ax.set_ylabel("Free energy (eV/atom)")
    ax.set_xlabel("Temperature (K)")
    ax.legend(frameon=False)

    return fig


@as_function_node
def CollectResults() -> pd.DataFrame:
    from calphy.postprocessing import gather_results

    results = gather_results(".")
    return results


def _fit_free_energies(temp_solid, fe_solid, temp_liquid, fe_liquid, fit_order):
    """Fit G(T) for both phases and return the fit coefficients and arrays."""
    t_s = np.asarray(temp_solid, dtype=float)
    f_s = np.asarray(fe_solid, dtype=float)
    t_l = np.asarray(temp_liquid, dtype=float)
    f_l = np.asarray(fe_liquid, dtype=float)
    solid_fit = np.polyfit(t_s, f_s, fit_order)
    liquid_fit = np.polyfit(t_l, f_l, fit_order)
    return (t_s, f_s, t_l, f_l), solid_fit, liquid_fit


def _melting_temperature(t_s, t_l, solid_fit, liquid_fit, n=1000):
    """Locate the solid-liquid free-energy crossing on the overlapping range.

    The melting point is the sign change of ``G_solid(T) - G_liquid(T)``.  A
    linear interpolation of the crossing is returned.  If the curves do not
    cross within the overlapping range, the closest-approach temperature is
    returned instead (never ``None``).
    """
    # Restrict to the range covered by BOTH phases to avoid extrapolation.
    tmin = float(max(t_s.min(), t_l.min()))
    tmax = float(min(t_s.max(), t_l.max()))
    grid = np.linspace(tmin, tmax, n)
    diff = np.polyval(solid_fit, grid) - np.polyval(liquid_fit, grid)

    sign_change = np.where(np.diff(np.sign(diff)))[0]
    if len(sign_change) > 0:
        i = sign_change[0]
        t0, t1 = grid[i], grid[i + 1]
        d0, d1 = diff[i], diff[i + 1]
        # linear interpolation of the zero crossing
        return float(t0 - d0 * (t1 - t0) / (d1 - d0))
    # no crossing: fall back to closest approach so we always return a number
    return float(grid[np.argmin(np.abs(diff))])


@as_function_node("T_melt")
def FindMeltingTemperature(
    temp_solid: list,
    fe_solid: list,
    temp_liquid: list,
    fe_liquid: list,
    fit_order: int = 4,
) -> float:
    """
    Find the solid-liquid phase transition temperature by locating the
    intersection of polynomial fits to G_solid(T) and G_liquid(T).

    The melting point is detected as the sign change of
    ``G_solid(T) - G_liquid(T)`` over the temperature range covered by both
    phases, then refined by linear interpolation.  This reliably finds a
    genuine crossing (the previous nearest-point search could return an
    endpoint and miss the intersection).

    Parameters
    ----------
    temp_solid : list
        Temperature array from SolidFreeEnergyWithTemp.
    fe_solid : list
        Free energy array from SolidFreeEnergyWithTemp.
    temp_liquid : list
        Temperature array from LiquidFreeEnergyWithTemp.
    fe_liquid : list
        Free energy array from LiquidFreeEnergyWithTemp.
    fit_order : int
        Polynomial order used for fitting.

    Returns
    -------
    float
        Melting temperature in K.
    """
    (t_s, _, t_l, _), solid_fit, liquid_fit = _fit_free_energies(
        temp_solid, fe_solid, temp_liquid, fe_liquid, fit_order
    )
    T_melt = _melting_temperature(t_s, t_l, solid_fit, liquid_fit)
    return T_melt


@as_function_node("fig")
def PlotSolidLiquidFreeEnergy(
    temp_solid: list,
    fe_solid: list,
    temp_liquid: list,
    fe_liquid: list,
    T_melt: Optional[float] = None,
    fit_order: int = 4,
):
    """
    Plot the solid and liquid free energies versus temperature, marking the
    melting temperature with a vertical dashed line and its value.

    Parameters
    ----------
    temp_solid, fe_solid : list
        Temperature and free energy of the solid phase.
    temp_liquid, fe_liquid : list
        Temperature and free energy of the liquid phase.
    T_melt : float, optional
        Melting temperature to mark.  If ``None`` it is computed from the fits.
    fit_order : int
        Polynomial order used for the fitted curves.
    """
    import matplotlib.pyplot as plt

    (t_s, f_s, t_l, f_l), solid_fit, liquid_fit = _fit_free_energies(
        temp_solid, fe_solid, temp_liquid, fe_liquid, fit_order
    )
    if T_melt is None:
        T_melt = _melting_temperature(t_s, t_l, solid_fit, liquid_fit)

    tmin = float(min(t_s.min(), t_l.min()))
    tmax = float(max(t_s.max(), t_l.max()))
    grid = np.linspace(tmin, tmax, 400)

    fig, ax = plt.subplots()
    ax.plot(t_s, f_s, "o", ms=4, color="#b71c1c", label="solid (data)")
    ax.plot(t_l, f_l, "o", ms=4, color="#0d47a1", label="liquid (data)")
    ax.plot(grid, np.polyval(solid_fit, grid), "-", color="#ef9a9a", label="solid fit")
    ax.plot(
        grid, np.polyval(liquid_fit, grid), "-", color="#90caf9", label="liquid fit"
    )

    if T_melt is not None and np.isfinite(T_melt):
        fe_at_melt = float(np.polyval(solid_fit, T_melt))
        ax.axvline(T_melt, ls="dashed", color="#37474f")
        ax.scatter([T_melt], [fe_at_melt], color="k", zorder=10)
        ymin, ymax = ax.get_ylim()
        ax.text(
            T_melt,
            ymin + 0.05 * (ymax - ymin),
            f"  $T_m$ = {T_melt:.0f} K",
            rotation=90,
            va="bottom",
        )

    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Free energy (eV/atom)")
    ax.legend(frameon=False)
    return fig
