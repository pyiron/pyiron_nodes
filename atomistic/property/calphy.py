import contextlib
import warnings
from dataclasses import dataclass, asdict, replace
from core import as_inp_dataclass_node, as_function_node, as_out_dataclass_node, group_node
from core.data_fields import EmptyArrayField
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
        Keep this ``True`` for the liquid leg: it is what makes calphy melt the
        input structure at ``temperature_high`` and verify that it actually
        melted.  With it off the liquid free energy is computed for whatever
        configuration was handed in, with no check at all.
    temperature_high: float
        https://calphy.org/en/latest/inputfile.html#temperature-high
        Melting temperature used by the melting cycle.  ``0`` lets calphy pick
        twice the upper end of the temperature range.
    spring_constants: Optional[float]
        https://calphy.org/en/latest/inputfile.html#spring-constants

    Notes
    -----
    ``n_equilibration_steps`` / ``n_switching_steps`` are a compromise between
    calphy's own defaults (25000 / 50000) and a demo that finishes in a
    reasonable time.  Short switching shows up as a large
    ``rs_max_dissipation`` in :class:`CalphyDiagnostics`; if that exceeds
    ~1e-4 eV/atom, raise these towards the calphy defaults.
    """

    md: Optional[MD] = None
    tolerance: Optional[Tolerance] = None
    nose_hoover: Optional[NoseHoover] = None
    berendsen: Optional[Berendsen] = None
    pressure: int = 0
    temperature: int = 300
    temperature_stop: int = 600
    npt: bool = True
    n_equilibration_steps: int = 10000
    n_switching_steps: int = 25000
    n_print_steps: int = 1000
    n_iterations: int = 3
    equilibration_control: str = "nose-hoover"
    melting_cycle: bool = True
    temperature_high: float = 0.0
    cores: Optional[int] = 1


@as_function_node("inp")
def ApplyTemperatureWindow(
    inp: InputClass,
    temperature_min: float = 0.0,
    temperature_max: float = 0.0,
):
    """Copy *inp* with its reversible-scaling temperature window replaced.

    A melting-point calculation runs the same two legs several times over
    different windows, and everything except the window stays fixed:
    ``n_switching_steps``, ``melting_cycle``, the thermostat settings, the
    pressure.  Building a fresh :class:`InputClass` per attempt repeats all of
    that, and two copies of the same settings can silently drift apart -- which
    matters, because free energies computed with different switching lengths are
    not comparable and nothing says so.  This keeps one ``InputClass`` for the
    settings and stamps the window on a copy.

    Parameters
    ----------
    inp : InputClass
        The shared settings.  Left untouched; a copy is returned.
    temperature_min, temperature_max : float
        The window, named to match :func:`EstimateCalphyTemperatureRange` and
        :func:`SuggestTemperatureWindow` so either drops straight in.  They
        become ``temperature`` (the thermostat setpoint, t0) and
        ``temperature_stop`` (reached by scaling the potential by lambda).

    Returns
    -------
    inp : InputClass
        A copy with ``temperature`` / ``temperature_stop`` set.
    """
    return replace(
        inp,
        temperature=float(temperature_min),
        temperature_stop=float(temperature_max),
    )


@as_out_dataclass_node
class CalphySystem:
    """Crystal, melt seed and potential -- what every calphy leg needs.

    These three travel together because they have to stay consistent, and the
    ways they can fall out of step are all silent:

    * a temperature window is only valid for the potential it was measured
      with, so an estimate made with one potential against legs run with
      another is meaningless without failing;
    * the liquid leg's cell must be the same cell as the crystal, or the two
      free energies are per-atom numbers for different systems and their
      crossing means nothing;
    * both legs must run the same potential, or the crossing is between two
      unrelated curves.

    Wiring one bundle instead of three ports makes that structural rather than
    something to remember.  Use :func:`BuildCalphySystem` to assemble one, and
    this class as a node to split it back into its parts.

    Attributes
    ----------
    structure : Atoms
        The crystal, for the solid leg and the superheating estimate.
    liquid_structure : Atoms
        A disordered starting point for the liquid leg -- normally the same cell
        rattled.  With ``InputClass.melting_cycle`` on, calphy melts it properly
        at ``temperature_high`` and verifies it melted, so this only has to be
        non-crystalline.
    potential : str
        LAMMPS potential name, as :func:`GetPotential` returns it.
    """

    structure: Optional[Atoms] = None
    liquid_structure: Optional[Atoms] = None
    potential: str = ""


@as_function_node("system")
def BuildCalphySystem(
    structure: Atoms,
    liquid_structure: Atoms,
    potential: str,
):
    """Bundle a crystal, its melt seed and their potential into one port.

    See :class:`CalphySystem` for why they belong together.

    Returns
    -------
    system : CalphySystem
        One value carrying all three, to be split with a ``CalphySystem`` node
        wherever the individual parts are needed.
    """
    return CalphySystem.pure_dataclass(
        structure=structure,
        liquid_structure=liquid_structure,
        potential=potential,
    )


def _generate_random_string(length: str) -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))


@contextlib.contextmanager
def _in_workdir(path):
    """Temporarily switch CWD to *path* (creating it if needed), then restore."""
    abs_path = os.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    old = os.getcwd()
    os.chdir(abs_path)
    try:
        yield abs_path
    finally:
        os.chdir(old)


def _single_pair_coeff(potential_name, species, coeff_lines, elements_in_structure):
    """Reduce a potential's ``pair_coeff`` lines to the one calphy can use.

    calphy issues exactly one ``pair_coeff`` command
    (``helpers.set_potential`` uses ``pair_coeff[0]``), and its
    reversible-scaling routine rebuilds *that one line* into a
    ``pair_style hybrid/scaled`` pair to scale the potential by λ.  A
    ``pair_style eam`` (funcfl) catalogue entry instead carries one diagonal
    ``pair_coeff i i <file>`` line per element, which cannot be expressed that
    way.  Silently keeping the first of them — what this used to do — declares
    atom types whose coefficients were never set, and LAMMPS aborts deep inside
    the run with ``All pair coeffs are not set``.

    The way out is that a structure rarely needs all of the potential's
    elements: a pure-Cu cell run with a Ni-Cu funcfl potential only needs the Cu
    file, and dropping the unused type leaves exactly one line, renumbered onto
    type 1.  When the structure really does need several elements from a
    per-type potential, there is nothing to reduce and this raises.

    Returns
    -------
    coeff : str
        The single ``pair_coeff`` line, ``pair_coeff`` keyword included.
    species : list
        The elements, in atom-type order, that the line applies to.
    """
    if len(coeff_lines) == 1:
        return coeff_lines[0], list(species)

    needed = [element for element in species if element in elements_in_structure]
    if len(needed) != 1:
        raise ValueError(
            f"potential {potential_name!r} sets its coefficients per atom type "
            f"({len(coeff_lines)} pair_coeff lines for elements {list(species)}), "
            "but calphy can only issue one pair_coeff command -- its "
            "reversible-scaling routine rewrites that single line to scale the "
            f"potential.  The structure needs {needed or 'none'} of those "
            "elements, so the extra types cannot be dropped either.  Use a "
            "potential that maps all types in one line (`eam/alloy`, `eam/fs`, "
            "`meam`, ...); `ListPotentials(structure=...)` shows the candidates."
        )

    element = needed[0]
    atom_type = str(species.index(element) + 1)
    matching = [
        line for line in coeff_lines if line.split()[1:3] == [atom_type, atom_type]
    ]
    if len(matching) != 1:
        raise ValueError(
            f"potential {potential_name!r} has no single `pair_coeff "
            f"{atom_type} {atom_type} ...` line for {element!r}; its pair_coeff "
            f"lines are {coeff_lines}"
        )
    # The dropped elements shift the numbering, so the survivor becomes type 1.
    coeff = " ".join(["pair_coeff", "1", "1"] + matching[0].split()[3:])
    return coeff, [element]


def _prepare_potential_and_structure(potential, structure, working_directory="."):
    import shutil
    from ase.data import atomic_masses, atomic_numbers
    from pyiron_lammps.structure import (
        LammpsStructure,
    )

    from pyiron_nodes.atomistic.engine.lammps import get_usable_potential_by_name

    potential_name = potential
    potential = get_usable_potential_by_name(potential_name=potential)

    config = [str(line).strip() for line in potential["Config"]]
    style_lines = [line for line in config if line.startswith("pair_style")]
    coeff_lines = [line for line in config if line.startswith("pair_coeff")]
    if not style_lines or not coeff_lines:
        raise ValueError(
            f"potential {potential_name!r} has no usable pair_style/pair_coeff "
            f"lines; its Config is {config}"
        )

    elements_struct_lst = set(structure.get_chemical_symbols())
    coeff_line, elements = _single_pair_coeff(
        potential_name, list(potential["Species"]), coeff_lines, elements_struct_lst
    )

    pair_style = [" ".join(style_lines[0].split()[1:])]
    pair_coeff = [" ".join(coeff_line.split()[1:])]

    lmp_structure = LammpsStructure()
    lmp_structure.potential = potential
    lmp_structure.atom_type = "atomic"
    lmp_structure.el_eam_lst = list(elements)
    lmp_structure.structure = structure

    masses = []
    for element_name in elements:
        if element_name in elements_struct_lst:
            masses.append(atomic_masses[atomic_numbers[element_name]])
        else:
            # A type the structure never uses; calphy still wants a mass for it.
            masses.append(1.0)

    abs_wd = os.path.abspath(working_directory)
    os.makedirs(abs_wd, exist_ok=True)
    file_name = os.path.join(abs_wd, _generate_random_string(7) + ".dat")
    lmp_structure.write_file(file_name=file_name)
    return pair_style, pair_coeff, elements, masses, file_name


def _prepare_input(inp, potential, structure, mode="fe", reference_phase="solid", working_directory="."):
    from calphy.input import Calculation

    pair_style, pair_coeff, elements, masses, file_name = (
        _prepare_potential_and_structure(potential, structure, working_directory)
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


def _create_simfolder(calc):
    """Create calphy's simulation folder, parents included.

    ``Calculation.create_folders()`` puts the run under ``<cwd>/calphy/<id>`` but
    creates it with a bare ``os.mkdir``, so it fails with ``FileNotFoundError``
    unless ``<cwd>/calphy`` happens to exist already.  Make the parent first.
    """
    import os

    folder = calc.get_folder_name()
    os.makedirs(os.path.dirname(folder), exist_ok=True)
    return calc.create_folders()


def _run_cleanup(simfolder, lattice, delete_folder=False):
    import shutil
    import os

    os.remove(lattice)
    if delete_folder:
        shutil.rmtree(simfolder)


@as_out_dataclass_node
class CalphyDiagnostics:
    """Everything calphy knows about the quality of a free-energy run.

    calphy writes all of this into its simulation folder and then it is thrown
    away.  A free energy on its own cannot be judged; these numbers say whether
    it can be trusted.

    The two switching directions are the most informative entry.  Reversible
    scaling sweeps the temperature forwards and then backwards; the two G(T)
    curves coincide only if the switching was slow enough and the phase stayed
    intact.  ``hysteresis`` is half their difference at every temperature and
    ``rs_max_dissipation`` its largest value — calphy warns above 1e-4 eV/atom.

    Attributes
    ----------
    phase : str
        ``"solid"`` or ``"liquid"``.
    simfolder : str
        calphy's simulation directory, kept for post-mortem inspection.
    temperature, free_energy : list
        The averaged sweep, identical to the node's own output ports.
    free_energy_forward, free_energy_backward : list
        G(T) from the forward and backward switching runs separately.
    free_energy_error : list
        Standard deviation over ``n_iterations`` at each temperature.  Zero
        unless ``n_iterations > 1``.
    hysteresis : list
        ``(G_forward - G_backward) / 2`` at each temperature, in eV/atom.
    rs_max_dissipation : float
        Largest absolute ``hysteresis``, in eV/atom.  Above ~1e-4 the sweep is
        too fast or the structure changed phase during it.
    fe_temperature, fe_free_energy, fe_error : float
        The thermodynamic-integration leg the sweep is anchored to.
    ti_work, reference_system, ideal_gas, einstein_crystal, com_correction, pv : float
        Decomposition of ``fe_free_energy``, in eV/atom.  Solid:
        ``fe = reference_system - ti_work + com_correction + pv`` with
        ``reference_system`` the Einstein crystal.  Liquid:
        ``fe = ideal_gas + reference_system - ti_work + pv`` with
        ``reference_system`` the Uhlenbeck-Ford fluid.
    spring_constants : list
        Einstein-crystal spring constants per element (solid only), in eV/A^2.
    vol_atom, density : float
        Equilibrated volume per atom (A^3) and number density (1/A^3).
    natoms, n_iterations : int
        Cell size and number of independent switching runs.
    calphy_warnings : list
        WARNING lines calphy logged during the run, plus any raised while
        collecting these diagnostics.
    """

    phase: str = ""
    simfolder: str = ""
    temperature: list = EmptyArrayField()
    free_energy: list = EmptyArrayField()
    free_energy_forward: list = EmptyArrayField()
    free_energy_backward: list = EmptyArrayField()
    free_energy_error: list = EmptyArrayField()
    hysteresis: list = EmptyArrayField()
    rs_max_dissipation: float = 0.0
    fe_temperature: float = 0.0
    fe_free_energy: float = 0.0
    fe_error: float = 0.0
    ti_work: float = 0.0
    reference_system: float = 0.0
    ideal_gas: float = 0.0
    einstein_crystal: float = 0.0
    com_correction: float = 0.0
    pv: float = 0.0
    spring_constants: list = EmptyArrayField()
    vol_atom: float = 0.0
    density: float = 0.0
    natoms: int = 0
    n_iterations: int = 0
    calphy_warnings: list = EmptyArrayField()


def _rs_forward_backward(simfolder, f0, t0, natoms, pressure, n_iterations):
    """Reconstruct the forward and backward reversible-scaling G(T) curves.

    ``calphy.integrators.integrate_rs`` performs exactly this integration but
    averages the two directions before returning, so the hysteresis it computes
    survives only as a single number in the log.  Here the halves are kept
    apart.  The arithmetic mirrors ``integrate_rs`` line for line, including
    the ``scale_energy`` division by lambda that mode ``ts`` implies.

    The returned curves are the mean over the ``n_iterations`` switching runs,
    so their average reproduces ``temperature_sweep.dat`` exactly.  The
    dissipation, in contrast, is the *worst* of the individual runs: calphy
    logs ``np.min(es)``, the best one, which hides an iteration that drifted.
    Expect this number to be larger than the one in ``calphy.log`` whenever
    ``n_iterations > 1``.

    Returns
    -------
    temperature, free_energy_forward, free_energy_backward : np.ndarray
    max_dissipation : float
    """
    from calphy.integrators import kb
    from scipy.integrate import cumulative_trapezoid as cumtrapz

    p = pressure / (10000 * 160.21766208)
    wfs, wbs, diss = [], [], []
    flambda = None
    for i in range(1, n_iterations + 1):
        fdx, _, fvol, flambda = np.loadtxt(
            os.path.join(simfolder, "ts.forward_%d.dat" % i), unpack=True, comments="#"
        )
        bdx, _, bvol, blambda = np.loadtxt(
            os.path.join(simfolder, "ts.backward_%d.dat" % i), unpack=True, comments="#"
        )
        fdx = fdx / flambda + p * fvol / natoms
        bdx = bdx / blambda + p * bvol / natoms
        wf_i = cumtrapz(fdx, flambda, initial=0)
        # The backward run is integrated from its own end, which puts it back
        # on the forward lambda ordering -- the two arrays are then aligned.
        wb_i = cumtrapz(bdx[::-1], blambda[::-1], initial=0)
        wfs.append(wf_i)
        wbs.append(wb_i)
        diss.append(float(np.max(np.abs((wf_i - wb_i) / (2 * flambda)))))

    # calphy scales the accumulated work by lambda before adding it to the
    # reference free energy; keep that so the mean of the two curves returned
    # here reproduces temperature_sweep.dat exactly.
    wf = np.mean(wfs, axis=0) / flambda
    wb = np.mean(wbs, axis=0) / flambda
    temperature = t0 / flambda
    base = f0 / flambda + 1.5 * kb * temperature * np.log(flambda)
    return temperature, base + wf, base + wb, max(diss)


def _read_calphy_warnings(simfolder):
    """Return the WARNING lines calphy logged for this run."""
    log = os.path.join(simfolder, "calphy.log")
    if not os.path.isfile(log):
        return []
    with open(log) as fh:
        return [
            line.split("WARNING", 1)[1].strip()
            for line in fh
            if "WARNING" in line
        ]


def _run_routine(routine, job, simfolder):
    """Run a calphy routine, naming the run directory if it raises.

    ``MeltedError`` says only "System melted, increase size or reduce temp!" --
    it does not say at which temperature, nor where the trajectory that would
    answer that is.  The folder survives the failure, so point at it.
    """
    try:
        return routine(job)
    except Exception as exc:
        # The note is for a human reading the traceback; the attribute is for
        # CalphyMeltingTemperatureSearch, which catches the exception and never sees
        # one.  Both point at the same folder.
        exc.simfolder = simfolder
        exc.add_note(
            f"calphy run directory: {simfolder}\n"
            "If a solid sweep melted, CalphyMeltingFromTrajectory(simfolder=...) "
            "reads the dumped frames back and reports the temperature at which "
            "it happened -- use it to cap temperature_stop."
        )
        raise


def _free_energy_with_temp(inp, potential, structure, phase, working_directory):
    """Run one reversible-scaling leg and return ``(free_energy, temperature, diagnostics)``.

    Shared by :func:`SolidFreeEnergyWithTemp`, :func:`LiquidFreeEnergyWithTemp`
    and :func:`CalphyMeltingTemperatureSearch`, which has to run the same two legs
    repeatedly at shifted windows.  Keeping one body means the adaptive search
    cannot drift away from what the single-leg nodes do.

    The lattice file is removed in a ``finally``: a search that shifts its window
    several times would otherwise leave one behind per failed attempt.
    """
    if phase == "solid":
        from calphy.solid import Solid as Phase
    else:
        from calphy.liquid import Liquid as Phase
    from calphy.routines import routine_ts

    calc = _prepare_input(
        inp,
        potential,
        structure,
        mode="ts",
        reference_phase=phase,
        working_directory=working_directory,
    )
    try:
        with _in_workdir(working_directory):
            simfolder = _create_simfolder(calc)
            job = Phase(calculation=calc, simfolder=simfolder)
            job = _run_routine(routine_ts, job, simfolder)
            temperature_array, free_energy_array, error_array = np.loadtxt(
                os.path.join(simfolder, "temperature_sweep.dat"),
                unpack=True,
                usecols=(0, 1, 2),
            )
            diagnostics = _collect_diagnostics(
                job,
                calc,
                simfolder,
                phase,
                temperature=temperature_array,
                free_energy=free_energy_array,
                fe_error=error_array,
            )
    finally:
        if os.path.isfile(calc.lattice):
            os.remove(calc.lattice)
    return free_energy_array.tolist(), temperature_array.tolist(), diagnostics


def _collect_diagnostics(
    job, calc, simfolder, phase, temperature=None, free_energy=None, fe_error=None
):
    """Assemble a :class:`CalphyDiagnostics` bundle from a finished calphy job.

    Everything is read off the job object or out of *simfolder*; nothing is
    recomputed with LAMMPS.  Any failure is captured into ``calphy_warnings``
    rather than raised -- a parsing slip must not discard the result of a run
    that took hours.
    """
    warnings_ = []
    kwargs = dict(phase=phase, simfolder=str(simfolder))
    try:
        warnings_ = _read_calphy_warnings(simfolder)
        n_iterations = int(getattr(calc, "n_iterations", 1) or 1)
        natoms = int(getattr(job, "natoms", 0) or 0)
        results = job.report.get("results", {}) if hasattr(job, "report") else {}

        kwargs.update(
            fe_temperature=float(calc._temperature),
            fe_free_energy=float(job.fe),
            fe_error=float(job.ferr),
            ti_work=float(job.w),
            reference_system=float(job.fref),
            ideal_gas=float(getattr(job, "fideal", 0.0) or 0.0),
            einstein_crystal=float(results.get("einstein_crystal", 0.0) or 0.0),
            com_correction=float(results.get("com_correction", 0.0) or 0.0),
            pv=float(job.pv),
            spring_constants=list(np.atleast_1d(job.k).astype(float))
            if job.k is not None
            else [],
            vol_atom=float(job.vol / natoms) if natoms else 0.0,
            density=float(job.rho) if job.rho is not None else 0.0,
            natoms=natoms,
            n_iterations=n_iterations,
        )

        if temperature is not None:
            kwargs.update(
                temperature=list(np.asarray(temperature, dtype=float)),
                free_energy=list(np.asarray(free_energy, dtype=float)),
                free_energy_error=list(np.asarray(fe_error, dtype=float))
                if fe_error is not None
                else [],
            )
            t_rs, fe_fwd, fe_bwd, e_diss = _rs_forward_backward(
                simfolder,
                job.fe,
                float(calc._temperature),
                natoms,
                float(calc._pressure or 0.0),
                n_iterations,
            )
            kwargs.update(
                free_energy_forward=list(fe_fwd),
                free_energy_backward=list(fe_bwd),
                hysteresis=list((fe_fwd - fe_bwd) / 2.0),
                rs_max_dissipation=e_diss,
            )
            if e_diss > 1e-4:
                warnings_.append(
                    f"reversible-scaling dissipation {e_diss:.2e} eV/atom exceeds "
                    f"1e-4 -- switching too fast or the {phase} phase changed "
                    f"during the sweep"
                )
    except Exception as exc:  # never lose an expensive run over a parse error
        warnings_.append(f"diagnostics collection failed: {exc!r}")

    kwargs["calphy_warnings"] = warnings_
    return CalphyDiagnostics.pure_dataclass(**kwargs)


@as_function_node(isolate=True)
def SolidFreeEnergy(inp, structure: Atoms, potential: str, working_directory: str = "calphy_workdir", store: bool = True) -> float:
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
    working_directory: str
        Directory for calphy I/O files (created if absent).

    Returns:
    --------
    free_energy: float
        Free energy in eV/atom
    diagnostics: CalphyDiagnostics
        Reference-system decomposition, spring constants, cell volume and any
        warnings calphy logged.  The sweep fields stay empty -- this node runs
        a single thermodynamic integration, not a temperature sweep.
    """
    from calphy.solid import Solid
    from calphy.routines import routine_fe

    calc = _prepare_input(inp, potential, structure, mode="fe", reference_phase="solid", working_directory=working_directory)
    with _in_workdir(working_directory):
        simfolder = _create_simfolder(calc)
        job = Solid(calculation=calc, simfolder=simfolder)
        job = _run_routine(routine_fe, job, simfolder)
        diagnostics = _collect_diagnostics(job, calc, simfolder, "solid")
    _run_cleanup(simfolder, calc.lattice)
    free_energy = job.report["results"]["free_energy"].tolist()
    return free_energy, diagnostics


@as_function_node(isolate=True)
def LiquidFreeEnergy(
    inp, structure: Atoms, potential: str, working_directory: str = "calphy_workdir", store: bool = True
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
    working_directory: str
        Directory for calphy I/O files (created if absent).

    Returns:
    --------
    free_energy: float
        Free energy in eV/atom
    diagnostics: CalphyDiagnostics
        Reference-system decomposition, cell volume and any warnings calphy
        logged.  The sweep fields stay empty -- this node runs a single
        thermodynamic integration, not a temperature sweep.
    """
    from calphy.liquid import Liquid
    from calphy.routines import routine_fe

    calc = _prepare_input(
        inp, potential, structure, mode="fe", reference_phase="liquid", working_directory=working_directory
    )
    with _in_workdir(working_directory):
        simfolder = _create_simfolder(calc)
        job = Liquid(calculation=calc, simfolder=simfolder)
        job = _run_routine(routine_fe, job, simfolder)
        diagnostics = _collect_diagnostics(job, calc, simfolder, "liquid")
    _run_cleanup(simfolder, calc.lattice)
    free_energy = job.report["results"]["free_energy"].tolist()
    return free_energy, diagnostics


@as_function_node(isolate=True)
def SolidFreeEnergyWithTemp(inp, structure: Atoms, potential: str, working_directory: str = "calphy_workdir", store: bool = True):
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
    working_directory: str
        Directory for calphy I/O files (created if absent).

    Returns:
    --------
    free_energy: list
        Free energy in eV/atom.
    temperature: list
        Temperature in K.
    diagnostics: CalphyDiagnostics
        Forward and backward switching curves, their hysteresis, the
        thermodynamic-integration decomposition and any warnings calphy
        logged.  Wire this into ``CalphyDiagnosticsTable`` or
        ``PlotCalphyHysteresis`` to judge whether the sweep converged.
    """
    free_energy, temperature, diagnostics = _free_energy_with_temp(
        inp, potential, structure, "solid", working_directory
    )
    return free_energy, temperature, diagnostics


@as_function_node(isolate=True)
def LiquidFreeEnergyWithTemp(inp, structure: Atoms, potential: str, working_directory: str = "calphy_workdir", store: bool = True):
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
    working_directory: str
        Directory for calphy I/O files (created if absent).

    Returns:
    --------
    free_energy: list
        Free energy in eV/atom.
    temperature: list
        Temperature in K.
    diagnostics: CalphyDiagnostics
        Forward and backward switching curves, their hysteresis, the
        thermodynamic-integration decomposition and any warnings calphy
        logged.  Wire this into ``CalphyDiagnosticsTable`` or
        ``PlotCalphyHysteresis`` to judge whether the sweep converged.
    """
    free_energy, temperature, diagnostics = _free_energy_with_temp(
        inp, potential, structure, "liquid", working_directory
    )
    return free_energy, temperature, diagnostics


@as_function_node("df")
def CalphyDiagnosticsTable(diagnostics: CalphyDiagnostics) -> pd.DataFrame:
    """Render the scalar entries of a :class:`CalphyDiagnostics` as a table.

    One row per quantity, plus one row per warning calphy raised, so a run can
    be judged at a glance in the GUI.  The array-valued fields (the switching
    curves) are summarised rather than listed -- plot them with
    :func:`PlotCalphyHysteresis`.
    """
    d = diagnostics
    rows = [
        ("phase", d.phase, ""),
        ("free energy at T0", d.fe_free_energy, "eV/atom"),
        ("  reference temperature T0", d.fe_temperature, "K"),
        ("  statistical error", d.fe_error, "eV/atom"),
        ("  switching work", d.ti_work, "eV/atom"),
        ("  reference system", d.reference_system, "eV/atom"),
        ("  ideal gas", d.ideal_gas, "eV/atom"),
        ("  Einstein crystal", d.einstein_crystal, "eV/atom"),
        ("  centre-of-mass correction", d.com_correction, "eV/atom"),
        ("  pV", d.pv, "eV/atom"),
        ("max switching hysteresis", d.rs_max_dissipation, "eV/atom"),
        ("volume per atom", d.vol_atom, "A^3"),
        ("density", d.density, "1/A^3"),
        ("spring constants", ", ".join(f"{k:.4g}" for k in d.spring_constants), "eV/A^2"),
        ("atoms", d.natoms, ""),
        ("switching runs", d.n_iterations, ""),
        ("simulation folder", d.simfolder, ""),
    ]
    for w in d.calphy_warnings:
        rows.append(("WARNING", w, ""))
    return pd.DataFrame(rows, columns=["quantity", "value", "unit"])


@as_function_node("fig")
def PlotCalphyHysteresis(diagnostics: CalphyDiagnostics):
    """Show the forward and backward switching curves and their difference.

    Reversible scaling sweeps the temperature up and then back down.  If the
    switching was slow enough and the phase survived it, the two curves lie on
    top of each other and the lower panel sits inside the +-1e-4 eV/atom band
    calphy uses as its own warning threshold.  A curve that opens up towards
    one end of the range usually means the phase changed there.
    """
    import matplotlib.pyplot as plt

    d = diagnostics
    t = np.asarray(d.temperature, dtype=float)
    if t.size == 0:
        fig, ax = plt.subplots()
        ax.text(
            0.5,
            0.5,
            "no temperature sweep in these diagnostics\n"
            "(single-temperature free energy run)",
            ha="center",
            va="center",
        )
        ax.set_axis_off()
        return fig

    fwd = np.asarray(d.free_energy_forward, dtype=float)
    bwd = np.asarray(d.free_energy_backward, dtype=float)
    avg = np.asarray(d.free_energy, dtype=float)
    hyst = np.asarray(d.hysteresis, dtype=float)

    fig, (ax, ax2) = plt.subplots(
        2, 1, sharex=True, gridspec_kw={"height_ratios": [2, 1]}, figsize=(6, 6)
    )
    ax.plot(t, fwd, color="#b71c1c", label="forward")
    ax.plot(t, bwd, color="#0d47a1", label="backward")
    ax.plot(t, avg, color="#37474f", ls="dashed", label="average")
    ax.set_ylabel("Free energy (eV/atom)")
    ax.set_title(f"{d.phase} — reversible scaling")
    ax.legend(frameon=False)

    ax2.axhspan(-1e-4, 1e-4, color="#c8e6c9", label="calphy threshold")
    ax2.axhline(0.0, color="#37474f", lw=0.8)
    ax2.plot(t, hyst, color="#6a1b9a")
    ax2.set_ylabel("Hysteresis (eV/atom)")
    ax2.set_xlabel("Temperature (K)")
    ax2.legend(frameon=False)
    fig.tight_layout()
    return fig


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


def _crossing_is_bracketed(t_solid, t_liquid, solid_fit, liquid_fit):
    """Is there a genuine sign change of G_solid - G_liquid over both ranges?

    Without this, the closest-approach fallback in :func:`_melting_temperature`
    is indistinguishable from a result: two curves that never meet still return
    a temperature, and it is usually a window endpoint.
    """
    overlap = (
        max(t_solid.min(), t_liquid.min()),
        min(t_solid.max(), t_liquid.max()),
    )
    gap = [np.polyval(solid_fit, t) - np.polyval(liquid_fit, t) for t in overlap]
    return bool(gap[0] * gap[1] < 0)  # a bool, not np.bool_


def _linear_crossing(temp_solid, fe_solid, temp_liquid, fe_liquid):
    """Crossing of straight-line fits to both branches, or ``None`` if parallel.

    Only for *re-centring* a search window that missed the melting point.  A
    high-order fit is the right way to read T_m off curves that bracket it and
    the wrong way to guess where it is beyond their ends, where the fit
    diverges; over the ~100 K a shifted window moves, G(T) is straight enough.
    """
    slope_s, offset_s = np.polyfit(
        np.asarray(temp_solid, dtype=float), np.asarray(fe_solid, dtype=float), 1
    )
    slope_l, offset_l = np.polyfit(
        np.asarray(temp_liquid, dtype=float), np.asarray(fe_liquid, dtype=float), 1
    )
    if abs(slope_s - slope_l) < 1e-12:
        return None
    crossing = (offset_l - offset_s) / (slope_s - slope_l)
    return float(crossing) if crossing > 0 else None


def _melting_temperature_error(
    temp_solid, fe_solid, err_solid, temp_liquid, fe_liquid, err_liquid, t_melt
):
    """Propagate the per-temperature free-energy errors onto T_m.

    The two curves cross at a shallow angle, so a small error in either free
    energy becomes a large one in the temperature: dividing by the slope
    difference (i.e. by the entropy of fusion) is what converts eV/atom into
    kelvin.  Returns 0.0 when calphy reported no errors at all, which is what
    ``n_iterations = 1`` gives.
    """
    err_s = np.atleast_1d(np.asarray(err_solid, dtype=float))
    err_l = np.atleast_1d(np.asarray(err_liquid, dtype=float))
    if err_s.size == 0 or err_l.size == 0 or not np.any(err_s) or not np.any(err_l):
        return 0.0
    t_s = np.asarray(temp_solid, dtype=float)
    t_l = np.asarray(temp_liquid, dtype=float)
    order_s, order_l = np.argsort(t_s), np.argsort(t_l)
    e_s = float(np.interp(t_melt, t_s[order_s], err_s[order_s]))
    e_l = float(np.interp(t_melt, t_l[order_l], err_l[order_l]))
    slope_s = np.polyfit(t_s, np.asarray(fe_solid, dtype=float), 1)[0]
    slope_l = np.polyfit(t_l, np.asarray(fe_liquid, dtype=float), 1)[0]
    if abs(slope_s - slope_l) < 1e-12:
        return float("nan")
    return float(np.hypot(e_s, e_l) / abs(slope_s - slope_l))


#: Default distance, as a fraction of the window width, between a proven bound
#: on T_m and the nearest edge of the next window.  A bound is a temperature at
#: which a phase was already on its way out (or already the stable one), so a
#: window whose edge sits exactly on it invites the same outcome again.
_BOUND_BACKOFF = 0.1


def _melt_bound(exc, t_min, t_max):
    """What a solid leg that raised ``MeltedError`` proves about T_m.

    Returns ``(bound, observed)``: an upper bound on the melting temperature,
    and the temperature the crystal was actually seen to go at, or ``None`` if
    that could not be established.

    ``MeltedError`` says only that it happened.  Where matters, because the
    three places it can happen bound T_m very differently:

    * **during the sweep** — calphy dumps as it goes and checks afterwards, so
      the trajectory on disk is complete and :func:`_melting_from_trajectory`
      places the event on a temperature axis for free (see
      :func:`CalphyMeltingFromTrajectory`).  That temperature is the bound, and
      it is the one worth having: it can be hundreds of kelvin below ``t_max``.
    * **during equilibration** (``calphy/solid.py:251``), which runs at
      ``t_min`` — then the solid cannot even be held at the bottom of the
      window, and ``t_min`` is a much stronger bound than ``t_max``.  No sweep
      ran, so the absence of ``ts.forward_*.dat`` is what identifies this case.
    * anywhere, with the dumps unreadable — then all that is known is that the
      solid failed somewhere at or below ``t_max``.

    Never raises.  A search must not die because its consolation prize was
    unavailable.
    """
    simfolder = getattr(exc, "simfolder", None)
    if simfolder is None or not os.path.isdir(simfolder):
        return t_max, None
    swept = any(
        name.startswith("ts.forward_") and name.endswith(".dat")
        for name in os.listdir(simfolder)
    )
    if not swept:
        return t_min, None
    try:
        observed, _ = _melting_from_trajectory(simfolder)
    except Exception:
        return t_max, None
    if not np.isfinite(observed):
        # Every dumped frame was still crystalline: the check that fired ran on
        # the final configuration, past the last frame.  Only t_max is implied.
        return t_max, None
    return float(observed), float(observed)


def _next_centre(preferred, t_lower, t_upper, width, backoff):
    """Centre the next window, honouring what is already known about T_m.

    *preferred* is where the free energies say the crossing is, or ``None`` when
    the attempt aborted and produced none.  The bracket wins over it: an
    extrapolation across hundreds of kelvin is a guess, while ``t_lower`` /
    ``t_upper`` are things a phase demonstrated.  With both bounds in hand and
    no usable prediction this bisects, so the bracket at least halves per
    attempt instead of creeping by a fixed step.
    """
    lo = None if t_lower is None else t_lower + 0.5 * width + backoff
    hi = None if t_upper is None else t_upper - 0.5 * width - backoff
    if lo is not None and hi is not None:
        if lo > hi:
            # The bracket is narrower than a window: stop trying to fit inside
            # it and simply cover it.
            return 0.5 * (t_lower + t_upper)
        target = preferred if preferred is not None else 0.5 * (lo + hi)
        return float(min(max(target, lo), hi))
    if hi is not None:
        return hi if preferred is None else min(preferred, hi)
    if lo is not None:
        return lo if preferred is None else max(preferred, lo)
    return preferred  # unreachable: every handled outcome sets a bound


def _suggest_window(t_predicted, width, min_lambda=0.5):
    """Centre a window of *width* on *t_predicted*, clamped to be runnable.

    Placement is symmetric on purpose.  Reversible scaling superheats the solid
    at the top of the window by as much as it undercools the liquid at the
    bottom, so there is no safer side to lean towards; leaning either way trades
    one abort for the other.

    The ``min_lambda`` clamp raises ``temperature_min`` rather than lowering
    ``temperature_max``: λ = temperature_min/temperature_max is what governs the
    switching accuracy, and a window that had to be clamped is one whose centre
    is near the 10 K floor, where the upper end is the trustworthy bound.
    """
    half = 0.5 * float(width)
    t_max = float(t_predicted) + half
    t_min = max(float(t_predicted) - half, 10.0)
    if t_min / t_max < min_lambda:
        t_min = min_lambda * t_max
    return t_min, t_max


@as_function_node
def SuggestTemperatureWindow(
    temp_solid: list,
    fe_solid: list,
    temp_liquid: list,
    fe_liquid: list,
    fit_order: int = 1,
    window_width: float = 0.0,
    min_lambda: float = 0.5,
    max_extrapolation: float = 0.5,
):
    """Melting temperature and the window to re-run in, from two legs that ran.

    A temperature window can miss the melting point without anything going
    wrong: above T_m the liquid is the stable phase and sits at
    ``temperature_min`` quite happily, while a defect-free periodic solid
    superheats past ``temperature_max``, so both legs finish and calphy raises
    nothing at all.  What comes out is two free-energy curves that never cross.
    :func:`FindMeltingTemperature` then returns their closest approach, which is
    a window endpoint.

    The curves are not useless, though — they are the best measurement of T_m
    available, they just place it outside the range that was sampled.  This node
    reads that off them and turns it into the window that *would* have contained
    it:

        window above T_m   ->  G_liquid below G_solid throughout  ->  go down
        window below T_m   ->  G_solid below G_liquid throughout  ->  go up

    It is what :func:`CalphyMeltingTemperatureSearch` uses internally to re-centre, and
    it is exposed separately so that a hand-wired
    :func:`SolidFreeEnergyWithTemp` / :func:`LiquidFreeEnergyWithTemp` pair gets
    the same second pass.  ``temperature_min`` / ``temperature_max`` are named to
    match :func:`EstimateCalphyTemperatureRange`, so they drop straight into
    ``InputClass(temperature=..., temperature_stop=...)``.

    **A straight line is used for the extrapolation regardless of
    ``fit_order``.**  A high-order polynomial is the right way to read a crossing
    off curves that bracket it and the wrong way to guess where one is beyond
    their ends, where the fit diverges rather than continues.  G(T) = H - TS is
    close to straight over the range a shifted window moves.

    Parameters
    ----------
    temp_solid, fe_solid, temp_liquid, fe_liquid : list
        The two sweeps, as the ``*FreeEnergyWithTemp`` nodes return them.
    fit_order : int
        Polynomial order used only when the crossing is genuinely bracketed.
    window_width : float
        Width of the proposed window in K.  ``0`` keeps the width of the
        supplied sweeps: a width both phases already survived is the one to
        keep, and widening is what pushes them back into metastability.
    min_lambda : float
        Floor on ``temperature_min / temperature_max`` for the proposal.
    max_extrapolation : float
        How far outside the sampled range the crossing may fall, as a fraction
        of that range, before this refuses to answer.  Beyond it the straight
        line is being asked to predict a region it has no evidence about.

    Returns
    -------
    T_melt : float
        The crossing if it is bracketed, otherwise the linear extrapolate.
    temperature_min, temperature_max : float
        The window to run next.  When the crossing is already bracketed these
        come back centred on it, which is where a confirmation run belongs.
    bracketed : bool
        ``False`` means ``T_melt`` is a prediction, not a measurement.

    Raises
    ------
    ValueError
        If the two branches never cross above 0 K -- they are parallel, or the
        liquid is the lower branch and falling away faster -- or if the crossing
        lies further than ``max_extrapolation`` outside the sampled range.  Both
        cases mean the legs cannot say where the melting point is, and a
        plausible-looking number would be worse than a refusal.
    """
    (t_s, f_s, t_l, f_l), solid_fit, liquid_fit = _fit_free_energies(
        temp_solid, fe_solid, temp_liquid, fe_liquid, fit_order
    )
    lo = float(max(t_s.min(), t_l.min()))
    hi = float(min(t_s.max(), t_l.max()))
    span = hi - lo
    width = float(window_width) or span

    bracketed = _crossing_is_bracketed(t_s, t_l, solid_fit, liquid_fit)
    if bracketed:
        T_melt = _melting_temperature(t_s, t_l, solid_fit, liquid_fit)
    else:
        predicted = _linear_crossing(temp_solid, fe_solid, temp_liquid, fe_liquid)
        if predicted is None:
            raise ValueError(
                f"straight fits to G_solid(T) and G_liquid(T) over {lo:.0f}-"
                f"{hi:.0f} K do not cross at any temperature above 0 K, so no "
                "window contains a melting point. Either the two entropies came "
                "out equal, or the liquid is the lower branch and falling away "
                "faster -- neither is physical for a solid and its melt: check "
                "that the solid leg really ran on the crystal and the liquid leg "
                "on a melted structure (InputClass.melting_cycle must be True)."
            )
        distance = max(lo - predicted, predicted - hi, 0.0)
        if distance > max_extrapolation * span:
            raise ValueError(
                f"the free-energy curves put the crossing at {predicted:.0f} K, "
                f"{distance:.0f} K outside the {lo:.0f}-{hi:.0f} K they were "
                f"sampled over (limit: {max_extrapolation:g} x the "
                f"{span:.0f} K span). A straight line extrapolated that far is "
                "a guess, not a measurement -- move the window there in steps, "
                "or start from EstimateCalphyTemperatureRange."
            )
        T_melt = predicted

    temperature_min, temperature_max = _suggest_window(T_melt, width, min_lambda)
    return T_melt, temperature_min, temperature_max, bracketed


@as_function_node
def CalphyMeltingTemperatureSearch(
    inp,
    structure: Atoms,
    potential: str,
    liquid_structure: Optional[Atoms] = None,
    bound_backoff: float = 0.0,
    max_attempts: int = 6,
    fit_order: int = 1,
    min_lambda: float = 0.5,
    working_directory: str = "calphy_workdir",
    store: bool = True,
):
    """Melting temperature from a temperature window that repairs itself.

    A single pair of reversible-scaling legs only yields a melting point if the
    window happens to contain it, and a window that contains it necessarily
    pushes *both* phases into metastability: the solid is superheated at the top,
    the liquid undercooled at the bottom.  Get it wrong in either direction and
    calphy does not return a poor answer, it aborts —  ``MeltedError`` if the
    solid went, ``SolidifiedError`` if the liquid froze.  Neither says where the
    window should have been.

    A window can also miss the melting point without failing at all.  Above T_m
    the liquid is stable at ``temperature_min``, and a defect-free periodic
    solid superheats past ``temperature_max``, so both legs finish cleanly and
    produce two curves that never cross.  That silent case is the one a
    superheating-based window estimate lands in, and it is the *most*
    informative outcome of the four: the curves say where the crossing is, just
    not inside the range sampled.

    So this node runs the two legs and turns every outcome into a bound on T_m,
    then places the next window inside the bracket those bounds define:

    ===========================  ================================  ==============
    outcome                      what it proves                    next window
    ===========================  ================================  ==============
    ``MeltedError`` (solid)      T_m below where the dump shows    top just under
                                 the crystal went (no MD cost)     that
    ``SolidifiedError`` (liquid) T_m above ``temperature_min``     bottom just
                                                                   over it
    G_liquid below throughout    T_m below ``temperature_min``     on the
                                                                   extrapolated
                                                                   crossing
    G_solid below throughout     T_m above ``temperature_max``     likewise
    curves cross inside          T_m measured                      done
    ===========================  ================================  ==============

    The bracket, not the extrapolation, has the final word: an extrapolate is a
    straight line asked about a region it never saw, while a bound is something
    a phase demonstrated.  With both bounds known and no usable prediction the
    search bisects, so the bracket at least halves per attempt.

    This is the algorithm of calphy's own ``mode: melting_temperature``
    (``calphy.routines.MeltingTemp``), reimplemented here so that both legs keep
    producing :class:`CalphyDiagnostics`, so the window can come from
    :func:`EstimateCalphyTemperatureRange`, and — the reason it is not simply
    called — so that running out of attempts returns the best extrapolate with
    ``converged = False`` instead of raising ``Maximum number of tries
    reached``.  A bad potential is then still reported on rather than silently
    costing an hour of MD for nothing.

    **Cost is the whole window search, not one calculation.**  Every attempt is
    two full calphy legs, so this is ``max_attempts`` times the price of a
    hand-tuned pair in the worst case.  Give it a good starting window — that is
    what :func:`EstimateCalphyTemperatureRange` is for — and it converges in one
    or two.  Watch ``report`` to see how many it took.

    Parameters
    ----------
    inp : InputClass
        ``temperature`` / ``temperature_stop`` are the *starting* window; the
        rest of the settings are used unchanged for every attempt.
    structure : Atoms
        The crystal.  Used for the solid leg, and for the liquid leg too unless
        ``liquid_structure`` is given.
    liquid_structure : Atoms, optional
        Starting point for the liquid leg, e.g. ``Rattle(structure)``.  With
        ``inp.melting_cycle`` on (the default) calphy melts it at
        ``temperature_high`` regardless, so this rarely matters.
    bound_backoff : float
        Gap in K to leave between a proven bound on T_m and the nearest edge of
        the next window.  ``0`` uses a tenth of the window width.  A bound is a
        temperature at which a phase was already giving way, so a window edge
        sitting exactly on it tends to reproduce the same abort and buy nothing.
    max_attempts : int
        Cap on the number of solid+liquid pairs.  Reaching it is not an error:
        the last extrapolate is returned with ``converged = False``.
    fit_order : int
        Polynomial order for reading the crossing off the final curves.  1 is
        the default because over a window narrow enough for both phases to
        survive, G(T) is close to straight and a high-order fit mostly follows
        the switching noise.  Re-centring always uses a straight line.
    min_lambda : float
        Floor on ``temperature_min / temperature_max`` for every window tried,
        so a downward shift cannot walk into the regime where reversible
        scaling stops being reversible.

    Returns
    -------
    T_melt : float
        Melting temperature in K.
    T_melt_error : float
        Propagated from calphy's per-temperature free-energy errors, so it needs
        ``inp.n_iterations > 1`` to be non-zero.  It is a *statistical* error
        only: it says nothing about switching being too fast, which
        ``rs_max_dissipation`` in the diagnostics does.
    converged : bool
        ``True`` only if the final window actually brackets the crossing.
        ``False`` means ``T_melt`` is an extrapolation — usable as the next
        starting guess, not as a result.
    report : pandas.DataFrame
        One row per attempt: the window tried, what happened, and the melting
        temperature implied.  This is the audit trail for a search that may have
        moved a long way from the window it was given.
    temperature_min, temperature_max : float
        A window centred on ``T_melt``, ready to wire back into
        ``InputClass(temperature=..., temperature_stop=...)``.  When
        ``converged`` is ``False`` this is what to re-run with; when it is
        ``True`` it is where a refinement run with longer switching belongs,
        since a crossing near a window edge is worth re-centring on.  Computed
        by :func:`SuggestTemperatureWindow`, which does the same job for a
        hand-wired pair of legs.
    temp_solid, fe_solid, temp_liquid, fe_liquid : list
        The final sweeps, in the same form the single-leg nodes return them, so
        they can be fed straight to :func:`PlotSolidLiquidFreeEnergy`.
    solid_diagnostics, liquid_diagnostics : CalphyDiagnostics
        For the final attempt only.  Check ``rs_max_dissipation`` here before
        trusting ``T_melt``: a converged window says the crossing is in range,
        not that the free energies on either side of it are accurate.

    Raises
    ------
    ValueError
        Only if no attempt ever completed both legs, i.e. there is no
        free-energy data at all to place a crossing with.  ``report`` is
        included in the message.
    """
    from dataclasses import replace

    from calphy.errors import MeltedError, SolidifiedError

    t_lo, t_hi = float(inp.temperature), float(inp.temperature_stop)
    if t_hi <= t_lo:
        raise ValueError(
            f"inp.temperature_stop ({t_hi:g} K) must be above inp.temperature "
            f"({t_lo:g} K): the two bound the reversible-scaling window the "
            "search starts from."
        )
    width = t_hi - t_lo
    centre = 0.5 * (t_hi + t_lo)
    backoff = float(bound_backoff) or _BOUND_BACKOFF * width
    liquid_seed = structure if liquid_structure is None else liquid_structure

    # The bracket on T_m.  Everything the search learns is expressed here, and
    # the next window is placed inside it rather than a fixed distance from the
    # last one -- a window 700 K too high is 700 K too high whatever the step is.
    t_lower = None  # T_m is above this
    t_upper = None  # T_m is below this

    rows = []
    tried = set()
    result = None  # last attempt that got both legs through
    n_attempts = 0

    for attempt in range(1, int(max_attempts) + 1):
        n_attempts = attempt
        t_min, t_max = _suggest_window(centre, width, min_lambda)

        # Two bounds can point at each other -- a solid that melts at the top of
        # every window that keeps the liquid frozen at the bottom of the next
        # one.  Re-running a window already tried cannot end differently, so
        # stop instead of spending the remaining attempts on the same cycle.
        window = (round(t_min), round(t_max))
        if window in tried:
            rows.append(
                {
                    "attempt": attempt,
                    "temperature_min": round(t_min, 1),
                    "temperature_max": round(t_max, 1),
                    "outcome": "window search cycling",
                    "T_melt": float("nan"),
                }
            )
            break
        tried.add(window)

        leg_inp = replace(
            inp, temperature=int(round(t_min)), temperature_stop=int(round(t_max))
        )
        row = {
            "attempt": attempt,
            "temperature_min": round(t_min, 1),
            "temperature_max": round(t_max, 1),
            "outcome": "",
            "T_melt": float("nan"),
        }

        try:
            fe_s, t_s, diag_s = _free_energy_with_temp(
                leg_inp, potential, structure, "solid", working_directory
            )
        except MeltedError as exc:
            # The solid gave way somewhere below t_max.  calphy dumped the sweep
            # as it went, so *where* is already on disk -- reading it back costs
            # no MD and replaces a blind step with a measurement.
            bound, observed = _melt_bound(exc, t_min, t_max)
            t_upper = bound if t_upper is None else min(t_upper, bound)
            # With a measurement the bound carries the whole move.  Without one
            # it is only "somewhere in this window", which on its own would
            # creep down by the back-off alone, so keep the half-window step
            # that a blind search has to fall back on.
            hint = None if observed is not None else centre - 0.5 * width
            rows.append(
                {
                    **row,
                    "outcome": "solid melted"
                    if observed is None
                    else f"solid melted @ {observed:.0f} K",
                }
            )
            centre = _next_centre(hint, t_lower, t_upper, width, backoff)
            continue
        try:
            fe_l, t_l, diag_l = _free_energy_with_temp(
                leg_inp, potential, liquid_seed, "liquid", working_directory
            )
        except SolidifiedError:
            # The liquid froze at t_min.  Nothing to read back: it normally goes
            # during equilibration, before the sweep that would have dumped a
            # trajectory.  "T_m is above t_min" is the whole of the information.
            t_lower = t_min if t_lower is None else max(t_lower, t_min)
            rows.append({**row, "outcome": "liquid froze"})
            centre = _next_centre(
                centre + 0.5 * width, t_lower, t_upper, width, backoff
            )
            continue

        (t_arr_s, _, t_arr_l, _), solid_fit, liquid_fit = _fit_free_energies(
            t_s, fe_s, t_l, fe_l, fit_order
        )
        T_melt = _melting_temperature(t_arr_s, t_arr_l, solid_fit, liquid_fit)
        converged = _crossing_is_bracketed(t_arr_s, t_arr_l, solid_fit, liquid_fit)
        result = (T_melt, converged, t_s, fe_s, t_l, fe_l, diag_s, diag_l)
        if converged:
            rows.append({**row, "outcome": "bracketed", "T_melt": round(T_melt, 1)})
            break

        # Both legs ran and the curves never met, so one phase is the stable one
        # across the whole window and the melting point is on the other side of
        # it.  That is thermodynamics, not a heuristic: it bounds T_m as firmly
        # as an abort does, and it is the case a bad starting window most often
        # lands in -- neither phase fails, there is simply no crossing to find.
        overlap_lo = max(t_arr_s.min(), t_arr_l.min())
        overlap_hi = min(t_arr_s.max(), t_arr_l.max())
        midpoint = 0.5 * (overlap_lo + overlap_hi)
        liquid_is_stable = np.polyval(solid_fit, midpoint) > np.polyval(
            liquid_fit, midpoint
        )
        if liquid_is_stable:
            t_upper = t_min if t_upper is None else min(t_upper, t_min)
        else:
            t_lower = t_max if t_lower is None else max(t_lower, t_max)

        predicted = _linear_crossing(t_s, fe_s, t_l, fe_l)
        if predicted is None:
            # Equal entropies: the branches are parallel and never meet at any
            # temperature.  Moving the window cannot fix that, so stop and let
            # the closest approach stand as the (unconverged) answer.
            rows.append({**row, "outcome": "curves parallel, no crossing"})
            break
        result = (predicted, False, t_s, fe_s, t_l, fe_l, diag_s, diag_l)
        rows.append(
            {
                **row,
                "outcome": "liquid stable throughout"
                if liquid_is_stable
                else "solid stable throughout",
                "T_melt": round(predicted, 1),
            }
        )
        centre = _next_centre(predicted, t_lower, t_upper, width, backoff)
    else:
        rows.append(
            {
                "attempt": max_attempts,
                "temperature_min": float("nan"),
                "temperature_max": float("nan"),
                "outcome": f"gave up after {max_attempts} attempts",
                "T_melt": float("nan"),
            }
        )

    report = pd.DataFrame(rows)
    if result is None:
        bracket = (
            f"{'-inf' if t_lower is None else format(t_lower, '.0f')} K and "
            f"{'+inf' if t_upper is None else format(t_upper, '.0f')} K"
        )
        raise ValueError(
            "no attempt completed both a solid and a liquid leg, so there are "
            "no free energies to place a melting point with. The windows tried "
            f"were:\n{report.to_string(index=False)}\n"
            f"After {n_attempts} attempts the melting point is only known to lie "
            f"between {bracket}. A solid that melts and a liquid that freezes in "
            "the same window means no window of this width works for this "
            "potential: narrow it (lower EstimateCalphyTemperatureRange's "
            "window_fraction), or the cell is too small to hold either phase."
        )

    (
        T_melt,
        converged,
        temp_solid,
        fe_solid,
        temp_liquid,
        fe_liquid,
        solid_diagnostics,
        liquid_diagnostics,
    ) = result
    T_melt_error = _melting_temperature_error(
        temp_solid,
        fe_solid,
        solid_diagnostics.free_energy_error,
        temp_liquid,
        fe_liquid,
        liquid_diagnostics.free_energy_error,
        T_melt,
    )
    temperature_min, temperature_max = _suggest_window(T_melt, width, min_lambda)
    return (
        T_melt,
        T_melt_error,
        converged,
        report,
        temperature_min,
        temperature_max,
        temp_solid,
        fe_solid,
        temp_liquid,
        fe_liquid,
        solid_diagnostics,
        liquid_diagnostics,
    )


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
    if not _crossing_is_bracketed(t_s, t_l, solid_fit, liquid_fit):
        # Silence here is the trap this node is easiest to fall into: the
        # fallback value looks like every other answer it returns.
        warnings.warn(
            f"G_solid and G_liquid do not cross between "
            f"{max(t_s.min(), t_l.min()):.0f} K and "
            f"{min(t_s.max(), t_l.max()):.0f} K, so {T_melt:.0f} K is their "
            "closest approach, not a melting point. The window does not contain "
            "the transition -- CalphyMeltingTemperatureSearch moves it until it does.",
            RuntimeWarning,
            stacklevel=2,
        )
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


def _highest_stable_temperature(
    is_stable, t_lo: float, t_hi: float, n_refinements: int
) -> Optional[float]:
    """Bisect for the highest temperature that still satisfies ``is_stable``.

    Returns the largest midpoint found stable — always a temperature that was
    actually tested, never an interpolation — or ``None`` if every probe came
    back unstable.  ``t_lo`` itself is never tested, so it cannot be returned;
    a caller that seeds ``t_lo`` above the stability edge gets ``None`` and can
    widen the bracket rather than trusting an untested bound.
    """
    best = None
    for _ in range(n_refinements):
        mid = 0.5 * (t_lo + t_hi)
        if is_stable(mid):
            best = t_lo = mid
        else:
            t_hi = mid
    return None if best is None else float(best)


def _solid_fraction(atoms) -> float:
    """Fraction of atoms in a crystalline environment, from pyscal's q6 test.

    This is the same classification calphy applies in ``check_if_melted``
    (``pyscal``'s ``find.solids`` with an automatic neighbour cutoff), so a
    window this node accepts is a window calphy's own melt check accepts.
    calphy reaches it through a LAMMPS dump file; building the system from
    arrays gives an identical answer without touching the disk.
    """
    from pyscal3 import System

    system = System(
        atoms={
            "positions": [list(p) for p in atoms.get_positions()],
            "types": [1] * len(atoms),
        },
        box=[list(v) for v in np.asarray(atoms.get_cell())],
    )
    try:
        system.find.neighbors(method="cutoff", cutoff=0)
    except RuntimeError:
        # The adaptive cutoff fails on strongly disordered configurations.
        system.find.neighbors(method="cutoff", cutoff=5.0)
    system.find.solids(cluster=False)
    return float(np.sum(system.atoms.solid)) / len(atoms)


@as_function_node
def EstimateCalphyTemperatureRange(
    structure: Atoms,
    engine,
    t_start: float = 100.0,
    t_stop: float = 3000.0,
    heating_rate_k_per_ps: float = 75.0,
    check_every: int = 100,
    timestep_fs: float = 2.0,
    pressure_bar: float = 1.0,
    solid_fraction_threshold: float = 0.7,
    hold_solid_fraction: float = 0.95,
    n_hold_steps: int = 5000,
    n_refinements: int = 3,
    window_fraction: float = 0.85,
    min_lambda: float = 0.5,
):
    """Estimate a calphy temperature window an NPT crystal actually survives.

    A coarse Berendsen **NPT** heating ramp locates the temperature at which
    the crystal loses long-range order, then a bisection finds the highest
    temperature at which the *pristine* crystal survives a hold long enough to
    stand in for a calphy run.  That temperature becomes
    ``temperature_stop``; ``temperature`` is placed below it.  Both feed
    :class:`InputClass` directly.

    **The ensemble matters more than anything else here.**  A crystal cannot
    melt without expanding by roughly 5 % linearly, so at fixed volume it
    superheats enormously.  Measured on a 500-atom fcc Al cell at
    ``a = 4.05 Å`` with ``1995--Angelo-J-E--Ni-Al-H``, a potential whose Al
    melts between 300 K and 500 K at 1 bar: NVT keeps it fully crystalline
    (solid fraction 1.00) through 700 K, while NPT at 1 bar melts it
    completely (0.00) already at 500 K.  An earlier NVT version of this node
    returned 1095–2343 K for ``1999--Mishin-Y--Al`` (melting point ≈ 1000 K) —
    a window whose *lower* bound sat above the melting point, so no crossing
    could be found at all.  Everything below is constant-pressure.

    **Order parameter.**  Melting is detected from the pyscal q6 solid
    fraction, which is the same test calphy applies before it raises
    ``MeltedError``, so a window accepted here is one calphy accepts.  Atomic
    displacement is reported but not used as a criterion: it cannot detect
    freezing, because a cooling liquid's displacement falls smoothly through
    any fixed threshold as its diffusivity drops.  A displacement-based
    cooling ramp therefore reports crystallisation that never occurred — for
    both potentials above it "found" one while the solid fraction sat at 0.00
    for all 20 plateaux.

    **Why a hold, not a cooling ramp.**  Short plateaux superheat too, just
    less than NVT: Angelo melts at 680 K on a 2 ps-per-plateau ramp but a
    50 ps hold melts it at 500 K.  A window whose top came from the ramp would
    hand calphy a ``temperature_stop`` its much longer runs cannot survive,
    which is the ``MeltedError`` this node is meant to prevent.  So the ramp
    only brackets the search, and ``n_hold_steps`` — which should be
    comparable to ``InputClass.n_equilibration_steps`` — decides the answer.
    Each probe restarts from the input structure, so a melted configuration
    never contaminates a later one, and a probe that has clearly melted exits
    without running out its remaining steps.

    **Cost.**  Dominated by the holds, so the bisection searches the top half
    of the bracket first (the melting point sits just under the ramp's
    superheating limit) and widens downwards only if every probe there melts.
    Ramp plus three 10 ps probes is roughly 20000 MD steps; on 256 atoms of
    fcc Al with a LAMMPS EAM potential that is 38 s for Angelo (melts at
    ~400 K, so a short ramp) and 66 s for Mishin (~1300 K, a long one).  An
    earlier version using 20 fixed 2 ps plateaux plus four 20 ps holds took
    ~600 s.  Being MD, it is not reproducible to the last kelvin: repeat runs
    on the same potential scatter by roughly 8 %, which is well inside the
    metastability margin the window is chosen with.
    If that is still too slow, raise ``heating_rate_k_per_ps`` — but
    note that a run which has *already* failed with ``MeltedError`` needs no MD
    at all: :func:`CalphyMeltingFromTrajectory` reads the answer out of the
    trajectory calphy already dumped.

    The melting point itself lies *below* ``temperature_max``: a defect-free
    periodic cell has no nucleation site, so it stays metastable somewhat
    above its melting point however long the hold.  The bound is deliberately
    on the cautious side of that margin — an upper bound a little below the
    melting point makes ``FindMeltingTemperature`` extrapolate, while one
    above the superheating limit aborts the whole calphy run with
    ``MeltedError``.  If the crossing does come out at or beyond
    ``temperature_max``, raise ``n_hold_steps`` and re-run rather than simply
    widening the window by hand.  Better still, hand ``temperature_min`` /
    ``temperature_max`` to :func:`CalphyMeltingTemperatureSearch`, which treats them
    as a *starting* window and shifts it until both phases survive and the
    crossing falls inside it.

    **The window must not be as wide as λ allows.**  ``temperature_min`` is a
    fixed ``window_fraction`` of ``temperature_max``, not ``min_lambda`` of it,
    because the *liquid* leg is anchored at ``temperature_min``: reversible
    scaling holds the thermostat there and reaches ``temperature_max`` by
    scaling the potential.  A window spanning λ = 0.5 therefore asks for a
    liquid at half the melting point, which freezes and aborts the run with
    ``SolidifiedError`` — measured for Cu with ``1985--Foiles-S-M--Ni-Cu``,
    where a 692–1384 K window melted the solid leg's structure fine and then
    lost the liquid.  ``window_fraction`` is bounded below by ``min_lambda``
    because
    reversible scaling holds the thermostat at ``temperature`` and scales the
    potential by λ = ``temperature``/T, and its accuracy collapses as λ
    shrinks: for fcc Al with Mishin's potential, sweeping 700→1100 K
    (λ = 0.64) gives a switching hysteresis of 2.2e-03 eV/atom, while
    100→1100 K (λ = 0.09) gives 5.6e-02 — 550× calphy's own 1e-4 warning
    threshold, and a free energy biased by +18 meV/atom.  A wide window is not
    a safe default.

    Raises
    ------
    ValueError
        If the structure is already disordered at ``t_start``, or if no
        melting is seen below ``t_stop``.  Neither case admits a window that
        brackets the melting point, and returning a plausible-looking one
        silently is the failure this node exists to prevent.

    Parameters
    ----------
    engine :
        ASE-compatible engine (must expose a ``.calculator`` attribute, e.g.
        ``GRACE`` or ``Ace``).
    t_start, t_stop : float
        Temperature search bounds (K).  The melting point must lie inside.
        ``t_start`` must be low enough that the crystal is stable there.
    heating_rate_k_per_ps : float
        Ramp rate.  This, not ``t_stop``, is the cost knob: the ramp stops at
        the melting point, so it costs ``(T_melt - t_start) / rate``.  Faster
        ramps superheat more and push ``t_superheat`` up, which only widens
        the bracket the holds then search; 75 K/ps was measured to bracket
        both a good and a bad Al potential.
    check_every : int
        MD steps between order-parameter evaluations.  Also the ramp's
        temperature increment (``rate * check_every * timestep``, 15 K at the
        defaults) and the granularity of the hold early-exit.
    timestep_fs : float
        MD timestep in femtoseconds.  2 fs is safe for most metals; use 1 fs
        for light elements (H, Li) or stiff potentials.
    pressure_bar : float
        Barostat target.  Match the ``pressure`` used in :class:`InputClass`.
    solid_fraction_threshold : float
        Crystalline fraction below which the *ramp* calls a plateau molten.
        The default 0.7 is calphy's own ``tolerance.solid_fraction``.
    hold_solid_fraction : float
        Crystalline fraction a *hold* must retain to count as stable.
        Deliberately much stricter than ``solid_fraction_threshold``, because
        the hold is a 10 ps stand-in for a calphy run of 35 ps or more: a
        configuration that has already decayed part of the way is mid-transition
        and will keep going.  Measured on fcc Al with Mishin's potential,
        holding at 1369 K left a solid fraction of 0.74 — which passes calphy's
        0.7 and then melts anyway.  0.9 is not strict enough either: it returned
        ``temperature_stop`` = 1308 K, which completes a calphy run without
        ``MeltedError`` but at a reversible-scaling dissipation of 9.9e-04
        eV/atom (calphy warns above 1e-4), whereas 0.95 gives 1227 K for the
        same cost.  Tightening it costs no extra MD — the bisection runs the
        same number of probes either way — so there is no reason to relax it.
    n_hold_steps : int
        Maximum MD steps per bisection probe; 5000 at 2 fs = 10 ps, against
        calphy's default 10 ps equilibration plus 25 ps switching.  A probe
        that melts exits early, so only the surviving ones cost the full
        length.  Doubling this to 20 ps was measured to move the answer for
        fcc Al by less than one bisection step, which is why the shorter probe
        is the default; raise it towards ``n_equilibration_steps +
        n_switching_steps`` for a stricter — and lower — upper bound.
    n_refinements : int
        Bisection steps.  Each halves the bracket, so 3 resolves the top half
        of a 0–1550 K bracket to about 100 K.  Coarser than it looks is fine:
        the target is the whole span between the melting point and the
        superheating limit, roughly 20 % of the temperature for a defect-free
        cell, not the melting point itself.
    window_fraction : float
        ``temperature_min / temperature_max``.  Sets how far the two legs are
        pushed into metastability: the solid is superheated by at most
        ``(1 - window_fraction)`` above the melting point and the liquid
        undercooled by at most the same.  0.85 keeps both within roughly 15 %,
        which a defect-free cell survives, while still being wide enough for a
        crossing to be resolved.  Narrower is safer but makes the two G(T)
        curves nearly parallel over the sampled range, so the crossing moves a
        lot for a small error in either.
    min_lambda : float
        Lower bound on ``window_fraction``, so an explicitly small fraction
        still cannot produce a window reversible scaling cannot handle.

    Returns
    -------
    temperature_min, temperature_max : float
        Bounds for :class:`InputClass`.
    ramp : pandas.DataFrame
        One row per plateau and per bisection probe: ``leg``
        (``"heating"`` / ``"hold"``), ``temperature``, ``solid_fraction``,
        ``displacement`` (Å, minimum-image corrected) and
        ``volume_per_atom`` (Å³).  Plotting ``solid_fraction`` against
        ``temperature`` shows the transition and how far the hold pulled the
        bound down from the ramp.
    """
    from ase import units
    from ase.md.nptberendsen import NPTBerendsen
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    def _mean_disp(atoms, ref_pos):
        """Mean per-atom displacement (Å) with minimum-image PBC correction."""
        cell = np.array(atoms.get_cell())
        frac = (atoms.get_positions() - ref_pos) @ np.linalg.inv(cell)
        frac -= np.round(frac)
        return float(np.mean(np.linalg.norm(frac @ cell, axis=1)))

    def _npt(atoms, temperature):
        return NPTBerendsen(
            atoms,
            timestep=timestep_fs * units.fs,
            temperature_K=float(temperature),
            pressure_au=pressure_bar * units.bar,
            taut=100.0 * units.fs,
            taup=1000.0 * units.fs,
            # Typical metallic compressibility, 1.4e-6 bar^-1, in ASE units.
            # The barostat only needs the right order of magnitude to relax
            # the cell.
            compressibility_au=1.4e-6 / units.bar,
            fixcm=True,
        )

    rows = []

    def _record(atoms, T, leg, ref):
        """Classify the current configuration and log one row."""
        fraction = _solid_fraction(atoms)
        rows.append(
            {
                "leg": leg,
                "temperature": float(T),
                "solid_fraction": fraction,
                "displacement": _mean_disp(atoms, ref),
                "volume_per_atom": atoms.get_volume() / len(atoms),
            }
        )
        return fraction

    atoms = structure.copy()
    atoms.calc = engine.calculator
    MaxwellBoltzmannDistribution(atoms, temperature_K=t_start, rng=None)
    dyn = _npt(atoms, t_start)
    dyn.run(check_every)  # equilibrate at t_start

    start_fraction = _solid_fraction(atoms)
    if start_fraction < solid_fraction_threshold:
        raise ValueError(
            f"the structure is already disordered at t_start = {t_start:g} K "
            f"and {pressure_bar:g} bar (solid fraction "
            f"{start_fraction:.2f} < {solid_fraction_threshold:g}), so "
            "there is no solid branch to bracket. Lower t_start, or check the "
            "potential: an alloy potential applied to a pure element often has "
            "a melting point far below the real one."
        )

    # ── Continuous heating ramp: bracket the search from above ─────────────
    # The setpoint advances every ``check_every`` steps, which at the default
    # 75 K/ps and 2 fs is a 15 K increment -- a linear ramp for all practical
    # purposes, and it stops the moment the crystal goes, instead of running
    # out a coarse plateau grid.
    kelvin_per_check = heating_rate_k_per_ps * timestep_fs * check_every / 1000.0
    t_superheat = None
    T = t_start
    while T < t_stop:
        T = min(T + kelvin_per_check, t_stop)
        dyn.set_temperature(temperature_K=float(T))
        ref = atoms.get_positions().copy()
        dyn.run(check_every)
        if _record(atoms, T, "ramp", ref) < solid_fraction_threshold:
            t_superheat = float(T)
            break
    if t_superheat is None:
        raise ValueError(
            f"no melting between {t_start:g} K and {t_stop:g} K at "
            f"{pressure_bar:g} bar — the melting point is not bracketed and any "
            "window returned would be a guess. Raise t_stop, or lower "
            "heating_rate_k_per_ps so the barostat can follow the expansion."
        )

    # ── Refine: highest temperature a pristine crystal survives a hold ─────
    def _survives_hold(T):
        probe = structure.copy()
        probe.calc = engine.calculator
        MaxwellBoltzmannDistribution(probe, temperature_K=float(T), rng=None)
        hold = _npt(probe, T)
        for _ in range(max(1, n_hold_steps // check_every)):
            ref = probe.get_positions().copy()
            hold.run(check_every)
            # Give up on a probe as soon as it is unambiguously molten; a
            # crystal above its superheating limit goes within a couple of ps,
            # and the remaining steps would only confirm it.
            if _solid_fraction(probe) < solid_fraction_threshold:
                break
        return _record(probe, T, "hold", ref) >= hold_solid_fraction

    # The melting point sits just under the ramp's superheating limit, so
    # bisect the top half of the bracket first and only widen if that misses.
    temperature_max = _highest_stable_temperature(
        _survives_hold, 0.5 * t_superheat, t_superheat, n_refinements
    )
    if temperature_max is None and 0.5 * t_superheat > t_start:
        temperature_max = _highest_stable_temperature(
            _survives_hold, t_start, 0.5 * t_superheat, n_refinements
        )
    if temperature_max is None:
        raise ValueError(
            f"the crystal melted at every probe between {t_start:g} K and "
            f"{t_superheat:g} K over {n_hold_steps * timestep_fs / 1000:g} ps, so "
            f"the melting point is at or below t_start = {t_start:g} K. Lower "
            "t_start, or check the potential."
        )
    temperature_min = max(max(window_fraction, min_lambda) * temperature_max, 10.0)

    ramp = pd.DataFrame(rows)
    return temperature_min, temperature_max, ramp


@as_function_node
def CalphyMeltingFromTrajectory(
    simfolder: str,
    solid_fraction_threshold: float = 0.7,
    iteration: int = 0,
    direction: str = "forward",
):
    """Recover where a calphy reversible-scaling sweep melted, from its dumps.

    ``MeltedError`` aborts a run without saying at which temperature the
    crystal went.  But calphy dumps the sweep trajectory as it goes (every
    ``n_print_steps``), and the melt check happens *after* the sweep finishes,
    so on failure both the trajectory and the λ log are complete on disk.
    Reversible scaling holds the thermostat at ``temperature`` and scales the
    potential by λ, so frame *k* was sampled at ``T = temperature / λ_k``;
    classifying each frame with the same pyscal q6 test calphy uses turns the
    dump into a solid-fraction-versus-temperature curve.

    This costs **no MD** — the run has already been paid for.  Where
    :func:`EstimateCalphyTemperatureRange` predicts a safe window from scratch,
    this reads the answer off a window that turned out not to be safe.  Use it
    to cap ``temperature_stop`` after a failure, and on a *successful* run to
    confirm the solid never partially melted: partial melting does not raise
    anything, it just biases the free energy (it does show up as a large
    ``rs_max_dissipation`` in :class:`CalphyDiagnostics`, but that cannot say
    where).

    Measured on 500 atoms of fcc Al, sweeping 300→900 K:
    ``1995--Angelo-J-E--Ni-Al-H`` (which raises ``MeltedError``) gives solid
    fractions 1.000, 1.000, 0.998, 0.906, 0.396, 0.000 at 300, 346, 409, 500,
    643, 900 K → ``t_melt`` = 558 K, consistent with the 300–500 K melting
    range an independent NPT run gives for that potential.  A
    ``1999--Mishin-Y--Al`` sweep over 700→1100 K stays at 1.00 throughout and
    returns ``nan``.

    **Limits.**  The resolution is one dump interval, which at
    ``n_switching_steps=25000`` and ``n_print_steps=1000`` is 26 frames — good
    to a few percent of the window, and the interpolation between the last
    crystalline frame and the first molten one is only as good as that spacing.
    It also only works for a failure *during* a sweep: a ``fe``-mode run has no
    temperature axis to place the melting on, and ``avg.dat`` starts logging
    after the sweep, too late to localise anything.

    Parameters
    ----------
    simfolder : str
        calphy run directory (the one named in the note attached to the
        exception; it holds ``input_file.yaml`` and the ``ts.*`` files).  The
        temperature range, ``n_print_steps`` and phase are read from there, so
        nothing needs restating here.
    solid_fraction_threshold : float
        Crystalline fraction below which a frame counts as molten; calphy's own
        ``tolerance.solid_fraction`` default is 0.7.
    iteration : int
        Which ``n_iterations`` sweep to read (1-based).  ``0`` picks the highest
        one present, which is where a failure aborted.
    direction : str
        ``"forward"`` (heating, λ: 1 → t0/t_stop) or ``"backward"`` (cooling).
        Melting normally happens on the forward leg.

    Returns
    -------
    t_melt : float
        Temperature (K) at which the solid fraction first crosses
        ``solid_fraction_threshold``, interpolated between the bracketing
        frames; ``nan`` if the sweep stayed crystalline throughout.  Set
        ``temperature_stop`` safely below this.
    sweep : pandas.DataFrame
        Per-frame ``step``, ``lambda``, ``temperature`` and ``solid_fraction``.
    """
    t_melt, sweep = _melting_from_trajectory(
        simfolder, solid_fraction_threshold, iteration, direction
    )
    return t_melt, sweep


def _melting_from_trajectory(
    simfolder, solid_fraction_threshold=0.7, iteration=0, direction="forward"
):
    """Body of :func:`CalphyMeltingFromTrajectory`, callable from Python.

    :func:`CalphyMeltingTemperatureSearch` has to read this off a leg that *raised*,
    where there is no node to wire it to -- the exception is caught inside the
    search.  Same extraction as :func:`_free_energy_with_temp`.
    """
    import yaml
    from ase.io import read

    if direction not in ("forward", "backward"):
        raise ValueError(f"direction must be 'forward' or 'backward', got {direction!r}")

    input_file = os.path.join(simfolder, "input_file.yaml")
    if not os.path.isfile(input_file):
        raise FileNotFoundError(
            f"{input_file} not found -- {simfolder} is not a calphy run directory"
        )
    with open(input_file) as fh:
        calculation = yaml.safe_load(fh)["calculations"][0]

    temperatures = np.atleast_1d(calculation["temperature"]).astype(float)
    if temperatures.size < 2:
        raise ValueError(
            f"{simfolder} is a single-temperature (mode 'fe') run: temperature "
            f"= {list(temperatures)}.  There is no sweep, so a melting event "
            "there cannot be placed on a temperature axis."
        )
    t0 = float(temperatures[0])

    n_print_steps = int(calculation.get("n_print_steps") or 0)
    if n_print_steps <= 0:
        raise ValueError(
            "calphy dumped no trajectory for this run (n_print_steps = "
            f"{n_print_steps}).  Set InputClass.n_print_steps > 0 -- it is what "
            "makes this diagnosis possible after the fact."
        )

    if not iteration:
        # A failed run aborted somewhere in the last iteration it started, and
        # the caller catching the exception has no way to know which that was.
        prefix = f"ts.{direction}_"
        written = [
            int(name[len(prefix) : -len(".dat")])
            for name in os.listdir(simfolder)
            if name.startswith(prefix)
            and name.endswith(".dat")
            and name[len(prefix) : -len(".dat")].isdigit()
        ]
        if not written:
            raise FileNotFoundError(
                f"no ts.{direction}_*.dat in {simfolder}: the run never reached "
                "the reversible-scaling sweep, so nothing was dumped to place a "
                "melting event on a temperature axis."
            )
        iteration = max(written)

    ts_file = os.path.join(simfolder, f"ts.{direction}_{iteration}.dat")
    traj_file = os.path.join(simfolder, f"traj.ts.{direction}_{iteration}.dat")
    for path in (ts_file, traj_file):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{path} not found.  Available: "
                f"{sorted(f for f in os.listdir(simfolder) if f.startswith('ts.') or f.startswith('traj.'))}"
            )

    # calphy prints one row per MD step to ts.*.dat, so row k * n_print_steps
    # holds the lambda in force when frame k was dumped.
    lambdas = np.loadtxt(ts_file, usecols=3, comments="#")
    frames = read(traj_file, index=":", format="lammps-dump-text")
    steps = np.minimum(np.arange(len(frames)) * n_print_steps, len(lambdas) - 1)
    frame_lambdas = lambdas[steps]

    sweep = pd.DataFrame(
        {
            "step": steps,
            "lambda": frame_lambdas,
            "temperature": t0 / frame_lambdas,
            "solid_fraction": [_solid_fraction(frame) for frame in frames],
        }
    )

    molten = np.flatnonzero(
        sweep["solid_fraction"].to_numpy() < solid_fraction_threshold
    )
    if molten.size == 0:
        return float("nan"), sweep

    first = int(molten[0])
    if first == 0:
        # Molten in the very first frame: the sweep never had a solid to lose,
        # so the best available statement is "at or below the start".
        return float(sweep["temperature"][0]), sweep
    f_hi, f_lo = sweep["solid_fraction"][first - 1], sweep["solid_fraction"][first]
    t_lo, t_hi = sweep["temperature"][first - 1], sweep["temperature"][first]
    weight = (f_hi - solid_fraction_threshold) / (f_hi - f_lo)
    t_melt = float(t_lo + weight * (t_hi - t_lo))
    return t_melt, sweep


@group_node(
    "T_melt",
    "bracketed",
    "fig",
    "solid_diagnostics",
    "liquid_diagnostics",
)
def MeltingTemperature(
    structure,
    potential: str = "",
    n_equilibration_steps: int = 10000,
    n_switching_steps: int = 25000,
    n_iterations: int = 1,
    stdev: float = 0.5,
    working_directory: str = "calphy_workdir",
):
    """Melting temperature from a bulk supercell and a potential, two-pass approach.

    Wraps the complete two-pass protocol into a single reusable block:

    1. **SuperheatingWindow** — a cheap NPT ramp measures the temperature at
       which the crystal loses order under the *same* potential calphy will use.
       That temperature becomes ``temperature_stop`` for pass 1.
    2. **Pass 1** — solid and liquid reversible-scaling legs run in the window
       the estimator proposed.  ``SuggestTemperatureWindow`` reads the crossing
       off the two G(T) curves (or extrapolates if they do not cross) and
       recommends the next window.
    3. **Pass 2** — the two legs run again in the window pass 1 suggested, now
       centred on the estimated crossing.  ``bracketed = True`` on pass 2 means
       the crossing was actually measured inside the window, which is the only
       version of T_m worth reporting.  ``False`` means the second window still
       missed and a third pass (or ``CalphyMeltingTemperatureSearch``) is needed.

    Both passes share a single ``InputClass`` via ``ApplyTemperatureWindow``,
    which stamps the window onto a copy before each pass so that
    ``n_switching_steps``, ``n_equilibration_steps``, and all other settings
    are guaranteed identical — free energies computed with different switching
    lengths cannot be compared, and two hand-written copies of the same settings
    can drift apart silently.

    Parameters
    ----------
    structure : Atoms
        Bulk supercell (typically 3×3×3 or 4×4×4 of the primitive cell to
        suppress surface effects).  Passed as-is to the solid leg; a rattled
        copy is used for the liquid leg start point.
    potential : str
        LAMMPS potential name as returned by ``GetPotential``.  The same name
        is used for the superheating estimate, both solid legs and both liquid
        legs — so the window and the free energies are always from the same
        force field.
    n_equilibration_steps : int
        calphy equilibration steps per leg.  calphy's default is 25000; the
        default here of 10000 is a compromise between speed and convergence.
    n_switching_steps : int
        calphy switching steps per leg.  calphy's default is 50000; 25000 is
        sufficient for a first result.  If ``rs_max_dissipation`` in the
        diagnostics exceeds ~1e-4 eV/atom, raise this towards 50000.
    n_iterations : int
        Independent switching runs per leg.  With ``n_iterations = 1`` (the
        default) there is no statistical error bar on T_m; raise to 3 or more
        for publication-quality results.
    stdev : float
        Displacement standard deviation for ``Rattle``, used to create the
        disordered starting point for the liquid leg.  With
        ``InputClass.melting_cycle = True`` (the default), calphy melts the
        rattled structure at ``temperature_high`` before the liquid leg starts,
        so the stdev only needs to produce a non-crystalline seed — the default
        0.5 Å works for most metals.
    working_directory : str
        Parent directory for calphy's simulation folders.  Each leg writes into
        a randomly-named subdirectory inside it, so all legs can share one path.

    Returns
    -------
    T_melt : float
        Melting temperature in K from pass 2.  Trust this number only when
        ``bracketed = True``.
    bracketed : bool
        ``True`` if the G_solid / G_liquid crossing falls inside the pass-2
        window, i.e. both phases were metastable simultaneously during the
        measurement.  ``False`` means T_m is an extrapolation — use it as the
        starting point for ``CalphyMeltingTemperatureSearch`` rather than as a result.
    fig : Figure
        G(T) plot for pass 2, showing the solid and liquid free energies and
        the fitted crossing at T_melt.  Check visually that the curves are
        smooth and actually cross inside the plotted range.
    solid_diagnostics, liquid_diagnostics : CalphyDiagnostics
        Per-leg quality bundles from pass 2.  The most important field is
        ``rs_max_dissipation``: above ~1e-4 eV/atom the reversible-scaling
        sweep was too fast.  Wire either into ``CalphyDiagnosticsTable`` or
        ``PlotCalphyHysteresis`` to inspect.
    """
    from pyiron_nodes.atomistic.property.calphy import (
        ApplyTemperatureWindow,
        EstimateCalphyTemperatureRange,
        InputClass,
        LiquidFreeEnergyWithTemp,
        PlotSolidLiquidFreeEnergy,
        SolidFreeEnergyWithTemp,
        SuggestTemperatureWindow,
    )
    from pyiron_nodes.atomistic.engine.ase import LammpsEngine
    from pyiron_nodes.atomistic.structure.transform import Rattle
    from core import Workflow

    inner = Workflow("MeltingTemperature")

    # Melt seed: a rattled copy of the supercell as the liquid leg's start.
    # melting_cycle=True (the InputClass default) re-melts this at temperature_high
    # and verifies it melted, so the Rattle only has to break crystalline order.
    inner.melt_seed = Rattle(structure=structure, stdev=stdev)

    # Shared sampling settings — one object for both passes so n_switching_steps
    # and n_equilibration_steps are guaranteed identical everywhere.
    inner.settings = InputClass(
        n_equilibration_steps=n_equilibration_steps,
        n_switching_steps=n_switching_steps,
        n_iterations=n_iterations,
    )

    # Superheating window: NPT ramp with the *same* potential calphy will use.
    # A window measured with a different potential is silently meaningless.
    inner.engine = LammpsEngine(potential=potential)
    inner.window = EstimateCalphyTemperatureRange(
        structure=structure,
        engine=inner.engine,
        t_start=100.0,
        t_stop=6000.0,
    )

    # ── Pass 1 ────────────────────────────────────────────────────────────────
    # Run in the window the superheating estimate proposed.  The two G(T) curves
    # may not cross — above T_m the liquid is stable at temperature_min and the
    # defect-free solid superheats past temperature_max, so both legs finish
    # cleanly.  SuggestTemperatureWindow reads where the crossing is (or would be)
    # and returns the window centred on it for pass 2.

    inner.inp1 = ApplyTemperatureWindow(
        inp=inner.settings,
        temperature_min=inner.window.outputs.temperature_min,
        temperature_max=inner.window.outputs.temperature_max,
    )
    inner.solid1 = SolidFreeEnergyWithTemp(
        inp=inner.inp1,
        structure=structure,
        potential=potential,
        working_directory=working_directory,
    )
    inner.liquid1 = LiquidFreeEnergyWithTemp(
        inp=inner.inp1,
        structure=inner.melt_seed,
        potential=potential,
        working_directory=working_directory,
    )
    inner.crossing1 = SuggestTemperatureWindow(
        temp_solid=inner.solid1.outputs.temperature,
        fe_solid=inner.solid1.outputs.free_energy,
        temp_liquid=inner.liquid1.outputs.temperature,
        fe_liquid=inner.liquid1.outputs.free_energy,
        fit_order=1,
    )

    # ── Pass 2 ────────────────────────────────────────────────────────────────
    # Run in the window pass 1 suggested (centred on the extrapolated or measured
    # crossing).  fit_order=2 here: once the window is likely to contain the
    # crossing, curvature in G(T) is real and worth fitting.

    inner.inp2 = ApplyTemperatureWindow(
        inp=inner.settings,
        temperature_min=inner.crossing1.outputs.temperature_min,
        temperature_max=inner.crossing1.outputs.temperature_max,
    )
    inner.solid2 = SolidFreeEnergyWithTemp(
        inp=inner.inp2,
        structure=structure,
        potential=potential,
        working_directory=working_directory,
    )
    inner.liquid2 = LiquidFreeEnergyWithTemp(
        inp=inner.inp2,
        structure=inner.melt_seed,
        potential=potential,
        working_directory=working_directory,
    )
    inner.crossing2 = SuggestTemperatureWindow(
        temp_solid=inner.solid2.outputs.temperature,
        fe_solid=inner.solid2.outputs.free_energy,
        temp_liquid=inner.liquid2.outputs.temperature,
        fe_liquid=inner.liquid2.outputs.free_energy,
        fit_order=2,
    )
    inner.fig = PlotSolidLiquidFreeEnergy(
        temp_solid=inner.solid2.outputs.temperature,
        fe_solid=inner.solid2.outputs.free_energy,
        temp_liquid=inner.liquid2.outputs.temperature,
        fe_liquid=inner.liquid2.outputs.free_energy,
        T_melt=inner.crossing2.outputs.T_melt,
        fit_order=2,
    )

    return (
        inner.crossing2.outputs.T_melt,
        inner.crossing2.outputs.bracketed,
        inner.fig,
        inner.solid2.outputs.diagnostics,
        inner.liquid2.outputs.diagnostics,
    )
