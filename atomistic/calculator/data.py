from dataclasses import field
from typing import Optional, Literal

from core import as_inp_dataclass_node, as_out_dataclass_node
from core.data_fields import DataArray, EmptyArrayField


# only conceptual, not truly implemented
def wfMetaData(log_level=0, doc=""):
    return {"log_level": log_level, "doc": doc}


@as_out_dataclass_node
class OutputSEFS:
    import numpy as np

    structures: Optional[np.ndarray] = field(
        default=None, metadata=wfMetaData(log_level=10)
    )
    energies: Optional[float] = field(default=None, metadata=wfMetaData(log_level=0))
    forces: Optional[np.ndarray] = field(default=None, metadata=wfMetaData(log_level=0))
    stresses: Optional[np.ndarray] = field(
        default=None, metadata=wfMetaData(log_level=10)
    )


@as_out_dataclass_node
class OutputCalcStatic:
    import numpy as np
    from ase import Atoms

    energy: Optional[float] = field(default=None, metadata=wfMetaData(log_level=0))
    force: Optional[np.ndarray] = field(default=None, metadata=wfMetaData(log_level=0))
    stress: Optional[np.ndarray] = field(
        default=None, metadata=wfMetaData(log_level=10)
    )
    structure: Optional[Atoms] = field(default=None, metadata=wfMetaData(log_level=10))

    atomic_energies: Optional[float] = field(
        default=None,
        metadata=wfMetaData(
            log_level=0, doc="per atom energy, only if supported by calculator"
        ),
    )


@as_out_dataclass_node
class OutputCalcStaticList:
    import numpy as np

    energies_pot: Optional[np.ndarray] = field(
        default=None, metadata=wfMetaData(log_level=0)
    )
    forces: Optional[np.ndarray] = field(default=None, metadata=wfMetaData(log_level=0))
    stresses: Optional[np.ndarray] = field(
        default=None, metadata=wfMetaData(log_level=10)
    )
    structures: Optional[np.ndarray] = field(
        default=None, metadata=wfMetaData(log_level=10)
    )
    is_converged: bool = False
    iter_steps: int = 0


@as_out_dataclass_node
class OutputCalcMinimize:
    initial: Optional[OutputCalcStatic] = field(
        default_factory=lambda: OutputCalcStatic.pure_dataclass(),
        metadata=wfMetaData(log_level=0),
    )
    final: Optional[OutputCalcStatic] = field(
        default_factory=lambda: OutputCalcStatic.pure_dataclass(),
        metadata=wfMetaData(log_level=0),
    )
    is_converged: bool = False
    iter_steps: int = 0


@as_out_dataclass_node
class OutputCalcMD:
    cells: DataArray = EmptyArrayField()
    energies_tot: DataArray = EmptyArrayField()
    energies_pot: DataArray = EmptyArrayField()
    forces: DataArray = EmptyArrayField()
    indices: DataArray = EmptyArrayField()
    natoms: DataArray = EmptyArrayField()
    positions: DataArray = EmptyArrayField()
    pressures: DataArray = EmptyArrayField()
    steps: DataArray = EmptyArrayField()
    temperatures: DataArray = EmptyArrayField()
    unwrapped_positions: DataArray = EmptyArrayField()
    velocities: DataArray = EmptyArrayField()
    volumes: DataArray = EmptyArrayField()
    species: DataArray = EmptyArrayField()


@as_inp_dataclass_node
class InputCalcMD:
    temperature: float = (
        300  # in K, we need more than one temperature field to support rescaling, but this is the default/initial temperature
    )
    n_ionic_steps: int = 10_000
    n_print: int = 100
    pressure: Optional[float] = None
    time_step: float = 1.0
    temperature_damping_timescale: Optional[float] = 100.0
    pressure_damping_timescale: Optional[float] = 1000.0
    seed: int = 42
    tloop: Optional[int] = (
        None  # number of steps to loop over for temperature rescaling, if applicable, should be int???
    )
    initial_temperature: Optional[float] = None  # FIXME
    langevin: bool = False
    delta_temp: Optional[float] = None
    delta_press: Optional[float] = None


@as_inp_dataclass_node
class InputCalcMinimize:
    """
        Sets parameters required for minimization.

    Parameters
    e_tol (float) – If the magnitude of difference between energies of two consecutive steps is lower than or equal to e_tol, the minimisation terminates. (Default is 0.0 eV.)

    f_tol (float) – If the magnitude of the global force vector at a step is lower than or equal to f_tol, the minimisation terminates. (Default is 1e-4 eV/angstrom.)

    max_iter (int) – Maximum number of minimisation steps to carry out. If the minimisation converges before max_iter steps, terminate at the converged step. If the minimisation does not converge up to max_iter steps, terminate at the max_iter step. (Default is 100000.)

    pressure (None/float/numpy.ndarray/list) – Target pressure. If set to None, an NVE or an NVT calculation is performed. A list of up to length 6 can be given to specify xx, yy, zz, xy, xz, and yz components of the pressure tensor, respectively. These values can mix floats and None to allow only certain degrees of cell freedom to change. (Default is None, run isochorically.)

    n_print (int) – Write (dump or print) to the output file every n steps (Default: 100)

    style ('cg'/'sd'/other values from Lammps docs) – The style of the numeric minimization, either conjugate gradient, steepest descent, or other keys permissible from the Lammps docs on ‘min_style’. (Default is ‘cg’ – conjugate gradient.)

    rotation_matrix (numpy.ndarray) – The rotation matrix from the pyiron to Lammps coordinate frame.
    """

    e_tol: float = 0.0
    f_tol: float = 1e-4
    max_iter: int = 1_000_000
    pressure: float = None
    n_print: int = 100
    style: str = "cg"


@as_inp_dataclass_node
class InputCalcStatic:
    pass  # LammpsControl.calc_static takes exactly zero arguments, and currently we
    # have the input objects matching their respective LammpsControl counterparts


@as_inp_dataclass_node
class InputSCF:
    """
    Basic inputs for a static (SCF) DFT calculation.
    """

    kpoints: str
    kpoint_offset: str = "0.5 0.5 0.5"
    functional: str = "PBE"
    energy_cutoff: float = 400.0
    smearing_type: Literal["gaussian", "methfessel-paxton", "fermi-dirac"] = "gaussian"
    smearing_width: float = 0.2
    smearing_order: int = 1
    electronic_convergence: float = 1e-6
    num_electronic_steps: int = 100


@as_inp_dataclass_node
class InputMinimizationVASP:
    """Ionic relaxation settings for VASP — minimization algorithms only (no MD).

    ``algorithm`` maps to IBRION: ConjugateGradient → 2, RMM-DIIS → 1,
    DampedMolecularDynamics → 3 (a damped relaxation scheme, not true MD).
    """

    algorithm: Literal["ConjugateGradient", "RMM-DIIS", "DampedMolecularDynamics"] = (
        "ConjugateGradient"  # IBRION 2 / 1 / 3
    )
    max_ionic_steps: int = 100  # NSW, number of ionic steps
    ionic_convergence: float = -0.01  # EDIFFG; negative = max force (eV/Å)
    isif: int = 2  # ISIF: 2 = ions only, 3 = ions + cell shape + volume


@as_inp_dataclass_node
class InputMDVASP:
    """Molecular dynamics settings for VASP with a Langevin thermostat.

    Field names follow the LAMMPS ``InputCalcMD`` convention rather than the raw
    INCAR tags. The mapping onto INCAR is done in
    ``pyiron_nodes.atomistic.engine.vasp_new._build_incar``:

    ensemble ``"NVT"``           → ISIF 2 (fixed cell)
    ensemble ``"NpT"``           → ISIF 3 (cell shape + volume free)
    temperature                  → TEBEG
    final_temperature            → TEEND (defaults to ``temperature``)
    n_ionic_steps                → NSW
    n_print                      → NBLOCK
    time_step                    → POTIM
    pressure                     → PSTRESS (GPa here, kBar in the INCAR)
    temperature_damping_timescale→ LANGEVIN_GAMMA = 1000 / tau, one entry per species
    pressure_damping_timescale   → LANGEVIN_GAMMA_L = 1000 / tau (NpT only)
    lattice_mass                 → PMASS (NpT only)
    seed                         → RANDOM_SEED

    Only the Langevin thermostat is supported for the moment (MDALGO 3), so
    there is no thermostat selector; ISYM is switched off, as symmetry must not
    be imposed during MD.
    """

    ensemble: Literal["NVT", "NpT"] = "NVT"
    temperature: float = 300.0  # in K, initial/target temperature
    final_temperature: Optional[float] = None  # in K, None → isothermal at temperature
    n_ionic_steps: int = 10_000
    n_print: int = 100
    time_step: float = 1.0  # in fs
    pressure: Optional[float] = None  # in GPa, required for NpT, ignored for NVT
    temperature_damping_timescale: float = 100.0  # in fs, Langevin friction on the ions
    pressure_damping_timescale: float = 1000.0  # in fs, Langevin friction on the cell
    lattice_mass: float = (
        1000.0  # in amu, fictitious mass of the cell degrees of freedom
    )
    seed: Optional[int] = None  # None → let VASP pick its own random seed


@as_inp_dataclass_node
class InputDipoleCorrection:
    """Dipole correction settings for VASP (e.g. asymmetric slabs)."""

    direction: int = 3  # IDIPOL: 1 / 2 / 3 = a / b / c lattice direction
    ldipol: bool = True  # LDIPOL: switch on the potential/dipole correction


@as_inp_dataclass_node
class AdditionalInputFlags:
    """Adding additional Input Flags that are not present in the predefined dataclasses."""

    key1: Optional[str] = None
    value1: Optional[str] = None
    key2: Optional[str] = None
    value2: Optional[str] = None
    key3: Optional[str] = None
    value3: Optional[str] = None
    key4: Optional[str] = None
    value4: Optional[str] = None
    key5: Optional[str] = None
    value5: Optional[str] = None

    def to_dict(self) -> dict:

        extra_flags = {}

        for i in range(1, 6):  # 1 to 5
            key = getattr(self, f"key{i}")
            value = getattr(self, f"value{i}")

            if key is not None:
                extra_flags[key] = value

        return extra_flags
