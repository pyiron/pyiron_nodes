from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import Incar, Kpoints
from pymatgen.io.vasp.outputs import Vasprun

from core import as_function_node
from pyiron_nodes.atomistic.calculator.data import (
    InputMDVASP,
    InputDipoleCorrection,
    InputMinimizationVASP,
    InputSCF,
    OutputCalcMD,
    OutputCalcMinimize,
    OutputCalcStatic,
    AdditionalInputFlags,
)

# ── INCAR enum lookups ────────────────────────────────────────────────────────


def _ISMEAR(smearing_type, smearing_order):
    if smearing_type == "fermi-dirac":
        return -1
    elif smearing_type == "gaussian":
        return 0
    elif smearing_type == "methfessel-paxton":
        if smearing_order < 1:
            raise ValueError("Methfessel-Paxton order must be >= 1")
        return smearing_order
    else:
        raise ValueError(
            f"Unknown smearing_type {smearing_type!r}; expected one of "
            "'gaussian', 'methfessel-paxton', 'fermi-dirac'"
        )


_IBRION_MINIMIZE = {
    "ConjugateGradient": 2,
    "RMM-DIIS": 1,
    "DampedMolecularDynamics": 3,
}

# ── POTCAR config ─────────────────────────────────────────────────────────────
# testing the comit


def read_potcar_config(config_file: Path) -> dict:
    config_data = {}
    try:
        with open(config_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    config_data[key.strip()] = value.strip()

        pyiron_vasp_resources = config_data.get("pyiron_vasp_resources", "")
        default_POTCAR_set = config_data.get("default_POTCAR_set", "")
        default_functional = config_data.get("default_functional", "PBE")

        # construct path directly: potpawPBE → potpaw_PBE
        potcar_dir = default_POTCAR_set.replace("potpaw", "potpaw_")
        config_data["default_POTCAR_path"] = os.path.join(
            pyiron_vasp_resources, potcar_dir
        )
        config_data["default_functional"] = default_functional

    except FileNotFoundError:
        pass  # config absent — callers must supply potcar_lib_path explicitly

    return config_data


_potcar_config = read_potcar_config(Path.home() / ".pyiron_vasp_config")
_default_functional: str = _potcar_config.get("default_functional", "PBE")
_default_potcar_lib_path: str = _potcar_config.get("default_POTCAR_path", "")
_default_vasp_command: str = _potcar_config.get("vasp_command", "mpiexec -n 1 vasp_std")

_RESOURCES_DIR = Path(__file__).parent / "vasp_resources"
_POTCAR_CSV = {
    "PBE": str(_RESOURCES_DIR / "vasp_pseudopotential_PBE_data.csv"),
    "LDA": str(_RESOURCES_DIR / "vasp_pseudopotential_LDA_data.csv"),
}


# ── Input resource dataclass ──────────────────────────────────────────────────


@dataclass
class VaspInput:
    """Combined VASP calculation settings produced by ``MergeVaspInput``.

    ``scf`` is always present; the others are layered on top when supplied. The
    sub-objects are kept nested (not flattened) so each category stays editable
    and new ones can be added without touching the existing ports.
    """

    scf: InputSCF
    minimization: Optional[InputMinimizationVASP] = None
    md: Optional[InputMDVASP] = None
    dipole_correction: Optional[InputDipoleCorrection] = None
    extra_incar: Optional[dict] = None


@dataclass
class VaspInputResources:
    structure: Atoms  # ASE Atoms — compatible with Bulk and other structure nodes
    calc: Optional[VaspInput]
    potcar_lib_path: str = field(
        default_factory=lambda: _default_potcar_lib_path
    )  # base path to POTCAR folders
    working_directory: Optional[str] = None
    potcar_symbols: Optional[list[str]] = None  # override default CSV symbol selection
    extra_incar: Optional[dict] = None  # additional INCAR tags beyond VaspInput


# ── Private helpers ───────────────────────────────────────────────────────────


def _ordered_elements(atoms: Atoms) -> list[str]:
    elements, prev = [], None
    for sym in atoms.get_chemical_symbols():
        if sym != prev:
            elements.append(sym)
            prev = sym
    return elements


def _get_potcar_paths(atoms: Atoms, functional: str, lib_path: str) -> list[str]:
    df = pd.read_csv(_POTCAR_CSV[functional])
    paths = []
    for el in _ordered_elements(atoms):
        row = df[(df["symbol"] == el) & (df["default"] == True)]  # noqa: E712
        potential_name = row.potential_name.values[0]
        paths.append(os.path.join(lib_path, potential_name, "POTCAR"))
    return paths


def _build_incar(
    calc: VaspInput, extra: dict | None = None, structure: Atoms | None = None
) -> Incar:
    if calc.minimization is not None and calc.md is not None:
        raise ValueError(
            "minimization and md are mutually exclusive — both control IBRION/NSW. "
            "Supply only one of them to MergeVaspInput."
        )

    scf = calc.scf

    # ── base SCF tags (every run) ─────────────────────────────────────────────
    tags = {
        "ENCUT": scf.energy_cutoff,
        "EDIFF": scf.electronic_convergence,
        "NELM": scf.num_electronic_steps,
        "ISMEAR": _ISMEAR(scf.smearing_type, scf.smearing_order),
        "SIGMA": scf.smearing_width,
        # static defaults — overridden below if minimization/md is supplied
        "IBRION": -1,
        "NSW": 0,
    }

    # ── ionic minimization (optional) ─────────────────────────────────────────
    if calc.minimization is not None:
        mini = calc.minimization
        tags["IBRION"] = _IBRION_MINIMIZE[mini.algorithm]
        tags["NSW"] = mini.max_ionic_steps
        tags["EDIFFG"] = mini.ionic_convergence
        tags["ISIF"] = mini.isif

    # ── molecular dynamics (optional, Langevin thermostat only) ───────────────
    if calc.md is not None:
        md = calc.md
        if structure is None:
            raise ValueError(
                "a structure is required to build an MD INCAR — LANGEVIN_GAMMA "
                "needs one friction coefficient per species in the POSCAR."
            )
        n_species = len(_ordered_elements(structure))

        tags["IBRION"] = 0
        tags["MDALGO"] = 3  # Langevin
        tags["ISYM"] = 0  # symmetry must not be imposed during MD
        tags["NSW"] = md.n_ionic_steps
        tags["NBLOCK"] = md.n_print
        tags["POTIM"] = md.time_step
        tags["TEBEG"] = md.temperature
        tags["TEEND"] = (
            md.temperature if md.final_temperature is None else md.final_temperature
        )
        # damping timescales are given in fs, LANGEVIN_GAMMA* are frictions in ps^-1
        tags["LANGEVIN_GAMMA"] = [1000.0 / md.temperature_damping_timescale] * n_species

        if md.ensemble == "NVT":
            tags["ISIF"] = 2  # ions move, cell is fixed
        elif md.ensemble == "NpT":
            if md.pressure is None:
                raise ValueError("ensemble 'NpT' requires md.pressure to be set (GPa).")
            tags["ISIF"] = 3  # ions + cell shape + volume
            tags["PSTRESS"] = md.pressure * 10.0  # GPa → kBar
            tags["PMASS"] = md.lattice_mass
            tags["LANGEVIN_GAMMA_L"] = 1000.0 / md.pressure_damping_timescale
        else:
            raise ValueError(
                f"Unknown ensemble {md.ensemble!r}; expected 'NVT' or 'NpT'."
            )

        if md.seed is not None:
            tags["RANDOM_SEED"] = [md.seed, 0, 0]  # VASP wants seed + two counters

    # ── dipole correction (optional) ──────────────────────────────────────────
    if calc.dipole_correction is not None:
        dip = calc.dipole_correction
        tags["LDIPOL"] = dip.ldipol
        tags["IDIPOL"] = dip.direction

    if extra:
        tags.update(extra)
    return Incar(tags)


# ── Output helpers ────────────────────────────────────────────────────────────
# ``ParseVaspOutput`` fills the same calculator dataclasses that the LAMMPS
# engine uses (``OutputCalcStatic`` / ``OutputCalcMinimize`` / ``OutputCalcMD``),
# so downstream nodes do not care which engine produced the data.

_kB = 8.617333262e-5  # eV/K
_kBar_to_GPa = 0.1


def _static_from_ionic_step(step: dict, structure: Atoms):
    """One ionic step → ``OutputCalcStatic`` (energy/force/stress/structure)."""
    out = OutputCalcStatic.pure_dataclass()
    out.energy = step["e_wo_entrp"]
    out.force = np.asarray(step["forces"])
    stress = step.get("stress")
    out.stress = None if stress is None else np.asarray(stress) * _kBar_to_GPa
    out.structure = structure
    return out


def _parse_velocities(vasprun_path: str) -> np.ndarray:
    """Per-step velocities from ``vasprun.xml``, empty if VASP did not write them.

    VASP only emits ``<varray name="velocities">`` for MD runs, and not for every
    VASP version, so this is best-effort: an empty array simply means the field
    stays unset on ``OutputCalcMD``.
    """
    import xml.etree.ElementTree as ET

    try:
        root = ET.parse(vasprun_path).getroot()
    except ET.ParseError:
        return np.array([])

    frames = []
    for calculation in root.findall("calculation"):
        for structure in calculation.findall("structure"):
            varray = structure.find('varray[@name="velocities"]')
            if varray is None:
                continue
            frames.append([[float(x) for x in v.text.split()] for v in varray])
    return np.asarray(frames) if frames else np.array([])


def _unwrap_positions(frac_coords: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Undo periodic wrapping, so diffusion coefficients can be read off directly.

    Takes fractional coordinates of shape ``(n_steps, n_atoms, 3)`` and removes
    the jumps introduced when an atom crosses a cell boundary (minimum image on
    successive frames), then converts back to Cartesian with each step's cell.
    """
    if len(frac_coords) == 0:
        return np.array([])

    deltas = np.diff(frac_coords, axis=0)
    deltas -= np.round(deltas)  # minimum image: drop the boundary crossings
    unwrapped_frac = np.concatenate(
        [frac_coords[:1], frac_coords[:1] + np.cumsum(deltas, axis=0)]
    )
    return np.einsum("nai,nij->naj", unwrapped_frac, cells)


def _md_from_vasprun(vr, ase_traj: list[Atoms], vasprun_path: str):
    """Full ionic trajectory → ``OutputCalcMD``, mirroring ``ParseLammpsOutput``."""
    out = OutputCalcMD.pure_dataclass()
    steps = vr.ionic_steps
    n_atoms = len(ase_traj[0])

    cells = np.asarray([a.get_cell().array for a in ase_traj])
    out.cells = cells
    out.positions = np.asarray([a.get_positions() for a in ase_traj])
    out.unwrapped_positions = _unwrap_positions(
        np.asarray([s.frac_coords for s in vr.structures]), cells
    )
    out.forces = np.asarray([s["forces"] for s in steps])
    out.volumes = np.asarray([a.get_volume() for a in ase_traj])
    out.natoms = np.full(len(steps), n_atoms)
    out.steps = np.arange(1, len(steps) + 1)

    energies_pot = np.asarray([s["e_wo_entrp"] for s in steps])
    out.energies_pot = energies_pot

    # 'kinetic' / 'total' only appear for real MD runs (IBRION 0)
    kinetic = [s.get("kinetic") for s in steps]
    if all(k is not None for k in kinetic):
        kinetic = np.asarray(kinetic, dtype=float)
        total = [s.get("total") for s in steps]
        out.energies_tot = (
            np.asarray(total, dtype=float)
            if all(t is not None for t in total)
            else energies_pot + kinetic
        )
        # equipartition over 3N-3 DOF — VASP keeps the centre of mass fixed
        n_dof = max(3 * n_atoms - 3, 1)
        out.temperatures = 2.0 * kinetic / (n_dof * _kB)
    else:
        out.energies_tot = energies_pot

    stresses = [s.get("stress") for s in steps]
    if all(s is not None for s in stresses):
        out.pressures = np.asarray(stresses, dtype=float) * _kBar_to_GPa

    out.velocities = _parse_velocities(vasprun_path)

    # same convention as ParseLammpsOutput: per-atom symbols, per-step species ids
    symbols = ase_traj[0].get_chemical_symbols()
    out.species = symbols
    species_order = _ordered_elements(ase_traj[0])
    species_ids = np.asarray([species_order.index(s) for s in symbols])
    out.indices = np.tile(species_ids, (len(steps), 1))

    return out


def _generate_hash(io_bundle: VaspInputResources) -> str:
    atoms = io_bundle.structure
    calc = io_bundle.calc

    flat_positions = [
        round(x, 6) for row in atoms.get_positions().tolist() for x in row
    ]
    flat_cell = [round(x, 6) for row in atoms.get_cell().array.tolist() for x in row]

    scf = calc.scf
    parts = [
        atoms.get_chemical_formula(),
        str(flat_positions),
        str(flat_cell),
        str(scf.functional),
        str(scf.energy_cutoff),
        str(scf.kpoints),
        str(scf.kpoint_offset),
        str(scf.electronic_convergence),
        str(scf.smearing_type),
        str(scf.smearing_width),
        str(scf.smearing_order),
        str(scf.num_electronic_steps),
        str(calc.minimization),
        str(calc.md),
        str(calc.dipole_correction),
        io_bundle.potcar_lib_path,
    ]

    if io_bundle.potcar_symbols:
        parts.append(str(io_bundle.potcar_symbols))

    if io_bundle.extra_incar:
        for k, v in sorted(io_bundle.extra_incar.items()):
            parts.append(f"{k}={v}")

    hash_string = "|".join(parts)
    return hashlib.sha256(hash_string.encode()).hexdigest()[:8]


# ── Nodes ─────────────────────────────────────────────────────────────────────


@as_function_node
def MergeVaspInput(
    scf: InputSCF,
    minimization: Optional[InputMinimizationVASP] = None,
    md: Optional[InputMDVASP] = None,
    dipole_correction: Optional[InputDipoleCorrection] = None,
    specific_inputs: Optional[AdditionalInputFlags] = None,
) -> VaspInput:
    """Combine the required SCF settings with any optional add-ons.

    ``scf`` is mandatory; ``minimization``, ``md`` and ``dipole_correction`` are
    optional. ``minimization`` and ``md`` are mutually exclusive (both drive the
    ionic loop) — that is enforced when the INCAR is built.
    """
    calc = VaspInput(
        scf=scf,
        minimization=minimization,
        md=md,
        dipole_correction=dipole_correction,
        extra_incar=specific_inputs.to_dict() if specific_inputs is not None else None,
    )
    return calc


@as_function_node
def CreateVaspInputResources(
    structure: Atoms,
    calc: VaspInput,
    potcar_lib_path: str = _default_potcar_lib_path,
    working_directory: Optional[str] = None,
    potcar_symbols: Optional[list[str]] = None,
) -> VaspInputResources:
    io_bundle = VaspInputResources(
        structure=structure,
        calc=calc,
        working_directory=working_directory,
        potcar_lib_path=potcar_lib_path,
        potcar_symbols=potcar_symbols,
        extra_incar=calc.extra_incar,
    )

    print("writing_input")
    print("working dir: ", io_bundle.working_directory)

    if io_bundle.working_directory is not None:
        workdir = io_bundle.working_directory
    else:
        workdir = _generate_hash(io_bundle)
        print("giving the hash name:", workdir)
    io_bundle.working_directory = workdir
    os.makedirs(workdir, exist_ok=True)

    # POSCAR
    pmg_structure = AseAtomsAdaptor.get_structure(io_bundle.structure)
    pmg_structure.to(fmt="poscar", filename=os.path.join(workdir, "POSCAR"))

    # INCAR
    incar = _build_incar(io_bundle.calc, io_bundle.extra_incar, io_bundle.structure)
    incar.write_file(os.path.join(workdir, "INCAR"))

    # POTCAR — look up paths from CSV, concatenate files into workdir/POTCAR
    potcar_paths = (
        [
            os.path.join(io_bundle.potcar_lib_path, s, "POTCAR")
            for s in io_bundle.potcar_symbols
        ]
        if io_bundle.potcar_symbols is not None
        else _get_potcar_paths(
            io_bundle.structure,
            io_bundle.calc.scf.functional,
            io_bundle.potcar_lib_path,
        )
    )
    with open(os.path.join(workdir, "POTCAR"), "wb") as wfd:
        for p in potcar_paths:
            with open(p, "rb") as fd:
                shutil.copyfileobj(fd, wfd)

    # KPOINTS — Gamma-centred mesh parsed from the "kx ky kz" string on InputSCF
    kpoints_path = os.path.join(workdir, "KPOINTS")
    mesh = [int(k) for k in io_bundle.calc.scf.kpoints.split()]
    if len(mesh) != 3:
        raise ValueError(
            f'scf.kpoints must be three integers like "4 4 4", got: '
            f"{io_bundle.calc.scf.kpoints!r}"
        )
    Kpoints.gamma_automatic(mesh).write_file(kpoints_path)

    return io_bundle


@as_function_node
def RunVaspCalculation(
    io_bundle: VaspInputResources,
    vasp_command: str = _default_vasp_command,
    run_script_path: Optional[str] = None,
    threads_per_core: int = 1,
    debug: bool = False,
):
    if not vasp_command:
        vasp_command = f"module load vasp && mpiexec -n {threads_per_core} vasp_std"

    if run_script_path is not None and os.path.exists(run_script_path):
        vasp_command = f"bash {run_script_path} {threads_per_core}"

    if debug:
        stdout = io_bundle.working_directory
        return io_bundle, stdout

    result = subprocess.run(
        vasp_command,
        cwd=io_bundle.working_directory,
        shell=True,
        universal_newlines=True,
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:
        error_path = os.path.join(io_bundle.working_directory, "error.msg")
        with open(error_path, "w") as f:
            f.write(result.stdout)
            if result.stderr:
                f.write(result.stderr)
        raise RuntimeError(
            f"VASP exited with code {result.returncode}.\n{result.stdout}"
        )

    stdout = result.stdout
    return io_bundle, stdout


@as_function_node
def ParseVaspOutput(
    io_bundle: VaspInputResources,
    vasprun_filename: str = "vasprun.xml",
):
    """Parse ``vasprun.xml`` into the calculator output dataclass for this run.

    Which dataclass lands on ``out`` follows the calc that was submitted, so the
    same downstream nodes work for VASP and LAMMPS:

    ``md``           → ``OutputCalcMD``       (full ionic trajectory)
    ``minimization`` → ``OutputCalcMinimize`` (initial + final, convergence)
    otherwise        → ``OutputCalcStatic``   (single point)

    ``trajectory`` (list of ASE ``Atoms``) and ``converged`` are kept alongside
    it — the ASE trajectory feeds visualisation nodes such as ``AnimateAse``.
    """
    from pymatgen.io.vasp.outputs import Vasprun

    vasprun_path = os.path.join(io_bundle.working_directory, vasprun_filename)
    vr = Vasprun(filename=vasprun_path, parse_dos=False, parse_projected_eigen=False)

    trajectory = [AseAtomsAdaptor.get_atoms(s) for s in vr.structures]
    converged = vr.converged
    calc = io_bundle.calc

    if calc is not None and calc.md is not None:
        out = _md_from_vasprun(vr, trajectory, vasprun_path)
    elif calc is not None and calc.minimization is not None:
        out = OutputCalcMinimize.pure_dataclass()
        out.initial = _static_from_ionic_step(vr.ionic_steps[0], trajectory[0])
        out.final = _static_from_ionic_step(vr.ionic_steps[-1], trajectory[-1])
        out.is_converged = vr.converged_ionic
        out.iter_steps = len(vr.ionic_steps)
    else:
        out = _static_from_ionic_step(vr.ionic_steps[-1], trajectory[-1])

    return out, trajectory, converged
