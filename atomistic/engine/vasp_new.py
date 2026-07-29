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
from vaspparser.vasp.output import Output, parse_vasp_output
from vaspparser.vasp.volumetric_data import VaspVolumetricData

from core import as_function_node
from pyiron_nodes.atomistic.calculator.data import (
    InputMDVASP,
    InputDipoleCorrection,
    InputMinimizationVASP,
    InputSCF,
    InputVaspDOS,
    InputVaspOutputFiles,
    OutputCalcMD,
    OutputCalcMinimize,
    OutputCalcStatic,
    OutputVaspDOS,
    AdditionalInputFlags,
)

# ── INCAR enum lookups ────────────────────────────────────────────────────────


def _ISMEAR(smearing_type, smearing_order):
    """Map a smearing scheme onto VASP's ``ISMEAR`` tag.

    ``smearing_order`` is only consulted for Methfessel-Paxton, where it is the
    expansion order ``N`` (ISMEAR = N). Raises ``ValueError`` on an unknown
    scheme or a Methfessel-Paxton order below 1.
    """
    if smearing_type == "tetrahedron":
        # tetrahedron method with Blöchl corrections — the accurate choice for a
        # DOS or a band structure, but it needs a Gamma-centred k-mesh and gives
        # no useful forces, so it is not the default
        return -5
    elif smearing_type == "fermi-dirac":
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
            "'gaussian', 'methfessel-paxton', 'fermi-dirac', 'tetrahedron'"
        )


_IBRION_MINIMIZE = {
    "ConjugateGradient": 2,
    "RMM-DIIS": 1,
    "DampedMolecularDynamics": 3,
}

# ── POTCAR config ─────────────────────────────────────────────────────────────
# testing the comit


def read_potcar_config(config_file: Path) -> dict:
    """Read the user's ``~/.pyiron_vasp_config`` into a dict of defaults.

    The file is ``key = value`` lines (``#`` comments and blanks ignored). Beyond
    the raw keys, this derives ``default_POTCAR_path`` by joining
    ``pyiron_vasp_resources`` with ``default_POTCAR_set`` (rewriting e.g.
    ``potpawPBE`` → ``potpaw_PBE``) and fills ``default_functional`` with ``PBE``
    when absent. A missing file yields an empty dict, so callers fall back to
    passing ``potcar_lib_path`` explicitly.
    """
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
    dos: Optional[InputVaspDOS] = None
    output_files: Optional[InputVaspOutputFiles] = None
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
    """Unique elements in the order they first appear along the atom list.

    This matches how the POSCAR/POTCAR group atoms — one entry per run of the
    same species — so the returned order lines up with the concatenated POTCAR
    and with per-species INCAR tags such as ``LANGEVIN_GAMMA``.
    """
    elements, prev = [], None
    for sym in atoms.get_chemical_symbols():
        if sym != prev:
            elements.append(sym)
            prev = sym
    return elements


def _get_potcar_paths(atoms: Atoms, functional: str, lib_path: str) -> list[str]:
    """POTCAR file paths for a structure, one per species, in POSCAR order.

    The default pseudopotential folder for each element is looked up in the
    bundled CSV for ``functional`` (``"PBE"`` or ``"LDA"``), then joined onto
    ``lib_path`` as ``<lib_path>/<potential_name>/POTCAR``. Concatenating the
    returned files in order gives the run's POTCAR.
    """
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
    """Translate a ``VaspInput`` into a pymatgen ``Incar``.

    The SCF settings are always written; each optional sub-input then layers its
    tags on top — ``minimization`` (IBRION/NSW/EDIFFG/ISIF), ``md`` (Langevin
    thermostat), ``dipole_correction`` (LDIPOL/IDIPOL), ``dos`` (NEDOS/LORBIT/
    EMIN/EMAX) and ``output_files`` (LCHARG/LWAVE/LVTOT/LVHAR). ``extra`` is a
    dict of raw INCAR tags applied last, so it overrides anything above.

    Parameters
    ----------
    calc
        The combined input produced by ``MergeVaspInput``.
    extra
        Raw INCAR overrides (e.g. ``calc.extra_incar``), applied on top.
    structure
        Required only for an MD run — ``LANGEVIN_GAMMA`` needs one friction
        coefficient per species, so the species count comes from here.

    Raises
    ------
    ValueError
        If both ``minimization`` and ``md`` are set (both drive the ionic loop),
        if an MD input is given without a ``structure``, or if an ``NpT`` MD run
        has no ``pressure``.
    """
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
        "ISPIN": 2 if scf.spin_polarized else 1,
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

    # ── density of states (optional) ──────────────────────────────────────────
    # VASP always writes a DOSCAR; these tags decide what goes into it, and so
    # what the ``dos`` port of ParseVaspOutput carries. LORBIT 11 doubles as the
    # switch for the per-ion magnetization table that feeds ``magnetic_moments``.
    if calc.dos is not None:
        dos = calc.dos
        tags["NEDOS"] = dos.n_points
        if dos.projected:
            tags["LORBIT"] = 11
        if dos.energy_min is not None:
            tags["EMIN"] = dos.energy_min
        if dos.energy_max is not None:
            tags["EMAX"] = dos.energy_max

    # ── which output files to write (optional) ────────────────────────────────
    # These decide which ``ParseVaspOutput`` ports can be filled at all: no
    # CHGCAR means no electron density, no LOCPOT means no electrostatic
    # potential. LCHARG defaults to .TRUE. in VASP, so leaving this out keeps
    # the previous behaviour.
    if calc.output_files is not None:
        files = calc.output_files
        tags["LCHARG"] = files.charge_density
        tags["LWAVE"] = files.wavefunctions
        if files.electrostatic_potential:
            tags["LVTOT"] = True
        if files.hartree_potential_only:
            tags["LVHAR"] = True

    if extra:
        tags.update(extra)
    return Incar(tags)


# ── Output helpers ────────────────────────────────────────────────────────────
# All output parsing goes through ``vaspparser`` — the VASP counterpart of the
# ``lammpsparser`` used by ``ParseLammpsOutput``. ``parse_vasp_output`` reads
# vasprun.xml, OUTCAR, OSZICAR, CONTCAR, CHGCAR and LOCPOT from the working
# directory and returns one hierarchical dictionary; everything below only
# reshapes that dictionary into the calculator dataclasses that the LAMMPS
# engine also fills (``OutputCalcStatic`` / ``OutputCalcMinimize`` /
# ``OutputCalcMD``), so downstream nodes do not care which engine produced the
# data.

_eVA3_to_GPa = 160.21766208  # vaspparser reports stresses in eV/Å³


class _SkippedVolumetricData(VaspVolumetricData):
    """Stand-in whose ``from_file`` does nothing.

    ``Output.collect`` reads CHGCAR/LOCPOT unconditionally when the files are
    present; substituting this class is how ``ParseVaspOutput`` skips a
    multi-gigabyte file the caller did not ask for.
    """

    def from_file(self, filename: str, normalize: bool = True):
        return None


def _output_parser_class(
    charge_density: bool, electrostatic_potential: bool, collected: list
):
    """``Output`` subclass that skips unwanted volumetric files and records itself.

    ``parse_vasp_output`` builds the parser internally and hands back only
    dictionaries, but the DOS port needs the ``ElectronicStructure`` object
    itself — the orbital names behind ``resolved_densities`` are not part of its
    ``to_dict()``. Appending to ``collected`` recovers that object;
    ``output_parser_class`` is the documented hook for exactly this.
    """

    class _Output(Output):
        def __init__(self):
            super().__init__()
            collected.append(self)
            if not charge_density:
                self.charge_density = _SkippedVolumetricData()
            if not electrostatic_potential:
                self.electrostatic_potential = _SkippedVolumetricData()

    return _Output


# VASP's fixed projection order in the DOSCAR; the number of orbital columns
# says which scheme was used (LORBIT 10 → totals, LORBIT 11 → lm-decomposed).
_DOSCAR_ORBITALS = {
    1: ["s"],
    3: ["s", "p", "d"],
    4: ["s", "p", "d", "f"],
    9: ["s", "py", "pz", "px", "dxy", "dyz", "dz2", "dxz", "dx2-y2"],
    16: [
        "s", "py", "pz", "px", "dxy", "dyz", "dz2", "dxz", "dx2-y2",
        "f-3", "f-2", "f-1", "f0", "f1", "f2", "f3",
    ],
}  # fmt: skip


def read_doscar(filename: str) -> Optional[OutputVaspDOS]:
    """Read a ``DOSCAR`` into ``OutputVaspDOS``.

    ``vaspparser`` has no DOSCAR parser — it takes the DOS from the ``<dos>``
    block of vasprun.xml instead — so this reads the file directly, for when the
    DOSCAR itself is what you want.

    The layout is six header lines, then ``NEDOS`` rows of total DOS
    (``E, DOS, intDOS`` for ISPIN 1; ``E, DOS↑, DOS↓, int↑, int↓`` for ISPIN 2),
    optionally followed by one repeated header plus ``NEDOS`` rows per ion when
    the run was run with LORBIT (the site- and orbital-projected DOS). In the
    projected rows the columns run orbital-major with spin varying fastest.

    Returns ``None`` if the file is missing or holds no DOS grid.
    """
    if not os.path.isfile(filename) or os.stat(filename).st_size == 0:
        return None

    with open(filename) as f:
        lines = f.readlines()
    if len(lines) < 7:
        return None

    n_ions = int(lines[0].split()[0])
    header = lines[5].split()
    n_points, efermi = int(header[2]), float(header[3])

    rows = np.array(
        [[float(x) for x in line.split()] for line in lines[6 : 6 + n_points]]
    )
    if len(rows) != n_points:
        return None

    dos = OutputVaspDOS.pure_dataclass()
    dos.energies = rows[:, 0]
    dos.efermi = efermi

    # 3 columns → ISPIN 1 (dos, int); 5 columns → ISPIN 2 (dos↑, dos↓, int↑, int↓)
    n_spin = 2 if rows.shape[1] == 5 else 1
    dos.total_densities = rows[:, 1 : 1 + n_spin].T
    dos.integrated_densities = rows[:, 1 + n_spin : 1 + 2 * n_spin].T

    # projected blocks: one repeated header line + n_points rows per ion
    block = 1 + n_points
    start = 6 + n_points
    if len(lines) < start + n_ions * block:
        return dos

    projected = []
    for ion in range(n_ions):
        first = start + ion * block + 1  # skip the repeated header
        ion_rows = np.array(
            [
                [float(x) for x in line.split()]
                for line in lines[first : first + n_points]
            ]
        )
        if len(ion_rows) != n_points:
            return dos
        # columns are orbital-major with spin varying fastest
        n_orbitals = (ion_rows.shape[1] - 1) // n_spin
        projected.append(
            ion_rows[:, 1 : 1 + n_orbitals * n_spin]
            .reshape(n_points, n_orbitals, n_spin)
            .transpose(2, 1, 0)  # → (n_spin, n_orbitals, n_points)
        )

    # (n_spin, n_atoms, n_orbitals, n_points), matching the vasprun-derived DOS
    dos.resolved_densities = np.stack(projected, axis=1)
    dos.orbitals = _DOSCAR_ORBITALS.get(dos.resolved_densities.shape[2])
    return dos


def _dos_from_electronic_structure(electronic_structure):
    """``ElectronicStructure`` → ``OutputVaspDOS``, the DOS from vasprun.xml.

    ``None`` when the run produced no DOS at all — the arrays come from the
    ``<dos>`` block of vasprun.xml, which is absent if only an OUTCAR survived.
    """
    if electronic_structure is None:
        return None
    energies = np.asarray(electronic_structure.dos_energies)
    if energies.size == 0:
        return None

    dos = OutputVaspDOS.pure_dataclass()
    dos.energies = energies
    dos.total_densities = np.asarray(electronic_structure.dos_densities)
    dos.integrated_densities = np.asarray(electronic_structure.dos_idensities)
    dos.efermi = electronic_structure.efermi

    # site- and orbital-projected DOS, only present with LORBIT 11
    resolved = electronic_structure.resolved_densities
    if resolved is not None:
        dos.resolved_densities = np.asarray(resolved)
        orbital_dict = electronic_structure.orbital_dict or {}
        dos.orbitals = [
            name for name, _ in sorted(orbital_dict.items(), key=lambda kv: kv[1])
        ]
    return dos


def _volumetric_from_dict(data: dict | None, structure: Atoms):
    """CHGCAR/LOCPOT sub-dictionary → ``VaspVolumetricData``.

    Returned instead of the bare array so the grid helpers stay available, e.g.
    ``electrostatic_potential.get_average_along_axis(ind=2)`` for a work
    function, or ``write_cube_file()`` for external visualisation.
    """
    if data is None:
        return None
    volumetric = VaspVolumetricData()
    volumetric.atoms = structure
    volumetric.total_data = data["total"]
    if data.get("diff") is not None:
        volumetric.diff_data = data["diff"]
    return volumetric


def _trajectory_from_output(output: dict, structure: Atoms) -> list[Atoms]:
    """Per-ionic-step positions and cells → list of ASE ``Atoms``."""
    generic = output["generic"]
    trajectory = []
    for positions, cell in zip(generic["positions"], generic["cells"]):
        atoms = structure.copy()
        atoms.set_cell(cell)
        atoms.set_positions(positions)
        trajectory.append(atoms)
    return trajectory


def _final_magmoms(output: dict) -> Optional[np.ndarray]:
    """Per-atom magnetic moments of the last ionic step, ``None`` if ISPIN = 1.

    Shape is ``(n_atoms,)`` for a collinear run and ``(n_atoms, 3)`` for a
    non-collinear one. The moments come from the OUTCAR, so they are missing
    when only vasprun.xml was written.
    """
    magmoms = output["generic"]["dft"].get("final_magmoms")
    if magmoms is None or len(magmoms) == 0:
        return None
    return np.asarray(magmoms[-1], dtype=float)


def _is_converged(output: dict, calc: Optional[VaspInput]) -> bool:
    """Electronic (and, for a relaxation, ionic) convergence of the last step.

    ``vaspparser`` reports no convergence flag of its own, so this applies the
    usual criterion: a loop that stopped before hitting its own step limit
    converged. An MD run always uses all NSW steps, so only the electronic loop
    is checked there.
    """
    generic = output["generic"]
    electronic = True
    scf_energies = generic["dft"].get("scf_energy_free")
    if calc is not None and scf_energies is not None and len(scf_energies) > 0:
        electronic = len(scf_energies[-1]) < calc.scf.num_electronic_steps

    ionic = True
    if calc is not None and calc.minimization is not None:
        max_steps = calc.minimization.max_ionic_steps
        ionic = max_steps <= 1 or len(generic["energy_tot"]) < max_steps

    return bool(electronic and ionic)


def _static_from_output(output: dict, index: int, structure: Atoms):
    """One ionic step → ``OutputCalcStatic`` (energy/force/stress/structure)."""
    generic = output["generic"]
    out = OutputCalcStatic.pure_dataclass()
    out.energy = float(generic["energy_tot"][index])
    out.force = np.asarray(generic["forces"][index])
    # stresses only come from the OUTCAR, so they are absent for a vasprun-only run
    stresses = generic.get("stresses")
    out.stress = (
        None if stresses is None else np.asarray(stresses[index]) * _eVA3_to_GPa
    )
    out.structure = structure
    return out


def _parse_velocities(vasprun_path: str) -> np.ndarray:
    """Per-step velocities from ``vasprun.xml``, empty if VASP did not write them.

    VASP only emits ``<varray name="velocities">`` for MD runs, and not for every
    VASP version, so this is best-effort: an empty array simply means the field
    stays unset on ``OutputCalcMD``. A missing file counts as best-effort too —
    ``vaspparser`` happily parses an MD run from the OUTCAR alone, in which case
    there is no vasprun.xml here to read.
    """
    import xml.etree.ElementTree as ET

    try:
        root = ET.parse(vasprun_path).getroot()
    except (ET.ParseError, OSError):
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


def _md_from_output(output: dict, trajectory: list[Atoms], vasprun_path: str):
    """Full ionic trajectory → ``OutputCalcMD``, mirroring ``ParseLammpsOutput``."""
    generic = output["generic"]
    out = OutputCalcMD.pure_dataclass()

    cells = np.asarray(generic["cells"])
    positions = np.asarray(generic["positions"])
    n_steps, n_atoms = positions.shape[0], positions.shape[1]

    out.cells = cells
    out.positions = positions
    # vaspparser hands out Cartesian positions; unwrapping happens in fractional
    # space, where a boundary crossing is just a jump of ±1
    frac_coords = np.einsum("nai,nij->naj", positions, np.linalg.inv(cells))
    out.unwrapped_positions = _unwrap_positions(frac_coords, cells)
    out.forces = np.asarray(generic["forces"])
    out.volumes = np.asarray(generic["volume"])
    out.natoms = np.full(n_steps, n_atoms)
    out.steps = np.asarray(generic["steps"])

    # energy_tot carries the kinetic energy on top of energy_pot for a real MD
    # run (IBRION 0); for anything else vaspparser sets the two equal
    out.energies_pot = np.asarray(generic["energy_pot"])
    out.energies_tot = np.asarray(generic["energy_tot"])

    # temperatures and stresses are OUTCAR-only quantities
    temperatures = generic.get("temperature")
    if temperatures is not None:
        out.temperatures = np.asarray(temperatures)
    stresses = generic.get("stresses")
    if stresses is not None:
        out.pressures = np.asarray(stresses) * _eVA3_to_GPa

    out.velocities = _parse_velocities(vasprun_path)

    # same convention as ParseLammpsOutput: per-atom symbols, per-step species ids
    symbols = trajectory[0].get_chemical_symbols()
    out.species = symbols
    species_order = _ordered_elements(trajectory[0])
    species_ids = np.asarray([species_order.index(s) for s in symbols])
    out.indices = np.tile(species_ids, (n_steps, 1))

    return out


def _generate_hash(io_bundle: VaspInputResources) -> str:
    """Deterministic 8-hex-character name for a run's working directory.

    Used as the fallback working directory when the caller does not supply one.
    The hash folds in everything that defines the calculation — structure
    (formula, positions, cell), every SCF field, the optional sub-inputs, the
    POTCAR library path and any ``extra_incar`` — so two identical calculations
    map to the same directory and any change produces a new one.
    """
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
        str(scf.spin_polarized),
        str(calc.minimization),
        str(calc.md),
        str(calc.dipole_correction),
        str(calc.dos),
        str(calc.output_files),
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
    dos: Optional[InputVaspDOS] = None,
    output_files: Optional[InputVaspOutputFiles] = None,
    specific_inputs: Optional[AdditionalInputFlags] = None,
) -> VaspInput:
    """Combine the required SCF settings with any optional add-ons.

    ``scf`` is mandatory; everything else is optional. ``minimization`` and
    ``md`` are mutually exclusive (both drive the ionic loop) — that is enforced
    when the INCAR is built. ``dos`` sets how finely the DOSCAR is resolved and
    ``output_files`` selects the CHGCAR/LOCPOT/WAVECAR files VASP writes, which
    is what makes the corresponding ``ParseVaspOutput`` ports available.
    """
    calc = VaspInput(
        scf=scf,
        minimization=minimization,
        md=md,
        dipole_correction=dipole_correction,
        dos=dos,
        output_files=output_files,
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
    """Write the four VASP input files and return the resource bundle.

    Creates ``working_directory`` (a hash of the calculation when none is given,
    see ``_generate_hash``) and writes POSCAR, INCAR, POTCAR and KPOINTS into it.
    POTCAR is the per-species pseudopotentials concatenated in POSCAR order,
    looked up from the bundled CSV under ``potcar_lib_path`` unless
    ``potcar_symbols`` names the folders explicitly. KPOINTS is a Gamma-centred
    mesh parsed from the ``"kx ky kz"`` string on ``InputSCF``.

    Parameters
    ----------
    structure
        The atomic structure (ASE ``Atoms``) to write to POSCAR.
    calc
        Combined VASP settings from ``MergeVaspInput``.
    potcar_lib_path
        Base directory holding the per-element POTCAR folders.
    working_directory
        Where the input files are written; defaults to a content hash.
    potcar_symbols
        Explicit POTCAR folder names, overriding the CSV lookup.

    Returns
    -------
    VaspInputResources
        The bundle (with ``working_directory`` resolved) that the run node takes.

    Raises
    ------
    ValueError
        If ``scf.kpoints`` is not three integers.
    """
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
    """Run VASP in the bundle's working directory.

    Executes ``vasp_command`` (or ``bash <run_script_path> <threads_per_core>``
    when ``run_script_path`` points at an existing file) with the working
    directory as CWD, so it picks up the POSCAR/INCAR/POTCAR/KPOINTS written by
    ``CreateVaspInputResources``. A non-zero exit writes stdout+stderr to
    ``error.msg`` and raises ``RuntimeError``.

    Parameters
    ----------
    io_bundle
        The resource bundle from ``CreateVaspInputResources``.
    vasp_command
        Shell command that launches VASP; empty falls back to a
        ``module load vasp && mpiexec`` line built from ``threads_per_core``.
    run_script_path
        Optional launcher script; when it exists it replaces ``vasp_command``.
    threads_per_core
        MPI rank count passed to the fallback command / launcher script.
    debug
        If ``True``, skip the launch and just return the working directory as
        stdout — useful for wiring up a workflow without running VASP.

    Returns
    -------
    (io_bundle, stdout)
        The same bundle (for chaining into ``ParseVaspOutput``) and the run's
        stdout (the working directory in ``debug`` mode).
    """
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
    dos_source: str = "vasprun",
    parse_electron_density: bool = True,
    parse_electrostatic_potential: bool = True,
):
    """Collect the whole working directory with ``vaspparser``.

    ``vaspparser.parse_vasp_output`` reads every VASP output file that is
    present — vasprun.xml and OUTCAR for energies/forces/magnetisation, OSZICAR
    for high-precision energies, CONTCAR for the final structure, CHGCAR and
    LOCPOT for the volumetric data — and returns them as one dictionary. This
    node reshapes that into the ports below.

    Ports
    -----
    out
        Calculator dataclass matching the calc that was submitted, so the same
        downstream nodes work for VASP and LAMMPS:
        ``md`` → ``OutputCalcMD`` (full ionic trajectory), ``minimization`` →
        ``OutputCalcMinimize`` (initial + final, convergence), otherwise
        ``OutputCalcStatic`` (single point).
    trajectory
        Every ionic step as an ASE ``Atoms`` — feeds visualisation nodes such as
        ``AnimateAse``.
    last_structure
        Final structure, taken from the CONTCAR when one was written (higher
        precision than the positions in vasprun.xml).
    total_energy
        Total energy of the last ionic step in eV; for an MD run this includes
        the kinetic energy of the ions.
    magnetic_moments
        Per-atom moments of the last ionic step, or ``None``. VASP prints these
        only when the run is spin-polarized (``InputSCF.spin_polarized``) *and*
        the per-ion magnetization table was requested (``InputVaspDOS.projected``,
        i.e. LORBIT 11) — ISPIN 2 on its own is not enough.
    dos
        ``OutputVaspDOS``: ``energies``, ``total_densities`` and
        ``integrated_densities``, plus ``resolved_densities``/``orbitals`` when
        the projection was switched on. ``None`` if the run wrote no DOS.
        ``dos_source`` picks where it comes from — ``"vasprun"`` (default) for
        the ``<dos>`` block that ``vaspparser`` reads out of vasprun.xml, or
        ``"doscar"`` to read the ``DOSCAR`` file directly. VASP writes the same
        quantity to both, so they should agree; if they do not, the run itself
        is worth a second look.
    electrostatic_potential, electron_density
        ``VaspVolumetricData`` from LOCPOT / CHGCAR, or ``None`` when the file
        was not written (see ``InputVaspOutputFiles``) or was skipped here. The
        grid itself is ``.total_data``; ``.diff_data`` holds the spin difference
        of a spin-polarized CHGCAR.
    converged
        Electronic convergence, and ionic convergence too for a relaxation.

    ``parse_electron_density`` / ``parse_electrostatic_potential`` exist because
    CHGCAR and LOCPOT are large and slow to read; switch them off to leave the
    files on disk untouched.
    """
    workdir = io_bundle.working_directory
    parsers: list = []
    output = parse_vasp_output(
        working_directory=workdir,
        structure=io_bundle.structure,
        output_parser_class=_output_parser_class(
            charge_density=parse_electron_density,
            electrostatic_potential=parse_electrostatic_potential,
            collected=parsers,
        ),
    )

    last_structure = Atoms.fromdict(output["structure"])
    trajectory = _trajectory_from_output(output, last_structure)
    total_energy = float(output["generic"]["energy_tot"][-1])
    magnetic_moments = _final_magmoms(output)
    if dos_source == "doscar":
        dos = read_doscar(os.path.join(workdir, "DOSCAR"))
    elif dos_source == "vasprun":
        dos = _dos_from_electronic_structure(
            parsers[0].electronic_structure if parsers else None
        )
    else:
        raise ValueError(
            f"Unknown dos_source {dos_source!r}; expected 'vasprun' or 'doscar'."
        )
    electron_density = _volumetric_from_dict(
        output.get("charge_density"), last_structure
    )
    electrostatic_potential = _volumetric_from_dict(
        output.get("electrostatic_potential"), last_structure
    )
    converged = _is_converged(output, io_bundle.calc)

    calc = io_bundle.calc
    if calc is not None and calc.md is not None:
        out = _md_from_output(output, trajectory, os.path.join(workdir, "vasprun.xml"))
    elif calc is not None and calc.minimization is not None:
        out = OutputCalcMinimize.pure_dataclass()
        out.initial = _static_from_output(output, 0, trajectory[0])
        out.final = _static_from_output(output, -1, last_structure)
        out.is_converged = converged
        out.iter_steps = len(trajectory)
    else:
        out = _static_from_output(output, -1, last_structure)

    return (
        out,
        trajectory,
        last_structure,
        total_energy,
        magnetic_moments,
        dos,
        electrostatic_potential,
        electron_density,
        converged,
    )
