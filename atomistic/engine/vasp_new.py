from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import Incar, Kpoints
from pymatgen.io.vasp.outputs import Vasprun

from core import as_function_node
from pyiron_nodes.atomistic.calculator.data import InputCalcDFT, OutputCalcStatic


# ── POTCAR config ─────────────────────────────────────────────────────────────

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
        config_data["default_POTCAR_path"] = os.path.join(pyiron_vasp_resources, potcar_dir)
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
class VaspInputResources:
    structure: Atoms        # ASE Atoms — compatible with Bulk and other structure nodes
    calc: InputCalcDFT
    potcar_lib_path: str = field(default_factory=lambda: _default_potcar_lib_path)  # base path to POTCAR folders
    working_directory: Optional[str] = None
    potcar_symbols: Optional[list[str]] = None  # override default CSV symbol selection
    extra_incar: Optional[dict] = None          # additional INCAR tags beyond InputCalcDFT


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


def _build_incar(calc: InputCalcDFT, extra: dict | None = None) -> Incar:
    
    if not calc.ionic_relaxation:
        ibrion = -1
    else:
        match calc.ionic_update_algorithm:
            case "MolecularDynamics":
                ibrion = 0
            case "RMM-DIIS": 
                ibrion = 1
            case "ConjugateGradient":
                ibrion = 2
            case "DampedMolecularDynamics":
                ibrion = 3
            case _:
                raise ValueError(
                    f"ionic_update_algorithm must be set when ionic_relaxation is True, "
                    f"got: {calc.ionic_update_algorithm!r}"
                )
    
    tags = {
        "ENCUT": calc.energy_cutoff,
        "EDIFF": calc.electronic_convergence,
        "EDIFFG": calc.ionic_convergence,
        "NSW": calc.max_ionic_steps,
        "IBRION": ibrion,
        "ISIF": calc.isif,
        "ISMEAR": calc.ismear,
        "SIGMA": calc.sigma,
        "ISPIN": calc.ispin,
        "ALGO": calc.algo,
        "PREC": calc.prec,
        "NCORE": calc.ncore,
    }
    if extra:
        tags.update(extra)
    return Incar(tags)

def _generate_hash(input_resources: VaspInputResources) -> str:
    atoms = input_resources.structure
    calc = input_resources.calc

    flat_positions = [round(x, 6) for row in atoms.get_positions().tolist() for x in row]
    flat_cell = [round(x, 6) for row in atoms.get_cell().array.tolist() for x in row]

    parts = [
        atoms.get_chemical_formula(),
        str(flat_positions),
        str(flat_cell),
        str(calc.energy_cutoff),
        str(calc.electronic_convergence),
        str(calc.ionic_convergence),
        str(calc.max_ionic_steps),
        str(calc.ionic_relaxation),
        str(calc.ionic_update_algorithm),
        str(calc.isif),
        str(calc.ismear),
        str(calc.sigma),
        str(calc.ispin),
        str(calc.algo),
        str(calc.prec),
        str(calc.ncore),
        str(calc.kpoints_mesh),
        str(calc.functional),
        input_resources.potcar_lib_path,
    ]

    if input_resources.potcar_symbols:
        parts.append(str(input_resources.potcar_symbols))

    if input_resources.extra_incar:
        for k, v in sorted(input_resources.extra_incar.items()):
            parts.append(f"{k}={v}")

    hash_string = "|".join(parts)
    return hashlib.sha256(hash_string.encode()).hexdigest()[:8]

# ── Nodes ─────────────────────────────────────────────────────────────────────

@as_function_node
def CreateVaspInputResources(
    structure: Atoms,
    calc: InputCalcDFT,
    potcar_lib_path: str = _default_potcar_lib_path,
    working_directory: Optional[str] = None,
    potcar_symbols: Optional[list[str]] = None,
    extra_incar: Optional[dict] = None,
) -> VaspInputResources:
    input_resources = VaspInputResources(
        structure=structure,
        calc=calc,
        working_directory=working_directory,
        potcar_lib_path=potcar_lib_path,
        potcar_symbols=potcar_symbols,
        extra_incar=extra_incar,
    )
    return input_resources


@as_function_node
def WriteVaspInputSet(input_resources: VaspInputResources) -> VaspInputResources:
    print("writing_input")
    print("working dir: ", input_resources.working_directory )
    if input_resources.working_directory is not None:
        workdir = input_resources.working_directory
    else:
        workdir = _generate_hash(input_resources)
        print("giving the hash name:", workdir)
    input_resources.working_directory = workdir
    os.makedirs(workdir, exist_ok=True)

    # POSCAR
    pmg_structure = AseAtomsAdaptor.get_structure(input_resources.structure)
    pmg_structure.to(fmt="poscar", filename=os.path.join(workdir, "POSCAR"))

    # INCAR
    incar = _build_incar(input_resources.calc, input_resources.extra_incar)
    incar.write_file(os.path.join(workdir, "INCAR"))

    # POTCAR — look up paths from CSV, concatenate files into workdir/POTCAR
    potcar_paths = (
        [os.path.join(input_resources.potcar_lib_path, s, "POTCAR") for s in input_resources.potcar_symbols]
        if input_resources.potcar_symbols is not None
        else _get_potcar_paths(input_resources.structure, input_resources.calc.functional, input_resources.potcar_lib_path)
    )
    with open(os.path.join(workdir, "POTCAR"), "wb") as wfd:
        for p in potcar_paths:
            with open(p, "rb") as fd:
                shutil.copyfileobj(fd, wfd)

    # KPOINTS
    kpoints_path = os.path.join(workdir, "KPOINTS")
    kpoint_string = input_resources.calc.kpoints_mesh
    mesh = [int(k) for k in kpoint_string.split()]
    if mesh is not None:
        kpoints = Kpoints.gamma_automatic(mesh)
        kpoints.write_file(kpoints_path)
    else:
        with open(kpoints_path, "w") as f:
            f.write("Automatic mesh\n0\nGamma\n1 1 1\n0 0 0\n")

    return input_resources


@as_function_node
def RunVaspCalculation(
    input_resources: VaspInputResources,
    vasp_command: str = _default_vasp_command,
    cores: int = 1,
    debug: bool = False,
):
    if not vasp_command:
        vasp_command = f"module load vasp && mpiexec -n {cores} vasp_std"

    if debug:
        stdout = input_resources.working_directory
        return input_resources, stdout

    result = subprocess.run(
        vasp_command,
        cwd=input_resources.working_directory,
        shell=True,
        universal_newlines=True,
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:
        error_path = os.path.join(input_resources.working_directory, "error.msg")
        with open(error_path, "w") as f:
            f.write(result.stdout)
            if result.stderr:
                f.write(result.stderr)
        raise RuntimeError(
            f"VASP exited with code {result.returncode}.\n{result.stdout}"
        )

    stdout = result.stdout
    return input_resources, stdout


@as_function_node
def ParseVaspOutput(
    input_resources: VaspInputResources,
    vasprun_filename: str = "vasprun.xml",
):
    from pymatgen.io.vasp.outputs import Vasprun

    vasprun_path = os.path.join(input_resources.working_directory, vasprun_filename)
    vr = Vasprun(filename=vasprun_path, parse_dos=False, parse_projected_eigen=False)

    trajectory = [AseAtomsAdaptor.get_atoms(s) for s in vr.structures]

    final = trajectory[-1]
    last_step = vr.ionic_steps[-1]

    out = OutputCalcStatic.pure_dataclass()
    out.energy = last_step["e_wo_entrp"]
    out.force = last_step["forces"]
    out.stress = last_step.get("stress")
    out.structure = final

    converged = vr.converged
    return out, trajectory, converged
