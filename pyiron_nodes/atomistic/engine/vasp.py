from __future__ import annotations

import os
import warnings
from pathlib import Path
import shutil
import subprocess
from typing import Optional
from dataclasses import dataclass, field

import pandas as pd

from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Incar, Kpoints
from pymatgen.io.vasp.outputs import Vasprun

from ase import Atoms

from pyiron_workflow import Workflow

from pyiron_atomistics.vasp.output import parse_vasp_output as pvo

from pyiron_snippets.logger import logger
from pyiron_snippets.resources import ResourceResolver

from pyiron_nodes.atomistic.engine.lammps import Shell
from pyiron_nodes.lammps import ShellOutput


def get_potcar_config_path(config_file: Optional[Path] = None) -> Optional[Path]:
    """
    Get the POTCAR library path from the config file.
    If the config file is not specified, it defaults to '.pyiron_vasp_config' in the user's home directory.

    Args:
        config_file (Path, optional): Path to the config file. Defaults to None.

    Returns:
        Path or None: The path where the POTCAR files are stored, or None if not found.
    """
    # If no config file is specified, use the default '.pyiron_vasp_config' in the user's home directory
    if config_file is None:
        config_file = Path.home().joinpath(".pyiron_vasp_config")

    # Read the config file and extract the path
    try:
        with open(config_file, "r") as f:
            for line in f:
                if line.startswith("pyiron_vasp_POTCAR_config"):
                    # Parse the line to get the path value
                    potcar_path = line.split("=")[1].strip()
                    return Path(potcar_path)
    except FileNotFoundError:
        warnings.warn(
            f"Configuration file not found: {config_file}, using default or manual path.",
            UserWarning,
        )
    except Exception as e:
        warnings.warn(f"Error reading potcar configuration file: {str(e)}", UserWarning)

    warnings.warn(
        "POTCAR path not found in configuration file, please check the configuration.",
        UserWarning,
    )
    return None


# Use the function to get the path
POTCAR_library_path = get_potcar_config_path()

# Other parts of vasp.py can now use POTCAR_library_path
POTCAR_specification_data = str(
    Path(__file__).parent.joinpath("vasp_pseudopotential_PBE_data.csv")
)


@dataclass
class VaspInput:
    structure: Structure
    incar: Incar
    pseudopot_lib_path: str = field(default=POTCAR_library_path)
    potcar_paths: Optional[list[str]] = None
    kpoints: Optional[Kpoints] = None


@Workflow.wrap.as_function_node("line_found")
def isLineInFile(filepath: str, line: str, exact_match: bool = True) -> bool:
    line_found = False  # Initialize the result as False
    try:
        with open(filepath, "r") as file:
            for file_line in file:
                if exact_match and line == file_line.strip():
                    line_found = True
                    break  # Exit loop if the line is found
                elif not exact_match and line in file_line:
                    line_found = True
                    break  # Exit loop if a partial match is found
    except FileNotFoundError:
        logger.logger.log(f"File '{filepath}' not found.")

    return line_found


def write_POSCAR(workdir: str, structure: Structure, filename: str = "POSCAR") -> str:
    poscar_path = os.path.join(workdir, filename)
    structure.to(fmt="poscar", filename=poscar_path)
    return poscar_path


def write_INCAR(workdir: str, incar: Incar, filename: str = "INCAR") -> str:
    incar_path = os.path.join(workdir, filename)
    incar.write_file(incar_path)
    return incar_path


def write_POTCAR(workdir: str, vasp_input, filename: str = "POTCAR") -> str:
    class PotcarNotGeneratedError(Exception):
        pass

    if vasp_input.potcar_paths is None:
        potcar_paths = get_default_POTCAR_paths(
            vasp_input.structure, vasp_input.pseudopot_lib_path
        )
    else:
        potcar_paths = vasp_input.potcar_paths

    potcar_path = os.path.join(workdir, filename)

    with open(potcar_path, "wb") as wfd:
        for f in potcar_paths:
            with open(f, "rb") as fd:
                shutil.copyfileobj(fd, wfd)

    return potcar_path


def write_KPOINTS(
    workdir: str, kpoints: Optional[Kpoints] = None, filename: str = "KPOINTS"
) -> str:
    kpoint_path = os.path.join(workdir, filename)
    if kpoints is not None:
        kpoints.write_file(kpoint_path)
    else:
        with open(kpoint_path, "w") as f:
            f.write(
                "Automatic mesh\n0\nGamma\n1 1 1\n0 0 0\n"
            )  # Example KPOINTS content
    return kpoint_path


@Workflow.wrap.as_function_node("workdir")
def create_WorkingDirectory(workdir: str, quiet: bool = False) -> str:
    # Check if workdir exists
    if not os.path.exists(workdir):
        os.makedirs(workdir)
        logger.log(f"made directory '{workdir}'")
    else:
        warnings.warn(
            f"Directory '{workdir}' already exists. Existing files may be overwritten."
        )
    return workdir


@Workflow.wrap.as_function_node("workdir")
def write_VaspInputSet(workdir: str, vasp_input: VaspInput) -> str:
    _ = write_POSCAR(workdir=workdir, structure=vasp_input.structure)
    _ = write_INCAR(workdir=workdir, incar=vasp_input.incar)
    _ = write_POTCAR(workdir=workdir, vasp_input=vasp_input)
    if vasp_input.kpoints is not None:
        _ = write_KPOINTS(workdir=workdir, kpoints=vasp_input.kpoints)

    return workdir


@Workflow.wrap.as_function_node("output")
def run_job(
    command: str,
    workdir: str | None = None,
    environment: Optional[dict[str, str]] = None,
    arguments: Optional[list[str]] = None,
) -> ShellOutput:
    if environment is None:
        environment = {}
    if arguments is None:
        arguments = []
    logger.log(print(f"run_job is in {os.getcwd()}"))
    environ = dict(os.environ)
    environ.update({k: str(v) for k, v in environment.items()})
    proc = subprocess.run(
        [command, *map(str, arguments)],
        capture_output=True,
        cwd=workdir,
        encoding="utf8",
        env=environ,
        shell=True,
    )
    output = ShellOutput()
    output.stdout = proc.stdout
    output.stderr = proc.stderr
    output.return_code = proc.returncode
    return output


@Workflow.wrap.as_function_node("output_dict")
def parse_VaspOutput(workdir: str) -> dict:
    logger.log(f"workdir of parse: {workdir}")
    return pvo(workdir)


@Workflow.wrap.as_function_node("convergence")
def check_convergence(
    workdir: str,
    filename_vasprun: str = "vasprun.xml",
    filename_vasplog: str = "vasp.log",
    backup_vasplog: str = "error.out",
) -> bool:
    converged = False
    line_converged = (
        "reached required accuracy - stopping structural energy minimisation"
    )

    try:
        vr = Vasprun(filename=os.path.join(workdir, filename_vasprun))
        converged = vr.converged
    except:
        try:
            converged = isLineInFile.node_function(
                filepath=os.path.join(workdir, filename_vasplog),
                line=line_converged,
                exact_match=False,
            )
        except:
            try:
                converged = isLineInFile.node_function(
                    filepath=os.path.join(workdir, backup_vasplog),
                    line=line_converged,
                    exact_match=False,
                )
            except:
                pass

    return converged


@Workflow.wrap.as_macro_node("vasp_output", "convergence_status")
def vasp_job(
    self,
    workdir: str,
    vasp_input: VaspInput,
    command: str = "module load vasp; module load intel/19.1.0 impi/2019.6; unset I_MPI_HYDRA_BOOTSTRAP; unset I_MPI_PMI_LIBRARY; mpiexec -n 1 vasp_std",
):
    self.working_dir = create_WorkingDirectory(workdir=workdir)
    self.vaspwriter = write_VaspInputSet(workdir=workdir, vasp_input=vasp_input)
    self.job = run_job(command=command, workdir=workdir)
    self.vasp_output = parse_VaspOutput(workdir=workdir)
    self.convergence_status = check_convergence(workdir=workdir)

    (
        self.working_dir
        >> self.vaspwriter
        >> self.job
        >> self.vasp_output
        >> self.convergence_status
    )
    self.starting_nodes = [self.working_dir]
    return self.vasp_output, self.convergence_status


def stack_element_string(structure) -> tuple[list[str], list[int]]:
    # site_element_list = [atom.symbol for atom in atoms]
    site_element_list = [site.species_string for site in structure]
    past_element = site_element_list[0]
    element_list = [past_element]
    element_count = []
    count = 0

    for element in site_element_list:
        if element == past_element:
            count += 1
        else:
            element_count.append(count)
            element_list.append(element)
            count = 1
            past_element = element

    element_count.append(count)
    return element_list, element_count


def get_default_POTCAR_paths(
    structure: Structure,
    pseudopot_lib_path: str,
    potcar_df: pd.DataFrame = pd.read_csv(POTCAR_specification_data),
) -> list[str]:
    ele_list, _ = stack_element_string(structure)
    potcar_paths = []
    for element in ele_list:
        ele_default_potcar_path = potcar_df[
            (potcar_df["symbol"] == element) & (potcar_df["default"] == True)
        ].potential_name.values[0]
        potcar_paths.append(
            os.path.join(pseudopot_lib_path, ele_default_potcar_path, "POTCAR")
        )

    return potcar_paths
