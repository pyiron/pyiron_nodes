import os
import shutil
import subprocess
from dataclasses import asdict
from typing import Optional, Literal

from ase.atoms import Atoms

from lammpsparser.compatibility.calculate import calc_md
from lammpsparser.compatibility.constraints import set_selective_dynamics
from lammpsparser.output import parse_lammps_output
from lammpsparser.structure import LammpsStructure

from pyiron_nodes.atomistic.calculator.data import InputCalcMD

from core import as_function_node

import pandas as pd

from ase import Atoms
from core import as_function_node
from dataclasses import dataclass

@dataclass
class LammpsIOBundle:
    structure: Atoms
    potential: str | pd.DataFrame
    working_directory: str = "."
    lammps_input_string: str = ''
    lammps_input_filename: str = 'lmp.in'
    lammps_structure_string: str = ''
    lammps_structure_filename: str = 'lammps.data'
    read_restart_filename: Optional[str] = None
    write_restart_filename: Optional[str] = None
    units: str = 'metal'
    resource_path: Optional[str] = None

@as_function_node
def ListPotentials(
    structure: Atoms, 
    resource_path: Optional[str] = None
):

    import os
    from lammpsparser.potential import view_potentials

    if resource_path is None:
        resource_path = os.path.join(os.environ["CONDA_PREFIX"], "share", "iprpy")

    potentials = list(view_potentials(structure, resource_path=resource_path)["Name"].values)

    return potentials

@as_function_node
def CreateLammpsStructure(
    structure: Atoms,
    potential: str | pd.DataFrame,
    units: Literal["metal", "real", "lj", "si", "cgs", "electron", "micro", "nano"] = "metal",
    working_directory: str = ".",
    atom_type: Literal["atomic", "amoeba", "angle", "apip", "atomic", "body", "bond", "charge", "dielectric", "dipole", "dpd", "edpd", "electron", "ellipsoid", "full", "line", "mdpd", "molecular", "oxdna", "peri", "smd", "sph", "sphere", "bpm/sphere", "spin", "tdpd", "tri", "template", "hybrid"] = "atomic",
    bond_dict: Optional[dict] = None,
    resource_path: Optional[str] = None
) -> LammpsIOBundle:
    from lammpsparser.compatibility.file import _get_potential

    print('units:', units)
    io_bundle = LammpsIOBundle(
        structure=structure,
        potential=potential,
        working_directory=working_directory,
        units=units,
        resource_path=resource_path
    )
    
    _, potential_replace, potential_elements = _get_potential(
        potential=potential, resource_path=resource_path
    )
    
    # CHECK if this makes sense
    if "atom_style" in potential_replace.keys():
        atom_type = potential_replace["atom_style"].split()[-1]

    lammps_str = LammpsStructure(bond_dict=bond_dict, units=units, atom_type=atom_type)
    lammps_str.el_eam_lst = potential_elements
    lammps_str.structure = structure

    io_bundle.lammps_structure_string = lammps_str._string_input

    return io_bundle

# TODO Make a separate function in case a full lammps input file is provided.
# Part of the move to make a separate node for a provided full lammps input file!!!

# def CreateLammpsInputFromFile():
#     from lammpsparser.compatibility.file import lammps_file_initialization, _get_potential, _modify_input_dict
        # lmp_str_lst = _modify_input_dict(
    #     input_control_file=input_control,
    #     lmp_str_lst=lmp_str_lst,
    # )

@as_function_node
def CreateLammpsMDInput(
    io_bundle: LammpsIOBundle,
    calc_dataclass: InputCalcMD,
    read_restart_filename: Optional[str] = None,
    write_restart_filename: Optional[str] = None,
):
    from lammpsparser.compatibility.file import lammps_file_initialization, _get_potential

    io_bundle.read_restart_filename = read_restart_filename
    io_bundle.write_restart_filename = write_restart_filename

    calc_kwargs = asdict(calc_dataclass)

    os.makedirs(io_bundle.working_directory, exist_ok=True)
    potential_lst, potential_replace, _ = _get_potential(
        potential=io_bundle.potential, resource_path=io_bundle.resource_path
    )

    # FIXME - temporary fix, should ideally use `read_restart_filename is not None`
    # Problem gets worse when the check box is ticked and the filename is empty
    read_restart_file = bool(read_restart_filename)
    write_restart_file = bool(write_restart_filename)    

    lmp_str_lst = []
    for l in lammps_file_initialization(
        structure=io_bundle.structure,
        units=io_bundle.units,
        read_restart_file=read_restart_file,
        restart_file=read_restart_filename,
    ):
        if l.strip().startswith("units") and "units" in potential_replace:
            lmp_str_lst.append(potential_replace["units"])
        elif l.strip().startswith("atom_style") and "atom_style" in potential_replace:
            lmp_str_lst.append(potential_replace["atom_style"])
            atom_type = potential_replace["atom_style"].split()[-1]
        elif l.strip().startswith("dimension") and "dimension" in potential_replace:
            lmp_str_lst.append(potential_replace["dimension"])
        else:
            lmp_str_lst.append(l)

    lmp_str_lst += potential_lst
    lmp_str_lst += ["variable dumptime equal {} ".format(calc_kwargs.get("n_print", 1))]
    lmp_str_lst += [
        "dump 1 all custom ${dumptime} dump.out id type xsu ysu zsu fx fy fz vx vy vz",
        'dump_modify 1 sort id format line "%d %d %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g"',
    ]

    lmp_str_lst += [
        k + " " + v
        for k, v in set_selective_dynamics(
            structure=io_bundle.structure, calc_md=True
        ).items()
    ]
    if "n_ionic_steps" in calc_kwargs.keys():
        n_ionic_steps = int(calc_kwargs.pop("n_ionic_steps"))
    else:
        n_ionic_steps = 1
    if read_restart_file:
        calc_kwargs["initial_temperature"] = 0.0

    calc_kwargs["units"] = io_bundle.units
    lmp_str_lst += calc_md(**calc_kwargs)

    if read_restart_file:
        lmp_str_lst += ["reset_timestep 0"]

    lmp_str_lst += ["run {} ".format(n_ionic_steps)]
    
    if read_restart_file:
        shutil.copyfile(
            os.path.abspath(read_restart_filename),
            os.path.join(io_bundle.working_directory, os.path.basename(read_restart_filename)),
        )

    if write_restart_file:
        lmp_str_lst.append(f"write_restart {os.path.basename(write_restart_filename)}")

    io_bundle.lammps_input_string = "\n".join(lmp_str_lst)

    return io_bundle

@as_function_node
def RunLammpsCalculation(
    io_bundle: LammpsIOBundle,
    lmp_command: Optional[str] = None,
    cores: int = 1,
    debug: bool = False
):
    #Writing
    os.makedirs(io_bundle.working_directory, exist_ok=True)
    with open(os.path.join(io_bundle.working_directory, io_bundle.lammps_input_filename), "w") as f:
        f.write(io_bundle.lammps_input_string)

    with open(os.path.join(io_bundle.working_directory, io_bundle.lammps_structure_filename), "w") as f:
        f.write(io_bundle.lammps_structure_string)

    #Running
    if not debug:
        if lmp_command is None:
            lmp_command = (
                os.getenv("ASE_LAMMPSRUN_COMMAND", f"mpiexec -n {cores} --oversubscribe lmp_mpi")
                + f" -in {io_bundle.lammps_input_filename}"
            )
        result = subprocess.run(
                lmp_command,
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
                f"LAMMPS exited with code {result.returncode}. "
                f"{result.stdout}"
            )
        output = result.stdout
    else:
        output = io_bundle.working_directory
    
    return io_bundle, output

@as_function_node
def ParseLammpsOutput(
    io_bundle: LammpsIOBundle,
    dump_h5_file_name: str = "dump.h5",
    dump_out_file_name: str = "dump.out",
    log_lammps_file_name: str = "log.lammps",
):
    from lammpsparser.compatibility.file import _get_potential

    _, _, species = _get_potential(
        potential=io_bundle.potential, resource_path=io_bundle.resource_path
    )
    output = parse_lammps_output(
            working_directory=io_bundle.working_directory,
            structure=io_bundle.structure,
            potential_elements=species,
            units=io_bundle.units,
            prism=None,
            dump_h5_file_name=dump_h5_file_name,
            dump_out_file_name=dump_out_file_name,
            log_lammps_file_name=log_lammps_file_name,
        )
    from pyiron_nodes.atomistic.calculator.data import OutputCalcMD

    out = OutputCalcMD.pure_dataclass()

    out.cells=output["generic"].get('cells')
    out.energies_tot=output["generic"].get('energies_tot')
    out.energies_pot=output["generic"].get('energies_pot')
    out.forces=output["generic"].get('forces')
    out.indices=output["generic"].get('indices')
    out.natoms=output["generic"].get('natoms')
    out.positions=output["generic"].get('positions')
    out.pressures=output["generic"].get('pressures')
    out.steps=output["generic"].get('steps')
    out.temperatures=output["generic"].get('temperature')
    out.unwrapped_positions=output["generic"].get('unwrapped_positions')
    out.velocities=output["generic"].get('velocities')
    out.volumes=output["generic"].get('volume')
    out.species=output["generic"].get('species')
    
    return out