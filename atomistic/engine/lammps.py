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
class LammpsInputResources:
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
) -> LammpsInputResources:
    from lammpsparser.compatibility.file import _get_potential

    # Parse potential to extract units and atom_style
    if isinstance(potential, pd.DataFrame):
        config_lines = potential.iloc[0]["Config"]
        for line in config_lines:
            line = line.strip()
            if line.startswith("units"):
                units = line.split()[1]
            elif line.startswith("atom_style"):
                atom_type = line.split()[1]

    input_resources = LammpsInputResources(
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
    #if "atom_style" in potential_replace.keys():
    #    atom_type = potential_replace["atom_style"].split()[-1]

    if atom_type == "full":
        # LammpsStructure does not support "full" atom_style, so an additional function write_lammps_data_full was added in this file and is used for the 'full' case
        # Does not include dihedrals or impropers, but should be sufficient for bonds and angles.
        # not hardcoded for tip3p water, but requires bond_dict to be provided. Example is in the electrochemistry/equilibrate.py file for WaterPotential node.
        
        structure_string = write_lammps_data_full(
            structure=structure,
            specorder=potential_elements,
            bond_dict=bond_dict,
            potential=potential
        )

    else:
        lammps_str = LammpsStructure(bond_dict=bond_dict, units=units, atom_type=atom_type)
        lammps_str.el_eam_lst = potential_elements
        lammps_str.structure = structure

        structure_string = lammps_str._string_input

    input_resources.lammps_structure_string = structure_string

    return input_resources

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
    input_resources: LammpsInputResources,
    calc_dataclass: InputCalcMD,
    read_restart_filename: Optional[str] = None,
    write_restart_filename: Optional[str] = None,
):
    from lammpsparser.compatibility.file import lammps_file_initialization, _get_potential

    input_resources.read_restart_filename = read_restart_filename
    input_resources.write_restart_filename = write_restart_filename

    calc_kwargs = asdict(calc_dataclass)

    os.makedirs(input_resources.working_directory, exist_ok=True)
    potential_lst, potential_replace, _ = _get_potential(
        potential=input_resources.potential, resource_path=input_resources.resource_path
    )

    # FIXME - temporary fix, should ideally use `read_restart_filename is not None`
    # Problem gets worse when the check box is ticked and the filename is empty
    read_restart_file = bool(read_restart_filename)
    write_restart_file = bool(write_restart_filename)    

    lmp_str_lst = []
    for l in lammps_file_initialization(
        structure=input_resources.structure,
        units=input_resources.units,
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


    # Handle potential: write to file if DataFrame, else inline
    if isinstance(input_resources.potential, pd.DataFrame):
        potential_filename = "potential.inp"
        potential_filepath = os.path.join(input_resources.working_directory, potential_filename)
        with open(potential_filepath, "w") as f:
            f.writelines(potential_lst)
        lmp_str_lst += [f"include {potential_filename}\n"]
    else:
        lmp_str_lst += potential_lst

    lmp_str_lst += ["variable dumptime equal {} ".format(calc_kwargs.get("n_print", 1))]
    lmp_str_lst += [
        "dump 1 all custom ${dumptime} dump.out id type xsu ysu zsu fx fy fz vx vy vz",
        'dump_modify 1 sort id format line "%d %d %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g"',
    ]

    if "n_ionic_steps" in calc_kwargs.keys():
        n_ionic_steps = int(calc_kwargs.pop("n_ionic_steps"))
    else:
        n_ionic_steps = 1
    if read_restart_file:
        calc_kwargs["initial_temperature"] = 0.0

    calc_kwargs["units"] = input_resources.units
    lmp_str_lst += calc_md(**calc_kwargs)

    lmp_str_lst += [
        k + " " + v
        for k, v in set_selective_dynamics(
            structure=input_resources.structure, calc_md=True
        ).items()
    ]

    if read_restart_file:
        lmp_str_lst += ["reset_timestep 0"]

    lmp_str_lst += ["run {} ".format(n_ionic_steps)]
    
    if read_restart_file:
        shutil.copyfile(
            os.path.abspath(read_restart_filename),
            os.path.join(input_resources.working_directory, os.path.basename(read_restart_filename)),
        )

    if write_restart_file:
        lmp_str_lst.append(f"write_restart {os.path.basename(write_restart_filename)}")

    input_resources.lammps_input_string = "\n".join(lmp_str_lst)

    return input_resources

@as_function_node
def RunLammpsCalculation(
    input_resources: LammpsInputResources,
    lmp_command: Optional[str] = None,
    cores: int = 1,
    debug: bool = False
):
    #Writing
    os.makedirs(input_resources.working_directory, exist_ok=True)
    with open(os.path.join(input_resources.working_directory, input_resources.lammps_input_filename), "w") as f:
        f.write(input_resources.lammps_input_string)

    with open(os.path.join(input_resources.working_directory, input_resources.lammps_structure_filename), "w") as f:
        f.write(input_resources.lammps_structure_string)

    #Running
    if not debug:
        if lmp_command is None:
            lmp_command = (
                os.getenv("ASE_LAMMPSRUN_COMMAND", f"mpiexec -n {cores} --oversubscribe lmp_mpi")
                + f" -in {input_resources.lammps_input_filename}"
            )
        result = subprocess.run(
                lmp_command,
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
                f"LAMMPS exited with code {result.returncode}. "
                f"{result.stdout}"
            )
        output = result.stdout
    else:
        output = input_resources.working_directory
    
    return input_resources, output

@as_function_node
def ParseLammpsOutput(
    input_resources: LammpsInputResources,
    dump_h5_file_name: str = "dump.h5",
    dump_out_file_name: str = "dump.out",
    log_lammps_file_name: str = "log.lammps",
):
    from lammpsparser.compatibility.file import _get_potential

    _, _, species = _get_potential(
        potential=input_resources.potential, resource_path=input_resources.resource_path
    )
    output = parse_lammps_output(
            working_directory=input_resources.working_directory,
            structure=input_resources.structure,
            potential_elements=species,
            units=input_resources.units,
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
    
    ase_trajectory = None

    if out.positions is not None:
        ase_trajectory = []

        # Get symbols once from initial structure
        all_symbols = input_resources.structure.get_chemical_symbols()

        for frame_idx in range(len(out.positions)):
            
            cell = out.cells[frame_idx] if out.cells is not None else None
            frame = Atoms(
                symbols=all_symbols,
                positions=out.positions[frame_idx],
                cell=cell,
                pbc=cell is not None,
            )
            ase_trajectory.append(frame)

    return out, ase_trajectory


# temporary here, should be included in LammpsStructure?
def write_lammps_data_full(
    structure: Atoms,
    specorder: list[str],
    bond_dict: dict,
    potential:  str | pd.DataFrame,
) -> str:
    """
    Build LAMMPS data file string in full atom_style.

    Full atom line format:
        atom-ID  mol-ID  atom-type  charge  x  y  z

    Returns
    -------
    str
        Complete LAMMPS data file content as a string.
    """
    from ase.neighborlist import neighbor_list
    from lammpsparser.structure import UnfoldingPrism, is_skewed
    from ase.data import atomic_masses, atomic_numbers
    from collections import defaultdict

    prism   = UnfoldingPrism(structure.cell, digits=15)
    coords  = [prism.pos_to_lammps(pos) for pos in structure.positions]
    symbols = structure.get_chemical_symbols()
    n_atoms = len(structure)

    charges = extract_charges_from_lammps_potential(
        potential.iloc[0]["Config"], specorder=specorder
    )

    species_lammps_id_dict = {el: idx + 1 for idx, el in enumerate(specorder)}

    # ------------------------------------------------------------------
    # Find bonds and angles via neighbor search using bond_dict
    # ------------------------------------------------------------------
    bond_list  = []  # [(atom_i, atom_j, bond_type), ...]
    angle_list = []  # [(atom_i, atom_j_center, atom_k, angle_type), ...]

    bond_type_map      = {}
    angle_type_map     = {}
    bond_type_counter  = 1
    angle_type_counter = 1

    neighbors = defaultdict(list)  # {atom_i: [atom_j, ...]}

    for center_el, specs in bond_dict.items():
        for spec_name, spec in specs.items():

            if "max_bond_num" in spec:
                cutoff      = spec["cutoff"]
                neighbor_el = spec["neighbor_type"]

                if spec_name not in bond_type_map:
                    bond_type_map[spec_name] = bond_type_counter
                    bond_type_counter += 1
                btype = bond_type_map[spec_name]

                i_lst, j_lst = neighbor_list(
                    "ij", structure,
                    cutoff={
                        (center_el, neighbor_el): cutoff,
                        (neighbor_el, center_el): cutoff,
                    },
                )
                seen = set()
                for i, j in zip(i_lst, j_lst):
                    if symbols[i] == center_el and symbols[j] == neighbor_el:
                        pair = (min(i, j), max(i, j))
                        if pair not in seen:
                            bond_list.append((pair[0], pair[1], btype))
                            seen.add(pair)
                            neighbors[pair[0]].append(pair[1])
                            neighbors[pair[1]].append(pair[0])

            if "max_angle_num" in spec:
                cutoff      = spec["cutoff"]
                neighbor_el = spec["neighbor_type"]

                if spec_name not in angle_type_map:
                    angle_type_map[spec_name] = angle_type_counter
                    angle_type_counter += 1
                atype = angle_type_map[spec_name]

                for center_idx, center_sym in enumerate(symbols):
                    if center_sym != center_el:
                        continue
                    nb = [
                        j for j in neighbors[center_idx]
                        if symbols[j] == neighbor_el
                    ]
                    for a in range(len(nb)):
                        for b in range(a + 1, len(nb)):
                            angle_list.append(
                                (nb[a], center_idx, nb[b], atype)
                            )

    # ------------------------------------------------------------------
    # Assign molecule IDs via union-find on bond connectivity
    # ------------------------------------------------------------------
    parent = list(range(n_atoms))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    for i, j, _ in bond_list:
        union(i, j)

    root_to_mol: dict = {}
    mol_counter = 1
    mol_ids     = []
    for i in range(n_atoms):
        root = find(i)
        if root not in root_to_mol:
            root_to_mol[root] = mol_counter
            mol_counter += 1
        mol_ids.append(root_to_mol[root])

    # ------------------------------------------------------------------
    # Build string
    # ------------------------------------------------------------------
    xhi, yhi, zhi, xy, xz, yz = prism.get_lammps_prism_str()

    lines = []

    # Header
    lines.append("Start File for LAMMPS TEST")
    lines.append(f"{n_atoms} atoms")
    lines.append(f"{len(specorder)} atom types")
    lines.append(f"{len(bond_list)} bonds")
    lines.append(f"{len(angle_list)} angles")
    lines.append(f"{len(bond_type_map)} bond types")
    lines.append(f"{len(angle_type_map)} angle types")
    lines.append("")

    # Cell
    lines.append(f"0. {xhi} xlo xhi")
    lines.append(f"0. {yhi} ylo yhi")
    lines.append(f"0. {zhi} zlo zhi")
    if is_skewed(structure):
        lines.append(f"{xy} {xz} {yz} xy xz yz")
    lines.append("")

    # Masses
    lines.append("Masses\n")
    for el, idx in species_lammps_id_dict.items():
        mass = atomic_masses[atomic_numbers[el]]
        lines.append(f"{idx:3d} {mass:f}  # ({el})")
    lines.append("")

    # Atoms — full style: atom-ID  mol-ID  atom-type  charge  x  y  z
    lines.append("Atoms  # full\n")
    for idx in range(n_atoms):
        el      = symbols[idx]
        atype   = species_lammps_id_dict[el]
        charge  = charges.get(el, 0.0)
        x, y, z = coords[idx]
        lines.append(
            f"{idx+1:6d} "
            f"{mol_ids[idx]:6d} "
            f"{atype:4d} "
            f"{charge:10.6f} "
            f"{x:.15f} {y:.15f} {z:.15f}"
        )
    lines.append("")

    # Bonds
    lines.append("Bonds\n")
    for b_id, (i, j, btype) in enumerate(bond_list, start=1):
        lines.append(f"{b_id:6d} {btype:4d} {i+1:6d} {j+1:6d}")
    lines.append("")

    # Angles
    lines.append("Angles\n")
    for a_id, (i, j, k, atype) in enumerate(angle_list, start=1):
        lines.append(f"{a_id:6d} {atype:4d} {i+1:6d} {j+1:6d} {k+1:6d}")
    lines.append("")

    return "\n".join(lines)   # ← caller decides what to do with it

def extract_charges_from_lammps_potential(lines, specorder):
    """
    Extract charges mapped to element symbols from LAMMPS potential Config lines.

    Parameters
    ----------
    lines : list of str
        Lines from potential["Config"].
    specorder : list of str
        Element symbols in potential order, e.g. ["O", "H"].

    Returns
    -------
    dict
        {element_symbol: charge_value}, e.g. {"O": -0.830, "H": 0.415}
    """
    import re

    # group_name -> charge
    group_charges = {}
    # element_symbol -> charge  (group name IS the element symbol here)
    result = {}

    # "group O type 2"  — group name is element symbol directly
    p_group_type = re.compile(r"^group\s+(\S+)\s+type\s+(\d+)", re.IGNORECASE)
    # "set group O charge -0.830"
    p_set_group  = re.compile(r"^set\s+group\s+(\S+)\s+charge\s+(-?\d*\.?\d+)", re.IGNORECASE)
    # "set type 1 charge -0.834"
    p_set_type   = re.compile(r"^set\s+type\s+(\d+)\s+charge\s+(-?\d*\.?\d+)", re.IGNORECASE)

    # type_id -> element symbol  (from "group O type 2")
    type_to_element = {}

    for raw_line in lines:
        line = raw_line.strip()

        # skip comments and empty lines
        if not line or line.startswith("#"):
            continue

        m = p_group_type.match(line)
        if m:
            element = m.group(1)   # "O", "H", "Pt", "Ne"
            type_id = int(m.group(2))
            type_to_element[type_id] = element
            continue

        m = p_set_group.match(line)
        if m:
            element = m.group(1)   # group name IS element symbol
            charge  = float(m.group(2))
            group_charges[element] = charge
            continue

        m = p_set_type.match(line)
        if m:
            type_id = int(m.group(1))
            charge  = float(m.group(2))
            element = type_to_element.get(type_id)
            if element is not None:
                group_charges[element] = charge
            continue

    # ------------------------------------------------------------------
    # Only return charges for elements in specorder
    # ------------------------------------------------------------------
    for element in specorder:
        if element in group_charges:
            result[element] = group_charges[element]
        else:
            print(f"  WARNING: no charge found for element '{element}', defaulting to 0.0")
            result[element] = 0.0

    print("Extracted charges:", result)
    return result