import string

from typing import Optional
from itertools import combinations
from datetime import datetime
from dataclasses import field
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from ase.build import bulk
from lammpsparser import get_potential_dataframe
from lammpsparser.compatibility.file import lammps_file_interface_function
from ase.io import read
from ase.atoms import Atoms

from core import as_function_node, as_out_dataclass_node


# only conceptual, not truly implemented
def wfMetaData(log_level=0, doc=""):
    return {"log_level": log_level, "doc": doc}


def _structure_from_parsed_output_helper(initial_structure: Atoms, parsed_output: dict, wrap: bool = False) -> Atoms:
    """Construct an `Atoms` object from parsed output data.

    Args:
        initial_structure: The initial atomic structure to use as a template.
        parsed_output: Parsed output containing atomic positions, cell, and indices.
        wrap: Whether to wrap the atomic positions to the simulation cell (default is False).
            Keeping the unwrapped positions is more beneficial if structures are passed between
            different LAMMPS simulations in one workflow to ensure continuity.

    Returns:
        An `Atoms` object with updated positions and cell.

    Example:
        >>> new_atoms = structure_from_parsed_output(atoms, lammps_output)

    """
    # Take a copy of the initial structure as template and update the relevant properties
    atoms_copy = initial_structure.copy()
    atoms_copy.set_array("indices", parsed_output["generic"]["indices"][-1])
    atoms_copy.set_positions(parsed_output["generic"]["positions"][-1])
    atoms_copy.set_velocities(parsed_output["generic"]["velocities"][-1])
    atoms_copy.set_cell(parsed_output["generic"]["cells"][-1])
    atoms_copy.set_pbc(True)
    if wrap:
        atoms_copy.wrap()

    return atoms_copy


@as_function_node
def get_atom_combinations(structure, equivalent_atoms, n_sol: int):  
    l, l_ind = np.max(structure.cell.diagonal()), np.argmax(structure.cell.diagonal())
    chars = string.ascii_lowercase
    symbols = string.ascii_uppercase
    if n_sol > 1:
        sites = np.array(['']*len(equivalent_atoms), dtype='<U3')
        
        for i, ii in enumerate(equivalent_atoms):
            inds = np.where(equivalent_atoms==ii)[0]
            for iii, ind in enumerate(inds):
                if structure.positions[ind,l_ind]<((l/2)+0.1):
                    sites[ind] = str(chars[i]+symbols[0])+str(iii)
                else:
                    sites[ind] = str(chars[i]+symbols[1])+str(iii)
        
        
        combinations_list = []
        for comb in combinations(np.unique(sites), n_sol):
            combinations_list.append('_'.join(comb))
        print(len(combinations_list))
    
        atom_combinations = []
        for i, comb in enumerate(combinations_list):
            atom_combinations.append([equivalent_atoms[np.where(sites==s)[0][0]] for s in comb.split('_')])
    
    else:
        sites = np.array(['']*len(equivalent_atoms), dtype='<U3')
        for i, ii in enumerate(equivalent_atoms):
            sites[i] = str(chars[i]+symbols[0]+'0')
    
        combinations_list = []
        for comb in combinations(np.unique(sites), n_sol):
            combinations_list.append('_'.join(comb))
        print(len(combinations_list))
        atom_combinations = equivalent_atoms

    return atom_combinations, combinations_list


@as_function_node
def structure_from_parsed_output(initial_structure: Atoms, parsed_output: dict, wrap: bool = False) -> Atoms:
    """Construct an `Atoms` object from parsed output data.

    Args:
        initial_structure: The initial atomic structure to use as a template.
        parsed_output: Parsed output containing atomic positions, cell, and indices.
        wrap: Whether to wrap the atomic positions to the simulation cell (default is False).
            Keeping the unwrapped positions is more beneficial if structures are passed between
            different LAMMPS simulations in one workflow to ensure continuity.

    Returns:
        An `Atoms` object with updated positions and cell.

    Example:
        >>> new_atoms = structure_from_parsed_output(atoms, lammps_output)

    """
    atoms = _structure_from_parsed_output_helper(initial_structure=initial_structure, parsed_output=parsed_output, wrap=wrap)
    return atoms


@as_function_node
def get_gb_energy(parsed_output, eng_ref: float):
    cell = parsed_output["generic"]["cells"][-1]
    energy = 1.602e4*(parsed_output["generic"]["energy_pot"][-1] - parsed_output["generic"]["natoms"][-1]*eng_ref)/(2*cell[1,1]*cell[2,2])
    return energy


@as_function_node
def calc_per_atom_energy(parsed_output):
    out = (parsed_output["generic"]["energy_pot"] / parsed_output["generic"]["natoms"])[-1]
    return out


@as_function_node
def get_gb_sigma_9_sites(n_sol: int):
    sites = np.array([ 'a00', 'a01', 'b0+', 'c0+', 'd0+', 'e0+', 'f0+', 'g0+', 'h0+', 'i0+',
                   'b0-', 'c0-', 'd0-', 'e0-', 'f0-', 'g0-', 'h0-', 'i0-',
                   'a10', 'a11', 'b1+', 'c1+', 'd1+', 'e1+', 'f1+', 'g1+',  'h1+', 'i1+',
                   'b1-', 'c1-', 'd1-', 'e1-', 'f1-', 'g1-', 'h1-', 'i1-'])

    indices =  np.array([66,   37, 104,  74,  64,  90,  99,  78,  58,  85,
                         41,   25,   1,  39,  45,  14,  7,   27, 
                         172, 143, 210, 180, 170, 196, 205, 184, 164, 191,
                         147, 131, 107, 145, 151, 120, 113, 133])
    
    combinations_list, atom_combinations = [], []
    for comb in combinations(np.linspace(0, len(sites)-1, len(sites), dtype=int), n_sol):
        combinations_list.append('_'.join(sites[list(comb)]))
        atom_combinations.append(indices[list(comb)])

    return combinations_list, atom_combinations


@as_function_node(["structure_with_solutes"])
def insert_solutes(structure, solute_element: str, indices: list = [0]): 
    structure_copy = structure.copy()
    el_lst = structure.get_chemical_symbols()
    for i in indices:
        el_lst[i] = solute_element
    structure_copy.set_chemical_symbols(el_lst)
    return structure_copy


@as_function_node
def get_list_of_potentials(structure):
    potential_lst = get_potential_dataframe(structure)["Name"].tolist()
    return potential_lst


@as_function_node
def lammps_structure_optimization(structure, potential: str):
    result = lammps_file_interface_function(
        working_directory=os.path.abspath("lmp"),
        structure=structure,
        potential=potential,
        calc_mode="minimize",
        calc_kwargs={},
        units="metal",
        lmp_command="lmp_mpi -in lmp.in"
    )[1]
    return result


@as_function_node
def read_structure_from_file(file_path: str):
    structure = read(file_path)
    return structure


@as_function_node("structure")
def repeat_xyz(structure: Atoms, repeat_x: int = 1, repeat_y: int = 1, repeat_z: int = 1) -> Atoms:
    """
    Repeat a crystal structure periodically along all lattice vectors.

    Parameters
    ----------
    structure : Atoms
        The ASE ``Atoms`` object to be repeated.
    repeat_x : int, optional
        Number of repetitions along the x-axis (default is ``1`` – no change).
    repeat_y : int, optional
        Number of repetitions along the y-axis (default is ``1`` – no change).
    repeat_z : int, optional
        Number of repetitions along the z-axis (default is ``1`` – no change).

    Returns
    -------
    Atoms
        A new ``Atoms`` object containing the repeated supercell.

    Task hint
    ----------
    Use this node when the workflow requires building a larger supercell
    from a primitive cell (e.g., "create a 2×2×2 bulk supercell"). Create a supercell 
    by repeating the input structure. Expand to a nxnxn supercell.
    """
    return structure.repeat([int(repeat_x), int(repeat_y), int(repeat_z)])


@as_out_dataclass_node
class GBOutput:
    initial_structure: Optional[Atoms] = field(default=None, metadata=wfMetaData(log_level=10))
    final_structure: Optional[Atoms] = field(default=None, metadata=wfMetaData(log_level=10))
    energy: Optional[float] = field(default=None, metadata=wfMetaData(log_level=0))


@as_function_node
def get_gb_output(initial_structure: Atoms, parsed_output: dict):
    final_structure = _structure_from_parsed_output_helper(initial_structure=initial_structure, parsed_output=parsed_output)
    output = GBOutput().pure_dataclass()
    output.initial_structure = initial_structure
    output.final_structure = final_structure
    output.energy=parsed_output["generic"]["energy_pot"][-1]
    return output