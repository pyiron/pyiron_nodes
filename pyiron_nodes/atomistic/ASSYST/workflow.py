# Standard Library Imports
import os
import numpy as np
from tqdm import tqdm
import pandas as pd

# Third-Party Imports
from ase.build import bulk
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import Incar
from pymatgen.transformations.standard_transformations import DeformStructureTransformation

# PyIron Workflow and Nodes
import pyiron_workflow as pwf
from pyiron_workflow import standard_nodes as std
from pyiron_workflow import for_node, Workflow
from pyiron_nodes.atomistic.engine.vasp import (
    get_default_POTCAR_paths, 
    write_POTCAR, 
    VaspInput, 
    vasp_job, 
    create_WorkingDirectory, 
    stack_element_string
)
from pyiron_atomistics.vasp.output import parse_vasp_output

@Workflow.wrap.as_function_node("incar")
def generate_modified_incar(incar, modifications):
    """
    Generates a modified INCAR dictionary by updating specific keys.

    Parameters:
    - incar (dict): Original INCAR dictionary to modify.
    - modifications (dict): Dictionary of keys and their corresponding new values to update.

    Returns:
    - dict: A modified INCAR dictionary with the specified changes.

    Example:
    --------
    Original INCAR:
        incar = {
            "ENCUT": 520,
            "EDIFF": 1e-5,
            "ISMEAR": 0,
            "SIGMA": 0.1
        }

    Modifications:
        modifications = {
            "ISIF": 2,
            "LREAL": "Auto",
            "NSW": 100
        }

    Call:
        modified_incar = generate_single_modified_incar(incar, modifications)

    Result:
        modified_incar = {
            "ENCUT": 520,
            "EDIFF": 1e-05,
            "ISMEAR": 0,
            "SIGMA": 0.1,
            "ISIF": 2,
            "LREAL": "Auto",
            "NSW": 100
        }
    """
    if not isinstance(modifications, dict):
        raise ValueError("Modifications must be provided as a dictionary.")
    
    # Create a copy of the original INCAR and apply modifications
    modified_incar = incar.copy()
    for key, value in modifications.items():
        modified_incar[key] = value

    return modified_incar

@Workflow.wrap.as_function_node("VaspInput")
def construct_next_VaspInput(vasp_output,
                             incar,
                             potcar_paths):
    
    vi = VaspInput(Structure.from_str(vasp_output.structures.iloc[0][-1], fmt="json"),
                   incar,
                   potcar_paths=potcar_paths)
    return vi

def select_indices_by_threshold(array, threshold):
    """
    Selects the indices of the first, last, and all values in the array
    that differ from the previous selected value by more than a specified threshold.

    Parameters:
    - array (iterable): The input array or list of values.
    - threshold (float): The minimum difference required to select a value.

    Returns:
    - list: Indices of selected values.
    """
    if len(array) == 0:
        return []

    # Initialize the list with the first index
    selected_indices = [0]

    # Iterate through the array and select indices based on the threshold
    for i in range(1, len(array)):
        if abs(array[i] - array[selected_indices[-1]]) > threshold:
            selected_indices.append(i)

    # Ensure the last index is included
    if len(array) - 1 not in selected_indices:
        selected_indices.append(len(array) - 1)

    return selected_indices

@Workflow.wrap.as_function_node
def collect_structures(df_list, energy_diff_threshold=0.2, job_names=["ISIF2", "ISIF5", "ISIF7"]):
    """
    Collects energies and structures from a list of workflow nodes based on a threshold.

    Parameters:
    - node_list (list): List of workflow nodes containing VASP outputs.
    - energy_diff_threshold (float): Threshold for selecting indices.
    - job_names (list): List of job names corresponding to each node in node_list.

    Returns:
    - filtered_energies (list): List of selected energies.
    - filtered_structures (list): List of selected pymatgen Structure objects.
    - filtered_scf_convergence (list): List of SCF convergence values.
    - filtered_job_names (list): List of generated job names like "ISIF2_base1".
    """
    # Initialize empty lists for energies, structures, SCF convergence, and job names
    all_energies = []
    all_structures = []
    all_scf_convergence = []
    all_job_names = []
    
    # Iterate over the workflow nodes
    for node_idx, (df, job_name) in enumerate(zip(df_list, job_names)):
        # Select indices based on the threshold
        selected_indices = select_indices_by_threshold(
            df.energy.iloc[0],
            threshold=energy_diff_threshold
        )
        
        # Extract structures, energies, SCF convergence, and job names for the selected indices
        structures = [
            Structure.from_str(
                df.structures.iloc[0][i],
                fmt="json"
            ) for i in selected_indices
        ]
        energies = [
            df.energy.iloc[0][i]
            for i in selected_indices
        ]
        scf_convergence = [
            df.scf_convergence[0][i]
            for i in selected_indices
        ]
        job_names_for_node = [
            f"{job_name}_base{node_idx + 1}_relaxstep{i + 1}" for i in range(len(selected_indices))
        ]
        
        # Append to the main lists
        all_energies.extend(energies)
        all_structures.extend(structures)
        all_scf_convergence.extend(scf_convergence)
        all_job_names.extend(job_names_for_node)
    
    # Filter out entries where SCF convergence is False
    filtered_energies = []
    filtered_structures = []
    filtered_scf_convergence = []
    filtered_job_names = []
    
    for energy, structure, scf, job_name in zip(all_energies, all_structures, all_scf_convergence, all_job_names):
        if scf:  # Only keep entries where SCF convergence is True
            filtered_energies.append(energy)
            filtered_structures.append(structure)
            filtered_scf_convergence.append(scf)
            filtered_job_names.append(job_name)
    
    return filtered_energies, filtered_structures, filtered_scf_convergence, filtered_job_names



@Workflow.wrap.as_function_node("VaspInputs")
def generate_VaspInputs(structure_list,
                        incar_list,
                        potcar_paths):
    VaspInput_list = []
    for idx, struct in enumerate(structure_list):
        print(struct)
        vi = VaspInput(struct,
                       incar_list[idx],
                       potcar_paths=potcar_paths[idx])
        VaspInput_list.append(vi)
    return VaspInput_list
    
def apply_triaxial_strain(structure, max_strain=0.8):
    """Apply random triaxial strain up to max_strain."""
    # Generating a diagonal strain matrix for triaxial strain
    strain_values = 1 + np.random.uniform(-max_strain, max_strain, 3)
    strain_matrix = np.diag(strain_values)
    transformation = DeformStructureTransformation(strain_matrix)
    return transformation.apply_transformation(structure)

def apply_shear_strain(structure, max_strain=0.8):
    """Apply random shear strain up to max_strain."""
    # For shear, we need a deformation matrix that includes off-diagonal shear components.
    shear_matrix = np.identity(3) + np.random.uniform(-max_strain, max_strain, (3, 3))
    np.fill_diagonal(shear_matrix, 1)  # Keeping the volume roughly the same
    transformation = DeformStructureTransformation(shear_matrix)
    return transformation.apply_transformation(structure)

def apply_rattle(structure, displacement=0.5, max_cell_strain=0.05):
    """Apply random displacement (RATTLE) to atoms and a small strain."""
    new_struct = structure.copy()
    # Random displacement
    for site in new_struct:
        displacement_vector = np.random.normal(0, displacement, 3)
        site.coords += displacement_vector
    
    # Random small strain
    strain_values = 1 + np.random.uniform(-max_cell_strain, max_cell_strain, 3)
    strain_matrix = np.diag(strain_values)
    transformation = DeformStructureTransformation(strain_matrix)
    return transformation.apply_transformation(new_struct)
    
@Workflow.wrap.as_function_node
def get_ASSYST_deformed_structures(
    structure_list,
    job_basename="pyxtal",
    n_stretch_permutations=5,
    n_rattle_permutations=5,
    shear_strain=0.8,
    triaxial_strain=0.8,
    rattle_displacement=0.1,
    rattle_strain=0.05,
):
    """
    Generate deformed structures by applying rattling, triaxial strain, and shear strain to each structure.

    Parameters:
    - structure_list (list): List of input structures.
    - job_basename (str): Base name for generated jobs.
    - n_stretch_permutations (int): Number of stretch (triaxial strain) permutations to generate.
    - n_rattle_permutations (int): Number of rattle permutations to generate.
    - shear_strain (float): Maximum shear strain to apply.
    - triaxial_strain (float): Maximum triaxial strain to apply.
    - rattle_displacement (float): Maximum displacement for rattling.
    - rattle_strain (float): Maximum strain for rattling.

    Returns:
    - all_structures (list): List of all deformed structures.
    - job_names (list): List of job names corresponding to the deformed structures.
    """
    all_structures = []  # Initialize list to collect all deformed structures
    job_names = []  # Initialize list to collect job names

    for idx, structure in enumerate(structure_list):
        rattled_structures = []
        triaxed_structures = []
        sheared_structures = []

        # Apply rattling
        for i in range(n_rattle_permutations):
            rattled = apply_rattle(
                structure,
                displacement=rattle_displacement,
                max_cell_strain=rattle_strain
            )
            rattled_structures.append(rattled)
            job_names.append(f"{job_basename[idx]}_rattle_relpath{idx}_{i}")

        # Apply triaxial strain
        for i in range(n_stretch_permutations):
            triaxed = apply_triaxial_strain(
                structure,
                max_strain=triaxial_strain
            )
            triaxed_structures.append(triaxed)
            job_names.append(f"{job_basename[idx]}_triaxial_relpath{idx}_{i}")

            # Apply shear strain for the same structure
            sheared = apply_shear_strain(
                structure,
                max_strain=shear_strain
            )
            sheared_structures.append(sheared)
            job_names.append(f"{job_basename[idx]}_shear_relpath{idx}_{i}")

        # Collect all generated structures
        all_structures.extend(rattled_structures)
        all_structures.extend(triaxed_structures)
        all_structures.extend(sheared_structures)

    return all_structures, job_names
