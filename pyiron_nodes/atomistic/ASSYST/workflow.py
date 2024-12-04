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
from pymatgen.transformations.standard_transformations import (
    DeformStructureTransformation,
)

# PyIron Workflow and Nodes
import pyiron_workflow as pwf
from pyiron_workflow import standard_nodes as std
from pyiron_workflow import for_node, Workflow
from pyiron_nodes.atomistic.engine.vasp import (
    VaspInput,
    vasp_job,
    get_multiple_input,
    generate_VaspInput,
    generate_modified_incar,
    construct_sequential_VaspInput_from_vaspoutput_structure
)
from pyiron_nodes.atomistic.ASSYST.structure_filter_utils import RCORE, is_valid_structure

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
def collect_structures(
    df_list, image_selection_eVatom_threshold=-1, job_names=["ISIF2", "ISIF5", "ISIF7"]
):
    """
    Collects energies and structures from a list of workflow nodes based on a threshold.

    Parameters:
    - df_list (list): List of DataFrames containing VASP outputs.
    - image_selection_eVatom_threshold (float): min difference in eV/atom threshold for selecting images.
    ## IMPORTANT:
    Default value is -1, meaning that we only take the last image.
    - job_names (list): List of job names corresponding to each DataFrame in df_list.

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
        structure = Structure.from_str(df.structures.iloc[0][-1], fmt="json")
        df["eV_atom"] = [row.energy / len(structure) for _, row in df.iterrows()]
        
        if image_selection_eVatom_threshold == -1:
            # Select only the last index
            selected_indices = [len(df.eV_atom.iloc[0]) - 1]
            print(f"ONLY LAST IMAGE OF EACH ISIF SELECTED : {selected_indices}")
        else:
            # Select indices based on the threshold
            selected_indices = select_indices_by_threshold(
                df.eV_atom.iloc[0], threshold=image_selection_eVatom_threshold
            )

        # Extract structures, energies, SCF convergence, and job names for the selected indices
        structures = [
            Structure.from_str(df.structures.iloc[0][i], fmt="json")
            for i in selected_indices
        ]
        for structure in structures:
            print(f"collected structure")
            print(structure)
        energies = [df.energy.iloc[0][i] for i in selected_indices]
        scf_convergence = [df.scf_convergence.iloc[0][i] for i in selected_indices]
        job_names_for_node = [
            f"{job_name}_accur_relaxstep{i}"
            for i in selected_indices
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

    for energy, structure, scf, job_name in zip(
        all_energies, all_structures, all_scf_convergence, all_job_names
    ):
        if scf:  # Only keep entries where SCF convergence is True
            filtered_energies.append(energy)
            filtered_structures.append(structure)
            filtered_scf_convergence.append(scf)
            filtered_job_names.append(job_name)
    
    return (
        filtered_energies,
        filtered_structures,
        filtered_scf_convergence,
        filtered_job_names,
    )


@Workflow.wrap.as_function_node("VaspInputs")
def generate_VaspInputs(structure_list, incar_list, potcar_paths):
    VaspInput_list = []
    for idx, struct in enumerate(structure_list):
        #print(struct)
        vi = VaspInput(struct, incar_list[idx], potcar_paths=potcar_paths[idx])
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
    min_dist=1.0,  # Minimum allowed interatomic distance
    core_overlap_tolerance=0.2, # 20% core overlap allowed.
):
    """
    Generate deformed structures by applying rattling, triaxial strain, and shear strain to each structure.
    Constraints include `min_dist` and core radii constraints (`RCORE`).

    Parameters:
    - structure_list (list): List of input pymatgen structures.
    - job_basename (str): Base name for generated jobs.
    - n_stretch_permutations (int): Number of stretch (triaxial strain) permutations to generate.
    - n_rattle_permutations (int): Number of rattle permutations to generate.
    - shear_strain (float): Maximum shear strain to apply.
    - triaxial_strain (float): Maximum triaxial strain to apply.
    - rattle_displacement (float): Maximum displacement for rattling.
    - rattle_strain (float): Maximum strain for rattling.
    - min_dist (float): Minimum allowed interatomic distance.

    Returns:
    - all_structures (list): List of all valid deformed structures.
    - job_names (list): List of job names corresponding to the deformed structures.
    """
    all_structures = []
    job_names = []

    for idx, structure in enumerate(structure_list):
        rattled_structures = []
        triaxed_structures = []
        sheared_structures = []

        # Apply rattling until the required number of valid structures is generated
        for i in range(n_rattle_permutations):
            attempts = 0
            while len(rattled_structures) < n_rattle_permutations:
                rattled = apply_rattle(
                    structure,
                    displacement=rattle_displacement,
                    max_cell_strain=rattle_strain,
                )
                if is_valid_structure(rattled, min_dist = min_dist, core_overlap_tolerance=core_overlap_tolerance):
                    rattled_structures.append(rattled)
                    job_names.append(f"{job_basename[idx]}_rattle_{len(rattled_structures)}")
                else:
                    print(f"Failed rattle generation: n={attempts}")
                attempts += 1
                if attempts > 100:  # Avoid infinite loops
                    break

        # Apply triaxial strain until the required number of valid structures is generated
        for i in range(n_stretch_permutations):
            attempts = 0
            while len(triaxed_structures) < n_stretch_permutations:
                triaxed = apply_triaxial_strain(structure, max_strain=triaxial_strain)
                if is_valid_structure(triaxed, min_dist = min_dist, core_overlap_tolerance=core_overlap_tolerance):
                    triaxed_structures.append(triaxed)
                    job_names.append(f"{job_basename[idx]}_triax_{len(triaxed_structures)}")
                else:
                    print(f"Failed triax generation: n={attempts}")
                attempts += 1
                if attempts > 100:  # Avoid infinite loops
                    break

            # Apply shear strain for the same structure
            attempts = 0
            while len(sheared_structures) < n_stretch_permutations:
                sheared = apply_shear_strain(structure, max_strain=shear_strain)
                if is_valid_structure(sheared, min_dist = min_dist, core_overlap_tolerance=core_overlap_tolerance):
                    sheared_structures.append(sheared)
                    job_names.append(f"{job_basename[idx]}_shear_{len(sheared_structures)}")
                else:
                    print(f"Failed sheared generation: n={attempts}")
                attempts += 1
                if attempts > 100:  # Avoid infinite loops
                    break

        # Collect all generated structures
        all_structures.extend(rattled_structures)
        all_structures.extend(triaxed_structures)
        all_structures.extend(sheared_structures)

    return all_structures, job_names

@pwf.as_function_node
def get_string(string):
    print(string)
    return string

@pwf.as_macro_node
def run_ASSYST_on_structure(
    wf,
    structure,
    incar,
    potcar_paths,
    ionic_steps=100,
    n_stretch_permutations=2,
    n_rattle_permutations=2,
    image_selection_eVatom_threshold = -1, 
    shear_strain=0.8,
    triaxial_strain=0.8,
    rattle_displacement=0.1,
    rattle_strain=0.05,
    core_overlap_tolerance=0.2,
    job_name="struct_pyxtal",
    vasp_command="module load vasp; module load intel/19.1.0 impi/2019.6; unset I_MPI_HYDRA_BOOTSTRAP; unset I_MPI_PMI_LIBRARY; mpiexec -n 40 vasp_std",
):
    wf.ISIF_7_modifier_dict = pwf.inputs_to_dict(
    input_specification={
        "ISIF": (None, 7),
        "NSW": (int, ionic_steps)
    })
    wf.ISIF7_incar = generate_modified_incar(wf.ISIF7_incar_nsw , wf.ISIF_7_modifier_dict)
    wf.ISIF7_input = generate_VaspInput(
        structure=structure, incar=incar, potcar_paths=potcar_paths
    )
    # This is really unpleasant, 
    wf.ISIF7_jobname = get_string(job_name + "/ISIF7")
    wf.ISIF7_job = vasp_job(
        workdir=wf.ISIF7_jobname, vasp_input=wf.ISIF7_input, command=vasp_command
    )
    wf.ISIF5_incar = generate_modified_incar(incar, {"ISIF": 5})
    wf.ISIF5_input = construct_sequential_VaspInput_from_vaspoutput_structure(
        wf.ISIF7_job.outputs.vasp_output,
        incar=wf.ISIF5_incar.outputs.incar,
        potcar_paths=potcar_paths,
    )
    wf.ISIF5_jobname = get_string(job_name + "/ISIF5")
    wf.ISIF5_job = vasp_job(
        workdir=wf.ISIF5_jobname, vasp_input=wf.ISIF5_input, command=vasp_command
    )
    wf.ISIF2_incar = generate_modified_incar(incar, {"ISIF": 2})
    wf.ISIF2_input = construct_sequential_VaspInput_from_vaspoutput_structure(
        wf.ISIF5_job.outputs.vasp_output,
        incar=wf.ISIF2_incar.outputs.incar,
        potcar_paths=potcar_paths,
    )
    wf.ISIF2_jobname = get_string(job_name + "/ISIF2")
    wf.ISIF2_job = vasp_job(
        workdir=wf.ISIF2_jobname, vasp_input=wf.ISIF2_input, command=vasp_command
    )
    # Need to feed the computed outputs into a different node in the shape of a list
    wf.ISIF_vaspoutputs = pwf.inputs_to_list(
        3,
        wf.ISIF2_job.outputs.vasp_output,
        wf.ISIF5_job.outputs.vasp_output,
        wf.ISIF7_job.outputs.vasp_output,
    )
    wf.ISIF_jobnames = pwf.inputs_to_list(
        3,
        wf.ISIF2_jobname,
        wf.ISIF5_jobname,
        wf.ISIF7_jobname,
    )
    wf.ASSYST_base_structures = collect_structures(
        df_list=wf.ISIF_vaspoutputs,
        image_selection_eVatom_threshold=image_selection_eVatom_threshold,
        job_names=wf.ISIF_jobnames,
    )
    # Generate the accurate incar that is used for potential training data
    wf.accurate_incar = generate_modified_incar(
        incar,
        {"KSPACING": 0.25, "EDIFFG": 1e-4, "EDIFF": 1e-5, "LREAL": False, "NSW": 0},
    )

    wf.n_base_jobs = pwf.standard_nodes.Length(
        wf.ASSYST_base_structures.outputs.filtered_structures
    )
    wf.get_incar_list = get_multiple_input(
        wf.accurate_incar.outputs.incar, n=wf.n_base_jobs
    )
    wf.get_potcar_paths_base = get_multiple_input(potcar_paths, n=wf.n_base_jobs)
    # Generate the VaspInputs which will be used for the calculations of base structure
    wf.ASSYST_base_VaspInputs = generate_VaspInputs(
        structure_list=wf.ASSYST_base_structures.outputs.filtered_structures,
        incar_list=wf.get_incar_list.outputs.objects_list,
        potcar_paths=wf.get_potcar_paths_base,
    )
    wf.ASSYST_base_structure_jobs = for_node(
        vasp_job,
        zip_on=("vasp_input", "workdir"),
        vasp_input=wf.ASSYST_base_VaspInputs.outputs.VaspInputs,
        workdir=wf.ASSYST_base_structures.outputs.filtered_job_names,
        command=vasp_command,
    )

    wf.ASSYST_permutation_structures = get_ASSYST_deformed_structures(
        wf.ASSYST_base_structures.outputs.filtered_structures,
        job_basename=wf.ASSYST_base_structures.outputs.filtered_job_names,
        n_stretch_permutations=n_stretch_permutations,
        n_rattle_permutations=n_rattle_permutations,
        shear_strain=shear_strain,
        triaxial_strain=triaxial_strain,
        rattle_displacement=rattle_displacement,
        rattle_strain=rattle_strain,
        core_overlap_tolerance=core_overlap_tolerance
    )
    wf.n_perm_jobs = pwf.standard_nodes.Length(
        wf.ASSYST_permutation_structures.outputs.job_names
    )
    wf.get_potcar_paths_perms = get_multiple_input(potcar_paths, n=wf.n_perm_jobs)
    wf.get_permutations_incar_list = get_multiple_input(
        wf.accurate_incar.outputs.incar, n=wf.n_perm_jobs
    )
    wf.ASSYST_permutation_VaspInputs = generate_VaspInputs(
        structure_list=wf.ASSYST_permutation_structures.outputs.all_structures,
        incar_list=wf.get_permutations_incar_list.outputs.objects_list,
        potcar_paths=wf.get_potcar_paths_perms,
    )
    wf.ASSYST_permutation_structure_jobs = for_node(
        vasp_job,
        zip_on=("vasp_input", "workdir"),
        vasp_input=wf.ASSYST_permutation_VaspInputs.outputs.VaspInputs,
        workdir=wf.ASSYST_permutation_structures.outputs.job_names,
        command=vasp_command,
    )
    return wf.ASSYST_permutation_structure_jobs, wf.ASSYST_base_structure_jobs
