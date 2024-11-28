from pyiron_nodes.atomistic.ASSYST.workflow import generate_modified_incar, generate_VaspInputs, construct_next_VaspInput, collect_structures, get_ASSYST_deformed_structures
import pyiron_workflow as pwf
from pyiron_workflow import for_node, Workflow

import shutil
import os
# Third-Party Imports
from ase.build import bulk
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import Incar

from pyiron_nodes.atomistic.engine.vasp import (
    get_default_POTCAR_paths, 
    write_POTCAR, 
    VaspInput, 
    vasp_job, 
    create_WorkingDirectory, 
    stack_element_string
)

vasp_command="module load vasp; module load intel/19.1.0 impi/2019.6; unset I_MPI_HYDRA_BOOTSTRAP; unset I_MPI_PMI_LIBRARY; mpiexec -n 40 vasp_std"
potcar_paths = ["/cmmc/u/hmai/vasp_potentials_54/Fe_sv/POTCAR"]
bulk_Fe = bulk("Fe", cubic=True, a=2.7)
bulk_Fe = AseAtomsAdaptor().get_structure(bulk_Fe)
bulk_Fe.perturb(0.1)
structure_folder = "struct_pyxtal_1"

incar = Incar.from_dict({
    "ALGO": "Fast",
    "AMIX": 0.01,
    "AMIX_MAG": 0.1,
    "BMIX": 0.0001,
    "BMIX_MAG": 0.0001,
    "EDIFF": 1e-05,
    "EDIFFG": -0.01,
    "ENCUT": 400,
    "GGA": "Pe",
    "IBRION": 2,
    "ISIF": 7,
    "ISMEAR": 1,
    "ISPIN": 2,
    "ISTART": 0,
    "KPAR": 2,
    "LORBIT": 10,
    "LPLANE": False,
    "LREAL": False,
    "MAGMOM": "20*3.0 1*-0.01 27*3.0",
    "NCORE": 4,
    "NELM": 120,
    "NSIM": 1,
    "NSW": 0,
    "PREC": "Accurate",
    "SIGMA": 0.2,
    "SYSTEM": "He-20-d-2.4"
})
incar["MAGMOM"] = "2*3"

vi = VaspInput(bulk_Fe, incar, potcar_paths=["/cmmc/u/hmai/vasp_potentials_54/Fe_sv/POTCAR"])

try:
    shutil.rmtree("struct_pyxtal_1")
    shutil.rmtree("vasp_ASSYST_struct")
except:
    a = 0



wf = pwf.Workflow("vasp_ASSYST_struct")
wf.ISIF7_incar = generate_modified_incar(incar,
                                         {"ISIF": 7})
wf.ISIF7_input = generate_VaspInputs(structure_list = [bulk_Fe], 
                                     incar_list = [incar],
                                     potcar_paths=[potcar_paths])
wf.ISIF7_job = vasp_job(workdir=f"{structure_folder}/ISIF7",
                         vasp_input = wf.ISIF7_input.outputs.VaspInputs[0],
                         command=vasp_command
                        )
wf.ISIF5_incar = generate_modified_incar(incar,
                                         {"ISIF": 5})
wf.ISIF5_input = construct_next_VaspInput(wf.ISIF7_job.outputs.vasp_output,
                                incar = wf.ISIF5_incar.outputs.incar,
                                potcar_paths=potcar_paths)
wf.ISIF5_job = vasp_job(workdir=f"{structure_folder}/ISIF5",
                        vasp_input=wf.ISIF5_input,
                        command=vasp_command)
wf.ISIF2_incar = generate_modified_incar(incar,
                                         {"ISIF": 2})
wf.ISIF2_input = construct_next_VaspInput(wf.ISIF5_job.outputs.vasp_output,
                                incar = wf.ISIF2_incar.outputs.incar,
                                potcar_paths=potcar_paths)
wf.ISIF2_job = vasp_job(workdir=f"{structure_folder}/ISIF2",
                          vasp_input=wf.ISIF2_input,
                            command=vasp_command)
# Need to feed the computed outputs into a different node in the shape of a list
wf.ISIF_vaspoutputs = pwf.inputs_to_list(3,
                                         wf.ISIF2_job.outputs.vasp_output,
                                         wf.ISIF5_job.outputs.vasp_output,
                                         wf.ISIF7_job.outputs.vasp_output)

wf.ASSYST_base_structures = collect_structures(df_list = wf.ISIF_vaspoutputs,
                                               energy_diff_threshold = 0.1,
                                               job_names=[f"{os.getcwd()}/{structure_folder}/ISIF2",
                                                          f"{os.getcwd()}/{structure_folder}/ISIF5",
                                                          f"{os.getcwd()}/{structure_folder}/ISIF7"])
# Generate the accurate incar that is used for potential training data
wf.accurate_incar = generate_modified_incar(incar,
                                         {"KSPACING": 0.25,
                                          "EDIFFG": 1E-4,
                                          "EDIFF": 1E-5,
                                          "LREAL": False,
                                          "NSW": 0})
@pwf.as_function_node
def get_INCAR_list(incar, repetition):
    incar_list = [incar] * repetition
    return incar_list
wf.get_incar_list = get_INCAR_list(wf.accurate_incar.outputs.incar,
                                   repetition = len(wf.ASSYST_base_structures.outputs.filtered_job_names.value))
# Generate the VaspInputs which will be used for the calculations of base structure
wf.ASSYST_base_VaspInputs = generate_VaspInputs(structure_list=wf.ASSYST_base_structures.outputs.filtered_structures,
                                                incar_list= wf.get_incar_list.outputs.incar_list,
                                                potcar_paths=[potcar_paths] * len(wf.ASSYST_base_structures.outputs))

wf.ASSYST_base_structure_jobs = for_node(
    vasp_job,
    zip_on = ("vasp_input", "workdir", "command"),
    vasp_input = wf.ASSYST_base_VaspInputs.outputs.VaspInputs,
    workdir = wf.ASSYST_base_structures.outputs.filtered_job_names,
    command = [vasp_command] * len(wf.ASSYST_base_structures.outputs)
)

wf.ASSYST_permutation_structures = get_ASSYST_deformed_structures(
    wf.ASSYST_base_structures.outputs.filtered_structures,
    job_basename=wf.ASSYST_base_structures.outputs.filtered_job_names,
    n_stretch_permutations=5,
    n_rattle_permutations=5,
    shear_strain=0.8,
    triaxial_strain=0.8,
    rattle_displacement=0.1,
    rattle_strain=0.05,
)
# wf.failed = False
# wf.remove_child("get_permutations_incar_list")
wf.get_permutations_incar_list = get_INCAR_list(wf.accurate_incar.outputs.incar,
                                                len(wf.ASSYST_permutation_structures.outputs.job_names.value))
wf.remove_child("ASSYST_permutation_VaspInputs")
wf.ASSYST_permutation_VaspInputs = generate_VaspInputs(structure_list=wf.ASSYST_permutation_structures.outputs.all_structures,
                                                incar_list= wf.get_permutations_incar_list.outputs.incar_list,
                                                potcar_paths=[potcar_paths] * len(wf.ASSYST_permutation_structures.outputs.job_names.value))
# wf.remove_child("ASSYST_permutation_structure_jobs")
wf.ASSYST_permutation_structure_jobs = for_node(
    vasp_job,
    zip_on = ("vasp_input", "workdir", "command"),
    vasp_input = wf.ASSYST_permutation_VaspInputs.outputs.VaspInputs,
    workdir = wf.ASSYST_permutation_structures.outputs.job_names,
    command = [vasp_command] * len(wf.ASSYST_permutation_structures.outputs.job_names.value)
)
wf.pull()