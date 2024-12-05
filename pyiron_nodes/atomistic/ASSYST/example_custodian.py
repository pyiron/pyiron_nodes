from pyiron_nodes.atomistic.ASSYST.workflow import run_ASSYST_on_structure
from pyiron_workflow import Workflow
from ase.build import bulk
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import Incar
from pyiron_nodes.atomistic.engine.vasp import (
    VaspInput)
import os

def submit_to_slurm(
    node,
    /,
    job_name=None,
    output_file=None,
    error_file=None,
    time_limit="00:05:00",
    account="hmai",
    partition="s.cmmg",
    nodes=1,
    ntasks=32,
    cpus_per_task=1,
    memory="1GB",
):
    """
    An example of a helper function for running nodes on slurm.

    - Saves the node
    - Writes a slurm batch script that 
        - Loads the node
        - Runs it
        - Saves it again
    - Runs the batch script
    """
    if node.graph_root is not node:
        raise ValueError(
            f"Can only submit parent-most nodes, but {node.full_label} "
            f"has root {node.graph_root.full_label}"
        )
        
    node.save(backend="pickle")
    p = node.as_path()
    
    if job_name is None:
        job_name = node.full_label 
        job_name = job_name.replace(node.semantic_delimiter, "_")
        job_name = "pwf" + job_name
        
    script_content = f"""#!/bin/bash
#SBATCH --partition=s.cmmg
#SBATCH --ntasks={ntasks}  # Adjust CPU count as needed
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00  # Adjust wall time as needed
#SBATCH --output=time.out
#SBATCH --error=error.out
#SBATCH --job-name=resubmitter.sh  # Adjust job name as needed
#SBATCH --get-user-env=L
#SBATCH --mem-per-cpu=2000MB
#SBATCH --hint=nomultithread
##SBATCH --reservation=benchmarking

# Execute Python script inline
module purge
echo $PYTHONPATH
source /cmmc/ptmp/hmai/mambaforge/bin/activate pyiron_workflow
python - <<EOF
from pyiron_workflow import PickleStorage
node = PickleStorage().load(filename="{node.as_path().joinpath('picklestorage').resolve()}")  # Load
node.run()  # Run
node.save(backend="pickle")  # Save again
EOF
"""
    submission_script = p.joinpath("node_submission.sh")
    submission_script.write_text(script_content)
    import subprocess
    submission = subprocess.run(["sbatch", submission_script.resolve()])
    return submission

ntasks = 32
vasp_command = f"""module purge

module load intel/2024.0
module load impi/2021.11
module load mkl/2024.0

/cmmc/ptmp/hmai/mambaforge/bin/activate pymatgen

echo "import sys
from custodian.custodian import Custodian
from custodian.vasp.handlers import (
    VaspErrorHandler,
    NonConvergingErrorHandler,
    PositiveEnergyErrorHandler,
    FrozenJobErrorHandler,
)
from utils.custom_custodian_handlers import Han_CustomVaspErrorHandler
from custodian.vasp.jobs import VaspJob

output_filename = 'vasp.log'

handlers = [
    VaspErrorHandler(output_filename=output_filename),
    Han_CustomVaspErrorHandler(),
    NonConvergingErrorHandler(),
    #PositiveEnergyErrorHandler(),
    FrozenJobErrorHandler(output_filename=output_filename),
]

jobs = [VaspJob(sys.argv[1:], output_file=output_filename, suffix='')]
c = Custodian(handlers, jobs, max_errors=10, polling_time_step=3, monitor_freq=1)
c.run()">custodian_vasp.py

python custodian_vasp.py srun -c 1 -n {ntasks} --hint=nomultithread /cmmc/ptmp/hmai/vasp_compiled/intel_suite/vasp.6.4.3_intelsuite_march_znver4/bin/vasp_std >> vasp.log"""
#vasp_command="module load intel/2024.0; module load impi/2021.11; module load mkl/2024.0; srun -c 1 -n 32 --hint=nomultithread /cmmc/ptmp/hmai/vasp_compiled/intel_suite/vasp.6.4.3_intelsuite_march_znver4/bin/vasp_std >> vasp.log"
potcar_paths = ["/cmmc/u/hmai/vasp_potentials_54/Fe_sv/POTCAR"]
bulk_Fe = bulk("Fe", cubic=True, a=2.7)
bulk_Fe = AseAtomsAdaptor().get_structure(bulk_Fe)
bulk_Fe.perturb(0.1)
structure_folder = "struct_pyxtal_0"

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
    "NSW": 100,
    "PREC": "Accurate",
    "SIGMA": 0.2,
    "SYSTEM": "He-20-d-2.4"
})
incar["MAGMOM"] = "2*3"

vi = VaspInput(bulk_Fe, incar, potcar_paths=["/cmmc/u/hmai/vasp_potentials_54/Fe_sv/POTCAR"])

test_dir = "/cmmc/ptmp/hmai/test_pyiron_nodes/ASSYST/test1"
os.makedirs(test_dir, exist_ok=True)
curr_dir = os.getcwd()
os.chdir(test_dir)
wf = Workflow("struct_pyxtal")

wf.ASSYST = run_ASSYST_on_structure(bulk_Fe,
                        incar,
                        potcar_paths=["/cmmc/u/hmai/vasp_potentials_54/Fe_sv/POTCAR"],
                        #ionic_steps = 100,
                        n_stretch_permutations=2,
                        n_rattle_permutations=2,
                        shear_strain=0.8,
                        triaxial_strain=0.8,
                        rattle_displacement=0.1,
                        rattle_strain=0.05,
                        job_name="structure_custodian",
                        vasp_command=vasp_command)
# Send it off
submit_to_slurm(wf)
os.chdir(curr_dir)
