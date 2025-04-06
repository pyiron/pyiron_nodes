import os
from ase.build import bulk

from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import Incar

from pyiron_workflow import Workflow
from pyiron_nodes.atomistic.engine.vasp import VaspInput
from pyiron_nodes.atomistic.ASSYST.workflow import run_ASSYST_on_structure


vasp_command="module load vasp; module load intel/19.1.0 impi/2019.6; unset I_MPI_HYDRA_BOOTSTRAP; unset I_MPI_PMI_LIBRARY; mpiexec -n 40 vasp_std >> vasp.log"
potcar_paths = ["/cmmc/u/hmai/vasp_potentials_54/Fe_sv/POTCAR"]
bulk_Fe = bulk("Fe", cubic=True, a=2.7)
bulk_Fe = AseAtomsAdaptor().get_structure(bulk_Fe)
bulk_Fe.perturb(0.1)
structure_folder = "struct_pyxtal_5"

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
                        potcar_paths=potcar_paths,
                        ionic_steps = 100,
                        n_stretch_permutations=2,
                        n_rattle_permutations=2,
                        shear_strain=0.8,
                        triaxial_strain=0.8,
                        rattle_displacement=0.1,
                        rattle_strain=0.05,
                        job_name="lol",
                        vasp_command=vasp_command)
wf.run()
os.chdir(curr_dir)