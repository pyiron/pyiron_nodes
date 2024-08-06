#!/bin/bash
#SBATCH --partition=p.cmfe
#SBATCH --ntasks=40
#SBATCH --constraint='[swi1|swi1|swi2|swi3|swi4|swi5|swi6|swi7|swi8|swi9]'
#SBATCH --time=5:00:00
#SBATCH --job-name=ISIF_5_ISIF_7_pyxtal_struct_0.sh
#SBATCH --get-user-env=L

module load intel/19.1.0 impi/2019.6
module load vasp/5.4.4-buildFeb20

#export MODULEPATH=/cmmc/u/hmai/my_modules:$MODULEPATH

#mamba init
source ~/.bashrc
conda activate pymatgen
 
echo 'import sys

from custodian.custodian import Custodian
from custodian.vasp.handlers import VaspErrorHandler, UnconvergedErrorHandler, NonConvergingErrorHandler, PositiveEnergyErrorHandler
from custodian.vasp.jobs import VaspJob

output_filename = "vasp.log"
handlers = [VaspErrorHandler(output_filename=output_filename), UnconvergedErrorHandler(), NonConvergingErrorHandler(), PositiveEnergyErrorHandler()]
jobs = [VaspJob(sys.argv[1:], output_file=output_filename, suffix = ".relax_1", final=False, settings_override=[{"dict": "INCAR", "action": {"_set": {"KSPACING": 0.5}}}]),
        VaspJob(sys.argv[1:], output_file=output_filename, suffix = ".relax_2", final=False,
                settings_override = [{"file": "CONTCAR", "action": {"_file_copy": {"dest": "POSCAR"}}},
                    {"dict": "INCAR", "action": {"_set": {"KSPACING": 0.5, "EDIFF": 1E-5, "EDIFFG": 1E-4}}}], copy_magmom=True),
        VaspJob(sys.argv[1:], output_file=output_filename, suffix = "",
                settings_override = [{"dict": "INCAR", "action": {"_set": {"NSW": 0, "LAECHG": True, "LCHARGE": True, "NELM": 240, "EDIFF": 1E-5}}},
                                     {"file": "CONTCAR", "action": {"_file_copy": {"dest": "POSCAR"}}}])]
c = Custodian(handlers, jobs, max_errors=10)
c.run()'>custodian_vasp.py

if [ $(hostname) == 'cmti001' ];
then
        unset I_MPI_HYDRA_BOOTSTRAP;
        unset I_MPI_PMI_LIBRARY;
        python custodian_vasp.py mpiexec -n $1 vasp_std
else
        python custodian_vasp.py srun -n 40 --exclusive --mem-per-cpu=0 -m block:block,Pack vasp_std &> vasp.log
fi

echo '<net charge>
0.0 <-- specifies the net charge of the unit cell (defaults to 0.0 if nothing specified)
</net charge>
<periodicity along A, B, and C vectors>
.true. <--- specifies whether the first direction is periodic
.true. <--- specifies whether the second direction is periodic
.true. <--- specifies whether the third direction is periodic
</periodicity along A, B, and C vectors>
<atomic densities directory complete path>
/cmmc/u/hmai/chargemol_09_26_2017/atomic_densities/
</atomic densities directory complete path>
<charge type>
DDEC6 <-- specifies the charge type (DDEC3 or DDEC6)
</charge type>
<compute BOs>
.true. <-- specifies whether to compute bond orders or not
</compute BOs>'>job_control.txt

OMP_NUM_THREADS=40
export OMP_NUM_THREADS
export PATH=$PATH:/cmmc/u/hmai/chargemol_09_26_2017/atomic_densities/
export PATH=$PATH:/cmmc/u/hmai/chargemol_09_26_2017/chargemol_FORTRAN_09_26_2017/compiled_binaries/linux
Chargemol_09_26_2017_linux_parallel

# Cleanup the data so it doesn't flood the drive
rm CHG* CHGCAR* PROCAR* WAVECAR* EIGENVAL* REPORT* IBZKPT* REPORT* DOSCAR.*
#find . -type f \( -name "CHG*" -o -name "WAVECAR*" -o -name "PROCAR*" -o -name "IBZKPT*" -o -name "REPORT*" -o -name "EIGENVAL*" -o -name "AECCAR*" \) -delete