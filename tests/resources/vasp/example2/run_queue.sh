#!/bin/bash
#SBATCH --partition=p.cmfe
#SBATCH --ntasks=120
#SBATCH --constraint='[swi1|swi1|swi2|swi3|swi4|swi5|swi6|swi7|swi8|swi9|swe1|swe2|swe3|swe4|swe5|swe6|swe7]'
#SBATCH --time=5760
#SBATCH --output=time.out
#SBATCH --error=error.out
#SBATCH --job-name=pi_18411879
#SBATCH --chdir=/cmmc/u/hmai/project-NiGBs/Ni-bulk-solute-AP/In/In_hdf5/In
#SBATCH --get-user-env=L

pwd; 
echo Hostname: `hostname`
echo Date: `date`
echo JobID: $SLURM_JOB_ID

python -m pyiron_base.cli wrapper -p /cmmc/u/hmai/project-NiGBs/Ni-bulk-solute-AP/In/In_hdf5/In -j 18411879