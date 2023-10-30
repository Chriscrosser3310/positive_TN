#!/bin/bash
#SBATCH --time=10-00:00:00
#SBATCH -J iMPS_pTN
##SBATCH -o job.out
##SBATCH -e job.err
#SBATCH --mail-user=jchen9@caltech.edu
#SBATCH --mail-type=all
#SBATCH --partition=serial,parallel
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH -x pauling[001-002]
#SBATCH --mem=100G
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
# mkdir /scratch/local/Chriscrosser/

python -u iMPS_script.py [1,10,10] [0.1,2,10]