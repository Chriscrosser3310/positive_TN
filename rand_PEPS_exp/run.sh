#!/bin/bash
#SBATCH --time=10-00:00:00
#SBATCH -J positive_TN_PEPS
##SBATCH -o job.out
##SBATCH -e job.err
#SBATCH --mail-user=jchen9@caltech.edu
#SBATCH --mail-type=all
#SBATCH --partition=serial,parallel
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH -x pauling[001-002]
#SBATCH --mem=200G
#SBATCH --ntasks=3
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
# mkdir /scratch/local/Chriscrosser/

python -u rand_PEPS_exp.py 3 2 [2,3,4,5,6,7,8,9,10,11,12] 50