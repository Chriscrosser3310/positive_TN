#!/bin/bash
#SBATCH --time=10-00:00:00
#SBATCH -J positive_TN
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

python -u all_one_script.py 4 [10,15,20] 10 50 > 4_[10,15,20]_10_50.txt