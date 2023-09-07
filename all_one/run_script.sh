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
#SBATCH --ntasks=3
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
# mkdir /scratch/local/Chriscrosser/

srun --exclusive --ntasks=1 python -u all_one_script.py 2 [10,15,20] 10 50 > 2_[10,15,20]_10_50.txt &
srun --exclusive --ntasks=1 python -u all_one_script.py 3 [10,15,20] 10 50 > 3_[10,15,20]_10_50.txt &
srun --exclusive --ntasks=1 python -u all_one_script.py 4 [10,15,20] 10 50 > 4_[10,15,20]_10_50.txt &
wait