#!/bin/bash
#SBATCH --time=1-00:00:00
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
for u in 8
do
    :loop
    mail -s 'positive_TN' jchen9@caltech.edu < sbatch_buffer.txt
    ping localhost -n 61 > nul
    goto loop
done