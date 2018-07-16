#!/bin/bash
#SBATCH --ntasks=4
#SBATCH -t 24:00:00
#SBATCH -o slurm_logs/slurm-%j.out

STAN_EXEC_FNAME=${1}
DATA_FILE=${2}
OUTPUT_FILE=${3}
LOG_FILE=${4}

for j in `seq 1 4`;
do
    ./${STAN_EXEC_FNAME} variational iter=1000000 tol_rel_obj=0.01 output_samples=1000 \
		       data file=${DATA_FILE} output file=${OUTPUT_FILE}_chain${j}.csv \
		       &> ${LOG_FILE}_chain${j}.log &
done
wait
