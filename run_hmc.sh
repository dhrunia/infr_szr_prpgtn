#!/bin/bash
#SBATCH --ntasks=4
#SBATCH -t 24:00:00
#SBATCH -o slurm_logs/slurm-%j.out

STAN_EXEC_FNAME=${1}
DATA_FILE=${2}
OUTPUT_FILE=${3}
LOG_FILE=${4}
NWARMUP=${5}
NSAMPLING=${6}

for j in `seq 1 4`;
do

    ./${STAN_EXEC_FNAME} sample save_warmup=1 num_warmup=${NWARMUP} num_samples=${NSAMPLING} \
      adapt delta=0.8 algorithm=hmc engine=nuts max_depth=10 stepsize_jitter=1.0 \
      data file=${DATA_FILE} output file=${OUTPUT_FILE}_chain${j}.csv \
	&> ${LOG_FILE}_chain${j}.log &
done
wait
