#!/bin/bash
#SBATCH --ntasks=4
#SBATCH -o slurm_logs/slurm-%j.out

STAN_FNAME=$1;
SAMPLING_ITERS=$2;
WARMUP_ITERS=$3;
RES_DIR=$4;
NCHAINS=$5;
MAX_DEPTH=$6
DELTA=$7;
DATA_FNAME=$8;
HYP_TYPE=$9;
PRIOR_TYPE=${10};
for i in `seq 1 $NCHAINS`;
do
    ./${STAN_FNAME} id=$((100*${i})) sample num_samples=${SAMPLING_ITERS} \
      num_warmup=${WARMUP_ITERS} save_warmup=1 adapt delta=0.8 algorithm=hmc \
      engine=nuts max_depth=${MAX_DEPTH} data file=${RES_DIR}/Rfiles/${DATA_FNAME} \
      init=${RES_DIR}/Rfiles/param_init.R random seed=$((51*${i})) \
      output file=${RES_DIR}/${PRIOR_TYPE}_${HYP_TYPE}hyp_samples_md${MAX_DEPTH}_chain${i}.csv \
      refresh=10 &> ${RES_DIR}/logs/${PRIOR_TYPE}_${HYP_TYPE}hyp_md${MAX_DEPTH}_chain${i}.log &
done
wait
