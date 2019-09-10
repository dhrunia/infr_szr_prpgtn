#!/bin/bash
#SBATCH --ntasks=4
#SBATCH -o slurm_logs/slurm-%j.out
#SBATCH --time=48:00:00
#SBATCH --partition rhu

DATA_PATH=${1};
INIT_PATH=${2};
RES_DIR=${3};
STAN_FNAME=${4};
SAMPLING_ITERS=${5};
WARMUP_ITERS=${6};
NCHAINS=${7}; # $SLURM_NTASKS
DELTA=${8};
MAX_DEPTH=${9};
JITTER=${10};
FNAME_SUFFIX=${11};
LOG_DIR=${12};

for i in $(seq 1 ${SLURM_NTASKS});
do
    ./${STAN_FNAME} id=$((100*${i})) sample num_samples=${SAMPLING_ITERS} \
      num_warmup=${WARMUP_ITERS} save_warmup=1 adapt delta=${DELTA} algorithm=hmc \
      engine=nuts max_depth=${MAX_DEPTH} stepsize_jitter=${JITTER} \
      data file=${DATA_PATH} init=${INIT_PATH} random seed=$((987*${i})) \
      output file=${RES_DIR}/samples_${FNAME_SUFFIX}_md${MAX_DEPTH}_delta${DELTA}_jitter${JITTER}_chain${i}.csv refresh=10 \
      &> ${LOG_DIR}/${FNAME_SUFFIX}_md${MAX_DEPTH}_delta${DELTA}_jitter${JITTER}_chain${i}.log &
done
wait
