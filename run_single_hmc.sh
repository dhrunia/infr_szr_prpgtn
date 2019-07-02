#!/bin/bash
#SBATCH --ntasks=1
#SBATCH -o slurm_logs/slurm-%j.out
#SBATCH --time=4-12:00:00
#SBATCH --partition rhu

DATA_PATH=${1};
INIT_PATH=${2};
RES_DIR=${3};
STAN_FNAME=${4};
SAMPLING_ITERS=${5};
WARMUP_ITERS=${6};
DELTA=${7};
MAX_DEPTH=${8};
JITTER=${9};
FNAME_SUFFIX=${10};
LOG_DIR=${11};
CHAIN_NO=${12}

./${STAN_FNAME} id=$((100*${CHAIN_NO})) sample num_samples=${SAMPLING_ITERS} \
  num_warmup=${WARMUP_ITERS} save_warmup=1 adapt delta=${DELTA} algorithm=hmc \
  engine=nuts max_depth=${MAX_DEPTH} stepsize_jitter=${JITTER} \
  data file=${DATA_PATH} init=${INIT_PATH} random seed=$((987*${CHAIN_NO})) \
  output file=${RES_DIR}/samples_${FNAME_SUFFIX}_md${MAX_DEPTH}_delta${DELTA}_jitter${JITTER}_chain${CHAIN_NO}.csv refresh=10 \
  &> ${LOG_DIR}/${FNAME_SUFFIX}_md${MAX_DEPTH}_delta${DELTA}_jitter${JITTER}_chain${CHAIN_NO}.log &

wait
