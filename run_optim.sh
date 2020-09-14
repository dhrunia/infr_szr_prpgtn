#!/bin/bash
#SBATCH --ntasks=2
#SBATCH -o slurm_logs/slurm-%j.out
#SBATCH --time=24:00:00
#SBATCH --partition all

DATA_PATH=${1};
INIT_PATH=${2};
RES_DIR=${3};
LOG_DIR=${4};
STAN_FNAME=${5};
FNAME_SUFFIX=${6};
ITERS=${7};

for i in {1..2};
do
./${STAN_FNAME} optimize algorithm=lbfgs iter=${ITERS} save_iterations=0  \
data file=${DATA_PATH} \
init=${INIT_PATH} \
output file=${RES_DIR}/samples_${FNAME_SUFFIX}_run${i}.csv refresh=10 \
&> ${LOG_DIR}/snsrfit_ode_${FNAME_SUFFIX}_run${i}.log &
done

wait
