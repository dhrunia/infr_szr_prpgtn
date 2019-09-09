#!/bin/bash

#SBATCH --array=1-200
#SBATCH --partition=lowprio

ibt=$SLURM_ARRAY_TASK_ID

STAN_FNAME=$1
RESULTS_DIR=$2
INPUT_RFILE=$3

./${STAN_FNAME} optimize algorithm=lbfgs tol_param=1e-4 iter=20000 save_iterations=0  \
data file=${RESULTS_DIR}/RfilesBT/${INPUT_RFILE}_$ibt.R \
init=${RESULTS_DIR}/RfilesBT/param_init.R \
output file=${RESULTS_DIR}/OptimalBT/samples_${INPUT_RFILE}_${ibt}.csv refresh=10 \
&> ${RESULTS_DIR}/logs/snsrfit_ode_${INPUT_RFILE}_${ibt}.log

