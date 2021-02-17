#!/bin/bash
#SBATCH --account=ich001m
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=30
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --constraint=mc

DATA_PATH="${PWD}/..";
RES_DIR="${PWD}/../samples";
LOG_DIR="${PWD}/../logs";
STAN_FNAME="vep-snsrfit-ode-rk4";
ITERS=20000;

for snr in $(seq 0.1 0.1 5.0);
do
    srun --ntasks=1 --nodes=1 --mem-per-cpu=4G --cpus-per-task=1 --exclusive \
    ./${STAN_FNAME} optimize algorithm=lbfgs iter=${ITERS} save_iterations=0  \
    data file="${DATA_PATH}/Rfiles/fit_data_snsrfit_ode_snr${snr}.R" \
    init="${DATA_PATH}/Rfiles/param_init.R" \
    output file="${RES_DIR}/samples_snr${snr}.csv" refresh=10 \
    &> "${LOG_DIR}/snsrfit_ode_snr${snr}.log" &
done
wait