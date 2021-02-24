#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00
#SBATCH -o slurm_logs/slurm-%j.out

DATA_PATH=${1};
INIT_PATH=${2};
RES_PATH=${3};
LOG_PATH=${4};

./vep-snsrfit-ode-rk4 optimize algorithm=lbfgs iter=20000 save_iterations=0  \
data file=${DATA_PATH} \
init=${INIT_PATH} \
output file=$RES_PATH refresh=10 \
&> $LOG_PATH &

wait
