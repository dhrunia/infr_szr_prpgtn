#!/bin/bash -l
#SBATCH --account="ich042"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anirudh-nihalani.vattikonda@univ-amu.fr
#SBATCH --time=10:00:00
#SBATCH --nodes=10
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --output=logs/slurm/%j.out

DATA_DIR=$1
RES_DIR=$2
LOGS_DIR=$3
N_LAT=$4
L_MAX=$5
L_MAX_PARAMS=$6
SNR=$7
module load daint-gpu

for i in $(seq 1 10);
do
    srun --nodes=1 --ntasks-per-node=1 --exact python map_2dep_nf_job.py $DATA_DIR $RES_DIR $N_LAT $L_MAX $L_MAX_PARAMS $SNR $i > "${LOGS_DIR}/map_N_LAT${N_LAT}_L_MAX${L_MAX}_L_MAX_PARAMS${L_MAX_PARAMS}_SNR${SNR}_run${i}.log" 2>&1 &
done
wait
