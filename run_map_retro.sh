#!/bin/bash -l
#SBATCH --account="ich042"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anirudh-nihalani.vattikonda@univ-amu.fr
#SBATCH --time=20:00:00
#SBATCH --nodes=10
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --output=logs/slurm/%j.out

PAT_ID=$1
SZR_NAME=$2
DATA_DIR=$3
RES_DIR=$4
N_LAT=$5
L_MAX=$6
L_MAX_PARAMS=$7
LOGS_DIR=$8
module load daint-gpu

for i in $(seq 1 10);
do
    srun --nodes=1 --ntasks-per-node=1 --exact python map_2dep_nf_retro_job.py ${PAT_ID} ${SZR_NAME} ${DATA_DIR} ${RES_DIR} $N_LAT $L_MAX $L_MAX_PARAMS $i > "${LOGS_DIR}/map_${SZR_NAME}_N_LAT${N_LAT}_L_MAX${L_MAX}_L_MAX_PARAMS${L_MAX_PARAMS}_run${i}.log" 2>&1 &
done
wait
