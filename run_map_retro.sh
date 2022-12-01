#!/bin/bash -l
#SBATCH --account="ich042"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anirudh-nihalani.vattikonda@univ-amu.fr
#SBATCH --time=05:00:00
#SBATCH --nodes=10
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --output=logs/slurm/%j.out

PAT_ID=$1
SZR_NAME=$2
DATA_DIR=$3
RES_DIR=$4
LOGS_DIR=$5
module load daint-gpu

for i in $(seq 1 10);
do
    srun --nodes=1 --ntasks-per-node=1 --exact python map_2dep_nmm_retro_job.py ${PAT_ID} ${SZR_NAME} ${DATA_DIR} ${RES_DIR} $i > "${LOGS_DIR}/map_${SZR_NAME}_run${i}.log" 2>&1 &
done
wait
