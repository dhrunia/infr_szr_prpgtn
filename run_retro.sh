#!/bin/bash
N_LAT=128
L_MAX=32
L_MAX_PARAMS=16

for PAT_ID in $(ls -d datasets/data_jd/id* | cut -d "/" -f3);
do
    DATA_DIR="$PWD/datasets/data_jd/${PAT_ID}"
    RES_DIR="$PWD/results/exp90/${PAT_ID}"
    LOGS_DIR="${RES_DIR}/logs"
    mkdir -p $RES_DIR;
    mkdir -p $LOGS_DIR;
    for DATA_PATH in $(ls -d ${DATA_DIR}/fit/data_*.npz);
    do
        SZR_NAME=$(echo "$(basename $DATA_PATH)" | ( read s; echo ${s:5:-4}; ));
        sbatch run_map_retro.sh $PAT_ID $SZR_NAME $DATA_DIR $RES_DIR $N_LAT $L_MAX $L_MAX_PARAMS $LOGS_DIR;
        sleep 10;
    done
done