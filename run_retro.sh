#!/bin/bash
for PAT_ID in $(ls -d datasets/data_jd/id* | cut -d "/" -f3);
do
    DATA_DIR="$PWD/datasets/data_jd"
    RES_DIR="$PWD/results/exp95"
    LOGS_DIR="${RES_DIR}/${PAT_ID}/logs"
    mkdir -p $RES_DIR;
    mkdir -p $LOGS_DIR;
    for DATA_PATH in $(ls -d ${DATA_DIR}/${PAT_ID}/fit/data_*.npz);
    do
        SZR_NAME=$(echo "$(basename $DATA_PATH)" | ( read s; echo ${s:5:-4}; ));
        sbatch run_map_retro.sh $PAT_ID $SZR_NAME $DATA_DIR $RES_DIR $LOGS_DIR;
        sleep 10;
    done
done