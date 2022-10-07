#!/bin/bash

DATA_DIR="$PWD/datasets/syn_data/id004_bj/LMAX_lc_32"
RES_DIR="$PWD/results/exp88"
LOGS_DIR="$PWD/logs/exp88"
N_LAT=128
L_MAX=32
mkdir -p $RES_DIR;
mkdir -p $LOGS_DIR;
for L_MAX_PARAMS in $(seq 10 75);
do
    sbatch run_map.sh $DATA_DIR $RES_DIR $LOGS_DIR $N_LAT $L_MAX $L_MAX_PARAMS;
    sleep 10;
done
