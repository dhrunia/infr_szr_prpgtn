#!/bin/bash

DATA_DIR="$PWD/datasets/syn_data/id004_bj/LMAX_lc_32"
RES_DIR="$PWD/results/exp87.1"
LOGS_DIR="$PWD/logs/exp87"
for N_LAT in $(seq 64 128);
do
    sbatch run_map.sh $DATA_DIR $RES_DIR $LOGS_DIR $N_LAT 
done
