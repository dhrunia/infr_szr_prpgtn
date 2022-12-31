#!/bin/bash

DATA_DIR="$PWD/datasets/syn_data/id022_te/LMAX_lc_32"
RES_DIR="$PWD/results/exp109"
LOGS_DIR="${RES_DIR}/logs"
N_LAT=128
L_MAX=32
L_MAX_PARAMS=16
mkdir -p $RES_DIR;
mkdir -p $LOGS_DIR;
for SNR in $(seq 1 0.5 50);
do
    sbatch run_map.sh $DATA_DIR $RES_DIR $LOGS_DIR $N_LAT $L_MAX $L_MAX_PARAMS $SNR;
    sleep 10;
done
