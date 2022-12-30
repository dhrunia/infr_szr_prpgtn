#!/bin/bash

DATA_DIR="$PWD/datasets/syn_data/id022_te/LMAX_lc_32"
RES_DIR="$PWD/results/exp104"
LOGS_DIR="${RES_DIR}/logs"
L_MAX=32
L_MAX_PARAMS=16
SNR=50
mkdir -p $RES_DIR;
mkdir -p $LOGS_DIR;
for N_LAT in $(seq 64 128);
do
    sbatch run_map.sh $DATA_DIR $RES_DIR $LOGS_DIR $N_LAT $L_MAX $L_MAX_PARAMS $SNR;
    sleep 10;
done
