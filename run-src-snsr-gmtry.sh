#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8

DATADIR=/home/anirudhnihalani/results/exp1
mkdir -p $DATADIR/logs

for i in `seq 1 8`;
do
    ./vep-fe-rev-08a-fs sample save_warmup=1 num_warmup=20 num_samples=10 \
        algorithm=hmc engine=nuts max_depth=4 \
        data file=$DATADIR/fit_data_ns$1.R output refresh=1 file=$DATADIR/fitout_chain$1.csv &> $DATADIR/logs/ns$1_chain$i.out &
done
