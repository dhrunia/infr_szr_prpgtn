#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks={1}
#SBATCH -o slurm_logs/slurm-%j.out

DATADIR=/home/anirudhnihalani/vep.stan/results/exp1
mkdir -p $DATADIR/logs

for j in `seq 1 {1}`;
do
    ./vep-fe-rev-08a-fs id=$j sample save_warmup=1 num_warmup=200 num_samples=1000 algorithm=hmc engine=nuts max_depth={2} data file=$DATADIR/fit_data_ns{0}.R output file=$DATADIR/fitout_ns{0}_md{2}_chain$j.csv &> $DATADIR/logs/ns{0}_md{2}_chain$j.log &
    wait
done
