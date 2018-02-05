#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks={0}
#SBATCH --exclude=n[26-28]
#SBATCH -o slurm_logs/slurm-%j.out

DATADIR={1}
mkdir -p $DATADIR/logs

for j in `seq 1 {0}`;
do
    ./vep-fe-rev-08a-fs id=$j sample save_warmup=1 num_warmup=1000 num_samples=5000 algorithm=hmc engine=nuts max_depth={2} data file=$DATADIR/fit_data_ns{3}.R output file=$DATADIR/fitout_ns{3}_md{2}_chain$j.csv &> $DATADIR/logs/ns{3}_md{2}_chain$j.log &
done
wait    
