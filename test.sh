#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -o tmp/slurm-%j.out

./vep-forwardsim-2Depileptor sample algorithm=fixed_param num_samples=1 num_warmup=0 data \
file=tmp/sim_data.R output file=tmp/sim_out.csv &

wait

