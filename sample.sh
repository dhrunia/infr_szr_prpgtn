#!/bin/bash

# setup
model=$1
data=${2:-"$model.R"}
maxdepth=${3:-"7"}
if ! which stanc &> /dev/null; then . /gpfs/homeb/pcp0/pcp0025/stan/env2; fi
set -x

# compile
h=`pwd`; cd $CMDSTAN; make CC=g++ $h/$model; cd $h

# run
for i in `seq 8`; do
    ./$model id=$i \
        sample \
            save_warmup=1 num_warmup=200 num_samples=200 \
            algorithm=hmc engine=nuts max_depth=$maxdepth \
        data file=$data \
        output refresh=1 file=$i.csv &> $i.out &
done
wait

# summarize
stansummary --csv_file=summary.csv *.csv &> summary.txt
