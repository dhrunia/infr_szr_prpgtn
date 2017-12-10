#!/bin/bash

# setup
model=$1
data=${2:-"$model.R"}
site=${3:-"tvb"}

set -x
if ! which stanc &> /dev/null; then . $site; fi

# compile
h=`pwd`; cd $CMDSTAN; make CC=g++ $h/$model; cd $h

# run
for i in `seq 8`; do
    ./$model id=$i \
        sample \
            save_warmup=1 num_warmup=200 num_samples=200 \
            algorithm=hmc engine=nuts max_depth=14 \
        data file=$data \
        output refresh=1 file=$i.csv &
done
wait

# summarize
stansummary --csv_file=summary.csv *.csv &> summary.txt
