#!/bin/bash
work=$1
model=$2
data=$3
shift 3
args="$@"
set -ue
mkdir -p $work
cp $model.stan $work/
cp $data $work/data.R
here=`pwd`
cd $work
bsub -J $work -o sample.out $here/sample.sh $model data.R $args
