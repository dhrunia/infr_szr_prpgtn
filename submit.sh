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

site=tvb
if which bsub &> /dev/null; then site=juron; fi

case "$site" in
    tvb)
        srun -J $work -c 8 $here/sample.sh $model data.R $here/site/$site $args &> sample.out & ;;
    juron)
        bsub -J $work -o sample.out $here/sample.sh $model data.R $here/site/$site $args ;;
esac
