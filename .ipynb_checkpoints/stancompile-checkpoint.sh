#!/bin/bash
curr_dir=$(pwd)
echo $curr_dir
cd /home/hfw/mysoft/cmdstan-2.18.0
make $curr_dir/$1
cd $curr_dir
