#!/bin/sh

OUTPUT="/cs/usr/galhar/research/BrainNN/records"

cd ../

source env/bin/activate.csh

cd parallel_run


for i in {1..2};do
  sbatch -W -c1 --mem=500m --time=15 python3 single_run.py $i
done
wait
python3 collect.py