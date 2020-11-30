#!/bin/sh

OUTPUT="/cs/usr/galhar/research/BrainNN/records"

cd ../

source thesis_sed/bin/activate

cd src


for i in {1..2};do
  sbatch -W -c1 --mem=500m --time=15 python3 single_run.py $i
done
wait
python3 collect.py