#!/bin/sh


for i in {1..2};do
  sbatch -W -c1 --mem=500m --time=15 single_run $i
done
wait

cd ../
source env/bin/activate
cd parallel_run

python3 collect.py