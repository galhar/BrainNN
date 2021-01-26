#!/bin/sh


for i in $(eval echo {$1..$2}) ;do
  sbatch -c1 --mem=500m --time=20 single_run.sh $i
done
#wait

#cd ../
#source env/bin/activate
#cd parallel_run

#python3 collect.py
