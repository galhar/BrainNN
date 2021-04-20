#!/bin/sh
# Gal Harari

for i in $(eval echo {$1..$2}) ;do
  sbatch -c1 --mem=500m --time=300 --killable single_run.sh $i
done
#wait

#cd ../
#source env/bin/activate
#cd parallel_run

#python3 collect.py
