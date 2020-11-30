#!/bin/sh
cd ../
source env/bin/activate
cd parallel_run

python3 single_run.py "${@:1}"