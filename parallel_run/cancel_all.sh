squeue -u $USER | grep 3 | awk '{print$1}' | xargs -n 1 scancel 
