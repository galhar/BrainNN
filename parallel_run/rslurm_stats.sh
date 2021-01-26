SQUEUE_OUTPUT=$(squeue | grep $USER | sort -R)
echo "Running jobs:"
echo "$SQUEUE_OUTPUT" | wc -l | cat
ID=$(echo ${SQUEUE_OUTPUT} | head -n1 | awk '{print $1;}')
SLURM_FILE="slurm-${ID}.out"
echo "Info of the ${ID} job:"
echo "Time running -"
squeue | grep ${ID} | awk '{ print $6 }'
echo "File content -"
cat ${SLURM_FILE}
