SQUEUE_OUTPUT=$(squeue | grep galhar | sort -R)
echo "running jobs:"
echo "$SQUEUE_OUTPUT" | wc -l | cat
ID=$(echo ${SQUEUE_OUTPUT} | head -n1 | awk '{print $1;}')
SLURM_FILE="slurm-${ID}.out"
echo "info of the ${ID} job:"
cat ${SLURM_FILE}
