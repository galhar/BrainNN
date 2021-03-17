find . -name 'slurm*' -exec rm {} \;
find . -name 'EvaluationHookData*' -exec rm {} \;
find . -name 'collect_py_tmp*' -exec rm {} \;
find . -name 'NetSavedByHook*' -exec rm {} \;
rm -f tmp/*
