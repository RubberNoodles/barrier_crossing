#!/bin/bash

for K_S in 0.03 0.1 0.4 1.0; do
#for K_S in 0.4; do

for END_TIME in 0.0001 0.0003 0.00003; do
#for END_TIME in 0.0001; do

#for BATCH_SIZE in 3000 30000; do
for BATCH_SIZE in 3000; do

echo "${K_S}, ${END_TIME}, ${BATCH_SIZE}"

sbatch --job-name=we_opt_k${K_S}_t${END_TIME} \
we_opt.sbatch  "$1" "$2" "${K_S}" "${END_TIME}" "${BATCH_SIZE}"


#
sleep 1 # pause to be kind to the scheduler
done
done
done
#python3 reconstruct.py "$1" "$2"