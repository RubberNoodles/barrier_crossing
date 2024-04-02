#!/bin/bash

#for K_S in 0.03 0.1 0.4 1.0; do
for K_S in 0.4; do

for END_TIME in 0.0001 0.0003 0.00003; do
#for END_TIME in 0.0001; do

for BATCH_SIZE in 3000; do
#for BATCH_SIZE in 3000 10000; do

echo "${K_S}, ${END_TIME}, ${BATCH_SIZE}"

python3 free_energies.py --landscape_name "Double Well 10kT Barrier Brownian" --param_suffix 10kt --k_s "${K_S}" --end_time "${END_TIME}" --batch_size "${BATCH_SIZE}"



sleep 1 # pause to be kind to the scheduler
done
done
done
#python3 reconstruct.py "$1" "$2"