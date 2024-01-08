#!/bin/bash

for K_S in 0.03 0.1 0.4 1.0; do
for END_TIME in 0.001 0.0003 0.0001; do

# for K_S in 1.0; do
# for END_TIME in 0.0003 0.0001; do

echo "${K_S}, ${END_TIME}"

sbatch --job-name=reconstruct_k${K_S}_t${END_TIME} \
reconstruct.sbatch "$1" "$2" "${K_S}" "${END_TIME}"
#
sleep 1 # pause to be kind to the scheduler
done
done

#python3 reconstruct.py "$1" "$2"