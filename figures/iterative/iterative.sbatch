#!/bin/bash
#SBATCH --gres=gpu:8
#SBATCH -t 1-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p seas_gpu       # Partition to submit to
#SBATCH --mem-per-cpu=16000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o out_%A_eps_%a.out  # Requires ./log directory
#SBATCH -e err_%A_eps_%a.err  # File to which STDERR will be written, %j inserts jobid

# Array
# SBATCH --array=1-8

# Work in progress for the final barrier that we will be using.

echo "Looking for lock..."
n=0

while ! ln -s . lock; do 
        sleep 5
        n=$((n+1))
done

echo "Lock Found!"
trap "{ echo Process terminated prematurely.; rm lock }" ERR

# echo "Task $SLURM_ARRAY_TASK_ID is starting. Modules & Dependencies Loading..."
echo "Modules & Dependencies Loading..."

module load python/3.10.9-fasrc01
module load cuda/12.2.0-fasrc01
module load cudnn/8.9.2.26_cuda12-fasrc01

source ./../env/bin/activate
pip3 install -e "../../"
# num_extension=$1

# if [ ${SLURM_ARRAY_TASK_ID} -eq 1 ]; then
#   if [ -d "./data" ]
#   then 
#     rm -rf "./data"
#   fi

#   mkdir "data"
#   for (( ext_num=1; ext_num<=$num_extension; ext_num++ ))
#     do
#       mkdir "data/barrier_$ext_num"
#     done
# fi

echo "Installation complete. Releasing Lock. Running code..."

mv "lock" "deleteme" && rm "deleteme"

#python3 "./iterative.py" ${SLURM_ARRAY_TASK_ID}

python3 iterative.py "$1" "$2" "$3" "$4" "$5"