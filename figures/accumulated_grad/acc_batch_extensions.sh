#!/bin/bash
#SBATCH --gres=gpu:8
#SBATCH -t 2-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p seas_gpu         # Partition to submit to
#SBATCH --mem-per-cpu=32000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o out_%A_eps_%a.out  # Requires ./log directory
#SBATCH -e err_%A_eps_%a.err  # File to which STDERR will be written, %j inserts jobid
module load python/3.10.9-fasrc01
module load cuda/12.0.1-fasrc01
module load cudnn/8.8.0.121_cuda12-fasrc01
pip3 install -e ../../
pip3 install -r ./../../requirements.txt
pip3 install "jax[cuda12_cudnn88]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

num_extension=$1

if [${SLURM_ARRAY_TASK_ID} -eq 1 ]; then
  if [ -d "./data" ]
  then 
    rm -rf "./data"
  fi
fi

mkdir "data"

for (( ext_num=1; ext_num<=$num_extension; ext_num++ ))
do
  mkdir "data/ext_$ext_num"
done

python3 acc_batch_extensions.py ${SLURM_ARRAY_TASK_ID}

# if [ ${SLURM_ARRAY_TASK_ID} -eq $num_extension ]; then
# 	git add .
# 	git commit -m "Auto-commit."
# 	git push
# fi