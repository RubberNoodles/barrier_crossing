#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH -t 0-04:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p seas_gpu         # Partition to submit to
#SBATCH --mem-per-cpu=16000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o out_%A_eps_%a.out  # Requires ./log directory
#SBATCH -e err_%A_eps_%a.err  # File to which STDERR will be written, %j inserts jobid
module load python/3.10.9-fasrc01
module load cuda/12.2.0-fasrc01
module load cudnn/8.9.2.26_cuda12-fasrc01
pip3 install -e "../../"
pip3 install -r "./../../requirements.txt"
pip3 install --upgrade "jax[cuda12_pip]==0.4.16" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
#pip3 install --upgrade "https://storage.googleapis.com/jax-releases/cuda12/jaxlib-0.4.13+cuda12.cudnn89-cp310-cp310-manylinux2014_x86_64.whl" # old

python3 reconstruct.py "$1" "$2"