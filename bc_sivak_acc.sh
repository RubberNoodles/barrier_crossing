#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH -t 0-08:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p gpu_test   # Partition to submit to
#SBATCH --mem-per-cpu=64000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o logs/out_%A_eps_%a.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e logs/err_%A_eps_%a.err  # File to which STDERR will be written, %j inserts jobid
module load python/3.8.5-fasrc01
module load cuda/11.4.2-fasrc01
module load cudnn/8.2.2.26_cuda11.4-fasrc01
pip3 install -r requirements.txt
python3 new_sivak.py


