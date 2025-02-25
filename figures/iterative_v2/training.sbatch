#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -t 0-16:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p seas_gpu       # Partition to submit to
#SBATCH --mem-per-cpu=16000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o logs/train_out_%A.out  # Requires ./log directory
#SBATCH -e logs/train_err_%A.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --export=ALL

module load python/3.10.9-fasrc01
module load cuda/12.2.0-fasrc01
module load cudnn/8.9.2.26_cuda12-fasrc01

source ./../env/bin/activate


python3 train.py --landscape_name "$1" --param_suffix "$2" --end_time "$3"