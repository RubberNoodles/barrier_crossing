#!/bin/bash
#SBATCH --gres=gpu:4 #how many GPUs 
#SBATCH -t 0-01:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p gpu_test   # Partition to submit to
#SBATCH --mem-per-cpu=64000   # Memory pool for all cores in MB
#SBATCH -o test_output.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e test_errors.err  # File to which STDERR will be written, %j inserts jobid
module load python/3.8.5-fasrc01
module load cuda/11.4.2-fasrc01
module load cudnn/8.2.2.26_cuda11.4-fasrc01
pip3 install -r requirements.txt

python3 Optimize_new_grad.py