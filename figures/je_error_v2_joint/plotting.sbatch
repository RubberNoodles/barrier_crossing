#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH -t 0-8:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p gpu_test       # Partition to submit to
#SBATCH --mem-per-cpu=16000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o logs/plotting_out_%A.out  # Requires ./log directory
#SBATCH -e logs/plotting_err_%A.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --export=ALL

echo "Running sbatch: $1"
module load python/3.10.9-fasrc01
module load cuda/12.2.0-fasrc01
module load cudnn/8.9.2.26_cuda12-fasrc01

source ../env/bin/activate
echo "python3 plot_models.py $1 $2"
python3 plot_models.py $1 $2
echo "Plotting completed. Exiting..."