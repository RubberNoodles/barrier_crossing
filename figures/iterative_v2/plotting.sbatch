#!/bin/bash
#SBATCH -o logs/control_out_%A.out  # Requires ./log directory
#SBATCH -e logs/control_err_%A.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --export=ALL

echo "Running sbatch: $1"
module load python/3.10.9-fasrc01
module load cuda/12.2.0-fasrc01
module load cudnn/8.9.2.26_cuda12-fasrc01

source ../env/bin/activate

python3 plot_models.py $1