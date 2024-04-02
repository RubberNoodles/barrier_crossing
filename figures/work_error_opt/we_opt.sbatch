#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH -t 0-04:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p seas_gpu         # Partition to submit to
#SBATCH --mem-per-cpu=8000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o logs/out_%A_eps_%a.out  # Requires ./log directory
#SBATCH -e logs/err_%A_eps_%a.err  # File to which STDERR will be written, %j inserts jobid

echo "Looking for lock..."
n=0

while ! ln -s . ../lock; do 
        sleep 15
        n=$((n+1))
done
echo "Lock Found!"
trap "{ echo Process terminated prematurely.; rm ../lock }" ERR

module load python/3.10.9-fasrc01
module load cuda/12.2.0-fasrc01
module load cudnn/8.9.2.26_cuda12-fasrc01

source ./../env/bin/activate
pip3 install -e "../../"

echo "Installation complete. Releasing Lock. Running code..."

mv "../lock" "../deleteme" && rm "../deleteme"

python3 we_opt.py --landscape_name "$1" --param_suffix "$2" --k_s "$3" --end_time "$4"

deactivate
# python3 plotting.py