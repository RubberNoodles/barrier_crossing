#!/bin/bash
# reconstruct.sbatch
#
#SBATCH --gres=gpu:1
#SBATCH -t 0-02:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p seas_gpu         # Partition to submit to
#SBATCH --mem-per-cpu=8000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o log_out_%A.out  
#SBATCH -e log_err_%A.err 
module load python/3.10.9-fasrc01
module load cuda/12.2.0-fasrc01
module load cudnn/8.9.2.26_cuda12-fasrc01

echo "Looking for lock..."
n=0

while ! ln -s . ../lock; do 
        sleep 15
        n=$((n+1))
done
echo "Lock Found!"
trap "{ echo Process terminated prematurely.; rm ../lock }" ERR


source ./../env/bin/activate

pip3 install -e "../../"

echo "$1 $2 $3 $4"
#

echo "Installation complete. Releasing Lock. Running code..."

mv "../lock" "../deleteme" && rm "../deleteme"

python3 reconstruct.py --landscape_name "$1" --param_suffix "$2" --end_time "$3" --k_s "$4" --batch_size "$5"
deactivate