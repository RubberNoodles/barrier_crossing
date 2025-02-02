#!/bin/sh

while getopts "clt" flag; do
    case "$flag" in
        c) rm -rf results/*
        ;;
        l) rm -rf logs/*
        ;;
        t) train="true"
        ;;
        \?)
        echo "invalid argument"
        ;;
    esac
done

source ~/.bashrc
imp
source ../env/bin/activate

# barrier_heights=("5" "10" "15" "20" "25" "30" "35" "40") # Jan 21
# landscapes=("triple_well") # Jan 21
# times=(0.00003 0.0001) # Jan 21
# barrier_heights=('2.5' "10" "20" "40")
# barrier_heights=('2.5' "10" "20" "30" "40")
# barrier_heights=("5" "10" "15" "20" "25" "30" "35" "40") # Jan 22
# landscapes=("double_well" "asymmetric") # Jan 22
# times=(0.00003 0.0001) # Jan 22
# barrier_heights=("5" "10" "15" "20" "25" "30" "35" "40") # Jan 23
# landscapes=("triple_well" "double_well" "asymmetric") # Jan 23
# times=(0.0003) # Jan 23
barrier_heights=("5" "10" "15" "20" "25" "30" "35" "40") # Jan 23 v2
landscapes=("triple_well" "double_well" "asymmetric") # Jan 23 v2
times=(0.0003 0.0001 0.00003) # Jan 23 v2

iterate_over_params() {


    for landscape in "${landscapes[@]}"
    do
    for barrier_height in "${barrier_heights[@]}"
    do

            jobid=$(sbatch training.sbatch $landscape $barrier_height $t | awk '{print $NF}')
            # jobid=""

            echo "SLURM $jobid running: python3 train.py --landscape_name $landscape --param_suffix $barrier_height --end_time $t"
            echo "sbatch training.sbatch $landscape $barrier_height $t"
            sleep 1

            jobids+=($jobid)
    done
    done

}

for t in "${times[@]}" 
do  
    jobids=()
    export ITERATIVE_RESULT_DIR="results/$t"
    file_dir="$ITERATIVE_RESULT_DIR/files"

    if [[ "$train" == "true" ]]; then
        iterate_over_params
        jobid_string=""
        for jobid in "${jobids[@]}"
        do
            jobid_string+="$jobid,"
        done
        jobid_string="${jobid_string%?}"

        echo "sbatch -d afterok:$jobid_string -t 0-08:00 --mem-per-cpu=16000 plotting.sbatch $file_dir"
        sbatch -d "afterok:$jobid_string" -t 0-08:00 --mem-per-cpu=16000 plotting.sbatch $file_dir
        # sbatch plotting.sbatch $file_dir
    else
        echo "sbatch -t 0-08:00 --mem-per-cpu=16000 plotting.sbatch $file_dir"
        sbatch -t 0-08:00 --mem-per-cpu=16000 plotting.sbatch $file_dir
    fi


    
done
echo "All slurm commands run, exiting."
# Responsibiltiy of slurm to wait for all train to finish before running the reconstructions.