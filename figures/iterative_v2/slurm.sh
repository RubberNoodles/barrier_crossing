#!/bin/sh

while getopts "cl" flag; do
    case "$flag" in
        c) rm -rf results/*
        ;;
        l) rm -rf logs/*
        ;;
        \?)
        echo "invalid argument"
        ;;
    esac
done

source ~/.bashrc
imp
source ../env/bin/activate

barrier_heights=('2.5' "10" "20" "30" "40")
landscapes=("triple_well" "double_well" "asymmetric")
times=(0.00001 0.00005 0.0001 0.0002 0.0004)
for t in "${times[@]}" do

    jobids=()

    export ITERATIVE_RESULT_DIR="results/$t"

    for landscape in "${landscapes[@]}"
    do
    for barrier_height in "${barrier_heights[@]}"
    do

            jobid=$(sbatch training.sbatch $landscape $barrier_height $t | awk '{print $NF}')

            echo "SLURM $jobid running: python3 train.py --landscape_name $landscape --param_suffix $barrier_height --end_time $t"
            echo "sbatch training.sbatch $landscape $barrier_height $t"

            jobids+=($jobid)
    done
    done

    jobid_string=""

    for jobid in "${jobids[@]}"
    do
        jobid_string+="$jobid,"
    done
    jobid_string="${jobid_string%?}"

    echo $jobid_string
    file_dir="$ITERATIVE_RESULT_DIR/files"

    echo "sbatch -d afterok:$jobid_string -t 0-08:00 --mem-per-cpu=16000 plotting.sbatch $file_dir"
    sbatch -d "afterok:$jobid_string" -t 0-08:00 --mem-per-cpu=16000 plotting.sbatch $file_dir

echo "All slurm commands run, exiting."
# Responsibiltiy of slurm to wait for all train to finish before running the reconstructions.