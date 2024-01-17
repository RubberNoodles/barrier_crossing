#!/bin/bash

declare -a figures=( "asymetric_double_well" 
                     "double_well_2_5kt_barrier"
                     "double_well_10kt_barrier"
                     "double_well_25kt_barrier"
                     "triple_well"
)

declare -a figures=( "asymetric_double_well_brownian" 
                     "double_well_2_5kt_barrier_brownian"
                     "double_well_10kt_barrier_brownian"
                     "double_well_25kt_barrier_brownian"
                     "triple_well_brownian"
)


for figure in "${figures[@]}"; do
  # mkdir "output_data/$figure"
  # mkdir "output_data/$figure/coeffs"
  copy_commands=(
    "cp ../work_error_opt/output_data/$figure/error_coeffs.pkl output_data/$figure/coeffs/error.pkl"
    "cp ../work_error_opt/output_data/$figure/work_coeffs.pkl output_data/$figure/coeffs/work.pkl"
    "cp ../work_error_opt/output_data/$figure/split_coeffs.pkl output_data/$figure/coeffs/split.pkl"
  )

  for cmd in "${copy_commands[@]}"; do
    # Execute each command
    if not $cmd; then
      echo "COPY_FAILED"
    fi
  done
  # echo "$figure copy done"
done