#!/bin/bash

declare -a figures=( "asymetric_double_well" 
                     "double_well_2_5kt_barrier"
                     "double_well_10kt_barrier"
                     "triple_well"
)

for figure in "${figures[@]}"; do
  # mkdir "output_data/$figure"
  # mkdir "output_data/$figure/coeffs"
  cp "../work_error_opt/output_data/$figure/error_coeffs.pkl" "output_data/$figure/coeffs/error.pkl" 
  cp "../work_error_opt/output_data/$figure/work_coeffs.pkl" "output_data/$figure/coeffs/work.pkl" 
  cp "../work_error_opt/output_data/$figure/split_coeffs.pkl" "output_data/$figure/coeffs/split.pkl"
  echo "$figure copy done"
done