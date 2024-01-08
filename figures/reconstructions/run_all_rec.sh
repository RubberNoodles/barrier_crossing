#!/bin/bash

#deprecated for now

declare -A landscapes=( ["Double Well 2.5kT Barrier Brownian"]="2_5kt" 
                        ["Double Well 10kT Barrier Brownian"]="10kt"
                        ["Asymetric Double Well Brownian"]="asym"
                        ["Triple Well Brownian"]="triple_well"
                        )
for landscape in "${!landscapes[@]}"; do  
  # cat params_set/params_base_sc.py params_set/params_${landscapes[$landscape]} > params.py  

  echo "Task $landscape is starting. Modules & Dependencies Loading..."
  
  
  python3 reconstruct.py "$landscape" "${landscapes[$landscape]}"
    
  
done




# given a set of landscapes and a set of figures I want to output, run the corresponding slurm commands