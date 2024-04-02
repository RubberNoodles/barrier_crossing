#!/bin/bash

declare -A landscapes=( ["Double Well 2.5kT Barrier Brownian"]="2_5kt" 
                        ["Double Well 10kT Barrier Brownian"]="10kt"
                        ["Double Well 12kT Barrier Brownian"]="12kt"
                        ["Double Well 15kT Barrier Brownian"]="15kt"
                        ["Double Well 19kT Barrier Brownian"]="19kt"
                        ["Double Well 22kT Barrier Brownian"]="22kt"
                        ["Double Well 25kT Barrier Brownian"]="25kt"
                        ["Asymetric Double Well Brownian"]="asym"
                        ["Triple Well Brownian"]="triple_well"
                        )

#declare -A figures=( [work_error_opt]="we_params.sh" )
#declare -A figures=( [iterative]="it_params.sh" )
declare -A figures=( [reconstructions]="rc_params.sh" )

for figure in "${!figures[@]}"; do
  # Modify the params to fit the desired landscape
  
  for landscape in "${!landscapes[@]}"; do  
    # cat params_set/params_base_sc.py params_set/params_${landscapes[$landscape]} > params.py  
    echo "Task $figure on $landscape is starting. Modules & Dependencies Loading..."

    cd $figure
    
    ./${figures[$figure]} "$landscape" "${landscapes[$landscape]}"
    
    cd ../
  done
done
