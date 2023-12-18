#!/bin/bash

declare -A landscapes=( ["Double Well 2.5kT Barrier Brownian"]="2_5kt" 
                        ["Double Well 10kT Barrier Brownian"]="10kt"
                        ["Asymetric Double Well Brownian"]="asym"
                        ["Triple Well Brownian"]="triple_well"
                        )

#declare -A figures=( [reconstructions]="reconstruct.sh" [work_error_opt]="we_opt.sh" )
declare -A figures=( [work_error_opt]="we_opt.sh" )
#declare -A figures=( [iterative]="iterative.sh" )
#declare -A figures=( [reconstructions]="reconstruct.sh" )
# Need to fix iterative before I can do anything.

for figure in "${!figures[@]}"; do
  # Modify the params to fit the desired landscape
  
  
  for landscape in "${!landscapes[@]}"; do  
    # cat params_set/params_base_sc.py params_set/params_${landscapes[$landscape]} > params.py  

    echo "Task $figure on $landscape is starting. Modules & Dependencies Loading..."
    #echo 
    

    cd $figure
        
    

    # if [ $figure = "iterative" ]; then
    #   sbatch --test-only ${figures[$figure]} "$landscape"
    #   rm ../lock
    # else
    
    
    sbatch ${figures[$figure]} "$landscape" "${landscapes[$landscape]}"
      #python3 ${figures[$figure]} "$landscape" "${landscapes[$landscape]}"
      #rm ../lock
    #fi
    cd ../
  done
done



# given a set of landscapes and a set of figures I want to output, run the corresponding slurm commands