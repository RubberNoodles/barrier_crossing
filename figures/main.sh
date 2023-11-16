#!/bin/bash

declare -A landscapes=( ["Double Well 2.5kT Barrier"]="2_5kt" 
                        ["Double Well 10kT Barrier"]="10kt"
                        ["Asymetric Double Well"]="asym"
                        ["Triple Well"]="triple_well"
                        )

#declare -A figures=( [reconstructions]="reconstruct.sh" [work_error_opt]="we_opt.sh" )
declare -A figures=( [work_error_opt]="we_opt.sh" )
# Need to fix iterative before I can do anything.

for figure in "${!figures[@]}"; do
  # Modify the params to fit the desired landscape
  
  
  for landscape in "${!landscapes[@]}"; do  
   
    echo "Looking for lock..."
    n=0

    while ! ln -s . lock; do 
            sleep 15
            n=$((n+1))
    done
    echo "Lock Found!"
    

    # cat params_set/params_base_sc.py params_set/params_${landscapes[$landscape]} > params.py  
    
    trap "{ echo Process terminated prematurely.; rm lock }" ERR

    echo "Task $figure on $landscape is starting. Modules & Dependencies Loading..."
    #echo 
    

    cd $figure
        
    

    if [ $figure = "iterative" ]; then
      sbatch --test-only ${figures[$figure]} "$landscape"
      rm ../lock
    else
      sbatch ${figures[$figure]} "$landscape" "${landscapes[$landscape]}"
      #python3 ${figures[$figure]} "$landscape" "${landscapes[$landscape]}"
      #rm ../lock
    fi
    
    cd ../
  done
done



# given a set of landscapes and a set of figures I want to output, run the corresponding slurm commands