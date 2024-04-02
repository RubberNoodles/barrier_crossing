# From free energy values, look at error over different barrier heights

from barrier_crossing.utils import parse_args
import matplotlib.pyplot as plt
import jax.numpy as jnp

import code
import pickle
  

def plot_with_stddev(x,y, label=None, n=1, axis=0, ax=plt, dt=1., color = None):
  y = jnp.array(y).T
  stddev = jnp.std(y, axis)
  mn = jnp.mean(y, axis)
  # code.interact(local = locals())
  fill = ax.fill_between(x,
                  mn + n * stddev, mn - n * stddev, alpha=.3, color = color)
  line = ax.plot(x, mn, label=label, color = color)[0]
  return fill, line

  
if __name__ == "__main__":
  args, p  = parse_args()
  
  
  path_str = lambda height: f"double_well_{height.replace('.', '_')}kt_barrier_brownian"
  pkl_file = f"free_energies_t{args.end_time}_k{args.k_s}_b{args.batch_size}.pkl"
  
  heights = ["2.5", "10", "12", "15", "19", "22", "25"]
  
  
  model_types = [
    # "Linear", 
    # "Work; Position", 
    # "Work; Stiffness", 
    # "Work; Joint",
    "Error; Position",
    "Error; Stiffness",
    "Error; Joint",
    "Split Work; Position",
    "Split Work; Stiffness",
    "Split Work; Joint",
    "Split Error; Position",
    "Split Error; Stiffness",
    "Split Error; Joint",
    # "Near Equilibrium"
    
    ]
  plot_color_list = ['b', 'g', 'r' ,'c', 'm', 'y', 'k'] * 2
  model_dict = {model: [] for model in model_types}
  
  plot_heights = []
  for height in heights:
    
    parent_dir = f"output_data/{path_str(height)}/"
    
    try:
      with open(parent_dir + pkl_file, "rb") as f:
        fe_dict = pickle.load(f)
      
    except FileNotFoundError:
      print(f"Failed to open: {parent_dir + pkl_file}")
      continue
    
    
    found_data = True
    for model_type in model_types:
      fe = fe_dict.get(model_type, None)
      if fe is None:
        found_data = False
        break 
      else:
        model_dict[model_type].append(fe)
    if found_data:
      plot_heights.append(height)
    else:
      continue 
  fig = plt.figure(figsize = (12,12))
  ax = fig.add_subplot(1,1,1)
  
  for ind, (name, fe_values) in enumerate(model_dict.items()):
    plot_with_stddev(plot_heights, fe_values, label = name, ax = ax, color = plot_color_list[ind])
  
  ax.axhline(y = p.param_set.delta_E, label = "True FE", color = "red")
  ax.set_title("Free Energies")
  ax.set_xlabel("Barrier Height")
  ax.set_ylabel("∆F Estimate")
  
  ax.legend()
  
  fig.savefig(f"output_data/free_energies_t{args.end_time}_k{args.k_s}_b{args.batch_size}.png", bbox_inches = "tight")
  

    
    
      
    
    
  
  