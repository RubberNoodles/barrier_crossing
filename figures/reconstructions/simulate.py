"""Forward simulations to compute free energies."""
import code
import pickle
import tqdm

import barrier_crossing.simulate as bcs
import barrier_crossing.models as bcm

from barrier_crossing.utils import parse_args, make_trap_from_file, find_coeff_file

import jax.numpy as jnp
import jax.random as random

import matplotlib.pyplot as plt
# from figures.params import * # global variables;

  
if __name__ == "__main__":
  args, p  = parse_args()
  
  path = args.landscape_name.replace(" ", "_").replace(".", "_").lower()
  parent_dir = f"output_data/{path}/"
  coeff_dir = parent_dir + "coeffs/"
  
  if "2_5kt" in path:
    near_eq_file = "2.5kt_lr_opt.pkl"
  elif "10kt" in path:
    near_eq_file = "10kt_lr_opt.pkl"
  else:
    near_eq_file = None
  
  model_types = {
    "Linear": "linear", 
    "Work; Position": ["position","fwd"],
    "Work; Stiffness": ["stiffness","fwd"],
    "Work; Joint": ["joint","rev"],
    "Error; Position": ["position","rev"],
    "Error; Stiffness": ["stiffness","rev"],
    "Error; Joint": ["joint","rev"],
    "Split Work; Position": ["position","fwd", "split"],
    "Split Work; Stiffness": ["stiffness","fwd", "split"],
    "Split Work; Joint": ["joint","fwd", "split"],
    "Split Error; Position": ["position","rev", "split"],
    "Split Error; Stiffness": ["stiffness","rev", "split"],
    "Split Error; Joint": ["joint","rev", "split"],
    "Near Equilibrium": [near_eq_file] if near_eq_file else None
    }
  
  psm = bcm.ScheduleModel(p.param_set, p.r0_init, p.r0_final)
  ssm = bcm.ScheduleModel(p.param_set, p.ks_init, p.ks_final)
  
  plot_data = {}
  plot_color_list = ['b', 'g', 'r' ,'c', 'm', 'y', 'k'] * 2
  kde_colors = []
  plt.rc('font', size = 16)
  plt.subplots_adjust(right=0.8)
  
  ### Free energy std whisker plot. Save the free energies associated so that I can do
  # A plot that looks at free energies across different barrier heights. So I want to record
  # Free energies and their barrier heights.
  
  
  fig_fe = plt.figure(figsize = (10,7))
  
  ax_free_energies = fig_fe.add_subplot(1, 1, 1)
 
  ax_free_energies.set_title(f"{args.landscape_name} Free Energy Calculations | kₛ = { args.k_s} | end_time = { args.end_time }")
  ax_free_energies.set_xlabel("Protocol Type")
  ax_free_energies.set_ylabel("Free Energy")
  ax_free_energies.axhline( y= 0, label = "True ∆F")
  
  free_energies = {}
  for ind, (trap_name, model_args) in enumerate(model_types.items()):
    if model_args is None:
      continue
    dir_name, coeff_files = find_coeff_file(model_args, args)
    trap_fns = make_trap_from_file(dir_name, coeff_files, psm, ssm, p)
    
    if trap_fns is None:
      print(f"FAILURE: {trap_name} failed, continuing...")
      continue
    
    t = jnp.arange(p.param_set.simulation_steps)

    key = random.PRNGKey(1001)
    key, split = random.split(key)

    simulate_fwd = lambda keys: p.param_set.simulate_fn(
      *trap_fns,
      keys, 
      regime = "brownian",
      fwd = True)
    
    runs = 100
    
    free_energies[trap_name] = []
    for i in tqdm.tqdm(range(runs), desc = trap_name):
      key, split = random.split(key)
      total_work, (batch_trajectories, batch_works, _) = bcs.batch_simulate_harmonic(
          args.batch_size, simulate_fwd, key)
      
      # -beta_sc**(-1) * jnp.log(jnp.mean(jnp.exp(-beta_sc * total_works)))
      f_e = -p.beta**(-1) * jnp.log(jnp.mean(jnp.exp(-p.beta * total_work)))
      free_energies[trap_name].append(f_e)
  
  
  # code.interact(local = locals())
  ax_free_energies.boxplot(free_energies.values())
  ax_free_energies.set_xticklabels(free_energies.keys(),  rotation=45, ha='right')
  ax_free_energies.legend()
  fig_fe.savefig(parent_dir + f"free_energies_t{args.end_time}_k{args.k_s}_b{args.batch_size}.png", bbox_inches = "tight")
  with open(parent_dir +  f"free_energies_t{args.end_time}_k{args.k_s}_b{args.batch_size}.pkl", "wb") as f:
    pickle.dump(free_energies, f)
