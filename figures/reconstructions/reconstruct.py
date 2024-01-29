# Given coefficients (pickled); reconstruct and determine
# forward simulation and reconstructions statistics.
import time
import pickle
import sys
import os
import importlib
import subprocess

import barrier_crossing.energy as bc_energy
import barrier_crossing.protocol as bc_protocol
import barrier_crossing.simulate as bc_simulate
import barrier_crossing.optimize as bc_optimize
import barrier_crossing.iterate_landscape as bc_landscape
from barrier_crossing.utils import parse_args

import jax.numpy as jnp
import jax.random as random

import matplotlib.pyplot as plt

# from figures.params import * # global variables;

if __name__ == "__main__":
  args, p  = parse_args()
  sim_cut_steps = p.param_set.simulation_steps // 2
  
  path = args.landscape_name.replace(" ", "_").replace(".", "_").lower()
  parent_dir = f"output_data/{path}/"
  if not os.path.isdir(parent_dir):
    os.mkdir(parent_dir)
  if not os.path.isdir(parent_dir+"plotting"):
    os.mkdir(parent_dir + "plotting")
  
  if not os.path.isdir(parent_dir+"coeffs"):
    os.mkdir(parent_dir + "coeffs")
    cp_coeff_p = subprocess.Popen(['./collect_coeffs.sh'], shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if "COPY_FAILED" in cp_coeff_p.communicate()[0]:
      assert False, f"Coefficients for {parent_dir} not found."
  
  assert os.path.isdir(parent_dir+"coeffs"), "Reconstruction code requires protocol coefficients to run."
    
  coeff_dir = parent_dir + "coeffs/"
  
  coeff_files = {
    "Linear Protocol": "linear", 
    #"Optimized Linear": "opt_linear.pkl",
    "Work Optimized Protocol": "work.pkl",
    "Single Error Protocol":  "error.pkl",
    #"Iterative Protocol": "iterative.pkl",
    "Split Error Protocol": "split.pkl",
    # "Accumulated Error Protocol": "accumulated.pkl",
    # "Near Equilibrium Protocol": "near_eq.pkl"
    }

  plot_data = {}
  plot_color_list = ['b', 'g', 'r' ,'c', 'm', 'y', 'k', 'w']
  
  
  fig_rec = plt.figure(figsize = (8,8))
  fig_pro = plt.figure(figsize = (8,8))
  fig_hist = plt.figure(figsize = (8,8))
  
  ax_reconstruct = fig_rec.add_subplot(1, 1, 1)
  ax_protocol = fig_pro.add_subplot(1, 1, 1)
  ax_hist = fig_hist.add_subplot(1,1,1)

  ax_protocol.set_title(f"{args.landscape_name} Protocols")
  ax_protocol.set_xlabel("Timestep")
  ax_protocol.set_ylabel("Position")
  ax_reconstruct.set_title(f"{args.landscape_name} Energy Reconstructions | kₛ = { args.k_s} | end_time = { args.end_time }")
  ax_reconstruct.set_xlabel("Position")
  ax_reconstruct.set_ylabel("Energy")
  ax_hist.set_title(f"{args.landscape_name} Dissipated Work Distribution | kₛ = { args.k_s} | end_time = { args.end_time }")
  ax_hist.set_xlabel("Dissipated Work")

  no_trap = p.param_set.energy_fn(0.)
  time_vec = jnp.linspace(-20,20, 1000)
  if "triple" in args.landscape_name.lower():
    time_vec = jnp.linspace(-12,12, 1000)
  ax_reconstruct.plot(time_vec, jnp.squeeze(no_trap(time_vec.reshape(1,1,1000))) - no_trap([[p.r0_init]]), label = "Original", color = 'k')

  for ind, (trap_name, file_name) in enumerate(coeff_files.items()):
    if file_name == "linear":
      coeff = bc_protocol.linear_chebyshev_coefficients(p.r0_init, p.r0_final, p.param_set.simulation_steps, degree = 12, y_intercept = p.r0_init)
    else:
      try:
        with open(coeff_dir + file_name, "rb") as f:
          coeff = pickle.load(f)
      except FileNotFoundError:
        print(f"In order to run this code, you need a file of coefficients called {coeff_dir+file_name}")
        raise
    
    if file_name == "split.pkl":
      # We are going to only look at gradient values for the second set of coefficients
      _a = bc_protocol.make_trap_fxn( jnp.arange(sim_cut_steps), coeff[0], p.r0_init, p.r0_cut)
      _b = bc_protocol.make_trap_fxn( jnp.arange(p.param_set.simulation_steps - sim_cut_steps), coeff[1], p.r0_cut, p.r0_final)
      trap_fn = bc_protocol.trap_sum(p.param_set.simulation_steps, sim_cut_steps, _a, _b)
      
    else:
      trap_fn = bc_protocol.make_trap_fxn(jnp.arange(p.param_set.simulation_steps), coeff, p.r0_init, p.r0_final)
    
    time_vec = jnp.arange(p.param_set.simulation_steps-1)
    
    
    key = random.PRNGKey(int(time.time()))
    key, split = random.split(key)

    simulate_fwd_grad = lambda  trap_fn_arb, keys: p.param_set.simulate_fn(
      trap_fn_arb, 
      keys, 
      regime = "brownian",
      fwd = True)

    simulate_fwd = lambda keys: simulate_fwd_grad(trap_fn, keys)

    
    batch_size_grad = 2000
    bins = 70

    total_work, (batch_trajectories, batch_works, _) = bc_simulate.batch_simulate_harmonic(
        args.batch_size, simulate_fwd, key)
    
    # work distribution data
    mean_work = batch_works.mean()
    tail = total_work.mean() - total_work.min()
    #w_diss = jnp.cumsum(batch_works, axis = -1)[:, -1] - p.param_set.delta_E
    
    # Dissipated work = Work Used - Free Energy Difference
    w_diss = total_work - (p.param_set.energy_fn()([[p.r0_final]]) - p.param_set.energy_fn()([[p.r0_init]])) 
    
    
    # reconstructions stats
    midpoints, energies = bc_landscape.energy_reconstruction(
        jnp.cumsum(batch_works, axis=1), 
        batch_trajectories, 
        bins, 
        trap_fn, 
        p.param_set.simulation_steps, 
        args.batch_size, 
        args.k_s, 
        p.beta)
  
    landscape = (midpoints, energies)
    
    disc = bc_landscape.landscape_discrepancies(landscape, no_trap, no_trap([[0.]]), -10., 10.)
    bias = max(disc)
    
    print(f"{trap_name}: W_diss: {w_diss.mean()}, tail: {tail}, bias: {bias}")
    
    no_trap_rec_fn = bc_energy.V_biomolecule_reconstructed(0, midpoints, energies)
    
    first_well_pos, _ = bc_landscape.find_min_pos(no_trap_rec_fn, -10, 10)
    # max_rec = bc_landscape.find_max(landscape, -10., 10.)
    
    difference = no_trap_rec_fn([[first_well_pos]])
    energies_aligned = energies - difference
    
    # stats at extension values 
    extensions = [-10,-5,0] # temporary
    disc_samples = bc_landscape.landscape_discrepancies_samples(no_trap_rec_fn, no_trap, extensions)
    disc_samples = jnp.array(disc_samples)
    mean_disc_samples = disc_samples.mean()
    bias_samples = disc_samples.max()

    # loss values 
    if file_name == "split.pkl":
      grad_rev = lambda num_batches: bc_optimize.estimate_gradient_rev_split(
        num_batches,
        simulate_fwd_grad,
        p.r0_init,
        p.r0_final,
        p.r0_cut,
        sim_cut_steps,
        p.param_set.simulation_steps,
        p.beta)
      grad, (_, summary) = grad_rev(batch_size_grad)(*coeff, split)
      
    else:
      grad_rev = lambda num_batches: bc_optimize.estimate_gradient_rev(
        num_batches,
        simulate_fwd_grad,
        p.r0_init,
        p.r0_final,
        p.param_set.simulation_steps,
        p.beta)
      grad, (_, summary) = grad_rev(batch_size_grad)(coeff, split)
    

    plot_data[trap_name] = {
      "trap": trap_fn,
      "translation": difference,
      "work": total_work,
      "midpoints": midpoints,
      "energies": energies_aligned,
      "bias": bias,
      "mean_work": mean_work,
      "discrepancy": disc,
      "tail": tail,
      "work loss": summary[2].mean(), # TODO
      "error loss": summary[2].mean(), # TODO
      "accumulated loss": summary[2].mean(), # TODO
      "samples": {
        "mean_discrepancy": mean_disc_samples,
        "bias": bias_samples,
        "discrepancy": disc_samples
        }
      }
    
    color = plot_color_list[ind]
    ax_hist.axvline(x = w_diss.mean(), color = color)
    ax_hist.hist(w_diss, weights=jnp.ones(len(w_diss)) / len(w_diss), bins = 50, label = trap_name, alpha = 0.7, color = color)
    ax_protocol.plot(time_vec, trap_fn(time_vec), label = trap_name, color = color)
    ax_reconstruct.plot(midpoints, energies_aligned, label = trap_name, color = color)
    
  ax_hist.legend()
  ax_protocol.legend()
  ax_reconstruct.legend()
  fig_hist.savefig(parent_dir + f"plotting/work_distribution_k{args.k_s}_t{args.end_time}_b{args.batch_size}.png")
  fig_rec.savefig(parent_dir + f"plotting/reconstructions_k{args.k_s}_t{args.end_time}_b{args.batch_size}.png")
  fig_pro.savefig(parent_dir + f"plotting/protocol.png")

  with open(parent_dir + "/landscape_data.pkl", "wb") as f:
    for trap_name, data in plot_data.items():
        pass # not sure how to do this right now but there is an error    
        # pickle.dump(plot_data, f)
        # AttributeError: Can't pickle local object 'make_trap_fxn.<locals>.Get_r0'
        # plot_data[trap_name]["trap"] = 
    #pickle.dump(plot_data, f)




"""`
  # Table comparison of protocols --> Important graph to have! --> Adjust for different protocols etc

  # Table comparison (+ reconstruction info)
  _, (ax0, ax1) = plt.subplots(1, 2, figsize=[24, 12])
  step = jnp.arange(p.param_set.simulation_steps)
  for p_name in plot_data:
    data = plot_data[p_name]
    ax0.plot(step, data["trap"](step), label = f'{p_name}')
  ax0.legend()
  ax0.set_title(f"Different Protocol Trajectories; {extensions}")
  columns = ('Bias', 'Mean discrepancy',"Discrepancies Samples", 'Average total work', 'Tail length', 'Loss')
  rows = []
  table_data = []
  for p_name in plot_data:
    data = plot_data[p_name]
    rows.append(p_name)
    mean_disc = float(jnp.array(data["discrepancy"]).mean())
    disc_str = "\n"
    for ch in str(data["samples"]["discrepancy"]).split(","):
      disc_str += ch + "," + "\n"

    print(disc_str)
    disc_str = "N/A"
    table_data.append([data["bias"], mean_disc, disc_str, data["mean_work"], data["tail"], data["loss"]])
  n_rows = len(table_data)
  cell_text = []
  #colors = plt.cm.BuPu(jnp.linspace(0, 0.5, len(rows)))

  for row in range(n_rows):
    y_offset = table_data[row]
    cell_text.append([x for x in y_offset])

  ax1.axis('off')

  table = plt.table(cellText=cell_text,
                        rowLabels=rows,
                        colLabels=columns,loc = 'center')
  

  table.auto_set_font_size(False)
  table.set_fontsize(12)
  table.scale(1,10)
  
  plt.savefig(path+"protocol_info.png")

"""