# Given coefficients (pickled); reconstruct and determine
# reconstructions statistics.
import time
import tqdm
import pickle

import barrier_crossing.energy as bc_energy
import barrier_crossing.protocol as bc_protocol
import barrier_crossing.simulate as bc_simulate
import barrier_crossing.optimize as bc_optimize
import barrier_crossing.iterate_landscape as bc_landscape

import jax

import jax.numpy as jnp
import numpy as onp

import jax.random as random

import jax.example_libraries.optimizers as jopt

from jax_md import space

import matplotlib.pyplot as plt

from figures.params import * # global variables;

if __name__ == "__main__":
  
  coeff_files = {
    "Linear Protocol": "linear", 
    "Work Optimized Protocol": "work.pkl",
    "Single Error Protocol":  "error.pkl",
    #"Accumulated Error Protocol": "accumulated.pkl",
    }
  
  coeff_dir = "coeffs/"
  plot_data = {}
  
  plt.figure(figsize = (8,8))
  plt.title("S&C Energy Reconstructions")
  plt.xlabel("Position")
  plt.ylabel("Energy")

  no_trap_sivak = bc_energy.V_biomolecule_sivak(kappa_l, kappa_r, x_m, delta_E, 0, beta_sc)
  time_vec = jnp.linspace(-20,20, 1000)
  plt.plot(time_vec, jnp.squeeze(no_trap_sivak(time_vec.reshape(1,1,1000))), label = "Original")

  for trap_name, file_name in coeff_files.items():
    if file_name == "linear":
      coeff = bc_protocol.linear_chebyshev_coefficients(r0_init_sc, r0_final_sc, simulation_steps_sc, degree = 12, y_intercept = r0_init_sc)
    else:
      try:
        with open(coeff_dir + file_name, "rb") as f:
          coeff = pickle.load(f)
      except FileNotFoundError:
        print(f"In order to run this code, you need a file of coefficients called {coeff_dir+file_name}")
        raise
    
    trap_fn = bc_protocol.make_trap_fxn(jnp.arange(simulation_steps_sc), coeff, r0_init_sc, r0_final_sc)

    key = random.PRNGKey(int(time.time()))
    key, split = random.split(key)

    simulate_sivak_fwd_grad = lambda  trap_fn_arb, keys: bc_simulate.simulate_langevin_harmonic(
      energy_sivak, 
      init_position_fwd_sc, 
      trap_fn_arb,
      simulation_steps_sc, 
      Neq, 
      shift, 
      keys, 
      dt_sc,
      temperature_sc, mass_sc, gamma_sc # These parameters describe the state of the brownian system.
      )

    simulate_sivak_fwd = lambda keys: simulate_sivak_fwd_grad(trap_fn, keys)

    batch_size_sc_rec = 10
    batch_size_grad = 1000
    bins = 5

    total_work, (batch_trajectories, batch_works, _) = bc_simulate.batch_simulate_harmonic(
        batch_size_sc_rec, simulate_sivak_fwd, simulation_steps_sc, key)
    
    # work distribution data
    mean_work = batch_works.mean()
    tail = total_work.mean() - total_work.min()

    # reconstructions stats
    midpoints, energies = bc_landscape.energy_reconstruction(
        batch_works, 
        batch_trajectories, 
        bins, 
        trap_fn, 
        simulation_steps_sc, 
        batch_size_sc_rec, 
        k_s_sc, 
        beta_sc)
  
    landscape = (midpoints, energies)
    
    disc = bc_landscape.landscape_discrepancies(landscape, no_trap_sivak, no_trap_sivak([[0.]]), -10., 10.)
    bias = max(disc)
    
    max_rec = bc_landscape.find_max(landscape, -5., 5.)
    difference = no_trap_sivak([[0.]]) - max_rec
    energies_aligned = energies + difference
    
    no_trap_rec_fn = bc_energy.V_biomolecule_reconstructed(0, midpoints, energies_aligned)
    
    # stats at extension values 
    extensions = [-10,-5,0] # temporary
    disc_samples = bc_landscape.landscape_discrepancies_samples(no_trap_rec_fn, no_trap_sivak, extensions)
    disc_samples = jnp.array(disc_samples)
    mean_disc_samples = disc_samples.mean()
    bias_samples = disc_samples.max()

    # loss values 
    grad_rev = lambda num_batches: bc_optimize.estimate_gradient_rev(
      num_batches,
      simulate_sivak_fwd_grad,
      r0_init_sc,
      r0_final_sc,
      simulation_steps_sc,
      beta_sc)
      
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
      "work loss": summary[2].mean(),
      "error loss": summary[2].mean(),
      "accumulated loss": summary[2].mean(),
      "samples": {
        "mean_discrepancy": mean_disc_samples,
        "bias": bias_samples,
        "discrepancy": disc_samples
        }
      }
    

  plt.plot(midpoints, energies_aligned, label = trap_name)
  plt.legend()
  plt.savefig("plotting/reconstructions.png")

  with open("landscape_data.pkl", "wb") as f:
    for trap_name, data in plot_data.items():
        pass # not sure how to do this right now but there is an error    
        # pickle.dump(plot_data, f)
        # AttributeError: Can't pickle local object 'make_trap_fxn.<locals>.Get_r0'
        # plot_data[trap_name]["trap"] = 
    pickle.dump(plot_data, f)




"""`
  # Table comparison of protocols --> Important graph to have! --> Adjust for different protocols etc

  # Table comparison (+ reconstruction info)
  _, (ax0, ax1) = plt.subplots(1, 2, figsize=[24, 12])
  step = jnp.arange(simulation_steps_sc)
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