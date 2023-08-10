import time
import tqdm
import pickle
import csv
import sys

import jax
import jax.numpy as jnp
import numpy as onp
import jax.random as random
import jax.example_libraries.optimizers as jopt
from jax_md import space

import matplotlib.pyplot as plt

import barrier_crossing.energy as bc_energy
import barrier_crossing.protocol as bc_protocol
import barrier_crossing.simulate as bc_simulate
import barrier_crossing.optimize as bc_optimize
import barrier_crossing.iterate_landscape as bc_landscape

from figures.params import * # global variables;

if __name__ == "__main__":
  bh_index = int(sys.argv[1])
  kappa_l_list = [ x/(beta_sc*x_m**2) for x in [4., 5., 6.38629, 7., 8., 8.5, 9., 10.]]
  kappa_l = kappa_l_list[bh_index-1]
  kappa_r=kappa_l*10  
  
  
  scale_arr = [2 + i/2 for i in range(8)]
  scale = scale_arr[bh_index - 1]
  pos_e = [[5.4*x, 8*x**2 + 0.3 * x**3 - 6 * x ** 4 + x**6] for x in jnp.linspace(-6,6,1000)] # https://www.scielo.org.mx/pdf/rmf/v57n1/v57n1a8.pdf
  e_positions = jnp.array(pos_e)[:,0]
  e_energies = jnp.array(pos_e)[:,1] * scale

  # The plot on the energy function underneath will be coarse-grained due to few sample points.
  energy_custom_plot = bc_energy.V_biomolecule_reconstructed(k_s_sc, e_positions, e_energies)

  energy_sivak = bc_energy.V_biomolecule_reconstructed(k_s_sc, e_positions, e_energies)
  # energy_sivak = bc_energy.V_biomolecule_sivak(kappa_l, kappa_r, x_m, delta_E, k_s_sc, beta_sc)
  
  save_dir = f"data/barrier_{bh_index}/"
  # Protocol Coefficients
  lin_coeffs_sc = jnp.array([-1.14278407e-07,  3.33333325e+00,  5.03512065e-10,  2.32704592e-10,
        9.57856017e-10, -2.86552310e-10,  1.17483601e-09, -4.50110560e-10,
        1.21933308e-09, -4.65200489e-10,  7.08764991e-10, -1.05334935e-10,
        6.99122538e-10]) # Slightly shifted

  lin_coeffs_sc = bc_protocol.linear_chebyshev_coefficients(r0_init_sc, r0_final_sc, simulation_steps_sc, degree = 12, y_intercept = r0_init_sc)
  
  # Trap Functions. Reverse mode trap functions are for when we compute Jarzynski error with reverse protocol trajectories.
  trap_fn_fwd_sc = bc_protocol.make_trap_fxn(jnp.arange(simulation_steps_sc), lin_coeffs_sc, r0_init_sc, r0_final_sc)
  trap_fn_rev_sc = bc_protocol.make_trap_fxn_rev(jnp.arange(simulation_steps_sc), lin_coeffs_sc, r0_init_sc, r0_final_sc)
  
  true_sivak_simulation_fwd = lambda trap_fn: lambda keys: bc_simulate.simulate_langevin_harmonic(
      energy_sivak, 
      init_position_fwd_sc, 
      trap_fn,
      simulation_steps_sc, 
      Neq, 
      shift, 
      keys,
      dt_sc,
      temperature_sc, mass_sc, gamma_sc # These parameters describe the state of the system.
      )
  
  simulate_grad_sivak_rev = lambda energy_fn: lambda trap_fn, keys: bc_simulate.simulate_langevin_harmonic(
      energy_fn, 
      init_position_fwd_sc, 
      trap_fn,
      simulation_steps_sc, 
      Neq, 
      shift, 
      keys, 
      dt_sc,
      temperature_sc, mass_sc, gamma_sc # These parameters describe the state of the system.
      )

  max_iter = 8
  opt_steps_landscape = 1000 # 1000 + 
  bins = 75
  opt_batch_size = 20000 # 10k + 
  rec_batch_size = 10000

  # grad_no_E = lambda num_batches, energy_fn: bc_optimize.estimate_gradient_rev(
  #     num_batches,
  #     simulate_grad_sivak_rev(energy_fn),
  #     r0_init_sc, r0_final_sc, simulation_steps_sc,
  #     beta_sc)

  grad_no_E = lambda num_batches, energy_fn: bc_optimize.estimate_gradient_work(
      num_batches,
      simulate_grad_sivak_rev(energy_fn),
      r0_init_sc, r0_final_sc, simulation_steps_sc)

  lr = jopt.polynomial_decay(0.3, opt_steps_landscape, 0.001)
  optimizer = jopt.adam(lr)

  key = random.PRNGKey(int(time.time()))
  
  landscapes, coeffs = bc_landscape.optimize_landscape(
                      true_sivak_simulation_fwd,
                      lin_coeffs_sc, # First reconstruction; should one reconstruct with forward or reverse simulations? Does it matter?
                      grad_no_E,
                      key,
                      max_iter,
                      bins,
                      simulation_steps_sc,
                      opt_batch_size,
                      rec_batch_size,
                      opt_steps_landscape, 
                      optimizer,
                      r0_init_sc, r0_final_sc,
                      k_s_sc, beta_sc,
                      savefig = f"barrier_{bh_index}"
  )
  positions = jnp.array(landscapes[-1][0])

  plt.figure(figsize = (10,10))
  #energy_sivak_plot = bc_energy.V_biomolecule_sivak(kappa_l, kappa_r, x_m, delta_E, 0., beta_sc)
  energy_sivak_plot = bc_energy.V_biomolecule_reconstructed(0., e_positions, e_energies)
  true_E = []
  
  pos_vec = jnp.reshape(positions, (positions.shape[0], 1, 1))
  for j in range(positions.shape[0]):
    min_pos, _ = bc_landscape.find_min_pos(energy_sivak_plot, -8., 8.)
    true_E.append(energy_sivak_plot(pos_vec[j])-float(energy_sivak_plot([[min_pos]])))
  plt.plot(positions, true_E, label = "True Landscape")

  for num, (positions, energies) in enumerate(landscapes):
    min_e = jnp.min(energies[jnp.where((positions > -10) & (positions < 10))])
    if num == 0:
      label = "Linear"
    elif num == len(landscapes)-1:
      label = "Final Landscape"
    else:
      label = f"Iteration {num}"
    plt.plot(positions, energies - min_e, label = label)
  plt.legend()
  plt.xlabel("Position (x)")
  plt.ylabel("Free Energy (G)")
  plt.title(f"Iteratively Reconstructing Landscape; κ={scale:.2f}")
  plt.savefig(save_dir + "reconstruct_landscapes.png")
  
  plt.figure(figsize = (8,8))
  trap_fn = bc_protocol.make_trap_fxn(jnp.arange(simulation_steps_sc), lin_coeffs_sc, r0_init_sc, r0_final_sc)
  plt.plot(trap_fn(jnp.arange(simulation_steps_sc-1)), label = "Linear Protocol")
  for i, coeff in enumerate(coeffs):
      trap_fn = bc_protocol.make_trap_fxn(jnp.arange(simulation_steps_sc), coeff, r0_init_sc, r0_final_sc)
      plt.plot(trap_fn(jnp.arange(simulation_steps_sc-1)), label = f"Iteration {i}")
  plt.xlabel("Simulation Step")
  plt.ylabel("Position (x)")
  plt.title("Protocols Over Iteration")
  plt.legend()
  plt.savefig(save_dir + "opt_protocol_evolution.png")
  
  with open(save_dir + "coeffs.pkl", "wb") as f:
    pickle.dump(coeffs, f)
