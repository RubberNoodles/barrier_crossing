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
  save_dir = "results_0.1/"
  # Protocol Coefficients
  lin_coeffs_sc = bc_protocol.linear_chebyshev_coefficients(r0_init_sc, r0_final_sc, simulation_steps_sc, degree = 12, y_intercept = r0_init_sc)

  # Trap Functions. Reverse mode trap functions are for when we compute Jarzynski error with reverse protocol trajectories.
  trap_fn_fwd_sc = bc_protocol.make_trap_fxn(jnp.arange(simulation_steps_sc), lin_coeffs_sc, r0_init_sc, r0_final_sc)
  trap_fn_rev_sc = bc_protocol.make_trap_fxn_rev(jnp.arange(simulation_steps_sc), lin_coeffs_sc, r0_init_sc, r0_final_sc)
  
  simulate_sivak_fn_fwd = lambda energy_fn, keys: bc_simulate.simulate_brownian_harmonic(
      energy_fn, 
      init_position_fwd_sc, 
      trap_fn_fwd_sc,
      simulation_steps_sc, 
      Neq, 
      shift, 
      keys, 
      dt_sc,
      temperature_sc, mass_sc, gamma_sc # These parameters describe the state of the brownian system.
      )
  
  max_iter = 3
  opt_steps_landscape = 500
  bins = 100
  opt_batch_size = 10000
  rec_batch_size = 1000

  grad_no_E = lambda num_batches, energy_fn: bc_optimize.estimate_gradient_rev(
#      extensions,
      num_batches, energy_fn, init_position_rev_sc, 
      r0_init_sc, r0_final_sc, Neq, shift, 
      simulation_steps_sc, dt_sc, 
      temperature_sc, mass_sc, gamma_sc, beta_sc)

  lr = jopt.polynomial_decay(1., opt_steps_landscape, 0.01)
  optimizer = jopt.adam(lr)

  key = random.PRNGKey(int(time.time()))
  
  landscapes, coeffs, positions = bc_landscape.optimize_landscape(energy_sivak,
                      simulate_sivak_fn_fwd,
                      trap_fn_rev_sc, 
                      lin_coeffs_sc, 
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
                      k_s_sc, beta_sc)

  positions = jnp.array(positions)

  plt.figure(figsize = (10,10))
  
  energy_sivak_plot = bc_energy.V_biomolecule_sivak(kappa_l, kappa_r, x_m, delta_E, 0., beta_sc)
  true_E = []
  pos_vec = jnp.reshape(positions, (positions.shape[0], 1, 1))
  for j in range(positions.shape[0]):
    true_E.append(energy_sivak_plot(pos_vec[j])-float(energy_sivak_plot([[0.]])))
  plt.plot(positions, true_E, label = "True Landscape")

  for num, energies in enumerate(landscapes):
    max_e = jnp.max(energies[jnp.where((positions > -5) & (positions < 5))])
    plt.plot(positions, energies - max_e, label = f"Iteration {num}")

  plt.plot(positions, landscapes[-1] - max_e, label = "Final Landscape")
  plt.legend()
  plt.xlabel("Position (x)")
  plt.ylabel("Free Energy (G)")
  plt.title("Iteratively Reconstructing Landscape")
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
