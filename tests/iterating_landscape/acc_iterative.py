###################################################
# This code is a TEMPLATE for Sivak & Crooks
# testing. Last changed 12/18/2022. 
# WARNING: This code is NOT maintained.
###################################################

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


def plot_with_stddev(x, label=None, n=1, axis=0, ax=plt, dt=1.):
  stddev = jnp.std(x, axis)
  mn = jnp.mean(x, axis)
  xs = jnp.arange(mn.shape[0]) * dt

  ax.fill_between(xs,
                  mn + n * stddev, mn - n * stddev, alpha=.3)
  ax.plot(xs, mn, label=label)
_, ax = plt.subplots(1, 2, figsize=[16, 8])

if __name__ == "__main__":
  ###  LEGEND ###
  # S&C --> Parameters from Sivak and Crooke 2016

  N = 1
  dim = 1

  _, shift = space.free() # Defines how to move a particle by small distances dR.

  key = random.PRNGKey(int(time.time()))
  key, split = random.split(key, 2)  # RNG

  # ================= SIVAK & CROOKE =====================

  # Harmonic Trap Parameters S&C
  #k_s_sc = 0.4 # stiffness; 
  k_s_sc = 0.1 # stiffness; 
  r0_init_sc = -10. #nm; initial trap position
  r0_final_sc = 10. #nm; final trap position

  # Particle Parameters S&C
  mass_sc = 1e-17 # g
  init_position_fwd_sc = r0_init_sc*jnp.ones((N,dim)) #nm
  init_position_rev_sc = r0_final_sc*jnp.ones((N,dim))

  # Brownian Environment S&C
  temperature_sc = 4.183 #at 303K=30C S&C
  beta_sc=1.0/temperature_sc #1/(pNnm)
  D_sc = 0.44*1e6 #(in nm**2/s) 
  gamma_sc = 1./(beta_sc*D_sc*mass_sc) #s^(-1)


  # S&C Energy landscape params:
  x_m=10. #nm
  delta_E=7.0 #pN nm
  kappa_l=21.3863/(beta_sc*x_m**2) #pN/nm #for Ebarrier = 10kT and delta_E=0, as C&S use
  #kappa_l=6.38629/(beta*x_m**2) #pN/nm #for Ebarrier = 2.5kT and delta_E=0, as C&S use
  #kappa_l=2.6258/(beta*x_m**2)#barrier 0.625kT
  kappa_r=kappa_l #pN/nm; Symmetric wells.

  energy_sivak = bc_energy.V_biomolecule_sivak(kappa_l, kappa_r, x_m, delta_E, k_s_sc, beta_sc)
  
  # Protcol + Simulation Parameters
  end_time_gd = 1.
  dt_gd = 0.002
  simulation_steps_gd = int(end_time_gd / dt_gd)

  end_time_sc = 0.01
  # dt_sc = 2e-8 this might be exceeding floating point precision or something..
  end_time_sc = 0.01
  dt_sc = 5e-6
  simulation_steps_sc = int(end_time_sc / dt_sc)

  end_time_custom = 1.
  dt_custom = 0.001
  simulation_steps_custom = int(end_time_custom / dt_custom)

  # Equilibration Steps; in order to correctly apply Jarzynski, the system has to 
  # be in equilibrium, which is defined by equal free energy in all degrees of freedom
  Neq = 500

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
  
  max_iter = 4
  opt_steps_landscape = 200
  bins = 50
  opt_batch_size = 5000
  rec_batch_size = 1000

  _, shift = space.free()

  extensions = [-10, -5, -2, 0]

  grad_no_E = lambda num_batches, energy_fn: bc_optimize.estimate_gradient_acc_rev_extensions_scale(
      extensions,
      num_batches, energy_fn, init_position_rev_sc, 
      r0_init_sc, r0_final_sc, Neq, shift, 
      simulation_steps_sc, dt_sc, 
      temperature_sc, mass_sc, gamma_sc, beta_sc)

  lr = jopt.polynomial_decay(0.3, opt_steps_landscape, 0.01)
  optimizer = jopt.adam(lr)


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
    true_E.append(energy_sivak_plot(pos_vec[j])-39)
  plt.plot(positions, true_E, label = "True Landscape")

  for num, energies in enumerate(landscapes):
    plt.plot(positions, energies, label = f"Iteration {num}")

  plt.plot(positions, landscapes[-1], label = "Final Landscape")
  plt.legend()
  plt.xlabel("Position (x)")
  plt.ylabel("Free Energy (G)")
  plt.title("Iteratively Reconstructing Landscape")
  plt.savefig("./data/ac_plots/reconstruct_landscapes.png")
  
  with open("./data/ac_pkl/coeffs.pkl", "wb") as f:
    pickle.dump(coeffs, f)
