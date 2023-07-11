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

import math

import matplotlib.pyplot as plt

def plot_with_stddev(x, label=None, n=1, axis=0, ax=plt, dt=1.):
  stddev = jnp.std(x, axis)
  mn = jnp.mean(x, axis)
  xs = jnp.arange(mn.shape[0]) * dt

  ax.fill_between(xs,
                  mn + n * stddev, mn - n * stddev, alpha=.3)
  ax.plot(xs, mn, label=label)

if __name__ == "__main__":

  N = 1
  dim = 1

  # Harmonic Trap Parameters S&C
  k_s_sc = 0.4 # stiffness; 
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
  delta_E=1.0 #pN nm
  #kappa_l=21.3863/(beta_sc*x_m**2) #pN/nm #for Ebarrier = 10kT and delta_E=0, as C&S use
  kappa_l=6.38629/(beta_sc*x_m**2) #pN/nm #for Ebarrier = 2.5kT and delta_E=0, as C&S use
  #kappa_l=2.6258/(beta_sc*x_m**2)#barrier 0.625kT
  kappa_r=kappa_l #pN/nm; Symmetric wells.

  energy_sivak = bc_energy.V_biomolecule_sivak(kappa_l, kappa_r, x_m, delta_E, k_s_sc, beta_sc)

  end_time_sc = 1e-6
  dt_sc = 5e-10
  simulation_steps_sc = int(end_time_sc / dt_sc)

  Neq = 500
  batch_size = 10000 #test
  opt_steps = 5  #test

  # Start with linear coefficients describing two halves of the landscape
  r_cut = 0
  step_cut = 1000
  _, shift = space.free()
  coeffs1 = jnp.array(bc_protocol.linear_chebyshev_coefficients(r0_init_sc, r_cut, step_cut, degree = 12, y_intercept = -10))
  trap1 = bc_protocol.make_trap_fxn(jnp.arange(step_cut),coeffs1, r0_init_sc, r_cut)
  coeffs2 = jnp.array(bc_protocol.linear_chebyshev_coefficients(r_cut, r0_final_sc, simulation_steps_sc - step_cut, degree = 12, y_intercept = 0))
  trap2 = bc_protocol.make_trap_fxn(jnp.arange(simulation_steps_sc - step_cut),coeffs2, r_cut, r0_final_sc)
  
  lr = jopt.exponential_decay(0.3, opt_steps, 0.03)
  optimizer = jopt.adam(lr)

  simulate_langevin_notrap_rev = lambda trap_fn, keys: bc_simulate.simulate_langevin_harmonic(
    energy_sivak,
    init_position_rev_sc,
    trap_fn,
    simulation_steps_sc,
    Neq,
    shift,
    keys,
    dt_sc,
    temperature_sc, mass_sc, gamma_sc # These parameters describe the state of the brownian system.
    )
  
  coeffs1_optimized, coeffs2_optimized = bc_optimize.optimize_protocol_split(simulate_langevin_notrap_rev,coeffs1, coeffs2, r0_init_sc, r0_final_sc, r_cut, init_position_rev_sc,
                            step_cut, optimizer, batch_size, opt_steps, energy_sivak, Neq,
                            shift, dt_sc, temperature_sc,mass_sc, gamma_sc, beta_sc,simulation_steps_sc, save_path = None)

  with open('testing/langevin/coeffs1_split_optimized_cut1000_dE3_ks04.txt', 'w') as f:
    f.write(str(coeffs1_optimized))

  with open('testing/langevin/coeffs2_split_optimized_cut1000_dE3_ks04.txt', 'w') as f:
    f.write(str(coeffs2_optimized))
