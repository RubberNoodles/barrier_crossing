import time
import sys
import pickle
import os

import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from jax import random 

from jax_md import quantity, space

import jax.example_libraries.optimizers as jopt

from barrier_crossing.energy import V_biomolecule_geiger
from barrier_crossing.simulate import simulate_brownian_harmonic, batch_simulate_harmonic
from barrier_crossing.protocol import linear_chebyshev_coefficients, make_trap_fxn, make_trap_fxn_rev
from barrier_crossing.optimize import optimize_protocol, estimate_gradient_work, estimate_gradient_rev


def geiger_work_opt(task_num, batch_size, opt_steps):

  """ Optimize a set of coefficients for the first 12 Chebyshev polynomials to find
  the protocol that minimizes the amount of work used to move a brownian particle
  over a 2010 Geiger and Dellago double-well landscape. """

  save_filepath = f"geiger_output/epsilon_{task_num/2}/"
  if os.path.exists(save_filepath) == False:
    os.makedirs(save_filepath)
  N = 1
  dim = 1
  beta = 1.0
  k_s = 1.0 #stiffness
  sigma = 1.0/jnp.sqrt(beta * k_s)
  mass = 1.0
  gamma = 1.0
  epsilon = task_num/2 * (1.0/beta)
  
  end_time = 1.0
  dt = 0.002
  simulation_steps = int(end_time/dt)
  
  r0_init = 0.
  r0_final = 2.0 * sigma
  init_position = r0_init * jnp.ones((N,dim))
  
  eq_time = 6.
  Neq = int(eq_time / dt)
  
  _, shift_fn = space.free()

  key = random.PRNGKey(int(time.time()))
  key, split = random.split(key, 2)  

  trap_coeffs = linear_chebyshev_coefficients(r0_init,r0_final,simulation_steps, degree = 12, y_intercept = r0_init)
  trap_fn = make_trap_fxn(jnp.arange(simulation_steps), trap_coeffs, r0_init, r0_final)
  
  lr = jopt.exponential_decay(0.1, opt_steps, 0.01)
  optimizer = jopt.adam(lr)

  energy_fn = V_biomolecule_geiger(k_s, epsilon, sigma)
  
  grad_fxn = lambda num_batches: estimate_gradient_work(num_batches, energy_fn, init_position, r0_init, r0_final, Neq, shift_fn, simulation_steps, dt, 1/beta, mass, gamma)
  
  coeffs_, summaries, all_works = optimize_protocol(trap_coeffs, grad_fxn, optimizer, batch_size, opt_steps, save_filepath + "forward_")
  
  # Simulate the optimized protocol

  trap_fn_optimized = make_trap_fxn(jnp.arange(simulation_steps), coeffs_[-1][1], r0_init, r0_final)
  
  simulate_fn_fwd = lambda energy_fn, keys: simulate_brownian_harmonic(energy_fn,
    init_position, trap_fn_optimized, simulation_steps, Neq, shift_fn, keys, dt, temperature = 1/beta, mass = mass, gamma = gamma)

  # REDO SIMULATE FN
  total_works_forward_opt, _ = batch_simulate_harmonic(batch_size, energy_fn, simulate_fn_fwd, simulation_steps, 
  key)

  afile = open(save_filepath + 'works_forward_opt.pkl', 'wb')
  pickle.dump(total_works_forward_opt, afile)
  afile.close()



def geiger_error_opt(task_num, batch_size, opt_steps):

  """ Optimize a set of coefficients for the first 12 Chebyshev polynomials to find
  the protocol that minimizes the error in the Jarzynski equality obtained in experiments 
  where a brownian particle is moved
  over a 2010 Geiger and Dellago double-well landscape. """

  save_filepath = f"geiger_output/epsilon_{task_num/2}/"
  if os.path.exists(save_filepath) == False:
    os.makedirs(save_filepath)
    
  N = 1
  dim = 1
  beta = 1.0
  temperature = 1.0
  k_s = 1.0 #stiffness
  sigma = 1.0/jnp.sqrt(beta * k_s)
  mass = 1.0
  gamma = 1.0
  epsilon = task_num/2 * (1.0/beta)
  
  end_time = 1.0
  dt = 0.002
  simulation_steps = int(end_time/dt)
  
  r0_init = 0.
  r0_final = 2.0 * sigma
  init_position = r0_final * jnp.ones((N,dim))
  
  eq_time = 6.
  Neq = int(eq_time / dt)
  
  _, shift_fn = space.free()

  key = random.PRNGKey(int(time.time()))
  key, split = random.split(key, 2)  

  trap_coeffs = linear_chebyshev_coefficients(r0_init,r0_final,simulation_steps, degree = 12, y_intercept = r0_init)
  trap_fn = make_trap_fxn_rev(jnp.arange(simulation_steps), trap_coeffs, r0_init, r0_final)
  
  
  lr = jopt.exponential_decay(0.1, opt_steps, 0.01)
  optimizer = jopt.adam(lr)

  energy_fn = V_biomolecule_geiger(k_s, epsilon, sigma)
  
  grad_fxn = lambda num_batches: estimate_gradient_rev(num_batches, energy_fn, init_position, r0_init, r0_final, Neq, shift_fn, simulation_steps, dt, temperature, beta, mass, gamma)
  
  optimize_protocol(trap_coeffs, grad_fxn, optimizer, batch_size, opt_steps, save_path= save_filepath + "reverse_")
  
  coeffs_, summaries, all_works = optimize_protocol(trap_coeffs, grad_fxn, optimizer, batch_size, opt_steps, save_filepath)
  
  # Simulate the optimized protocol

  trap_fn_optimized = make_trap_fxn(jnp.arange(simulation_steps), coeffs_[-1][1], r0_init, r0_final)
  
  simulate_fn = lambda energy_fn, keys: simulate_brownian_harmonic(energy_fn,
    r0_init*jnp.ones((N,dim)), trap_fn_optimized, simulation_steps, Neq, shift_fn, keys, dt, temperature = 1/beta, mass = mass, gamma = gamma)
  # REDO SIMULATE FN
  total_works_reverse_opt, _ = batch_simulate_harmonic(batch_size, energy_fn, simulate_fn, simulation_steps, 
  key)

  afile = open(save_filepath + 'works_reverse_opt.pkl', 'wb')
  pickle.dump(total_works_reverse_opt, afile)
  afile.close()



def get_linear_works(task_num, batch_size):
  """ Finds the works obtained for  experiments where a brownian particle is moved
  over a 2010 Geiger and Dellago double-well landscape by a linear protocol/
  """

  save_filepath = f"geiger_output/epsilon_{task_num/2}/"
  if os.path.exists(save_filepath) == False:
    os.makedirs(save_filepath)
    
  N = 1
  dim = 1
  beta = 1.0
  temperature = 1.0
  k_s = 1.0 #stiffness
  sigma = 1.0/jnp.sqrt(beta * k_s)
  mass = 1.0
  gamma = 1.0
  epsilon = task_num/2 * (1.0/beta)
  
  end_time = 1.0
  dt = 0.002
  simulation_steps = int(end_time/dt)
  
  r0_init = 0.
  r0_final = 2.0 * sigma
  init_position = r0_final * jnp.ones((N,dim))
  
  eq_time = 6.
  Neq = int(eq_time / dt)
  
  _, shift_fn = space.free()

  key = random.PRNGKey(int(time.time()))
  key, split = random.split(key, 2)  

  trap_coeffs = linear_chebyshev_coefficients(r0_init,r0_final,simulation_steps, degree = 12, y_intercept = r0_init)
  trap_fn = make_trap_fxn(jnp.arange(simulation_steps), trap_coeffs, r0_init, r0_final)

  energy_fn = V_biomolecule_geiger(k_s, epsilon, sigma)
  simulate_fn = lambda energy_fn, keys: simulate_brownian_harmonic(energy_fn,
    init_position, trap_fn, simulation_steps, Neq, shift_fn, keys, dt, temperature = 1/beta, mass = mass, gamma = gamma)
  # REDO SIMULATE FN
  total_works_linear, _ = batch_simulate_harmonic(batch_size, energy_fn, simulate_fn, simulation_steps, 
  key)

  afile = open(save_filepath + 'works_linear.pkl', 'wb')
  pickle.dump(total_works_linear, afile)
  afile.close()
  
  



  













