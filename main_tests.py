import time
import sys

import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from jax import random 

from jax_md import quantity, space

from jax.experimental import optimizers as jopt

from barrier_crossing.energy import V_biomolecule_geiger
from barrier_crossing.simulate import simulate_brownian_harmonic, batch_simulate_harmonic
from barrier_crossing.protocol import linear_chebyshev_coefficients, make_trap_fxn, make_trap_fxn_rev
from barrier_crossing.optimize import optimize_protocol, estimate_gradient_fwd



def test_geiger_simulate():
  simulation_steps = 1000
  N = 1
  dim = 1
  beta = 1.
  mass = 1.
  gamma = 1.0
  
  r0_init = 0.
  r0_final = 2.
  init_position = r0_init*jnp.ones((N,dim))
  
  dt = 1e-6
  Neq = 100
  _, shift_fn = space.free()

  key = random.PRNGKey(int(time.time()))
  key, split = random.split(key, 2)  

  trap_coeffs = linear_chebyshev_coefficients(r0_init,r0_final,simulation_steps, degree = 12, y_intercept = r0_init)
  trap_fn = make_trap_fxn(jnp.arange(simulation_steps), trap_coeffs, r0_init, r0_final)
  simulate_fn_fwd = lambda energy_fn, keys: simulate_brownian_harmonic(
    energy_fn,
    init_position,
    trap_fn, simulation_steps, Neq,
    shift_fn,
    keys,
    dt,
    temperature = 1/beta,
    mass = mass,
    gamma = gamma)


  batch_size = 1000
  energy_fn = V_biomolecule_geiger(k_s = 0.4, epsilon = 1., sigma = 1.)

  tot_works, (trajectories, works) = batch_simulate_harmonic(batch_size, energy_fn, simulate_fn_fwd, trap_fn, simulation_steps, key)
  print("average work done in moving the particle: ",jnp.mean(tot_works))
  
def test_fwd_opt():
  
  # Make linear chebyshev
  
  end_time = 0.01
  dt = 1e-6
  simulation_steps = int(end_time/dt)
  
  N = 1
  dim = 1
  beta = 1.
  mass = 1.
  gamma = 1.0
  
  r0_init = 0.
  r0_final = 2.
  init_position = r0_init * jnp.ones((N,dim))
  
  eq_time = 0.01
  Neq = int(eq_time / dt)
  
  _, shift_fn = space.free()

  key = random.PRNGKey(int(time.time()))
  key, split = random.split(key, 2)  

  trap_coeffs = linear_chebyshev_coefficients(r0_init,r0_final,simulation_steps, degree = 12, y_intercept = r0_init)
  trap_fn = make_trap_fxn(jnp.arange(simulation_steps), trap_coeffs, r0_init, r0_final)
  
  batch_size = 100
  opt_steps = 10
  
  lr = jopt.exponential_decay(0.1, opt_steps, 0.001)
  optimizer = jopt.adam(lr)

  energy_fn = V_biomolecule_geiger(k_s = 0.4, epsilon = 1., sigma = 1.)
  
  grad_fxn = lambda num_batches: estimate_gradient_fwd(num_batches, energy_fn, init_position, r0_init, r0_final, Neq, shift_fn, simulation_steps, dt, 1/beta, mass, gamma)
  
  
  optimize_protocol(trap_coeffs, grad_fxn, optimizer, batch_size, opt_steps)