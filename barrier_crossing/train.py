import functools
import pickle
import time
import tqdm
import logging
import deprecation

import matplotlib.pyplot as plt

import jax
import jax_md

import jax.numpy as jnp
import numpy as onp

from jax import random
import jax.example_libraries.optimizers as jopt


from barrier_crossing.protocol import linear_chebyshev_coefficients, make_trap_fxn, make_trap_fxn_rev, trap_sum, trap_sum_rev
from barrier_crossing.simulate import simulate_brownian_harmonic, batch_simulate_harmonic
import barrier_crossing.loss as bc_loss
import barrier_crossing.models as bcm




def train(model: bcm.ScheduleModel, optimizer, batch_grad_fn, key, batch_size = 3000, num_epochs = 500): 
  
  state = optimizer.init_fn(model.coeffs)
  losses = []
  grad_fn = batch_grad_fn(batch_size)
  
  for j in tqdm.trange(num_epochs, desc = "Optimize Protocol: "):
    key, split = jax.random.split(key)
    grad, (_, summary) = grad_fn(optimizer.params_fn(state), split)
    
    state = optimizer.update_fn(j, grad, state)
    coeffs = optimizer.params_fn(state)
    
    model.coeffs = coeffs
    
    loss = summary[2]
    losses.append(loss)
  
  return jax.tree_util.tree_map(lambda *args: jnp.stack(args), *losses)

def find_error_samples(batch_size, energy_fn, simulate_fn, rev_trap_fn, simulation_steps, key, bins):
  """Given a (reversed) protocol and set of bins, return an array of the average time it takes for the particle to reach the midpoint
  of each of these bins.
  
  Returns:
    Array[]
  """
  total_works, (trajectories, works, log_probs) = batch_simulate_harmonic(batch_size,
                            energy_fn,
                            simulate_fn,
                            rev_trap_fn,
                            simulation_steps,
                            key)

  mean_traj = jnp.mean(trajectories, axis = 0)

  # Find timestep that mean trajectory is at the middle of each bin
  midpoint_timestep = []

  p_min = jnp.min(mean_traj)
  p_max = jnp.max(mean_traj)
  interval_len = (p_max - p_min)/bins

  midpoints = [ p_max - (bin_num + 0.5) * interval_len for bin_num in range(bins) ]

  time_step = 0

  for bin_num in range(bins):
    while float(mean_traj[time_step]) > midpoints[bin_num]:
      time_step = time_step + 1
    
    midpoint_timestep.append(time_step)

  midpoint_timestep = jnp.array(midpoint_timestep)
  return midpoint_timestep


optimze_protocol = train # For backwards compatibility