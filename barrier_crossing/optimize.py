
import functools
import pickle
import time
import tqdm
import logging

import matplotlib.pyplot as plt

import jax
import jax_md

import jax.numpy as jnp
from jax import random
from jax.experimental import optimizers as jopt

from jax_md import quantity, space

from barrier_crossing.energy import V_biomolecule_geiger
from barrier_crossing.protocol import linear_chebyshev_coefficients, make_trap_fxn, make_trap_fxn_rev
from barrier_crossing.simulate import simulate_brownian_harmonic

def single_estimate_fwd(energy_fn,
                        init_position, r0_init, r0_final,
                        Neq,
                        shift,
                        simulation_steps, dt,
                        temperature, mass, gamma):
  @functools.partial(jax.value_and_grad, has_aux=True) #the 'aux' is the summary
  def _single_estimate(coeffs, seed): 
      # Forward direction to compute the gradient of the work used.
      trap_fn = make_trap_fxn(jnp.arange(simulation_steps), coeffs, r0_init, r0_final)
      positions, log_probs, works = simulate_brownian_harmonic(
          energy_fn, 
          init_position,
          trap_fn,
          simulation_steps,
          Neq, shift, seed, 
          dt, temperature, mass, gamma
          )
      total_work = works.sum()
      tot_log_prob = log_probs.sum()
      summary = (positions, tot_log_prob, total_work)
      
      gradient_estimator = (tot_log_prob * jax.lax.stop_gradient(total_work) + total_work) 
      
      return gradient_estimator, summary
  return _single_estimate 

def estimate_gradient_fwd(batch_size,
                          energy_fn,
                          init_position, r0_init, r0_final,
                          Neq,
                          shift,
                          simulation_steps, dt,
                          temperature, mass, gamma):
  """Compute the total gradient with forward trajectories and loss based on work used.
    Returns:
      Callable with inputs ``coeffs`` as the Chebyshev coefficients that specify the protocol, and
      ``seed`` to set rng."""
  
  mapped_estimate = jax.vmap(single_estimate_fwd(energy_fn, init_position, r0_init, r0_final, Neq, shift, simulation_steps, dt, temperature, mass, gamma), [None, 0])

  @jax.jit #why am I allowed to jit something whose output is a function? Thought it had to be pytree output...
  def _estimate_gradient(coeffs, seed):
    seeds = jax.random.split(seed, batch_size)
    (gradient_estimator, summary), grad = mapped_estimate(coeffs, seeds)
    return jnp.mean(grad, axis=0), (gradient_estimator, summary)
  return _estimate_gradient


def single_estimate_rev(energy_fn,
                        init_position, r0_init, r0_final,
                        Neq, shift,
                        simulation_steps, dt,
                        temperature, mass, gamma, beta):
  @functools.partial(jax.value_and_grad, has_aux = True)
  def _single_estimate(coeffs, seed):
    trap_fn = make_trap_fxn(jnp.arange(simulation_steps), coeffs, r0_init, r0_final)
    positions, log_probs, works = simulate_brownian_harmonic(
        energy_fn, 
        init_position, trap_fn,
        simulation_steps,
        Neq, shift, seed, 
        dt, temperature, mass, gamma
        )
    total_work = works.sum()
    tot_log_prob = log_probs.sum()
    summary = (positions, tot_log_prob, jnp.exp(beta*total_work))

    gradient_estimator = (tot_log_prob) * jax.lax.stop_gradient(jnp.exp(beta * total_work)) + jax.lax.stop_gradient(beta * jnp.exp(beta*total_work)) * total_work
    return gradient_estimator, summary
  return _single_estimate 

def estimate_gradient_rev(batch_size,
                          energy_fn,
                          init_position, r0_init, r0_final,
                          Neq, shift,
                          simulation_steps, dt,
                          temperature, mass, gamma, beta):
  """Find e^(beta ΔW) which is proportional to the error <(ΔF_(batch_size) - ΔF)> = variance of ΔF_(batch_size) (2010 Geiger and Dellago). 
  Compute the total gradient with forward trajectories and loss based on work used.
    Returns:
      Callable with inputs ``coeffs`` as the Chebyshev coefficients that specify the protocol, and
      ``seed`` to set rng."""
  mapped_estimate = jax.vmap(single_estimate_rev(energy_fn, init_position, r0_init, r0_final, Neq, shift, simulation_steps, dt, temperature, mass, gamma, beta), [None, 0])  
  @jax.jit 
  def _estimate_gradient(coeffs, seed):
    seeds = jax.random.split(seed, batch_size)
    (gradient_estimator, summary), grad = mapped_estimate(coeffs, seeds)
    return jnp.mean(grad, axis=0), (gradient_estimator, summary)
  return _estimate_gradient

def find_bin_timesteps(energy_fn, simulate_fn, rev_trap_fn, simulation_steps, key, bins):
  """Given a (reversed) protocol and set of bins, return an array of the average time it takes for the particle to reach the midpoint
  of each of these bins.
  
  Returns:
    Array[]
  """
  total_works, (trajectories, works) = batch_simulate_harmonic(batch_size,
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

def single_estimate_acc(energy_fn,
                        init_position, r0_init, r0_final,
                        Neq, shift,
                        simulation_steps, dt,
                        temperature, mass, gamma, beta,
                        bin_timesteps):
    @functools.partial(jax.value_and_grad, has_aux = True)
    def _single_estimate(coeffs, seed):
      trap_fn = make_trap_fxn(jnp.arange(simulation_steps), coeffs, r0_init, r0_final)
      positions, log_probs, works = simulate_brownian_harmonic(
          energy_fn, 
          init_position, trap_fn,
          simulation_steps,
          Neq, shift, seed, 
          dt, temperature, mass, gamma
          )
      gradient_estimator = 0.
      summary = [positions, [], []]
      for time_slice in bin_timesteps:
        log_prob = jax.lax.dynamic_slice(log_probs, (0,), (time_slice,)).sum()
        work = jax.lax.dynamic_slice(works, (0,), (time_slice,)).sum()
        gradient_estimator += log_prob * jax.lax.stop_gradient(jnp.exp(beta*work)) + \
                jax.lax.stop_gradient(beta * jnp.exp(beta * work)) * work
        
        summary[1].append(log_prob)
        summary[2].append(jnp.exp(beta*work))
      return gradient_estimator, summary
    return _single_estimate

def estimate_gradient_acc(batch_size,
                          energy_fn,
                          init_position, r0_init, r0_final,
                          Neq, shift,
                          simulation_steps, dt,
                          temperature, mass, gamma, beta,
                          bin_timesteps):
  """Estimate the error across an entire landscape based on the number of bins.
  
  Returns:
    Callable(Array[], jax.random RNG key)
  """
  mapped_estimate = jax.vmap(single_estimate_acc(energy_fn, init_position, r0_init, r0_final, Neq, shift, simulation_steps, dt, temperature, mass, gamma, beta, bin_timesteps), [None, 0])  
  def _estimate_gradient(coeffs, seed):
    seeds = jax.random.split(seed, batch_size)
    (gradient_estimator, summary), grad = mapped_estimate(coeffs, seeds)
    return jnp.mean(grad, axis=0), (gradient_estimator, summary)
  return _estimate_gradient

def optimize_protocol(init_coeffs, batch_grad_fn, optimizer, batch_size, num_steps, save_path = None):
  """ Training loop to optimize the coefficients of a chebyshev polynomial that defines the 
  protocol of moving a harmonic trap over a free energy landscape.
  
  Args:
    batch_grad_fn: Callable[Float] -> Callable[Array[Float], JaxRNG Key]
      Function that takes in the number of batches and outputs a function that computes the gradient
    by running `batch_size` number of simulations and finding the gradient of the loss with respect to
    the coefficients of the Chebyshev polynomials. 
      Currently used for Engel's work reduction (forward) and Geiger & Dellago 2010 Part IV. D. (reverse)
    
    # TODO: describe other self-explanatory args
  Returns:
    ``coeffs``: List of tuples of (``optimization_step``, ``protocol coefficients``)
    ``summaries``: List of outputs from batch_harmonic_simulator
    ``all_works``: List of lists of amount of work (ΔW) for each trajectory
  """
  summaries = []
  coeffs_ = []
  all_works = []
  
  key = jax.random.PRNGKey(int(time.time()))
  key, split = jax.random.split(key, 2)
  
  init_state = optimizer.init_fn(init_coeffs)
  opt_state = optimizer.init_fn(init_coeffs)
  coeffs_.append((0,) + (optimizer.params_fn(opt_state),))

  grad_fn = batch_grad_fn(batch_size)
  
  for j in tqdm.trange(num_steps,position=1, desc="Optimize Protocol: ", leave = True):
    key, split = jax.random.split(key)
    grad, (_, summary) = grad_fn(optimizer.params_fn(opt_state), split)

    opt_state = optimizer.update_fn(j, grad, opt_state)
    all_works.append(summary[2])
    if j % 100 == 0:
        coeffs_.append(((j+1),) + (optimizer.params_fn(opt_state),))
    if j == (num_steps-1):
        coeffs_.append((j+1,) + (optimizer.params_fn(opt_state),))
      
  coeffs_.append((num_steps,) + (optimizer.params_fn(opt_state),))

  logging.info("init parameters: ", optimizer.params_fn(init_state))
  logging.info("final parameters: ", optimizer.params_fn(opt_state))

  all_works = jax.tree_multimap(lambda *args: jnp.stack(args), *all_works)

  # Pickle coefficients and optimization outputs in order to recreate schedules for future use.
  if save_path != None:
    afile = open(save_path+'coeffs.pkl', 'wb')
    pickle.dump(coeffs_, afile)
    afile.close()

    bfile = open(save_path+'works.pkl', 'wb')
    pickle.dump(all_works, bfile)
    bfile.close()
    
  return coeffs_, summaries, all_works
