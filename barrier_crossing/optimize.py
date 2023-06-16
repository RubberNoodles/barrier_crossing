
import functools
import pickle
import time
import tqdm
import logging

import matplotlib.pyplot as plt

import jax
import jax_md

import jax.numpy as jnp
import numpy as onp

from jax import random
import jax.example_libraries.optimizers as jopt

from jax_md import quantity, space

from barrier_crossing.energy import V_biomolecule_geiger
from barrier_crossing.protocol import linear_chebyshev_coefficients, make_trap_fxn, make_trap_fxn_rev, trap_sum, trap_sum_rev
from barrier_crossing.simulate import simulate_brownian_harmonic, batch_simulate_harmonic

def single_estimate_work(energy_fn,
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

def estimate_gradient_work(batch_size,
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
  
  mapped_estimate = jax.vmap(single_estimate_work(energy_fn, init_position, r0_init, r0_final, Neq, shift, simulation_steps, dt, temperature, mass, gamma), [None, 0])

  @jax.jit #why am I allowed to jit something whose output is a function? Thought it had to be pytree output...
  def _estimate_gradient(coeffs, seed):
    seeds = jax.random.split(seed, batch_size)
    (gradient_estimator, summary), grad = mapped_estimate(coeffs, seeds)
    return jnp.mean(grad, axis=0), (gradient_estimator, summary)
  return _estimate_gradient


def single_estimate_rev(simulate_fn_no_trap,
                        r0_init, r0_final, simulation_steps, 
                        beta):
  @functools.partial(jax.value_and_grad, has_aux = True)
  def _single_estimate(coeffs, seed):
    trap_fn = make_trap_fxn_rev(jnp.arange(simulation_steps), coeffs, r0_init, r0_final)
    positions, log_probs, works = simulate_fn_no_trap(trap_fn, seed)
    total_work = works.sum()
    tot_log_prob = log_probs.sum()
    summary = (positions, tot_log_prob, jnp.exp(beta*total_work))

    gradient_estimator = (tot_log_prob) * jax.lax.stop_gradient(jnp.exp(beta * total_work)) + jax.lax.stop_gradient(beta * jnp.exp(beta*total_work)) * total_work
    return gradient_estimator, summary
  return _single_estimate 

def estimate_gradient_rev(batch_size,
                          simulate_fn_no_trap,
                          r0_init, r0_final, simulation_steps, 
                          beta):
  """Find e^(beta ΔW) which is proportional to the error <(ΔF_(batch_size) - ΔF)> = variance of ΔF_(batch_size) (2010 Geiger and Dellago). 
  Compute the total gradient with forward trajectories and loss based on work used.
    Returns:
      Callable with inputs ``coeffs`` as the Chebyshev coefficients that specify the protocol, and
      ``seed`` to set rng."""
  mapped_estimate = jax.vmap(single_estimate_rev(simulate_fn_no_trap,
                          r0_init, r0_final, simulation_steps, 
                          beta), [None, 0])  
  @jax.jit 
  def _estimate_gradient(coeffs, seed):
    seeds = jax.random.split(seed, batch_size)
    (gradient_estimator, summary), grad = mapped_estimate(coeffs, seeds)
    return jnp.mean(grad, axis=0), (gradient_estimator, summary)
  return _estimate_gradient


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

def estimate_gradient_acc_rev_extensions(error_samples,batch_size,
                          energy_fn,
                          init_position, r0_init, r0_final,
                          Neq, shift,
                          simulation_steps, dt,
                          temperature, mass, gamma, beta):
  """ 
  DEPRECATED; DO NOT USE.
  
  New version of estimate_gradient_acc_rev, which takes
  samples of extensions instead of time steps in simulation.
  ToDo: change returns, so it's compatible with optimize_protocol
  """
  def _estimate_grad(coeffs, seed):
    grad_total = jnp.zeros(len(coeffs))
    gradient_estimator_total = []
    summary_total = []
    loss = 0.0
    for r in error_samples:
      grad_func = estimate_gradient_rev(batch_size,
                          energy_fn,
                          init_position, r, r0_final,
                          Neq, shift,
                          simulation_steps, dt,
                          temperature, mass, gamma, beta)
      grad, (gradient_estimator, summary) = grad_func(coeffs, seed)
      grad_total += grad
      gradient_estimator_total.append(gradient_estimator)
      summary_total.append(summary)
      loss += summary[2]
    return grad_total, loss, gradient_estimator_total, summary_total
  return _estimate_grad

def estimate_gradient_acc_rev_extensions_scale(error_samples,batch_size,
                          energy_fn,
                          init_position, r0_init, r0_final,
                          Neq, shift,
                          simulation_steps, dt,
                          temperature, mass, gamma, beta):
  """ 
  New version of estimate_gradient_acc_rev, which takes
  samples of extensions instead of time steps in simulation.
  ToDo: change returns, so it's compatible with optimize_protocol

  """
  def _estimate_grad(coeffs, seed):
    grad_total = jnp.zeros(len(coeffs))
    gradient_estimator_total = []
    summary_total = ([], [], 0.)
    loss = 0.0
    for r in error_samples:
      scale = (r0_final - r) / (r0_final - r0_init)
      coeffs_r = onp.array(coeffs) * scale
      coeffs_r[0] += r - r0_init
      coeffs_r = jnp.array(coeffs_r) # TODO: Test if this works.
      grad_func = estimate_gradient_rev(batch_size,
                          energy_fn,
                          init_position, r, r0_final,
                          Neq, shift,
                          simulation_steps, dt,
                          temperature, mass, gamma, beta)
      grad, (gradient_estimator, summary) = grad_func(coeffs_r, seed)
      grad_total += grad
      gradient_estimator_total.append(gradient_estimator)
      summary_total[0].append(summary[0])
      summary_total[1].append(summary[1])
      summary_total[2] += summary[2]
    return grad_total, (gradient_estimator_total, summary_total)
  return _estimate_grad

def optimize_protocol(init_coeffs, batch_grad_fn, optimizer, batch_size, num_steps, save_path = None, print_log = False):
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
    ``summaries``: List of outputs from batch_harmonic_simulator. Currently empty due to RAM constraints
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
    if print_log:
      # Only works when using accumulated gradient.
      print(f"\n Opt step {j} extension positions.")
      for extension in range(len(summary[0])):
        print(f"Number of steps: {jnp.squeeze(summary[0][extension]).shape[1]}")
        print(jnp.squeeze(summary[0][extension]).mean(axis=0)[-1])
    opt_state = optimizer.update_fn(j, grad, opt_state)
    all_works.append(summary[2])
      
    coeffs_.append((j,) + (optimizer.params_fn(opt_state),))

  logging.info("init parameters: ", optimizer.params_fn(init_state))
  logging.info("final parameters: ", optimizer.params_fn(opt_state))

  all_works = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *all_works)

  # Pickle coefficients and optimization outputs in order to recreate schedules for future use.
  if save_path != None:
    afile = open(save_path+'coeffs.pkl', 'wb')
    pickle.dump(coeffs_, afile)
    afile.close()

    bfile = open(save_path+'works.pkl', 'wb')
    pickle.dump(all_works, bfile)
    bfile.close()
    
  return coeffs_, summaries, all_works



#### TRUNCATED CODE


def single_estimate_rev_trunc(energy_fn,
                        init_position, r0_init, r0_final,
                        Neq, shift,
                        simulation_steps, dt,
                        temperature, 
                        mass, gamma, beta,
                        truncated_steps):
  @functools.partial(jax.value_and_grad, has_aux = True)
  def _single_estimate(coeffs, seed):
    trap_fn = make_trap_fxn_rev(jnp.arange(simulation_steps), coeffs, r0_init, r0_final)
    positions, log_probs, works = simulate_brownian_harmonic(
        energy_fn, 
        init_position, trap_fn,
        truncated_steps,
        Neq, shift, seed, 
        dt, temperature, mass, gamma
        )
    total_work = works.sum()
    tot_log_prob = log_probs.sum()
    summary = (positions, tot_log_prob, jnp.exp(beta*total_work))

    gradient_estimator = (tot_log_prob) * jax.lax.stop_gradient(jnp.exp(beta * total_work)) + jax.lax.stop_gradient(beta * jnp.exp(beta*total_work)) * total_work
    return gradient_estimator, summary
  return _single_estimate 

def estimate_gradient_rev_trunc(batch_size,
                          energy_fn,
                          init_position, r0_init, r0_final,
                          Neq, shift,
                          simulation_steps, dt,
                          temperature, mass, gamma, beta,
                          truncated_steps):
  """Find e^(beta ΔW) which is proportional to the error <(ΔF_(batch_size) - ΔF)> = variance of ΔF_(batch_size) (2010 Geiger and Dellago). 
  Compute the total gradient with forward trajectories and loss based on work used.
    Returns:
      Callable with inputs ``coeffs`` as the Chebyshev coefficients that specify the protocol, and
      ``seed`` to set rng."""
  mapped_estimate = jax.vmap(single_estimate_rev_trunc(energy_fn, init_position, r0_init, r0_final, Neq, shift, simulation_steps, dt, temperature, mass, gamma, beta, truncated_steps), [None, 0])  
  @jax.jit 
  def _estimate_gradient(coeffs, seed):
    seeds = jax.random.split(seed, batch_size)
    (gradient_estimator, summary), grad = mapped_estimate(coeffs, seeds)
    return jnp.mean(grad, axis=0), (gradient_estimator, summary)
  return _estimate_gradient

def estimate_gradient_acc_rev_trunc(error_samples,batch_size,
                          energy_fn,
                          init_position, r0_init, r0_final,
                          Neq, shift,
                          simulation_steps, dt,
                          temperature, mass, gamma, beta):
  """ 
  New version of estimate_gradient_acc_rev, which takes
  samples of extensions instead of time steps in simulation.
  ToDo: change returns, so it's compatible with optimize_protocol

  """
  def _estimate_grad(coeffs, seed):
    trap_fn = make_trap_fxn_rev(jnp.arange(simulation_steps), coeffs, r0_init, r0_final)
    simulate_fn = lambda energy_fn, key: simulate_brownian_harmonic(
          energy_fn, 
          init_position, trap_fn,
          simulation_steps,
          Neq, shift, key, 
          dt, temperature, mass, gamma
          )
    batch_size_ext = 5000
    total_works, (batch_trajectories, batch_works, batch_log_probs) = batch_simulate_harmonic(
          batch_size_ext, 
          energy_fn, 
          simulate_fn, 
          simulation_steps, 
          seed)
    mean_trajectory = jnp.mean(batch_trajectories, axis = 0) 
    grad_total = jnp.zeros(len(coeffs))
    
    gradient_estimator_total = []
    summary_total = [[], []]
    loss = jnp.zeros(batch_size)
    
    for r in error_samples:
      key, seed = jax.random.split(seed)
      extension_positions = jnp.where(mean_trajectory < r)[0]
      if extension_positions.any():
        r_step = int(extension_positions[0])
      else:
        r_step = simulation_steps
      if r_step < 50: # Prevent Pathological examples.
        r_step = simulation_steps
      grad_func = estimate_gradient_rev_trunc(batch_size,
                          energy_fn,
                          init_position, r, r0_final,
                          Neq, shift,
                          simulation_steps, dt,
                          temperature, mass, gamma, beta,
                          r_step)
      grad, (gradient_estimator, summary) = grad_func(coeffs, seed)

      grad_total += grad
      gradient_estimator_total.append(gradient_estimator)
      summary_total[0].append(summary[0])
      summary_total[1].append(summary[1])
      loss += summary[2]
    summary_total.append(loss)
    return grad_total, (gradient_estimator_total, summary_total)
  return _estimate_grad

def single_estimate_rev_split(energy_fn,
                        init_position, r0_init, r0_final, r_cut,
                        Neq, shift,
                        simulation_steps,step_cut, dt,
                        temperature, mass, gamma, beta):
  """
  New gradient for the optimization that splits the original 
  trap functions into two parts. 
  """
  @functools.partial(jax.value_and_grad, has_aux = True)
  def _single_estimate(coeffs_for_opt, coeffs_leave,seed):
    trap_leave = make_trap_fxn_rev(jnp.arange(step_cut), coeffs_leave, r0_init, r_cut)
    trap_opt = make_trap_fxn_rev(jnp.arange(simulation_steps - step_cut), coeffs_for_opt, r_cut, r0_final)
    trap_fn = trap_sum_rev(jnp.arange(simulation_steps),simulation_steps, step_cut,trap_leave, trap_opt)
    positions, log_probs, works = simulate_brownian_harmonic(energy_fn, init_position, trap_fn, 
                                                             simulation_steps, Neq, shift, seed, dt, temperature, mass, gamma)
    total_work = works.sum()
    tot_log_prob = log_probs.sum()
    summary = (positions, tot_log_prob, jnp.exp(beta*total_work))

    gradient_estimator = (tot_log_prob) * jax.lax.stop_gradient(jnp.exp(beta * total_work)) + jax.lax.stop_gradient(beta * jnp.exp(beta*total_work)) * total_work
    return gradient_estimator, summary
  return _single_estimate

def estimate_gradient_rev_split(batch_size,
                          energy_fn,
                          init_position, r0_init, r0_final, r_cut,
                          Neq, shift,
                          simulation_steps,step_cut, dt,
                          temperature, mass, gamma, beta):
  mapped_estimate = jax.vmap(single_estimate_rev_split(energy_fn, init_position, r0_init, r0_final,r_cut, Neq, shift, simulation_steps, step_cut, dt, temperature, mass, gamma, beta), [None,None, 0])  
  #@jax.jit
  def _estimate_gradient(coeffs_for_opt,coeffs_leave, seed):
    seeds = jax.random.split(seed, batch_size)
    (gradient_estimator, summary), grad = mapped_estimate(coeffs_for_opt,coeffs_leave, seeds)
    return jnp.mean(grad, axis=0), (gradient_estimator, summary)
  return _estimate_gradient

def plot_with_stddev(x, label=None, n=1, axis=0, ax=plt, dt=1.):
  stddev = jnp.std(x, axis)
  mn = jnp.mean(x, axis)
  xs = jnp.arange(mn.shape[0]) * dt

  ax.fill_between(xs,
                  mn + n * stddev, mn - n * stddev, alpha=.3)
  ax.plot(xs, mn, label=label)

def optimize_protocol_split(coeffs1, coeffs2, r0_init, r0_final, r_cut, init_position_rev,
                            step_cut, optimizer, batch_size, opt_steps, energy_fn, Neq,
                            shift, dt, temperature,mass, gamma, beta,simulation_steps, save_path = None):
  """
  New version of a training loop to optimize the chebyshev polynomial that defines the 
  protocol of moving a harmonic trap over a free energy landscape. Two sets of coefficients 
  that define part of the protocol before r_cut and after r_cut are optimized to minimize
  the Jarzynski error at extensions equal to r_cut and r0_final. 
  Outputs coeffs1_optimized, coeffs2_optimized and plots the two optimizations.

  """
  key = jax.random.PRNGKey(int(time.time()))
  key, split = jax.random.split(key, 2)
  
  # Optimize first part (coeffs1)
  
  batch_grad_fn1= lambda batch_size: estimate_gradient_rev(batch_size, energy_fn, init_position_rev, r0_init, r_cut, Neq, shift, step_cut, dt, temperature,
                                     mass, gamma, beta)
  coeffs1_optimize = []
  losses1 = []
  init_state1 = optimizer.init_fn(coeffs1)
  opt_state = optimizer.init_fn(coeffs1)
  coeffs1_optimize.append((0,) + (optimizer.params_fn(opt_state),))

  grad_fn = batch_grad_fn1(batch_size)
  opt_coeffs = []
  for j in tqdm.trange(opt_steps,position=1, desc="Optimize Protocol: ", leave = True):
    key, split = jax.random.split(key)
    grad, (_, summary) = grad_fn(optimizer.params_fn(opt_state), split)
    opt_state = optimizer.update_fn(j, grad, opt_state)
    losses1.append(summary[2])
    coeffs1_optimize.append((j,) + (optimizer.params_fn(opt_state),))
    logging.info("init parameters 1 : ", optimizer.params_fn(init_state1))
    logging.info("final parameters 1 : ", optimizer.params_fn(opt_state))

  losses1 = jax.tree_map(lambda *args: jnp.stack(args), *losses1)
  coeffs1_final = coeffs1_optimize[-1][1]
  
  # Plot optimization results 
  _, ax = plt.subplots(1, 2, figsize=[24, 12])

  plot_with_stddev(losses1.T, ax=ax[0])
  ax[0].set_title('Optimization Part 1')
  ax[0].set_xlabel('Number of Optimization Steps')
  ax[0].set_ylabel('Loss')

  trap_init = make_trap_fxn(jnp.arange(step_cut), coeffs1, r0_init,
                                     r_cut)
  ax[1].plot(jnp.arange(step_cut), trap_init(jnp.arange(step_cut)), label='Initial Guess')

  for j, coeff in coeffs1_optimize:
    if j%50 == 0 and j!=0:  
      trap_fn = make_trap_fxn(jnp.arange(step_cut),coeff,r0_init,r_cut)
      full_sched = trap_fn(jnp.arange(step_cut))
      ax[1].plot(jnp.arange(step_cut), full_sched, '-', label=f'Step {j}')

  trap_fn = make_trap_fxn(jnp.arange(step_cut), coeffs1_optimize[-1][1],r0_init,r_cut)
  full_sched = trap_fn(jnp.arange(step_cut))
  ax[1].plot(jnp.arange(step_cut), full_sched, '-', label=f'Final')
  ax[1].set_title('Schedule evolution')

  ax[0].legend()
  ax[1].legend()
  plt.savefig("./optimization1.png")

  # Optimize second part (coeffs2)
  batch_grad_fn2= lambda batch_size: estimate_gradient_rev_split(batch_size,energy_fn, init_position_rev, r0_init, r0_final, r_cut, Neq, shift, simulation_steps,step_cut, dt, temperature,
                                     mass, gamma, beta)
  init_coeffs = coeffs2
  coeffs_leave = coeffs1_final

  summaries2 = []
  coeffs2_optimize = []
  losses2 = []
  
  key = jax.random.PRNGKey(int(time.time()))
  key, split = jax.random.split(key, 2)
  
  init_state = optimizer.init_fn(init_coeffs)
  opt_state = optimizer.init_fn(init_coeffs)
  coeffs2_optimize.append((0,) + (optimizer.params_fn(opt_state),))

  grad_fn = batch_grad_fn2(batch_size)
  for j in tqdm.trange(opt_steps,position=1, desc="Optimize Protocol: ", leave = True):
    key, split = jax.random.split(key)
    grad, (_, summary) = grad_fn(optimizer.params_fn(opt_state),coeffs_leave, split)

    opt_state = optimizer.update_fn(j, grad, opt_state)
    losses2.append(summary[2])
      
    coeffs2_optimize.append((j,) + (optimizer.params_fn(opt_state),))

    logging.info("init parameters 2 : ", optimizer.params_fn(init_state))
    logging.info("final parameters 2 : ", optimizer.params_fn(opt_state))

  losses2 = jax.tree_map(lambda *args: jnp.stack(args), *losses2)
  coeffs2_final = coeffs2_optimize[-1][1]

  # Plot Second Optimization Results
  _, ax = plt.subplots(1, 2, figsize=[24, 12])

  plot_with_stddev(losses2.T, ax=ax[0])
  ax[0].set_title('Optimization Part 2')
  ax[0].set_xlabel('Number of Optimization Steps')
  ax[0].set_ylabel('Loss')

  trap_init = make_trap_fxn(jnp.arange(step_cut), coeffs2, r_cut,
                                     r0_final)
  ax[1].plot(jnp.arange(simulation_steps-step_cut), trap_init(jnp.arange(simulation_steps-step_cut)), label='Initial Guess')

  for j, coeff in coeffs2_optimize:
    if j%50 == 0 and j!=0:  
      trap_fn = make_trap_fxn(jnp.arange(simulation_steps - step_cut),coeff,r_cut,r0_final)
      full_sched = trap_fn(jnp.arange(simulation_steps - step_cut))
      ax[1].plot(jnp.arange(simulation_steps - step_cut), full_sched, '-', label=f'Step {j}')

  trap_fn = make_trap_fxn(jnp.arange(simulation_steps - step_cut), coeffs2_optimize[-1][1],r_cut, r0_final)
  full_sched = trap_fn(jnp.arange(simulation_steps - step_cut))
  ax[1].plot(jnp.arange(simulation_steps-step_cut), full_sched, '-', label=f'Final')
  ax[1].set_title('Schedule evolution')

  ax[0].legend()
  ax[1].legend()
  plt.savefig("./optimization2.png")

  return coeffs1_final, coeffs2_final

