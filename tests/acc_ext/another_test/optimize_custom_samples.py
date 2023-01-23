import time
import tqdm
import pickle
import functools
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

def init_trap_fn(x): # a "cubic" function for initial trap
    return (8*10**(-11))*(x-5000)**3

def mapper(x, min_x, max_x, min_to, max_to): #for cheb_coef
    return (x - min_x) / (max_x - min_x) * (max_to - min_to) + min_to

def cheb_coef(func, n, min, max): #returns chebyshev coefficients for a function
    coef = [0.0] * n
    for i in range(n):
        f = func(mapper(math.cos(math.pi * (i + 0.5) / n), -1, 1, min, max)) * 2 / n
        for j in range(n):
            coef[j] += f * math.cos(math.pi * j * (i + 0.5) / n)
    return jnp.array(coef)

def single_estimate_rev(energy_fn,
                        init_position, r0_init, r0_final,
                        Neq, shift,
                        simulation_steps, dt,
                        temperature, mass, gamma, beta):
  @functools.partial(jax.value_and_grad, has_aux = True)
  def _single_estimate(coeffs, seed):
    trap_fn = bc_protocol.make_trap_fxn_rev(jnp.arange(simulation_steps), coeffs, r0_init, r0_final)
    positions, log_probs, works = bc_simulate.simulate_brownian_harmonic(
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
    summary_total = []
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
      summary_total.append(summary)
      loss += summary[2]
    return grad_total, loss, gradient_estimator_total, summary_total
  return _estimate_grad


if __name__ == "__main__":

  N = 1
  dim = 1

  # Harmonic Trap Parameters S&C
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

  end_time_sc = 0.01
  dt_sc = 5e-6
  simulation_steps_sc = int(end_time_sc / dt_sc)

  Neq = 500
  batch_size = 10000 #test

  #init_coeffs = cheb_coef(init_trap_fn, 12, 0, 10000)
  #start from linear
  lin_coeffs = bc_protocol.linear_chebyshev_coefficients(r0_init_sc, r0_final_sc, simulation_steps_sc, degree = 12,
    y_intercept = r0_init_sc)
  init_coeffs = lin_coeffs

  _, shift = space.free() # Defines how to move a particle by small distances dR.
  # RNG
  key = random.PRNGKey(int(time.time()))
  key, split = random.split(key, 2)  

  #bins = 30 # how many samples on the landscape

  # uniform sampling:
  #error_samples = []
  #dr = simulation_steps_sc/bins
  #for i in range(bins):
  #  i += 1
  #error_samples.append(int(i * dr))

  # custom error samples (trying 6 most important points on landscape)
  bin_timesteps = [0, 0.1 * simulation_steps_sc, 0.2* simulation_steps_sc,simulation_steps_sc/4, 0.75* simulation_steps_sc, simulation_steps_sc]
  #custom extension samples
  error_samples = [-5,-3,-2,-1,0]


  batch_grad_fn = lambda num_batches: estimate_gradient_acc_rev_extensions_scale(
    error_samples,
    num_batches,
    energy_sivak,
    init_position_fwd_sc,
    r0_init_sc,
    r0_final_sc,
    Neq,
    shift,
    simulation_steps_sc,
    dt_sc,
    temperature_sc,
    mass_sc,
    gamma_sc, beta_sc)
  
  # Optimization:
  opt_steps = 300  #test

  lr = jopt.exponential_decay(0.3, opt_steps, 0.03)
  optimizer = jopt.adam(lr)

  summaries = []
  coeffs_ = []
  losses = []
  
  
  init_state = optimizer.init_fn(init_coeffs)
  opt_state = optimizer.init_fn(init_coeffs)
  coeffs_.append((0,) + (optimizer.params_fn(opt_state),))

  grad_fn = batch_grad_fn(batch_size)
  opt_coeffs = []
  for j in tqdm.trange(opt_steps,position=1, desc="Optimize Protocol: ", leave = True):
    key, split = jax.random.split(key)
    grad, loss, gradient_estimator_total, summary_total = grad_fn(optimizer.params_fn(opt_state), split)
    opt_state = optimizer.update_fn(j, grad, opt_state)
    losses.append(loss)
    coeffs_.append((j,) + (optimizer.params_fn(opt_state),))


  losses = jax.tree_map(lambda *args: jnp.stack(args), *losses)
  opt_coeffs.append(coeffs_[-1][1])
  
  # save to txt
  with open('coeffs_optimized_5points_firstwell_ks01_newgrad.txt', 'w') as f:
    f.write(str(opt_coeffs))

  _, ax = plt.subplots(1, 2, figsize=[24, 12])

  plot_with_stddev(losses.T, ax=ax[0])
  ax[0].set_title('Optimization')
  ax[0].set_xlabel('Number of Optimization Steps')
  ax[0].set_ylabel('Loss')

  trap_init = bc_protocol.make_trap_fxn(jnp.arange(simulation_steps_sc), init_coeffs, r0_init_sc,
                                     r0_final_sc,)
  ax[1].plot(jnp.arange(simulation_steps_sc), trap_init(jnp.arange(simulation_steps_sc)), label='Initial Guess')

  for j, coeff in coeffs_:
    if j%50 == 0 and j!=0:  
      trap_fn = bc_protocol.make_trap_fxn(jnp.arange(simulation_steps_sc),coeff,r0_init_sc,r0_final_sc)
      full_sched = trap_fn(jnp.arange(simulation_steps_sc))
      ax[1].plot(jnp.arange(simulation_steps_sc), full_sched, '-', label=f'Step {j}')

  # Plot final estimate:
  trap_fn = bc_protocol.make_trap_fxn(jnp.arange(simulation_steps_sc), coeffs_[-1][1],r0_init_sc,r0_final_sc)
  full_sched = trap_fn(jnp.arange(simulation_steps_sc))
  ax[1].plot(jnp.arange(simulation_steps_sc), full_sched, '-', label=f'Final')
  ax[1].set_title('Schedule evolution')

  ax[0].legend()
  ax[1].legend()
  plt.savefig("./optimization_5points_firstwell_ks01_newgrad.png")
