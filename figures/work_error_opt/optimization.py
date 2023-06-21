# Optimize coefficients for work and error distributions.
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

from figures.params import * # global variables;

def plot_with_stddev(x, label=None, n=1, axis=0, ax=plt, dt=1.):
  stddev = jnp.std(x, axis)
  mn = jnp.mean(x, axis)
  xs = jnp.arange(mn.shape[0]) * dt

  ax.fill_between(xs,
                  mn + n * stddev, mn - n * stddev, alpha=.3)
  ax.plot(xs, mn, label=label)

if __name__ == "__main__":
  path = "output_data/"

  lin_coeffs_sc = bc_protocol.linear_chebyshev_coefficients(r0_init_sc, r0_final_sc, simulation_steps_sc, degree = 12, y_intercept = r0_init_sc)
  # Trap Functions. Reverse mode trap functions are for when we compute Jarzynski error with reverse protocol trajectories.
  trap_fn_fwd_sc = bc_protocol.make_trap_fxn(jnp.arange(simulation_steps_sc), lin_coeffs_sc, r0_init_sc, r0_final_sc)
  trap_fn_rev_sc = bc_protocol.make_trap_fxn_rev(jnp.arange(simulation_steps_sc), lin_coeffs_sc, r0_init_sc, r0_final_sc)

  simulate_sivak_fwd_unf = lambda trap_fn, keys: bc_simulate.simulate_langevin_harmonic(
    energy_sivak, 
    init_position_fwd_sc, 
    trap_fn,
    simulation_steps_sc, 
    Neq, 
    shift, 
    keys, 
    dt_sc,
    temperature_sc, mass_sc, gamma_sc # These parameters describe the state of the brownian system.
    )

  simulate_sivak_rev_unf = lambda trap_fn, keys: bc_simulate.simulate_langevin_harmonic(
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
  
  grad_fwd = lambda num_batches: bc_optimize.estimate_gradient_work(
    num_batches,
    simulate_sivak_fwd_unf,
    r0_init_sc,
    r0_final_sc,
    simulation_steps_sc)
  
  grad_rev = lambda num_batches: bc_optimize.estimate_gradient_rev(
    num_batches,
    simulate_sivak_rev_unf,
    r0_init_sc,
    r0_final_sc,
    simulation_steps_sc,
    beta_sc)


  batch_size = 10000 # Number of simulations/trajectories simulated. GPU optimized.
  opt_steps = 10 # Number of gradient descent steps to take.
  
  lr = jopt.polynomial_decay(1., opt_steps, 0.001)
  optimizer = jopt.adam(lr)

  coeffs_work, summaries_work, losses_work = bc_optimize.optimize_protocol(lin_coeffs_sc, grad_fwd, optimizer, batch_size, opt_steps)

  coeffs_err, summaries_err, losses_err = bc_optimize.optimize_protocol(lin_coeffs_sc, grad_rev, optimizer, batch_size, opt_steps)

  per_5 = int(opt_steps/5)

  _, ax_work = plt.subplots(1, 2, figsize=[16, 8])
  plot_with_stddev(losses_work.T, ax=ax_work[0])

  # ax_work[0].set_title(f'Jarzynski Error over Optimization; Short trap; STD error sampling; {batch_size}; {opt_steps}.')
  ax_work[0].set_title(f'Dissipative Work; {batch_size}; {opt_steps}; {simulation_steps_sc}')
  ax_work[0].set_xlabel('Number of Optimization Steps')
  ax_work[0].set_ylabel('Error')

  trap_fn = bc_protocol.make_trap_fxn(jnp.arange(simulation_steps_sc), lin_coeffs_sc, r0_init_sc, r0_final_sc)
  init_sched = trap_fn(jnp.arange(simulation_steps_sc))
  ax_work[1].plot(jnp.arange(simulation_steps_sc), init_sched, label='Initial guess')

  for i, (_, coeff) in enumerate(coeffs_work):
    if i% per_5 == 0 and i!=0:
      trap_fn = bc_protocol.make_trap_fxn(jnp.arange(simulation_steps_sc),coeff,r0_init_sc,r0_final_sc)
      full_sched = trap_fn(jnp.arange(simulation_steps_sc))
      ax_work[1].plot(jnp.arange(simulation_steps_sc), full_sched, '-', label=f'Step {i}')

  # Plot final estimate:
  trap_fn = bc_protocol.make_trap_fxn(jnp.arange(simulation_steps_sc), coeffs_work[-1][1],r0_init_sc,r0_final_sc)
  full_sched = trap_fn(jnp.arange(simulation_steps_sc))
  ax_work[1].plot(jnp.arange(simulation_steps_sc), full_sched, '-', label=f'Final')


  ax_work[1].legend()#
  ax_work[1].set_title('Schedule evolution')

  plt.savefig(path+"work_optimization.png")
  
  _, ax_err = plt.subplots(1, 2, figsize=[16, 8])
  plot_with_stddev(losses_err.T, ax=ax_err[0])

  # ax_err[0].set_title(f'Jarzynski Error over Optimization; Short trap; STD error sampling; {batch_size}; {opt_steps}.')
  ax_err[0].set_title(f'Jarzynski Accumulated Error; {batch_size}; {opt_steps}; {simulation_steps_sc} steps')
  ax_err[0].set_xlabel('Number of Optimization Steps')
  ax_err[0].set_ylabel('Error')

  trap_fn = bc_protocol.make_trap_fxn(jnp.arange(simulation_steps_sc), lin_coeffs_sc, r0_init_sc, r0_final_sc)
  init_sched = trap_fn(jnp.arange(simulation_steps_sc))
  ax_err[1].plot(jnp.arange(simulation_steps_sc), init_sched, label='Initial guess')

  for i, (_, coeff) in enumerate(coeffs_err):
    if i % per_5 == 0 and i!=0:
      trap_fn = bc_protocol.make_trap_fxn(jnp.arange(simulation_steps_sc),coeff,r0_init_sc,r0_final_sc)
      full_sched = trap_fn(jnp.arange(simulation_steps_sc))
      ax_err[1].plot(jnp.arange(simulation_steps_sc), full_sched, '-', label=f'Step {i}')

  # Plot final estimate:
  trap_fn = bc_protocol.make_trap_fxn(jnp.arange(simulation_steps_sc), coeffs_err[-1][1],r0_init_sc,r0_final_sc)
  full_sched = trap_fn(jnp.arange(simulation_steps_sc))
  ax_err[1].plot(jnp.arange(simulation_steps_sc), full_sched, '-', label=f'Final')


  ax_err[1].legend()#
  ax_err[1].set_title('Schedule evolution')

  plt.savefig(path+"error_optimization.png")

  with open(path+"work_coeffs.pkl", "wb") as f:
    pickle.dump(coeffs_work[-1][1], f)
  
  with open(path+"error_coeffs.pkl", "wb") as f:
    pickle.dump(coeffs_err[-1][1], f)
  