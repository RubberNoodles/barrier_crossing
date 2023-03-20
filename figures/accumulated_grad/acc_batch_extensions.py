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

def plot_with_stddev(x, label=None, n=1, axis=0, ax=plt, dt=1.):
  stddev = jnp.std(x, axis)
  mn = jnp.mean(x, axis)
  xs = jnp.arange(mn.shape[0]) * dt

  ax.fill_between(xs,
                  mn + n * stddev, mn - n * stddev, alpha=.3)
  ax.plot(xs, mn, label=label)


if __name__ == "__main__":
  path = f"./data/ext_{int(sys.argv[1])}/"
  # Protocol Coefficients
  lin_coeffs_sc = bc_protocol.linear_chebyshev_coefficients(r0_init_sc, r0_final_sc, simulation_steps_sc, degree = 12, y_intercept = r0_init_sc)

  # Trap Functions. Reverse mode trap functions are for when we compute Jarzynski error with reverse protocol trajectories.
  trap_fn_fwd_sc = bc_protocol.make_trap_fxn(jnp.arange(simulation_steps_sc), lin_coeffs_sc, r0_init_sc, r0_final_sc)
  trap_fn_rev_sc = bc_protocol.make_trap_fxn_rev(jnp.arange(simulation_steps_sc), lin_coeffs_sc, r0_init_sc, r0_final_sc)
  
  with open("extensions.csv", 'r') as f:
    reader = csv.reader(f)
    count = 1
    for line in reader:
      if count == int(sys.argv[1]):
        extensions = [int(x) for x in line]
      count+=1
  
  grad_acc_rev = lambda num_batches: bc_optimize.estimate_gradient_acc_rev_trunc(
    extensions,
    num_batches,
    energy_sivak,
    init_position_rev_sc,
    r0_init_sc,
    r0_final_sc,
    Neq,
    shift,
    simulation_steps_sc,
    dt_sc,
    temperature_sc,
    mass_sc,
    gamma_sc,
    beta_sc)

  batch_size = 10000 # Number of simulations/trajectories simulated. GPU optimized.
  opt_steps = 500 # Number of gradient descent steps to take.

  lr = jopt.polynomial_decay(1., opt_steps, 0.001)
  optimizer = jopt.adam(lr)

  coeffs, summaries, losses = bc_optimize.optimize_protocol(lin_coeffs_sc, grad_acc_rev, optimizer, batch_size, opt_steps, print_log=True)
  
  _, ax = plt.subplots(1, 2, figsize=[16, 8])
  #avg_loss = jnp.mean(jnp.array(losses), axis = 0)
  plot_with_stddev(losses.T, ax=ax[0])

  # ax[0].set_title(f'Jarzynski Error over Optimization; Short trap; STD error sampling; {batch_size}; {opt_steps}.')
  ax[0].set_title(f'Jarzynski Accumulated Error; {batch_size}; {opt_steps}; {simulation_steps_sc} steps; {extensions}')
  ax[0].set_xlabel('Number of Optimization Steps')
  ax[0].set_ylabel('Error')

  trap_fn = bc_protocol.make_trap_fxn(jnp.arange(simulation_steps_sc), lin_coeffs_sc, r0_init_sc, r0_final_sc)
  init_sched = trap_fn(jnp.arange(simulation_steps_sc))
  ax[1].plot(jnp.arange(simulation_steps_sc), init_sched, label='Initial guess')

  for i, (_, coeff) in enumerate(coeffs):
    if i % 5 == 0 and i!=0:
      trap_fn = bc_protocol.make_trap_fxn(jnp.arange(simulation_steps_sc),coeff,r0_init_sc,r0_final_sc)
      full_sched = trap_fn(jnp.arange(simulation_steps_sc))
      ax[1].plot(jnp.arange(simulation_steps_sc), full_sched, '-', label=f'Step {i}')

  # Plot final estimate:
  trap_fn = bc_protocol.make_trap_fxn(jnp.arange(simulation_steps_sc), coeffs[-1][1],r0_init_sc,r0_final_sc)
  full_sched = trap_fn(jnp.arange(simulation_steps_sc))
  ax[1].plot(jnp.arange(simulation_steps_sc), full_sched, '-', label=f'Final')


  ax[1].legend()#
  ax[1].set_title('Schedule evolution')

  plt.savefig(path+"optimization.png")
  
  with open(path+"opt_coeffs.pkl", "wb") as f:
    pickle.dump(coeffs, f)
  
  with open(path+"opt_losses.pkl", "wb") as f:
    pickle.dump(losses, f)
