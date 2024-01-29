# Optimize coefficients for work and error distributions.
import pickle
import os

import barrier_crossing.protocol as bc_protocol
import barrier_crossing.optimize as bc_optimize
from barrier_crossing.utils import parse_args

import jax.numpy as jnp

import jax.example_libraries.optimizers as jopt

import matplotlib.pyplot as plt

# from figures.params import * # global variables;

def plot_with_stddev(x, label=None, n=1, axis=0, ax=plt, dt=1.):
  stddev = jnp.std(x, axis)
  mn = jnp.mean(x, axis)
  xs = jnp.arange(mn.shape[0]) * dt

  ax.fill_between(xs,
                  mn + n * stddev, mn - n * stddev, alpha=.3)
  ax.plot(xs, mn, label=label)

if __name__ == "__main__":
  args, p = parse_args()
  
  
  path = f"output_data/{args.landscape_name.replace(' ', '_').replace('.', '_').lower()}/"
  if not os.path.isdir(path):
    os.mkdir(path)
  

  sim_cut_steps = p.param_set.simulation_steps // 2
  
  lin_coeffs = bc_protocol.linear_chebyshev_coefficients(p.r0_init, p.r0_final, p.param_set.simulation_steps, degree = 12, y_intercept = p.r0_init)
  # Trap Functions. Reverse mode trap functions are for when we compute Jarzynski error with reverse protocol trajectories.
  trap_fn_fwd = bc_protocol.make_trap_fxn(jnp.arange(p.param_set.simulation_steps), lin_coeffs, p.r0_init, p.r0_final)
  trap_fn_rev = bc_protocol.make_trap_fxn_rev(jnp.arange(p.param_set.simulation_steps), lin_coeffs, p.r0_init, p.r0_final)
  
  coeffs_split_1 = jnp.array(bc_protocol.linear_chebyshev_coefficients(p.r0_init, p.r0_cut, sim_cut_steps, degree = 12, y_intercept = -10))
  coeffs_split_2 = jnp.array(bc_protocol.linear_chebyshev_coefficients(p.r0_cut, p.r0_final, p.param_set.simulation_steps - sim_cut_steps, degree = 12, y_intercept = 0))
  
  simulate_fwd_unf = lambda trap_fn, keys: p.param_set.simulate_fn(
    trap_fn, 
    keys, 
    regime = "brownian",
    fwd = True)
  
  simulate_rev_unf = lambda trap_fn, keys: p.param_set.simulate_fn(
    trap_fn, 
    keys, 
    regime = "brownian",
    fwd = False)
  
  simulate_split_unf = lambda simulation_steps: lambda trap_fn, keys: p.param_set.simulate_fn(
    trap_fn, 
    keys, 
    regime = "brownian",
    fwd = False,
    simulation_steps = simulation_steps)
  
  grad_fwd = lambda num_batches: bc_optimize.estimate_gradient_work(
    num_batches,
    simulate_fwd_unf,
    p.r0_init,
    p.r0_final,
    p.param_set.simulation_steps)
  
  grad_rev = lambda num_batches: bc_optimize.estimate_gradient_rev(
    num_batches,
    simulate_rev_unf,
    p.r0_init,
    p.r0_final,
    p.param_set.simulation_steps,
    p.beta)

  batch_size = 10000 # Number of simulations/trajectories simulated. GPU optimized.
  opt_steps = 200 # Number of gradient descent steps to take.
  
  lr = jopt.polynomial_decay(0.1, opt_steps, 0.001)
  optimizer = jopt.adam(lr)

  coeffs_work, summaries_work, losses_work = bc_optimize.optimize_protocol(lin_coeffs, grad_fwd, optimizer, batch_size, opt_steps)
  print(f"first 10 work coefficients: \n{coeffs_work[:10]}")
  coeffs_err, summaries_err, losses_err = bc_optimize.optimize_protocol(lin_coeffs, grad_rev, optimizer, batch_size, opt_steps)
  print(f"first 10 error coefficients: \n{coeffs_err[:10]}")
  
  coeffs_split = bc_optimize.optimize_protocol_split(
    simulate_split_unf, 
    coeffs_split_1, 
    coeffs_split_2, 
    p.r0_init, 
    p.r0_final, 
    p.r0_cut,
    sim_cut_steps, 
    optimizer, 
    batch_size, 
    opt_steps, 
    p.beta, 
    p.param_set.simulation_steps, 
    file_path = path)
  
  per_5 = int(opt_steps/5)

  _, ax_work = plt.subplots(1, 2, figsize=[16, 8])
  plot_with_stddev(losses_work.T, ax=ax_work[0])

  # ax_work[0].set_title(f'Jarzynski Error over Optimization; Short trap; STD error sampling; {batch_size}; {opt_steps}.')
  ax_work[0].set_title(f'Dissipative Work; {batch_size}; {opt_steps}; {p.param_set.simulation_steps}')
  ax_work[0].set_xlabel('Number of Optimization Steps')
  ax_work[0].set_ylabel('Error')

  trap_fn = bc_protocol.make_trap_fxn(jnp.arange(p.param_set.simulation_steps), lin_coeffs, p.r0_init, p.r0_final)
  inithed = trap_fn(jnp.arange(p.param_set.simulation_steps))
  ax_work[1].plot(jnp.arange(p.param_set.simulation_steps), inithed, label='Initial guess')

  for i, (_, coeff) in enumerate(coeffs_work):
    if i% per_5 == 0 and i!=0:
      trap_fn = bc_protocol.make_trap_fxn(jnp.arange(p.param_set.simulation_steps),coeff,p.r0_init,p.r0_final)
      fullhed = trap_fn(jnp.arange(p.param_set.simulation_steps))
      ax_work[1].plot(jnp.arange(p.param_set.simulation_steps), fullhed, '-', label=f'Step {i}')

  # Plot final estimate:
  trap_fn = bc_protocol.make_trap_fxn(jnp.arange(p.param_set.simulation_steps), coeffs_work[-1][1],p.r0_init,p.r0_final)
  fullhed = trap_fn(jnp.arange(p.param_set.simulation_steps))
  ax_work[1].plot(jnp.arange(p.param_set.simulation_steps), fullhed, '-', label=f'Final')


  ax_work[1].legend()#
  ax_work[1].set_title(f'{args.landscape_name} Schedule Evolution')

  plt.savefig(path+"work_optimization.png")
  
  _, ax_err = plt.subplots(1, 2, figsize=[16, 8])
  plot_with_stddev(losses_err.T, ax=ax_err[0])

  # ax_err[0].set_title(f'Jarzynski Error over Optimization; Short trap; STD error sampling; {batch_size}; {opt_steps}.')
  ax_err[0].set_title(f'Jarzynski Error; {batch_size}; {opt_steps}; {p.param_set.simulation_steps} steps')
  ax_err[0].set_xlabel('Number of Optimization Steps')
  ax_err[0].set_ylabel('Error')

  trap_fn = bc_protocol.make_trap_fxn(jnp.arange(p.param_set.simulation_steps), lin_coeffs, p.r0_init, p.r0_final)
  inithed = trap_fn(jnp.arange(p.param_set.simulation_steps))
  ax_err[1].plot(jnp.arange(p.param_set.simulation_steps), inithed, label='Initial guess')

  for i, (_, coeff) in enumerate(coeffs_err):
    if i % per_5 == 0 and i!=0:
      trap_fn = bc_protocol.make_trap_fxn(jnp.arange(p.param_set.simulation_steps),coeff,p.r0_init,p.r0_final)
      fullhed = trap_fn(jnp.arange(p.param_set.simulation_steps))
      ax_err[1].plot(jnp.arange(p.param_set.simulation_steps), fullhed, '-', label=f'Step {i}')

  # Plot final estimate:
  trap_fn = bc_protocol.make_trap_fxn(jnp.arange(p.param_set.simulation_steps), coeffs_err[-1][1],p.r0_init,p.r0_final)
  fullhed = trap_fn(jnp.arange(p.param_set.simulation_steps))
  ax_err[1].plot(jnp.arange(p.param_set.simulation_steps), fullhed, '-', label=f'Final')


  ax_err[1].legend()#
  ax_err[1].set_title(f'{args.landscape_name} Schedule Evolution')

  plt.savefig(path + "error_optimization.png")

  with open(path + "work_coeffs.pkl", "wb") as f:
    pickle.dump(coeffs_work[-1][1], f)
  
  with open(path + "error_coeffs.pkl", "wb") as f:
    pickle.dump(coeffs_err[-1][1], f)
    
  with open(path + "split_coeffs.pkl", "wb") as f:
    pickle.dump(coeffs_split, f)
  
  print(p.param_set)