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


def plot_with_stddev(x, label=None, n=1, axis=0, ax=plt):
  stddev = jnp.std(x, axis)
  mn = jnp.mean(x, axis)

  ax.fill_between(jnp.arange(mn.shape[0]),
                  mn + n * stddev, mn - n * stddev, alpha=.3)
  ax.plot(mn, '-o', label=label)

def plot_optimization_fwd(num_optimization_steps, all_works, coeffs_, simulation_steps, r0_init, r0_final, save_filepath = None):
  #plot work distns 
  plt.figure(figsize=[12, 12])
  for j in range(opt_steps):
    if(j%100 == 0):
      work_dist = all_works[j,:]
      plt.hist(work_dist,10,label=f'Step {j}')

  plt.legend()

  ##### PLOT LOSS AND SCHEDULE EVOLUTION #####
  _, (ax0, ax1) = plt.subplots(1, 2, figsize=[24, 12])
  plot_with_stddev(all_works.T, ax=ax0)
  #ax0.set_ylim([0.1,1.0])
  #ax0.set_xlim([0,200])
  ax0.set_title('Total work')

  trap_fn = make_trap_fxn(jnp.arange(simulation_steps), init_coeffs, r0_init, r0_final)
  init_sched = trap_fn(jnp.arange(simulation_steps))
  ax1.plot(jnp.arange(simulation_steps), init_sched, label='initial guess')

  for j, coeffs in coeffs_:
    #if(j%50 == 0):
    trap_fn = make_trap_fxn(jnp.arange(simulation_steps),coeffs,r0_init,r0_final)
    full_sched = trap_fn(jnp.arange(simulation_steps))
    ax1.plot(jnp.arange(simulation_steps), full_sched, '-', label=f'Step {j}')

  #plot final estimate:
  trap_fn = make_trap_fxn(jnp.arange(simulation_steps),coeffs_[-1][1],r0_init,r0_final)
  full_sched = trap_fn(jnp.arange(simulation_steps))
  ax1.plot(jnp.arange(simulation_steps), full_sched, '-', label=f'Final')

  ax1.legend()#
  ax1.set_title('Schedule evolution')
  if save_filepath != None:
    plt.savefig(save_filepath+ "forward_optimization.png")
  plt.show()


def plot_simulations():
  """DEPRECATED
  ###plots """

  #what the underlying 'molecular' potential looks like:

  x=jnp.linspace(-4,10,200)
  xvec=jnp.reshape(jnp.linspace(-4,10,200), [200,1,1])
  k_splot = 0.
  Vfn = V_biomolecule(0, 0, 0, 0, k_splot, beta, epsilon, sigma) # returns in pN nm
  V = []
  for j in range(len(xvec)):
    V.append(Vfn(xvec[j], r0=0.))
  plt.figure(figsize=(10,10))
  plt.plot(x,V,'-o')
  plt.savefig(save_filepath+ "potential.png")
  plt.show()
  ####PLOT RESULTS FORWARD#####

  _, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=[30, 12])
  ax0.plot(dt*1000*jnp.arange(simulation_steps),trap_fn(jnp.arange(simulation_steps)), '-o')
  ax0.set(xlabel="time (ms)")
  ax0.set_title('Initial trap schedule')



  for j in range(batch_size):
      if j % 1000 == 0:
        ax1.plot(dt*1000*jnp.arange(simulation_steps), trajectories[j][:,0,0])
  #ax1.legend()#
  ax1.set(xlabel="time (ms)")
  ax1.set_title('Particle positions')


  for j in range(batch_size):
    if j % 1000 == 0:
      ax2.plot(dt*1000*jnp.arange(simulation_steps)+1, works[j], '-o')
  #ax2.plot(dt*1000*jnp.arange(simulation_steps)+1, summary[1], '-o')
  ax2.set(xlabel="time (ms)")
  ax2.set_title('Energy increments')
  plt.savefig(save_filepath+ "forward_sim.png")
  plt.show()

  ####PLOT RESULTS BACKWARD#####
  back_sim_steps = jnp.flip(jnp.arange(simulation_steps))
  _, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=[30, 12])
  ax0.plot(dt*1000*back_sim_steps,trap_fn(jnp.flip(back_sim_steps)), '-o')
  ax0.set(xlabel="time (ms)")
  ax0.set_title('Initial backwards trap schedule')



  for j in range(batch_size):
      if j % 1000 == 0:
        ax1.plot(dt*1000*back_sim_steps, trajectories[j][:,0,0])
  #ax1.legend()#
  ax1.set(xlabel="time (ms)")
  ax1.set_title('Backward Particle positions')


  for j in range(batch_size):
    if j % 1000 == 0:
      ax2.plot(dt*1000*back_sim_steps+1, works[j], '-o')
  #ax2.plot(dt*1000*jnp.arange(simulation_steps)+1, summary[1], '-o')
  ax2.set(xlabel="time (ms)")
  ax2.set_title('Backward Energy increments')
  plt.savefig(save_filepath+ "backward_sim.png")
  plt.show()

  # ##### PLOT WORK DISTRIBUTION #####
  # plt.figure(figsize=[12, 12])
  # # plt.hist(jnp.array(tot_works)*beta,20,alpha=1.0,color='g')

  # plt.xlabel("Work (kbT)")
  # plt.ylabel("counts")
  # plt.legend()
  # plt.savefig(save_filepath+ "USELESSwork_distribution.png")
  # plt.show()
  # print("forward mean:", jnp.mean(jnp.array(tot_works)*beta), "kbT")