import time
import sys

import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from jax import random 

from jax_md import quantity, space

import jax.example_libraries.optimizers as jopt

from barrier_crossing.energy import V_biomolecule_geiger
from barrier_crossing.simulate import simulate_brownian_harmonic, batch_simulate_harmonic
from barrier_crossing.protocol import linear_chebyshev_coefficients, make_trap_fxn, make_trap_fxn_rev
from barrier_crossing.optimize import optimize_protocol, estimate_gradient_fwd
from barrier_crossing.iterate_landscape import optimize_landscape

def test_geiger_simulate():
  """ Simulate moving a Brownian Particle moving across a double-well landscape
   as described by 2010 Geiger and Dellago. Parameters are currently arbitrary
  """

  N = 1 # Number of particles.
  dim = 1 # Number of dimensions that the particle can move in.
  
  dt = 1e-6 # Time-step
  Neq = 100 # Number of steps we use to equilibrate the particle. 
  simulation_steps = 1000 # Number of time steps to run the simulation for.
  
  r0_init = 0. # Initial position of the trap
  r0_final = 2. # Final position of the trap
  init_position = r0_init*jnp.ones((N,dim)) # Initial position of the particle
  
  beta = 1. # 1/temperature
  mass = 1. # mass of the particle (UNITS?)
  gamma = 1. # friction coefficient (see brownian simulator)
  
  _, shift_fn = space.free() # Describes the simulation bounding/periodic conditions. Free means no boundaries.

  key = random.PRNGKey(int(time.time())) # Set-up RNG
  key, split = random.split(key, 2)  
  
  # Describe linear movement of the harmonic trap from r0_init to r0_final
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

  # Run `batch_size` number of simulations with JAX-optimized looping
  total_works, (trajectories, works, log_probs) = batch_simulate_harmonic(batch_size, energy_fn, simulate_fn_fwd, trap_fn, simulation_steps, key)
  
  print("average work done in moving the particle: ",jnp.mean(total_works))
  
def test_fwd_opt():
  """ Optimize a set of coefficients for the first 12 Chebyshev polynomials to find
  the protocol that minimizes the amount of work used to move a brownian particle
  over a 2010 Geiger and Dellago double-well landscape. """

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

def test_opt_landscape():
  end_time = 0.01
  dt = 1e-4
  simulation_steps = int(end_time/dt)
  
  N = 1
  dim = 1
  beta = 1.
  mass = 1.
  gamma = 1.0
  k_s = 1.
  epsilon = 1.
  sigma = 1.
  
  r0_init = 0.
  r0_final = 2.
  init_position = r0_init * jnp.ones((N,dim))
  
  eq_time = 0.01
  Neq = int(eq_time / dt)
  
  _, shift_fn = space.free()

  key = random.PRNGKey(int(time.time()))
  key, split = random.split(key, 2)  

  energy_fn = V_biomolecule_geiger(k_s, epsilon, sigma)

  init_trap_coeffs = linear_chebyshev_coefficients(r0_init,r0_final,simulation_steps, degree = 12, y_intercept = r0_init)
  init_trap_fn = make_trap_fxn(jnp.arange(simulation_steps), init_trap_coeffs, r0_init, r0_final)
  
  batch_size = 500
  opt_steps = 10
  
  lr = jopt.exponential_decay(0.1, opt_steps, 0.001)
  optimizer = jopt.adam(lr)

  simulate_fn = lambda energy_fn, keys: simulate_brownian_harmonic(energy_fn,
    init_position, init_trap_fn, simulation_steps, Neq, shift_fn, keys, dt,
     temperature = 1/beta, mass = mass, gamma = gamma)
  
  # =========== LANDSCAPE OPTIMIZATION LOOP ==============
  
  max_iter = 5
  bins = 10
  
  grad_no_E = lambda num_batches, energy_fn: estimate_gradient_fwd(num_batches, energy_fn, init_position, r0_init, r0_final, Neq, shift_fn, simulation_steps, dt, 1/beta, mass, gamma)
  
  landscapes, coeffs, positions = optimize_landscape(energy_fn,
                     simulate_fn,
                     init_trap_fn,
                     init_trap_coeffs,
                     grad_no_E,
                     key,
                     max_iter,
                     bins,
                     simulation_steps,
                     batch_size,
                     opt_steps, optimizer,
                     r0_init, r0_final,
                     k_s, beta)

  positions = jnp.array(positions)
    
  plt.figure(figsize = (10,10))
  true_E = []
  pos_vec = jnp.reshape(positions, (positions.shape[0], 1, 1))
  for j in range(positions.shape[0]):
    true_E.append(energy_fn(pos_vec[j]))
  plt.plot(positions, true_E, label = "True Landscape")

  for num, energies in enumerate(landscapes):
    plt.plot(positions, energies, label = f"Iteration {num}")
  
  plt.plot(positions, landscapes[-1], label = "Final Landscape")
  plt.legend()
  plt.xlabel("Position (x)")
  plt.ylabel("Free Energy (G)")
  
  plt.show()

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