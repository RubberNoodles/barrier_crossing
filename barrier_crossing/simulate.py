import time
import sys

import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from jax import random 

from jax_md import quantity, space

from barrier_crossing.energy import V_biomolecule_geiger, V_simple_spring, brownian
from barrier_crossing.protocol import linear_chebyshev_coefficients, make_trap_fxn, make_trap_fxn_rev
"""#Simulations

##Brownian particle simulation

###Parameters:
"""

"""###Run Simulation:"""

def simulate_brownian_harmonic(energy_fn,
                               init_position,
                               trap_fn,
                               simulation_steps,
                               Neq,
                               shift,
                               key,
                               dt=1e-5,
                               temperature=1e-5, mass=1.0, gamma=1.0):
  """Simulation of Brownian particle being dragged by a moving harmonic trap. Previously called run_brownian_opt
  Args:
    energy_fn: the function that governs particle energy. Here, an external harmonic potential
    r0_init: initial position of the trap, for which equilibration is done
    Neq: number of equilibration steps
    shift: shift_fn governing the Brownian simulation
    key: random key
    num_steps: total # simulation steps
    dt: integration time step 
    temperature: simulation temperature kT

  Returns:
    total work required to drag the particle, from eq'n 17 in Jarzynski 2008
  """

  def equilibrate(init_state, Neq, apply, r0_init):
    @jax.jit
    def scan_eq(state, step):
      state = apply(state, step, r0=r0_init)
      return state, 0
    state, _ = jax.lax.scan(scan_eq,init_state,jnp.arange(Neq))
    return state
    
  def increment_work(state, step):
        return (energy_fn(state.position, r0=trap_fn(step)) - energy_fn(state.position, r0=trap_fn(step-1)))

  @jax.jit
  def scan_fn(state, step):
    dW = increment_work(state, step) #increment based on position BEFORE 'thermal kick' a la Crooks
    # Dynamically pass r0 to apply, which passes it on to energy_fn
    state = apply(state, step, r0=trap_fn(step))
    return state, (state.position, state.log_prob, dW)

  r0_init = trap_fn(0)
  key, split = random.split(key)  

  init, apply = brownian(energy_fn, shift, dt=dt, T_schedule=temperature, gamma=gamma)
  apply = jax.jit(apply)

  eq_state = equilibrate(init(split, init_position, mass=mass), Neq, apply, r0_init)
  state = eq_state
  state, (positions, log_probs, works) = jax.lax.scan(scan_fn,state,(jnp.arange(simulation_steps-1)+1))
  #print("Log probability:")
  #print(log_probs)
  works = jnp.concatenate([jnp.reshape(0., [1]), works])
  positions = jnp.concatenate([jnp.reshape(eq_state.position, [1,1,1]), positions])
  return positions, log_probs, works


def batch_simulate_harmonic(batch_size,
                            energy_fn,
                            simulate_fn,
                            trap_fn,
                            simulation_steps,
                            key,
                            trap_coeffs = None, 
                            cheby_degree = 12 
                            ): 
  
  """ Makes a function to describe the trap schedule and runs the simulation code on it
  in JAX optimized batches.
  Args:
    energy_fn: Callable(particle_position, r0) -> float. Gives the energy of a particle at a particular
      position + trap at position r0
    simulate_fn: Callable(Energy_fn, trap_schedule) 
      -> final BrownianState, (Array[particle_position], Array[log probability], Array[work])
      Function that simulates moving the particle along the given trap_schedule given a specified
      energy function.
    batch_size: Integer specifying how many different trajectories to simulate.
    key: rng, jax.random.
    init_position: array specifying the initial position of the particles. Generally an array of shape
      (1,1).
    mass: float specifying mass of the particle.
    beta: float describing reciprocal temperature, equal to 1/K_bT.
    gamma: A float specifying the friction coefficient between the particles
      and the solvent.
    r0_init, r0_final: floats specifying initial and final trap positions.
    simulation_steps: Integer specifying number of steps to run the simulation for.
  Returns:
    Output of simulate_fn, each element as an array of length batch_size.
  """

  key, split = jax.random.split(key)
  #see JAX-MD documentation for details on how these energy/force/displacement functions work:
  # force_fn = quantity.force(energy_fn) # This is not useful code right now.
  _, shift_fn = space.free()

  total_works = []

  # To generate a bunch of samples, we 'map' across seeds.
  mapped_sim = jax.soft_pmap(lambda keys : simulate_fn(energy_fn, keys))
  seeds = jax.random.split(split, batch_size)
  trajectories, _, works = mapped_sim(seeds) #seed is array with diff seed for each run. I'm discarding the log prob data, here

  total_works = jnp.sum(works, 1)
  
  return total_works, (trajectories, works)

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

  ##### PLOT WORK DISTRIBUTION #####
  plt.figure(figsize=[12, 12])
  # plt.hist(jnp.array(tot_works)*beta,20,alpha=1.0,color='g')

  plt.xlabel("Work (kbT)")
  plt.ylabel("counts")
  plt.legend()
  plt.savefig(save_filepath+ "USELESSwork_distribution.png")
  plt.show()
  print("forward mean:", jnp.mean(jnp.array(tot_works)*beta), "kbT")

  """## Optimization of trap protocol

  ### Parameters
  """