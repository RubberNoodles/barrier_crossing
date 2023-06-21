import time
import sys

import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from jax import random 

from jax_md import quantity, space

from barrier_crossing.energy import V_biomolecule_geiger, V_simple_spring, brownian, nvt_langevin
from barrier_crossing.protocol import linear_chebyshev_coefficients, make_trap_fxn, make_trap_fxn_rev

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
    positions: particle positions (trajectory) for each simulation step.
    log_probs: log probability of a particle being at each point on the trajectory.
    works: total work required to drag the particle, from eq'n 17 in Jarzynski 2008
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
  key, split = jax.random.split(key)  

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

def simulate_langevin_harmonic(energy_fn,
                               init_position,
                               trap_fn,
                               simulation_steps,
                               Neq,
                               shift,
                               key,
                               dt=1e-5,
                               temperature=1e-5, mass=1.0, gamma=1.0):
  """Simulation of LANGEVIN particle being dragged by a moving harmonic trap.
  Args:
    energy_fn: the function that governs particle energy. Here, an external harmonic potential
    r0_init: initial position of the trap, for which equilibration is done
    Neq: number of equilibration steps
    shift: shift_fn governing the LANGEVIN simulation
    key: random key
    num_steps: total # simulation steps
    dt: integration time step
    temperature: simulation temperature kT

  Returns:
    positions: particle positions (trajectory) for each simulation step.
    log_probs: log probability of a particle being at each point on the trajectory.
    works: total work required to drag the particle, from eq'n 17 in Jarzynski 2008
  """

  def equilibrate(init_state, Neq, apply, r0_init):
    @jax.jit
    def scan_eq(state, step):
      state = apply(state, r0=r0_init)
      return state, 0
    state, _ = jax.lax.scan(scan_eq,init_state,jnp.arange(Neq))
    # for step in jnp.arange(Neq):
    #   state, _ = scan_eq(state, step)
    return state

  def increment_work(state, step):
        return (energy_fn(state.position, r0=trap_fn(step)) - energy_fn(state.position, r0=trap_fn(step-1)))

  @jax.jit
  def scan_fn(state, step):
    dW = increment_work(state, step) #increment based on position BEFORE 'thermal kick' a la Crooks
    # Dynamically pass r0 to apply, which passes it on to energy_fn
    state = apply(state, r0=trap_fn(step))
    return state, [state.position, state.log_prob, dW]

  r0_init = trap_fn(0)
  key, split = jax.random.split(key)

  init, apply = nvt_langevin(energy_fn, shift, dt=dt, kT=temperature, gamma=gamma)

  eq_state = equilibrate(init(split, init_position, mass=mass), Neq, apply, r0_init)
  state = eq_state
  state, (positions, log_probs, works) = jax.lax.scan(scan_fn,state,(jnp.arange(simulation_steps-1)+1))

  works = jnp.concatenate([jnp.reshape(0., [1]), works])
  positions = jnp.concatenate([jnp.reshape(eq_state.position, [1,1,1]), positions])
  return positions, log_probs, works

def batch_simulate_harmonic(batch_size,
                            simulate_fn,
                            simulation_steps,
                            key,
                            memory_limit = None): 
  """Given trap and simulation functions, run code to simulate a brownian particle moved by trap
  in JAX optimized batches.
  Args:
    energy_fn: Callable(particle_position, r0) -> float. Gives the energy of a particle at a particular
      position + trap at position r0
    simulate_fn: Callable(Energy_fn, keys)
      -> final BrownianState, (Array[particle_position], Array[log probability], Array[work])
      Function simulating particle moving.
    batch_size: Integer specifying how many different trajectories to simulate.
    key: rng, jax.random.
    simulation_steps: Integer specifying number of steps to run the simulation for.
  Returns:
    Array[], Tuple(Array[Array[]], Array[Array[]], Array[])

    total_works: dissipative work for each trajectory
    trajectories: array of shape ``(batch_size, simulation_steps)`` with the particle positions
      for each trajectory.
    works: array of shape ``(batch_size, simulation_steps)`` containing works for each trajectory for each simulation step
    log_probs: Log probability of the trajectory.
  """
  # TODO: Find out why you cannot take the gradient of this function.

  key, split = jax.random.split(key)
  #see JAX-MD documentation for details on how these energy/force/displacement functions work:
  # force_fn = quantity.force(energy_fn) # This is not useful code right now.

  total_works = []

  # To generate a bunch of samples, we 'map' across seeds.
  mapped_sim = jax.vmap(lambda keys : simulate_fn(keys) ) 
  seeds = jax.random.split(split, batch_size)
  trajectories, log_probs, works = mapped_sim(seeds) #seed is array with diff seed for each run.

  total_works = jnp.sum(works, 1)
  
  return total_works, (trajectories, works, log_probs)