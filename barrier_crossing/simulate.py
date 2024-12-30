"""
Simulator functions following structure in JAX-MD, except including more parameters
to track during gradient descent.
"""
import jax
import jax.numpy as jnp
from typing import Callable, Union

from barrier_crossing.energy import brownian, nvt_langevin

def simulator(dynamics_simulator):
  """Simulation of particle being dragged by a moving harmonic trap. First equilibrate
    system with Neq simulation steps, and pull particle over energy/force landscape.

    Returns callable with the following specifications:
      Args:
        energy_fn: the function that governs particle energy.
        init_position: [[f32]], initial position of the particle
        trap_fn: Protocol for harmonic trap
        simulation_steps: total # simulation steps
        Neq: number of equilibration steps
        shift: shift_fn governing the Brownian simulation
        key: random key
        dt: integration time step
        temperature: simulation temperature in kT
        mass: mass of particle in g
        gamma: friction coefficient

      Returns:
        positions: particle positions (trajectory) for each simulation step.
        log_probs: log probability of a particle being at each point on the trajectory.
        works: total work required to drag the particle, from eq'n 17 in Jarzynski 2008
    """
  def _simulate(energy_fn,
                init_position,
                trap_fn,
                ks_trap_fn,
                simulation_steps,
                Neq,
                shift,
                key,
                dt=1e-5,
                temperature=1e-5, mass=1.0, gamma=1.0):
    
    if not isinstance(ks_trap_fn, Callable):
      # Constant trap
      assert isinstance(ks_trap_fn, float)
      _tmp = ks_trap_fn
      ks_trap_fn = lambda step: _tmp
    def equilibrate(init_state, Neq, apply, r0_init):
      @jax.jit
      def scan_eq(state, step):
        state = apply(state, step, k_s = ks_trap_fn(0), r0=r0_init)
        return state, 0
      state, _ = jax.lax.scan(scan_eq,init_state,jnp.arange(Neq))
      return state

    def increment_work(state, step):
          return (energy_fn(state.position, k_s = ks_trap_fn(step), r0=trap_fn(step)) - energy_fn(state.position, k_s = ks_trap_fn(step-1), r0=trap_fn(step-1)))

    @jax.jit
    def scan_fn(state, step):
      dW = increment_work(state, step) #increment based on position BEFORE 'thermal kick' a la Crooks
      # Dynamically pass r0 to apply, which passes it on to energy_fn
      state = apply(state, step, r0 = trap_fn(step), k_s = ks_trap_fn(step))
      return state, (state.position, state.log_prob, dW)

    r0_init = trap_fn(0)
    key, split = jax.random.split(key)

    init, apply = dynamics_simulator(energy_fn, shift, dt=dt, T_schedule=temperature, gamma=gamma)
    apply = jax.jit(apply)

    eq_state = equilibrate(init(split, init_position, mass=mass, k_s = ks_trap_fn(0)), Neq, apply, r0_init)
    state = eq_state
    state, (positions, log_probs, works) = jax.lax.scan(scan_fn,state,(jnp.arange(simulation_steps-1)+1))
    works = jnp.concatenate([jnp.reshape(0., [1]), works])
    positions = jnp.concatenate([jnp.reshape(eq_state.position, [1,1,1]), positions])
    return positions, log_probs, works
  
  return _simulate


def batch_simulate_harmonic(batch_size,
                            simulate_fn,
                            key,
                            memory_limit = None): 
  """Given trap and simulation functions, run code to simulate a particle moved by trap
  in JAX optimized batches.
  Args:
    energy_fn: Callable(particle_position, r0) -> float. Gives the energy of a particle at a particular
      position + trap at position r0
    simulate_fn: Callable(Energy_fn, keys)
      -> final ParticleState, (Array[particle_position], Array[log probability], Array[work])
      Function simulating particle moving.
    batch_size: Integer specifying how many different trajectories to simulate.
    key: rng, jax.random.
    memory_limit: WIP
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

  total_works = []

  # To generate a bunch of samples, we 'map' across seeds.
  mapped_sim = jax.vmap(lambda keys : simulate_fn(keys) ) 
  
  seeds = jax.random.split(split, batch_size)
  trajectories, log_probs, works = mapped_sim(seeds) #seed is array with diff seed for each run.

  total_works = jnp.sum(works, axis = 1)
  
  return total_works, (trajectories, works, log_probs)

### Fixed simulators for ease of use

simulate_brownian_harmonic = simulator(brownian)
simulate_langevin_harmonic = simulator(nvt_langevin)