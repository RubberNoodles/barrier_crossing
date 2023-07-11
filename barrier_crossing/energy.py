import collections
from typing import Any, Callable, TypeVar, Union, Tuple, Dict, Optional

import jax.numpy as jnp
from jax import random, lax, jit

from jax_md import dataclasses, simulate, quantity, util, interpolate, space
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

from jax_md import util

from barrier_crossing.protocol import make_trap_fxn, make_trap_fxn_rev


# Typing

Array = util.Array
f32 = util.f32
f64 = util.f64

Box = space.Box

ShiftFn = space.ShiftFn

T = TypeVar('T')
InitFn = Callable[..., T]
ApplyFn = Callable[[T], T]
Simulator = Tuple[InitFn, ApplyFn]

"""##Custom Brownian simulator

This is a modification of the JAX-MD Brownian simulator that also returns the log probability of any trajectory that runs. This is needed in order to compute gradients correctly (eq'n 12 in my arXiv paper)
"""
@dataclasses.dataclass
class BrownianState:
  position: Array
  mass: float
  rng: Array
  log_prob: float

def brownian(energy_or_force,
             shift,
             dt,
             T_schedule,
             gamma=0.1):
  """Simulation of Brownian dynamics.
  This code simulates Brownian dynamics which are synonymous with the overdamped
  regime of Langevin dynamics. However, in this case we don't need to take into
  account velocity information and the dynamics simplify. Consequently, when
  Brownian dynamics can be used they will be faster than Langevin. As in the
  case of Langevin dynamics our implementation follows [1].
  Args:
    energy_or_force: A function that produces either an energy or a force from
      a set of particle positions specified as an ndarray of shape
      [n, spatial_dimension].
    shift: A function that displaces positions, R, by an amount dR. Both R
      and dR should be ndarrays of shape [n, spatial_dimension].
    dt: Floating point number specifying the timescale (step size) of the
      simulation.
    T_schedule: Either a floating point number specifying a constant temperature
      or a function specifying temperature as a function of time.
    gamma: A float specifying the friction coefficient between the particles
      and the solvent.
  Returns:
    Tuple of `init_fn` to initialize the simulator and `apply_fn` to move orward one step
      in time.
    [1] E. Carlon, M. Laleman, S. Nomidis. "Molecular Dynamics Simulation."
        http://itf.fys.kuleuven.be/~enrico/Teaching/molecular_dynamics_2015.pdf
        Accessed on 06/05/2019.
  """

  force_fn = quantity.canonicalize_force(energy_or_force)

  dt, gamma = util.static_cast(dt, gamma)

  T_schedule = interpolate.canonicalize(T_schedule)

  def _dist(state, t, **kwargs):
    nu = jnp.float32(1) / (state.mass * gamma)
    F = force_fn(state.position, t=t, **kwargs)
    mean =  F * nu * dt
    variance = jnp.float32(2) * T_schedule(t) * dt * nu
    return tfd.Normal(mean, jnp.sqrt(variance))
  
  def init_fn(key, R, mass=jnp.float32(1)):
    state = BrownianState(R, mass, key, 0.)
    state = simulate.canonicalize_mass(state)
    return state

  def apply_fn(state, t=jnp.float32(0), **kwargs):
    dist = _dist(state, t, **kwargs)
    key, split = random.split(state.rng)

    # We have to stop gradients here, otherwise the gradient with respect to
    # energy/force is zero. The following is a simple repro of the issue:
    #  def sample_log_prob(mean, key):
    #    d = tfd.Normal(mean, 1.)
    #    s = d.sample(seed=key)
    #    return d.log_prob(s)
    #  jax.grad(sample_log_prob)(0., key)  # Always 0 !
    dR = lax.stop_gradient(dist.sample(seed=split))

    log_prob = dist.log_prob(dR).sum()
    R = shift(state.position, dR, t=t, **kwargs)
    return BrownianState(R, state.mass, key, log_prob)

  return init_fn, apply_fn


""" Custom Langevin Simulator

Modification of the Langevin simulator from JAX-MD==0.2.5 but includes the log 
probability of a movement occurring as a state variable.
"""

@dataclasses.dataclass
class NVTLangevinState:
  """
  
  A struct containing state information for the Langevin thermostat.

  Attributes:
    position: The current position of the particles. An ndarray of floats with
      shape `[n, spatial_dimension]`.
    momentum: The momentum of particles. An ndarray of floats with shape
      `[n, spatial_dimension]`.
    force: The (non-stochastic) force on particles. An ndarray of floats with
      shape `[n, spatial_dimension]`.
    mass: The mass of particles. Will either be a float or an ndarray of floats
      with shape `[n]`.
    rng: The current state of the random number generator.
    log_prob: The log_probability of being at the current position given system
      parameters
  """
  position: Array
  momentum: Array
  force: Array
  mass: Array
  rng: Array
  log_prob: float

  @property
  def velocity(self) -> Array:
    return self.momentum / self.mass


@simulate.dispatch_by_state
def stochastic_step(state: NVTLangevinState, dt:float, kT: float, gamma: float):
  """A single stochastic step (the `O` step)."""
  c1 = jnp.exp(-gamma * dt)
  c2 = jnp.sqrt(kT * (1 - c1**2))
  momentum_dist = simulate.Normal(c1 * state.momentum, c2**2 * state.mass)
  dR = momentum_dist.sample(split)
  key, split = random.split(state.rng)
  return state.set(momentum=dR, log_prob=momentum_dist.log_prob(dR).sum(), rng=key)


def nvt_langevin(energy_or_force_fn: Callable[..., Array],
                 shift_fn: ShiftFn,
                 dt: float,
                 kT: float,
                 gamma: float=0.1,
                 center_velocity: bool=True,
                 **sim_kwargs) -> Simulator:
  """Forked from JAX-MD
  
  Simulation in the NVT ensemble using the BAOAB Langevin thermostat.

  Samples from the canonical ensemble in which the number of particles (N),
  the system volume (V), and the temperature (T) are held constant. Langevin
  dynamics are stochastic and it is supposed that the system is interacting
  with fictitious microscopic degrees of freedom. An example of this would be
  large particles in a solvent such as water. Thus, Langevin dynamics are a
  stochastic ODE described by a friction coefficient and noise of a given
  covariance.

  Our implementation follows the paper [#davidcheck] by Davidchack, Ouldridge,
  and Tretyakov.

  Args:
    energy_or_force: A function that produces either an energy or a force from
      a set of particle positions specified as an ndarray of shape
      `[n, spatial_dimension]`.
    shift_fn: A function that displaces positions, `R`, by an amount `dR`. Both
      `R` and `dR` should be ndarrays of shape `[n, spatial_dimension]`.
    dt: Floating point number specifying the timescale (step size) of the
      simulation.
    kT: Floating point number specifying the temperature in units of Boltzmann
      constant. To update the temperature dynamically during a simulation one
      should pass `kT` as a keyword argument to the step function.
    gamma: A float specifying the friction coefficient between the particles
      and the solvent.
    center_velocity: A boolean specifying whether or not the center of mass
      position should be subtracted.
  Returns:
    See above.

  .. rubric:: References
  .. [#carlon] R. L. Davidchack, T. E. Ouldridge, and M. V. Tretyakov.
    "New Langevin and gradient thermostats for rigid body dynamics."
    The Journal of Chemical Physics 142, 144114 (2015)
  """
  force_fn = quantity.canonicalize_force(energy_or_force_fn)

  # def _dist(state, **kwargs):
  #   _kT = kwargs.pop('kT', kT)
  #   nu = f32(1) / (state.mass * gamma)
  #   F = force_fn(state.position, **kwargs)
  #   mean =  F * nu * dt
  #   variance = f32(2) * _kT * dt * nu
  #   return simulate.Normal(mean, jnp.sqrt(variance))

  @jit
  def init_fn(key, R, mass=f32(1.0), **kwargs):
    _kT = kwargs.pop('kT', kT)
    key, split = random.split(key)
    force = force_fn(R, **kwargs)
    state = NVTLangevinState(R, None, force, mass, key, 0)
    #print(state.position)
    state = simulate.canonicalize_mass(state)
    #print(state.position)
    return simulate.initialize_momenta(state, split, _kT)

  @jit
  def step_fn(state, **kwargs):
    _dt = kwargs.pop('dt', dt)
    _kT = kwargs.pop('kT', kT)
    dt_2 = _dt / 2

    key, split = random.split(state.rng)

    state = simulate.momentum_step(state, dt_2)
    state = simulate.position_step(state, shift_fn, dt_2, **kwargs)
    state = simulate.stochastic_step(state, _dt, _kT, gamma)
    state = simulate.position_step(state, shift_fn, dt_2, **kwargs)
    state = state.set(force=force_fn(state.position, **kwargs))
    state = simulate.momentum_step(state, dt_2)

    # dist = _dist(state, **kwargs)
    # dR = lax.stop_gradient(dist.sample(key=split))
    # state = state.set(log_prob=dist.log_prob(dR).sum())
    return state

  return init_fn, step_fn


# 2016 Sivak and Crooks energy landscape. [ ] TODO: Full Citations.
def V_biomolecule_sivak(kappa_l, kappa_r, x_m, delta_E, k_s, beta):
  """Returns:
      Function that takes arguments of `position` and `trap_position` to output the total
        energy of the system for those parameters."""
  def total_energy(position, r0=0.0, **unused_kwargs):
      x = position[0][0]
      Em = -(1./beta)*jnp.log(jnp.exp(-0.5*beta*kappa_l*(x+x_m)**2)+jnp.exp(-(0.5*beta*kappa_r*(x-x_m)**2+beta*delta_E)))
      #moving harmonic potential:
      Es = k_s/2 * (x-r0) ** 2
      return Em + Es
  return total_energy

# 2010 Geiger and Dellago energy landscape.
def V_biomolecule_geiger(k_s, epsilon, sigma):
  """Returns:
    Function that takes arguments of `position` and `trap_position` to output the total
      energy of the system for those parameters."""
  def total_energy(position, r0=0.0, **unused_kwargs):
      x = position[0][0]
      #underlying energy landscape:
      # 1/beta * log(e^(-0.5 beta kappa_1 (x + x_m)^2) + )
      Em = epsilon * ((1 - (x/sigma - 1)**2)**2 + (epsilon/2)*((x/sigma) - 1))
      #moving harmonic potential:
      Es = k_s/2 * (x-r0) ** 2
      return Em + Es
  return total_energy

def V_biomolecule_reconstructed(k_s, positions, energies):
  """Returns a function that takes position and outputs the free energy of 
  a particle on a reconstructed landscape. 
  
  Args:
    positions: Array[Floats]. An array of particle positions
    energies: Array[Floats]
      Same shape as positions, and the i-th value of the array is equal to
    the free energy of a particle at the i-th position.
  
  Returns: Callable[particle_position, r0 = trap_position]
  """
  # Assuming a particle at positions[i] has G = energies[i]

  start = positions[0]
  end = positions[-1]
  dx = positions[1]-positions[0]

  def total_energy(particle_position, r0 = 0., **unused_kwargs):
    x = particle_position[0][0]
    bin_num = (x-start)/dx
    bin_num = bin_num.astype(int)

    # interpolate
    t = (x-(bin_num * dx + start))/dx
    Em = t * energies[bin_num+1] + (1-t) * energies[bin_num]
  
    # moving harmonic potential
    Es = k_s/2 * (x-r0) ** 2
    return Em + Es
  return total_energy

# Currently unused
def V_simple_spring(r0,k,box_size):
  def spring_energy(position, **unused_kwargs):
    dR=(position[0][0]-r0)
    return k/2 * dR ** 2
  return spring_energy

