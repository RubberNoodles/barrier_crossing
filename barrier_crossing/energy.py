import collections

import jax
import jax.numpy as jnp
from jax import random, lax, jit

import jax_md as jmd
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

from barrier_crossing.protocol import make_trap_fxn, make_trap_fxn_rev

"""##Custom Brownian simulator

This is a modification of the JAX-MD Brownian simulator that also returns the log probability of any trajectory that runs. This is needed in order to compute gradients correctly (eq'n 12 in my arXiv paper)
"""

class BrownianState(collections.namedtuple('BrownianState',
                                           'position mass rng log_prob')):
  pass

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
    shift_fn: A function that displaces positions, R, by an amount dR. Both R
      and dR should be ndarrays of shape [n, spatial_dimension].
    dt: Floating point number specifying the timescale (step size) of the
      simulation.
    T_schedule: Either a floating point number specifying a constant temperature
      or a function specifying temperature as a function of time.
    quant: Either a quantity.Energy or a quantity.Force specifying whether
      energy_or_force is an energy or force respectively.
    gamma: A float specifying the friction coefficient between the particles
      and the solvent.
  Returns:
    Tuple of `init_fn` to initialize the simulator and `apply_fn` to move orward one step
      in time.
    [1] E. Carlon, M. Laleman, S. Nomidis. "Molecular Dynamics Simulation."
        http://itf.fys.kuleuven.be/~enrico/Teaching/molecular_dynamics_2015.pdf
        Accessed on 06/05/2019.
  """

  force_fn = jmd.quantity.canonicalize_force(energy_or_force)

  dt, gamma = jmd.util.static_cast(dt, gamma)

  T_schedule = jmd.interpolate.canonicalize(T_schedule)

  def _dist(state, t, **kwargs):
    nu = jnp.float32(1) / (state.mass * gamma)
    F = force_fn(state.position, t=t, **kwargs)
    mean =  F * nu * dt
    variance = jnp.float32(2) * T_schedule(t) * dt * nu
    return tfd.Normal(mean, jnp.sqrt(variance))
  
  def init_fn(key, R, mass=jnp.float32(1)):
    mass = jmd.simulate.canonicalize_mass(mass)
    return BrownianState(R, mass, key, 0.)

  def apply_fn(state, t=jnp.float32(0), **kwargs):
    dist = _dist(state, t, **kwargs)
    key, split = jax.random.split(state.rng)

    # We have to stop gradients here, otherwise the gradient with respect to
    # energy/force is zero. The following is a simple repro of the issue:
    #  def sample_log_prob(mean, key):
    #    d = tfd.Normal(mean, 1.)
    #    s = d.sample(seed=key)
    #    return d.log_prob(s)
    #  jax.grad(sample_log_prob)(0., key)  # Always 0 !
    dR = jax.lax.stop_gradient(dist.sample(seed=split))

    log_prob = dist.log_prob(dR).sum()
    R = shift(state.position, dR, t=t, **kwargs)
    return BrownianState(R, state.mass, key, log_prob)

  return init_fn, apply_fn

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

# Currently unused
def V_simple_spring(r0,k,box_size):
  def spring_energy(position, **unused_kwargs):
    #dR = jnp.mod((position - r0) + box_size * f32(0.5), box_size) - f32(0.5) * box_size
    dR=(position[0][0]-r0)
    return k/2 * dR ** 2
  return spring_energy

