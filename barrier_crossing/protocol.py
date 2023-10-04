import typing

import scipy.special as sps

import jax.numpy as jnp
import numpy as onp

from jax import lax

def chebyshev_coefficients(degree):
  return onp.stack([onp.concatenate(
        [onp.zeros(degree - j), sps.chebyt(j, True)])
    for j in range(degree + 1)])
  
def chebyshev_coefficients_nonmonic(degree):
  return onp.stack([onp.concatenate(
        [onp.zeros(degree - j), sps.chebyt(j, False)])
    for j in range(degree + 1)])
  
class Chebyshev(typing.NamedTuple):
  weights: jnp.array

  @property
  def degree(self):
    return self.weights.shape[0] - 1

  @property
  def coefficients(self):
    return chebyshev_coefficients(self.degree)

  def _powers(self, x):
    def _multiply_by_x(y, _):
      y *= x
      return y, y
    ones = jnp.ones_like(x)
    _, powers = lax.scan(
        _multiply_by_x, ones, None, length=self.degree, reverse=True)
    return jnp.concatenate([powers, ones[jnp.newaxis]], axis=0)

  def __call__(self, x):
    """`x` is an array of values in (0, 1)."""
    x = 2. * x - 1.  # Rescale (0, 1) -> (-1, 1)
    x_powers = self._powers(x)
    return jnp.einsum(
        'w,wp,px->x', self.weights, self.coefficients, x_powers)

def make_schedule_chebyshev(time_vec,coeffs,r0_init,r0_final):
  time_vec = time_vec[1:-1]
  scaled_time_vec = time_vec/time_vec[-1]
  vals = Chebyshev(coeffs)(scaled_time_vec)
  vals = jnp.concatenate([vals, jnp.reshape(r0_final, [1])])
  vals = jnp.concatenate([jnp.reshape(r0_init, [1]), vals])
  return vals


def linear_chebyshev_coefficients(r0_init, r0_final, simulation_steps, degree=12, y_intercept = 0):
  """Find Chebyshev coefficients for a line with equation y = mx + y_intercept 
  where m = (r0_final - r0_init)/sim_steps.
    Returns coefficients in ndarray with length `degree`"""
  slope = (r0_final - r0_init)/simulation_steps
  vals = slope*jnp.arange(0,simulation_steps+1) + y_intercept
  xscaled=jnp.arange(simulation_steps+1)/simulation_steps

  p = onp.polynomial.Chebyshev.fit(xscaled, vals, degree, domain=[0,1])

  return p.coef

def make_trap_fxn(time_vec,coeffs,r0_init,r0_final):
  """ Returns function with slices/index inputs that returns
  the normal trap position depending on the Chebyshev polynomial protocol. """
  positions = make_schedule_chebyshev(time_vec,coeffs,r0_init,r0_final)
  def Get_r0(step):
    return positions[step]
  return Get_r0

def make_trap_fxn_rev(time_vec,coeffs,r0_init,r0_final):
  """ Returns function with slices/index inputs that returns
    the reverse time trap position depending on the Chebyshev polynomial protocol. """
  positions = jnp.flip(make_schedule_chebyshev(time_vec,coeffs,r0_init,r0_final))
  def Get_r0(step):
    return positions[step]
  return Get_r0
  
def make_custom_trap_fxn(time_vec, timestep_trap_position, r0_init, r0_final):
  """Create trap function. Return Callable to interpolate between 
  (time, position)` pairs of `time_trap_position
  
  Note that custom trap functions CANNOT be optimized.
  """
  simulation_steps = time_vec.shape[0]
  custom_positions = jnp.interp(jnp.arange(simulation_steps), timestep_trap_position[:,0], timestep_trap_position[:,1])
  custom_positions = custom_positions.at[0].set(r0_init)
  custom_positions = custom_positions.at[-1].set(r0_final)
  def Get_r0(step):
    if step >= simulation_steps:
      return r0_final
    return custom_positions[step]
  return Get_r0
    
def make_custom_trap_fxn_rev(time_vec, timestep_trap_position, r0_init, r0_final):
  """Create trap function. Return Callable to interpolate between 
  (time, position)` pairs of `time_trap_position"""
  simulation_steps = time_vec.shape[0]
  custom_positions = jnp.interp(jnp.arange(simulation_steps), timestep_trap_position[:,0], timestep_trap_position[:,1])
  custom_positions = custom_positions.at[0].set(r0_init)
  custom_positions = custom_positions.at[-1].set(r0_final)
  custom_positions = jnp.flip(custom_positions)
  def Get_r0(step):
    if step >= simulation_steps:
      return r0_init
    return custom_positions[step]
  return Get_r0 

def trap_sum(simulation_steps, cut, trap1, trap2): 
  """
  Make a trap function out of two "pieces" : trap1 and trap2.
  cut is the simulation step at which trap1 ends and trap2 starts.
  """
  positions1 = trap1(jnp.arange(cut))
  positions2 = trap2(jnp.arange(simulation_steps-cut))
  def Get_r0(step):
    positions = []
    positions.append(jnp.where(step < cut, positions1[step], positions2[step-cut]))
    #return jnp.array(positions).flatten()
    return positions[0]
  return Get_r0

def trap_sum_rev(timevec, simulation_steps, cut, trap1, trap2): 
  """
  Make a trap function out of two "pieces" : trap1 and trap2.
  This is the reverse direction, so
  cut is the simulation step at which trap 2 ends and trap 1 starts
  """
  positions1 = trap1(jnp.arange(cut))
  positions2 = trap2(jnp.arange(simulation_steps-cut))
  def Get_r0(step):
    positions = []
    positions.append(jnp.where(step < cut, positions2[step], positions1[step-cut]))
    #return jnp.array(positions).flatten()
    return positions[0]
  return Get_r0
