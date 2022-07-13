
"""##Protocol parametrizations"""

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
    _, powers = jax.lax.scan(
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

# Finding the Chebyshev coefficients for a line with equation y = mx + y_intercept
# where m = (r0_final - r0_init)/sim_steps
def linear_chebyshev_coefficients(r0_init, r0_final, simulation_steps, degree=12, y_intercept = 0):
  slope = (r0_final - r0_init)/simulation_steps
  vals = slope*jnp.arange(0,simulation_steps+1) + y_intercept
  xscaled=jnp.arange(simulation_steps+1)/simulation_steps

  p = onp.polynomial.Chebyshev.fit(xscaled, vals, degree, domain=[0,1])

  return p.coef # This is the coefficients that we are looking for

"""## Making Trap Trajectory (Forward & Reverse)"""

def make_trap_fxn(timevec,coeffs,r0_init,r0_final):
  positions = make_schedule_chebyshev(timevec,coeffs,r0_init,r0_final)
  def Get_r0(step):
    return positions[step]
  return Get_r0

def make_trap_fxn_rev(timevec,coeffs,r0_init,r0_final):
  """ Returns function with slices/index inputs that returns
    the reverse time trap position depending on the chebyshev """
  positions = jnp.flip(make_schedule_chebyshev(timevec,coeffs,r0_init,r0_final))
  def Get_r0(step):
    return positions[step]
  return Get_r0
  
