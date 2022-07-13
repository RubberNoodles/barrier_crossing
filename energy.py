# TODO: Imports

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
    See above.
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
    mass = jmd.quantity.canonicalize_mass(mass)
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

def V_biomolecule(kappa_l, kappa_r, x_m, delta_E, k_s, beta, epsilon, sigma):
  def total_energy(position, r0=0.0, **unused_kwargs):
      x = position[0][0]
      #underlying energy landscape:
      # 1/beta * log(e^(-0.5 beta kappa_1 (x + x_m)^2) + )
      Em = epsilon * ((1 - (x/sigma - 1)**2)**2 + (epsilon/2)*((x/sigma) - 1))
      #moving harmonic potential:
      Es = k_s/2 * (x-r0) ** 2
      return Em + Es
  return total_energy

def V_simple_spring(r0,k,box_size):
  def spring_energy(position, **unused_kwargs):
    #dR = jnp.mod((position - r0) + box_size * f32(0.5), box_size) - f32(0.5) * box_size
    dR=(position[0][0]-r0)
    return k/2 * dR ** 2
  return spring_energy


"""##Functions for estimating gradients of the MD simulations"""

def run_brownian_opt(energy_fn, coeffs, init_position, r0_init, r0_final, Neq, shift, key, simulation_steps, dt=1e-5, temperature=1e-5, mass=1.0, gamma=1.0):
  """Simulation of Brownian particle being dragged by a moving harmonic trap.
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
  sim_step = jnp.arange(simulation_steps)  
  trap_fn = make_trap_fxn(sim_step,coeffs,r0_init,r0_final)

  def equilibrate(init_state, Neq, apply, r0_init):
    @jit
    def scan_eq(state, step):
      state = apply(state, step, r0=r0_init)
      return state, 0
    state, _ = lax.scan(scan_eq,init_state,jnp.arange(Neq))
    return state
    
  def increment_work(state, step):
        return (energy_fn(state.position, r0=trap_fn(step)) - energy_fn(state.position, r0=trap_fn(step-1)))

  @jit
  def scan_fn(state, step):
    dW = increment_work(state, step) #increment based on position BEFORE 'thermal kick' a la Crooks
    # Dynamically pass r0 to apply, which passes it on to energy_fn
    state = apply(state, step, r0=trap_fn(step))
    return state, (state.position, state.log_prob, dW)

  key, split = random.split(key)  

  init, apply = brownian(energy_fn, shift, dt=dt, T_schedule=temperature, gamma=gamma)
  apply = jit(apply)

  eq_state = equilibrate(init(split, init_position, mass=mass), Neq, apply, r0_init)
  state = eq_state
  state, (positions, log_probs, works) = lax.scan(scan_fn,state,(jnp.arange(simulation_steps-1)+1))
  #print("Log probability:")
  #print(log_probs)
  works = jnp.concatenate([jnp.reshape(0., [1]), works])
  positions = jnp.concatenate([jnp.reshape(eq_state.position, [1,1,1]), positions])
  return positions, log_probs, works

