import copy
import time
import logging
import tqdm

import matplotlib.pyplot as plt

import jax.numpy as jnp
import jax.random as random
import jax

from jax.experimental import optimizers as jopt

from jax_md import space

from barrier_crossing.energy import V_biomolecule_geiger, V_biomolecule_reconstructed
from barrier_crossing.protocol import make_trap_fxn, make_trap_fxn_rev, linear_chebyshev_coefficients
from barrier_crossing.simulate import simulate_brownian_harmonic, batch_simulate_harmonic
from barrier_crossing.optimize import estimate_gradient_fwd, optimize_protocol, find_bin_timesteps

def energy_reconstruction(works, trajectories, bins, trap_fn, simulation_steps, batch_size, k_s, beta):
  """
  Outputs a (midpoints, free_energies) tuple for reconstructing 
  free-energy landscapes.
  t - time step
  l - bin index (0 <= l < number of bins)
  """
  logging.info("Reconstructing the landscape...")
  traj_min = traj_min = min(trajectories[:,0]) 
  traj_max = max(trajectories[:, simulation_steps]) 
  bin_width = (traj_max-traj_min)/bins

  def exponential_avg(t): #find exponential average using eq. 3.20
    exp_works_t = jnp.exp(-beta * works)[:,t] #find exponential of work at t
    return (1/batch_size)*(jnp.sum(exp_works_t, 0)) 
  def numerator(t,l): # approximate numerator in eq. 3.16 using eq. 3.20
    exp_works_t = jnp.exp(-beta * works)[:,t] #find exponential of work at t
    traj_t = trajectories[:,t] #trajectories at time t
    heaviside = jnp.where((traj_min + l * bin_width <= traj_t) & (traj_t <=traj_min + (l+1) * bin_width), 1 , 0)
    return (1/batch_size) * jnp.sum(exp_works_t * heaviside)
  def potential_energy(l, t):
    q_l = traj_min + (l + 0.5) * bin_width # midpoint of the lth extension bin
    return 0.5 * k_s * (trap_fn(t) - q_l)**2
  def free_energy_q(l): # find free energy for lth bin
    sum1 = 0.0
    sum2 = 0.0
    for t in range(simulation_steps):
      sum1 += numerator(t,l)/ exponential_avg(t)
      sum2 += (jnp.exp(-beta*potential_energy(l,t)))/ exponential_avg(t)
    return (-(1/beta) * jnp.log(sum1/sum2))

  midpoints = []
  free_energies = []
  for k in tqdm.trange(bins, position=0, desc="Reconstruct Landscape Bins: "):
    logging.info(f"Reconstructing for bin {k+1}")
    energy = free_energy_q(k)[0][0]
    free_energies.append(energy)
    midpoints.append((traj_min[0][0] + (k+0.5)*bin_width[0][0]))
  
  return (midpoints, free_energies)


def landscape_diff(ls_1, ls_2):
  # TODO depending on the form might need to change which index to take
  #assert(ls_1.shape == ls_2.shape)
  return jnp.linalg.norm(jnp.array(ls_2[1])-jnp.array(ls_1[1]))


def optimize_landscape(ground_truth_energy_fn,
                      simulate_fn,
                      init_trap_fn,
                      init_trap_coeffs,
                      grad_fn_no_E,
                      key,
                      max_iter,
                      bins,
                      simulation_steps,
                      batch_size,
                      opt_steps,
                      optimizer,
                      r0_init,
                      r0_final,
                      k_s,
                      beta):
  """Iteratively reconstruct a black box energy landscape from simulated trajectories. Optimize a protocol
  with respect to reconstruction error (or a proxy such as average work used) on the reconstructed landscapes 
  to create a protocol that will allow for more accurate reconstructions.
  Args:
    ground_truth_energy_fn: Energy landscape we are trying to reconstruct. Callable(particle_position, trap_position) ->
      energy. 
        Intended to act as an unknown landscape with outputs of particle trajectory that would be obtained
      from experimentation.
    simulate_fn: Callable(Energy_fn, trap_schedule) 
      -> final BrownianState, (Array[particle_position], Array[log probability], Array[work])
        Function that simulates moving the particle along the given trap_schedule given a specified
      energy function.
    init_trap_fn: Callable(time_step) -> trap_position
    init_trap_coeffs: Array[Float]. Coefficients of Chebyshev polynomial of initial protocol
    grad_fn_no_E: Callable(batch_size, energy_fn) -> Callable(coeffs, seed, *args)
        Function that allows input of arbitrary energy, function. Intended to be used as follows
      ``grad_fxn = lambda num_batches: grad_fn_no_E(num_batches, energy_fn_guess)``
    key: rng
    max_iter: Integer specifying number of iterations of reconstruction.
    bins: Integer specifying number of checkpoints/locations we try to estimate the energy landscape at.
  
  Returns:
    ``landscapes``: List of landscapes length ``max_iter``.
    ``coeffs``: List of Chebyshev coefficients that are optimal at each iteration. 
  """

  old_landscape = None
  new_landscape = None

  landscapes = []
  coeffs = []

  iter_num = 0
  diff = 1e10 # large number

  trap_fn = init_trap_fn
  trap_coeffs = init_trap_coeffs
  
  for iter_num in tqdm.trange(max_iter, position=2, desc="Optimize Landscape: "):
    if new_landscape:
      old_landscape = (copy.deepcopy(new_landscape[0]),copy.deepcopy(new_landscape[1])) # TODO

    _, (trajectories, works, log_probs) = batch_simulate_harmonic(batch_size,
                            ground_truth_energy_fn,
                            simulate_fn,
                            trap_fn,
                            simulation_steps,
                            key)
    
    logging.info("Creating landscape.")
    new_landscape = energy_reconstruction(works, trajectories, bins, trap_fn, simulation_steps, batch_size, k_s, beta) # TODO: 
    landscapes.append(new_landscape[1])

    positions, energies = new_landscape
    
    energy_fn_guess = V_biomolecule_reconstructed(k_s, jnp.array(positions), jnp.array(energies))
    
    # Optimize a protocol with this new landscape
    logging.info("Optimiziing protocol from linear using reconstructed landscape.")
    bin_timesteps = find_bin_timesteps(energy_fn_guess, simulate_fn, trap_fn, simulation_steps, key, bins)
    
    grad_fxn = lambda num_batches: grad_fn_no_E(num_batches, energy_fn_guess, bin_timesteps)
    lin_trap_coeffs = linear_chebyshev_coefficients(r0_init,r0_final,simulation_steps, degree = 12, y_intercept = r0_init)
    coeffs_, _, losses = optimize_protocol(lin_trap_coeffs, grad_fxn, optimizer, batch_size, opt_steps)
    
    final_coeff = coeffs_[-1][1]
    coeffs.append(final_coeff)
    
    trap_fn = make_trap_fxn(jnp.arange(simulation_steps), final_coeff, r0_init, r0_final)
    
    if iter_num > 0:
      diff = landscape_diff(old_landscape, new_landscape) # take the norm of the difference
      logging.info(f"Difference between prior landscape: {diff:.4f}")

    iter_num += 1
  
  positions = new_landscape[0]
  return landscapes, coeffs, positions
# TODO: Some plotting code for the ladnscape
    
    
