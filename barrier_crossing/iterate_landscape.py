import copy
import time
import logging
import tqdm

import matplotlib.pyplot as plt

import jax.numpy as jnp
import jax.random as random
import jax

import jax.example_libraries.optimizers as jopt

from jax_md import space

from barrier_crossing.energy import V_biomolecule_geiger, V_biomolecule_reconstructed
from barrier_crossing.protocol import make_trap_fxn, make_trap_fxn_rev, linear_chebyshev_coefficients
from barrier_crossing.simulate import simulate_brownian_harmonic, batch_simulate_harmonic
from barrier_crossing.optimize import estimate_gradient_work, optimize_protocol, find_error_samples

def plot_with_stddev(x, label=None, n=1, axis=0, ax=plt, dt=1.):
  stddev = jnp.std(x, axis)
  mn = jnp.mean(x, axis)
  xs = jnp.arange(mn.shape[0]) * dt

  ax.fill_between(xs,
                  mn + n * stddev, 
                  mn - n * stddev, 
                  alpha=.3)
  ax.plot(xs, mn, label=label)
  
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

  # Use less than 10 gb of memory:
  num_pages = int(simulation_steps * batch_size * batch_size / 1e9)
  if num_pages == 0:
    page_size = 0
  else:
    page_size = int(simulation_steps / num_pages)

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
    num = 0.
    denom = 0.
    page_start = 0

    batch_num = jax.vmap(lambda t: numerator(t,l)/ exponential_avg(t))
    batch_denom =jax.vmap(lambda t: (jnp.exp(-beta*potential_energy(l,t))) / exponential_avg(t))

    
    for _ in range(num_pages):
      assert(page_size > 0)
      page_end = int(page_start + page_size)
      page_range = jnp.arange(page_start, page_end)
      num += batch_num(page_range).sum()
      denom += batch_denom(page_range).sum() 
      
      page_start = page_end

    page_range = jnp.arange(page_start, simulation_steps)
    if page_range.shape[0] != 0:
      num += batch_num(page_range).sum()
      denom += batch_denom(page_range).sum()
    
    return (-(1/beta) * jnp.log(num/denom))

  midpoints = []
  free_energies = []
  for k in tqdm.trange(bins, position=0, desc="Reconstruct Landscape Bins: "):
    logging.info(f"Reconstructing for bin {k+1}")
    energy = free_energy_q(k)
    free_energies.append(energy)
    midpoints.append((traj_min[0][0] + (k+0.5)*bin_width[0][0]))
  
  return (jnp.array(midpoints), jnp.array(free_energies))

def find_max_pos(landscape, barrier_pos):
  """
  Finds the position of the barrier (maximum)
  in the reconstructed landscape.
  Inputs: barrier_pos is the position of the barrier in
  the true landscape 
  """
  max = 0
  for position in range(barrier_pos-5,barrier_pos+5,1): # maximum expected at around 0
    if landscape([[position]]) > landscape([[max]]):
      max = position
  return max

def find_min_pos(landscape, min_1_guess, min_2_guess):
  """
  Finds the positions of the minima 
  in the reconstructed landscape.
  Inputs: min_1_guess and min_2_guess are expected positions of the wells
  Returns : (position of first well, position of second well)
  """
  min1 = min_1_guess
  min2 = min_2_guess
  for position in jnp.linspace(min_1_guess-3,min_1_guess+3,100):
    if landscape([[position]]) < landscape([[min1]]):
      min1 = position
  for position2 in jnp.linspace(min_2_guess-3, min_2_guess+3, 100):
    if landscape([[position2]]) < landscape([[min2]]):
      min2 = position2
  return (min1, min2)

def landscape_error(ls, true_ls, r0_init, r0_final, barrier_pos):
  """
  Inputs: 
  r0_init and r0_final are the positions of the wells (minima)
  in the true landscape
  barrier_pos is the position of the barrier in the true landscape
  Returns:
  first_well_error : percentage difference between the depth of the first well
                     in the landscape and the true depth of the first well
    
  second_well_error : percentage difference between the depth of the second well
                     in the landscape and the true depth of the second well

  delta_E_error : percentage difference between the delta_E
                     in the landscape and the true delta_E
  """
  barrier_pos = find_max_pos(ls, barrier_pos)
  min1, min2 = find_min_pos(ls,r0_init,r0_final)
  first_well_depth = ls([[barrier_pos]])-ls([[min1]])
  second_well_depth = ls([[barrier_pos]]) - ls([[min2]])
  delta_E = abs(first_well_depth - second_well_depth)
  first_well_true = true_ls([[barrier_pos]])-true_ls([[r0_init]])
  second_well_true = true_ls([[barrier_pos]])-true_ls([[r0_final]])
  delta_E_true = abs(true_ls([[r0_init]])-true_ls([[r0_final]]))
  first_well_error = 100*abs(first_well_true - first_well_depth)/first_well_true
  second_well_error = 100*abs(second_well_true - second_well_depth)/second_well_true
  delta_E_error = 100*abs(delta_E_true - delta_E)/delta_E_true
  return first_well_error, second_well_error, delta_E_error


def find_max(landscape, init, final):
  """
  Find maximum energy value of a landscape
  in the range (init,final).
  Inputs:
  landscape in the form of a tuple (midpoints, energies)
  """
  max = -1000
  midpoints = landscape[0]
  energies = landscape[1]
  for i in range(len(landscape[0])):
    if init < midpoints[i] < final and energies[i] > max:
      max = energies[i]
  return max


def landscape_discrepancies(ls, true_ls, true_max, r_min, r_max):

  """
  Aligns landscape (ls) with true_ls and finds distance between
  the landscape and true_ls at each midpoint specified for ls.

  Inputs: landscape in the form (midpoints, energies) and 
          true landscape in functional form,
          true_max is the maximum energy on true landscape, 
          r_min and r_max denote the range of extensions considered
          for the discrepancy calculation.
  
  Outputs: list of landscape discrepancies at each midpoint 
           in the range (r_min, r_max)
  """

  # Find difference at max point to align landscapes
  ls_max = find_max(ls, r_min, r_max)
  diff = true_max - ls_max
 
  # only consider points in range (min, max)
  midpoints = []
  energies = []
  for i in range(len(ls[0])):
    if r_min <= ls[0][i] <= r_max:
      midpoints.append(ls[0][i])
      energies.append(ls[1][i] + diff)
  
  discrepancies = []
  x = 0
  for energy in energies:
    discrepancies.append(abs(true_ls([[midpoints[x]]]) -energy))
    x += 1
  
  return discrepancies


def landscape_discrepancies_samples(ls, true_ls, error_samples):
  """
  Finds the distance (discrepancy) between ls and true_ls at points
  specified by error_samples.
  Inputs: ls and true_ls in functional forms (already aligned), 
          error_samples is a list of extensions
  Outputs: list of discrepancies (corresponding to error samples)
  """
  discrepancies = []
  for r in error_samples:
    discrepancies.append(abs(true_ls([[r]]) - ls([[r]])))
  return discrepancies


def landscape_diff(ls_1, ls_2):
  # TODO depending on the form might need to change which index to take
  #assert(ls_1.shape == ls_2.shape)
  return jnp.linalg.norm(jnp.array(ls_2[1])-jnp.array(ls_1[1]))


def optimize_landscape(
                      simulate_fn,
                      init_coeffs,
                      grad_fn_no_E,
                      key,
                      max_iter,
                      bins,
                      simulation_steps,
                      opt_batch_size,
                      reconstruct_batch_size,
                      opt_steps,
                      optimizer,
                      r0_init,
                      r0_final,
                      k_s,
                      beta,
                      savefig = False):
  """Iteratively reconstruct a de novo energy landscape from simulated trajectories. Optimize a protocol
  with respect to reconstruction error (or a proxy such as average work used) on the reconstructed landscapes 
  to create a protocol that will allow for more accurate reconstructions.
  
  Args:
    simulate_fn: Callable(trap_schedule) -> Callable(keys) 
      -> final BrownianState, (Array[particle_position], Array[log probability], Array[work])
        Function that simulates moving the particle along the given trap_schedule given a de novo
      energy function.
    init_coeffs: Array[] Chebyshev coefficients of trap.
    grad_fn_no_E: Callable(batch_size, energy_fn) -> Callable(coeffs, seed, *args)
        Function that takes input of arbitrary energy function. Intended to be used as follows
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

  trap_fn = make_trap_fxn(jnp.arange(simulation_steps), init_coeffs, r0_init, r0_final)
  
  for iter_num in tqdm.trange(max_iter, position=2, desc="Optimize Landscape: "):
    if new_landscape:
      old_landscape = (copy.deepcopy(new_landscape[0]),copy.deepcopy(new_landscape[1])) # TODO

    _, (trajectories, works, log_probs) = batch_simulate_harmonic(reconstruct_batch_size,
                            simulate_fn(trap_fn),
                            simulation_steps,
                            key) 
    
    logging.info("Creating landscape.")
    new_landscape = energy_reconstruction(works, trajectories, bins, trap_fn, simulation_steps, reconstruct_batch_size, k_s, beta) # TODO: 
    landscapes.append(new_landscape)

    positions, energies = new_landscape
    
    energy_fn_guess = V_biomolecule_reconstructed(k_s, jnp.array(positions), jnp.array(energies))
    
    # Optimize a protocol with this new landscape
    logging.info("Optimizing protocol from linear using reconstructed landscape.")
    # error_samples = find_error_samples(energy_fn_guess, simulate_fn, trap_fn, simulation_steps, key, bins)
    
    grad_fxn = lambda num_batches: grad_fn_no_E(num_batches, energy_fn_guess)
   
    init_trap_coeffs = init_coeffs
    
    coeffs_, _, losses = optimize_protocol(init_trap_coeffs, grad_fxn, optimizer, opt_batch_size, opt_steps)

    if savefig:
      _, ax = plt.subplots(1, 2, figsize=[24, 12])

      #avg_loss = jnp.mean(jnp.array(losses), axis = 0)
      plot_with_stddev(losses.T, ax=ax[0])

      # ax[0].set_title(f'Jarzynski Error over Optimization; Short trap; STD error sampling; {batch_size}; {opt_steps}.')
      ax[0].set_title(f"Iterative Iter Num {iter_num}")
      ax[0].set_xlabel('Number of Optimization Steps')
      ax[0].set_ylabel('Error')

      trap_fn = make_trap_fxn(jnp.arange(simulation_steps), init_trap_coeffs, r0_init, r0_final)
      init_sched = trap_fn(jnp.arange(simulation_steps))
      ax[1].plot(jnp.arange(simulation_steps), init_sched, label='Initial guess')

      for i, (_, coeff) in enumerate(coeffs_):
        print_per = max(1, int(opt_steps/5))
        if i% print_per == 0 and i!=0:
          trap_fn = make_trap_fxn(jnp.arange(simulation_steps),coeff,r0_init,r0_final)
          full_sched = trap_fn(jnp.arange(simulation_steps))
          ax[1].plot(jnp.arange(simulation_steps), full_sched, '-', label=f'Step {i}')

      # Plot final estimate:
      trap_fn = make_trap_fxn(jnp.arange(simulation_steps), coeffs_[-1][1],r0_init,r0_final)
      full_sched = trap_fn(jnp.arange(simulation_steps))
      ax[1].plot(jnp.arange(simulation_steps), full_sched, '-', label=f'Final')


      ax[1].legend()#
      ax[1].set_title('Schedule evolution')
      plt.savefig(f"losses_iter{iter_num}_{savefig}.png")

    final_coeff = coeffs_[-1][1]
    coeffs.append(final_coeff)
    
    trap_fn = make_trap_fxn(jnp.arange(simulation_steps), final_coeff, r0_init, r0_final)
    
    if iter_num > 0:
      diff = landscape_diff(old_landscape, new_landscape) # take the norm of the difference
      logging.info(f"Difference between prior landscape: {diff:.4f}")

    iter_num += 1
    
  _, (trajectories, works, log_probs) = batch_simulate_harmonic(reconstruct_batch_size,
                            simulate_fn(trap_fn),
                            simulation_steps,
                            key) 
  
  logging.info("Creating landscape.")
  new_landscape = energy_reconstruction(works, trajectories, bins, trap_fn, simulation_steps, reconstruct_batch_size, k_s, beta) # TODO: 
  landscapes.append(new_landscape)
  
  return landscapes, coeffs
    
    
