"""
Module to reconstruct landscapes with optimized algorithm. In addition, helper
code to classify landscape reconstruction quality, as well as `optimize_landscape`
code for iterative reconstruction.
"""
import code
import copy
import tqdm
from typing import Callable

import matplotlib.pyplot as plt

import numpy as onp

import jax.numpy as jnp
import jax
import jax.random as random

from barrier_crossing.models import ScheduleModel
from barrier_crossing.energy import ReconstructedLandscape
from barrier_crossing.simulate import batch_simulate_harmonic

def plot_with_stddev(x, label=None, n=1, axis=0, ax=plt, dt=1.):
  stddev = jnp.std(x, axis)
  mn = jnp.mean(x, axis)
  xs = jnp.arange(mn.shape[0]) * dt

  ax.fill_between(xs,
                  mn + n * stddev, 
                  mn - n * stddev, 
                  alpha=.3)
  ax.plot(xs, mn, label=label)
  
def energy_reconstruction(works, trajectories, bins, trap_fn, ks_trap_fn, simulation_steps, batch_size, beta):
  """Outputs a (midpoints, free_energies) tuple for reconstructing 
  free-energy landscapes according to formalism by Hummer & Szabo 2001.

  Args:
      works Array[batch_size, simulation_steps]: Along the second axis, the amount of work done for the
        given trajectory up until the given simulation step.
      trajectories Array[]: Particle positions at each simulation step.
      bins int: Number of bins to reconstruct.
      trap_fn Callable(time_step) -> float: Function to determine trap position
      simulation_steps int: Number of simulation steps. Equal to simulation_length/dt.
      batch_size int: Number of distinct random trajectories. Requires at least 100 for most simulations.
      k_s float: Trap stiffness.
      beta float: Inverse temperature (1/kT).

  Returns:
      midpoints Array[]: midpoints of equally spaced bins along the reaction coordinate.
      energies Array[]: Energies[n] corresponds to the reconstructed free energy at midpoints[n].
  
  ------
  1. G. Hummer and A. Szabo. Free energy reconstruction from nonequilibrium single-molecule pulling experiments.
    Proceedings of the National Academy of Sciences of the United States of America, 98(7):3658–3661, Mar 2001.
  """
  if not isinstance(ks_trap_fn, Callable):
    _tmp = ks_trap_fn
    ks_trap_fn = lambda step: _tmp
  
  if works.shape != (batch_size, simulation_steps):
    raise ValueError(f"Array `works` should be of shape ({batch_size, simulation_steps}), got ({works.shape}) instead.")
  
  exp_works = jnp.exp( - beta * works)
  eta = jnp.mean(exp_works, axis = 0)
  t = jnp.arange(simulation_steps)

  traj_min = trajectories.min()
  traj_max = trajectories.max() + 1e-3
  bin_width = (traj_max-traj_min)/bins
  
  traj_mask = jnp.squeeze(jnp.floor(bins * ((trajectories-traj_min)/(traj_max - traj_min))))

  traj_bins = traj_mask.astype(int) + jnp.outer(jnp.ones(batch_size), t) * bins
  midpoints = jnp.array([traj_min + (i+0.5) * bin_width for i in range(bins)])

  bin_arr = onp.zeros(bins * simulation_steps)

  onp.add.at(bin_arr, traj_bins.astype(int), exp_works)

  bin_arr = bin_arr / batch_size
  numerator = bin_arr.reshape(simulation_steps, bins).T @ jnp.reciprocal(eta)
  # u(z,t): at each timestep and midpoint, we have a deflection energy
  vectorized_trap = jax.vmap(trap_fn)
  repeat_trap_positions = jnp.tile(vectorized_trap(t), (bins, 1))
  deflect = jnp.exp(-beta * ks_trap_fn(t) * 0.5 * jnp.square( repeat_trap_positions.T - midpoints).T)
  
  denominator = deflect @ jnp.reciprocal(eta)
  energies = -(1/beta) * jnp.log(jnp.divide(numerator, denominator))
  return jnp.array(midpoints), jnp.array(energies)

def find_max_pos(landscape, barrier_pos):
  """
  Finds the position of the barrier (maximum)
  in the reconstructed landscape.
  Inputs: barrier_pos is the position of the barrier in
  the true landscape 
  """
  max = 0
  for position in range(barrier_pos-5,barrier_pos+5,1): # maximum expected at around 0
    if landscape(position) > landscape(max):
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
    if landscape(position) < landscape(min1):
      min1 = position
  for position2 in jnp.linspace(min_2_guess-3, min_2_guess+3, 100):
    if landscape(position2) < landscape(min2):
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
  first_well_depth = ls(barrier_pos)-ls(min1)
  second_well_depth = ls(barrier_pos) - ls(min2)
  delta_E = abs(first_well_depth - second_well_depth)
  first_well_true = true_ls(barrier_pos)-true_ls(r0_init)
  second_well_true = true_ls(barrier_pos)-true_ls(r0_final)
  delta_E_true = abs(true_ls(r0_init)-true_ls(r0_final))
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


def landscape_discrepancies(ls, true_ls, first_well, r_min, r_max):

  """
  Aligns landscape (ls) with first well and finds distance between
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
  
  no_trap_rec_fn = ReconstructedLandscape(*ls).molecule_energy
  first_well_guess = no_trap_rec_fn(r_min)

  diff = first_well - first_well_guess
    
  # only consider points in range (min, max)
  midpoints = []
  energies = []
  for i in range(len(ls[0])):
    if r_min <= ls[0][i] <= r_max:
      midpoints.append(ls[0][i])
      energies.append(ls[1][i] + diff)
      
    
  discrepancies = []
  for midpoint, energy in zip(midpoints, energies):
    discrepancies.append(abs(true_ls(midpoint) - energy))
  
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
    discrepancies.append(abs(true_ls(r) - ls(r)))
  return discrepancies


def landscape_diff(ls_1, ls_2):
  # TODO depending on the form might need to change which index to take
  #assert(ls_1.shape == ls_2.shape)
  return jnp.linalg.norm(jnp.array(ls_2[1])-jnp.array(ls_1[1]))

def interpolate_inf(energy):
  energy = onp.array(energy)
  nans = onp.isinf(energy)
  x= lambda z: z.nonzero()[0]
  energy[nans]= onp.interp(x(nans), x(~nans), energy[~nans])
  return jnp.array(energy)

def optimize_landscape(max_iter: int, 
                       reconstruct_batch_size: int,
                       position_model: ScheduleModel, 
                       stiffness_model: ScheduleModel,
                       true_simulation: Callable,
                       guess_sim_pos: Callable,
                       guess_sim_ks: Callable,
                       grad_fn_pos_unf: Callable,
                       grad_fn_ks_unf: Callable,
                       reconstruct_fn: Callable,
                       train_fn: Callable,                
                       key,
                       num_reconstructions=100):
  """
  Iteratively reconstruct a de novo energy landscape from simulated trajectories. Optimize a protocol
  with respect to reconstruction error (or a proxy such as average work used) on the reconstructed landscapes 
  to create a protocol that will allow for more accurate reconstructions.
  
  Args:
    max_iter: Number of iterations.
    position_model, stiffness_model: ScheduleModel which will be iteratively optimized.
    true_simulation: Callable(trap_schedule) -> Callable(keys) 
      -> final BrownianState, (Array[particle_position], Array[log probability], Array[work])
        Function that simulates moving the particle along the given trap_schedule given a de novo
      energy function.
    guess_sim_pos/ks: Callable(energy_fn) -> SimulationFn. At each iteration, this simulate function will
      be endowed with the reconstructed "guess" for an energy function that is computed from HS reconstructions.
    grad_fn_no_E: Callable(batch_size, energy_fn) -> Callable(coeffs, seed, *args)
      Function that takes input of arbitrary energy function. Intended to be used as follows
      ``grad_fxn = lambda num_batches: grad_fn_no_E(num_batches, energy_fn_guess)``
    reconstruct_fn: Callable(batch_cum_works, trajectories, trap_functions, rng) -> Array[Positions, Energies]. Using
      HS formalism and outputs from a batch simulation, compute the "guess" for the free energy landscape.
    train_fn: Callable(model, grad_fn, key) -> Array[losses]. Compute training loop on model with loss function specified
      by `grad_fn`.
    key: rng
    
  Returns:
    ``landscapes``: List of landscapes length ``max_iter``.
    ``coeffs``: List of Chebyshev coefficients that are optimal at each iteration. 
    ``losses``: List of length `num_iter` with loss arrays for each optimization iteration.
  """
  old_landscape = None
  new_landscape = None
  landscapes = []
  
  iter_num = 0
  
  # Compute bias?
  losses = {"position": [], "stiffness": []}
  coeffs = {"position": [], "stiffness": []}
  
  position_sched = position_model.protocol(position_model.coeffs)
  stiffness_sched = stiffness_model.protocol(stiffness_model.coeffs)
  trap_fns = [position_sched, stiffness_sched]
  
  for iter_num in tqdm.trange(max_iter, desc="Optimize Landscape: "):
    if new_landscape:
      old_landscape = (copy.deepcopy(new_landscape[0]),copy.deepcopy(new_landscape[1]))
    
    es = []
    for _ in tqdm.trange(num_reconstructions, position = 3, desc = "Reconstructing Landscapes"):
      key, _ = random.split(key)
      
      _, (trajectories, works, _) = batch_simulate_harmonic(reconstruct_batch_size,
                                true_simulation(*trap_fns),
                                key) 
      # lambda batch_works, trajectories, position_fn, stiffness_fn :energy_reconstruction(batch_works, trajectories, bins, position_fn, stiffness_fn, simulation_steps, reconstruct_batch_size, k_s, beta)
      mid, e = reconstruct_fn(jnp.cumsum(works, axis = 1), trajectories, *trap_fns, reconstruct_batch_size)
      es.append(e)
    
    all_es = jnp.vstack(es)
    
    energies = jnp.mean(all_es, axis = 0)
    if jnp.isinf(all_es).any():
        print("Infinities found in energy reconstruction. Continuing through interpolation...")
        energies = interpolate_inf(energies) 
    
    new_landscape = (mid, energies)
    landscapes.append(new_landscape)
    
    energy_fn_guess = ReconstructedLandscape(jnp.array(mid), jnp.array(energies)).total_energy
    
    
    # train(model, optimizer, grad_fn, key, batch_size = batch_size, num_epochs = num_epochs)
    position_model.pop_hist()
    stiffness_model.pop_hist()
    position_model.mode = "rev"
    stiffness_model.mode = "fwd"
    
    guess_sim_pos_fn = guess_sim_pos(energy_fn_guess)
    guess_sim_ks_fn = guess_sim_ks(energy_fn_guess)
    
    simulate_stiffness_fn = lambda stiffness_trap, key: guess_sim_ks_fn(position_model.protocol(position_model.coeffs), stiffness_trap, key)
    grad_fn = grad_fn_ks_unf(stiffness_model, simulate_stiffness_fn)
    losses_stiffness = train_fn(stiffness_model, grad_fn, key)
    
    simulate_position_fn = lambda position_trap, key: guess_sim_pos_fn(position_trap, stiffness_model.protocol(stiffness_model.coeffs), key)
    grad_fn = grad_fn_pos_unf(position_model, simulate_position_fn)
    losses_position = train_fn(position_model, grad_fn, key)

    losses["position"].append(losses_position)
    losses["stiffness"].append(losses_stiffness)
    
    coeffs["position"].append(position_model.coeffs)
    coeffs["stiffness"].append(stiffness_model.coeffs)
    
    position_model.mode = "fwd"
    stiffness_model.mode = "fwd"
    
    position_sched = position_model.protocol(position_model.coeffs)
    stiffness_sched = stiffness_model.protocol(stiffness_model.coeffs)
    
    trap_fns = [position_sched, stiffness_sched]
    
    iter_num += 1
    
  return landscapes, coeffs, losses