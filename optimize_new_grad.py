import time
import tqdm
import pickle

import barrier_crossing.energy as bc_energy
import barrier_crossing.protocol as bc_protocol
import barrier_crossing.simulate as bc_simulate
import barrier_crossing.optimize as bc_optimize
import barrier_crossing.iterate_landscape as bc_landscape

import jax

import jax.numpy as jnp
import numpy as onp

import jax.random as random

import jax.example_libraries.optimizers as jopt

from jax_md import space

import math

import matplotlib.pyplot as plt

def plot_with_stddev(x, label=None, n=1, axis=0, ax=plt, dt=1.):
  stddev = jnp.std(x, axis)
  mn = jnp.mean(x, axis)
  xs = jnp.arange(mn.shape[0]) * dt

  ax.fill_between(xs,
                  mn + n * stddev, mn - n * stddev, alpha=.3)
  ax.plot(xs, mn, label=label)

def init_trap_fn(x): # a "cubic" function for initial trap
    return (8*10**(-11))*(x-5000)**3

def mapper(x, min_x, max_x, min_to, max_to): #for cheb_coef
    return (x - min_x) / (max_x - min_x) * (max_to - min_to) + min_to

def cheb_coef(func, n, min, max): #returns chebyshev coefficients for a function
    coef = [0.0] * n
    for i in range(n):
        f = func(mapper(math.cos(math.pi * (i + 0.5) / n), -1, 1, min, max)) * 2 / n
        for j in range(n):
            coef[j] += f * math.cos(math.pi * j * (i + 0.5) / n)
    return jnp.array(coef)
  
def midpoints_to_timesteps(energy_fn, simulate_fn, rev_trap_fn, simulation_steps, key, batch_size, midpoints):
  """Given a (reversed) protocol and and array midpoints, return an array of the average time it takes for the particle to reach
  each midpoint.
  Returns:
    Array[]
  """
  total_works, (trajectories, works, log_probs) = batch_simulate_harmonic(batch_size,
                            energy_fn,
                            simulate_fn,
                            simulation_steps,
                            key)

  mean_traj = jnp.mean(trajectories, axis = 0)
  midpoint_timestep = []
  time_step = 0

  for midpoint in midpoints:
    while float(mean_traj[time_step]) > midpoint:
      time_step = time_step + 1
    
    midpoint_timestep.append(time_step)

  midpoint_timestep = jnp.array(midpoint_timestep)
  return midpoint_timestep

  
if __name__ == "__main__":

  N = 1
  dim = 1

  # Harmonic Trap Parameters S&C
  k_s_sc = 0.7 # stiffness; 
  r0_init_sc = -10. #nm; initial trap position
  r0_final_sc = 15. #nm; final trap position

  # Particle Parameters S&C
  mass_sc = 1e-17 # g
  init_position_fwd_sc = r0_init_sc*jnp.ones((N,dim)) #nm
  init_position_rev_sc = r0_final_sc*jnp.ones((N,dim))

  # Brownian Environment S&C
  temperature_sc = 4.183 #at 303K=30C S&C
  beta_sc=1.0/temperature_sc #1/(pNnm)
  D_sc = 0.44*1e6 #(in nm**2/s) 
  gamma_sc = 1./(beta_sc*D_sc*mass_sc) #s^(-1)


  # S&C Energy landscape params:
  x_m=10. #nm
  delta_E=7.0 #pN nm
  kappa_l=21.3863/(beta_sc*x_m**2) #pN/nm #for Ebarrier = 10kT and delta_E=0, as C&S use
  #kappa_l=6.38629/(beta*x_m**2) #pN/nm #for Ebarrier = 2.5kT and delta_E=0, as C&S use
  #kappa_l=2.6258/(beta*x_m**2)#barrier 0.625kT
  kappa_r=kappa_l #pN/nm; Symmetric wells.
  min = -10. # x-coordinate of the minimum of first well
  max = 10. # x-coordinate of the minimum of second well

  energy_sivak = bc_energy.V_biomolecule_sivak(kappa_l, kappa_r, x_m, delta_E, k_s_sc, beta_sc)

  end_time_sc = 0.01
  dt_sc = 5e-6
  simulation_steps_sc = int(end_time_sc / dt_sc)

  Neq = 500
  batch_size = 10000 

  init_coeffs = cheb_coef(init_trap_fn, 12, 0, 10000)

  _, shift = space.free() # Defines how to move a particle by small distances dR.
  # RNG
  key = random.PRNGKey(int(time.time()))
  key, split = random.split(key, 2)  

  bins = 30 # how many samples on the landscape

  # uniform sampling:
  #error_samples = []
  #dr = simulation_steps_sc/bins
  #for i in range(bins):
  #  i += 1
  #error_samples.append(int(i * dr))
  
  # Sample uniformly, but only in the interval between the two wells (since r0_final_sc >10.):
  dr = (max - min)/bins
  midpoints = [ max - (bin_num + 0.5) * dr for bin_num in range(bins) ]
  init_trap_fn = bc_protocol.make_trap_fxn_rev(jnp.arange(simulation_steps_sc), init_coeffs, 
                                 r0_init_sc, r0_final_sc)

  simulate_fn = lambda energy_fn, keys: simulate_brownian_harmonic(energy_fn,
    init_position_rev_sc, init_trap_fn, simulation_steps_sc, Neq, shift, keys, dt_sc,
    temperature_sc, mass_sc, gamma_sc)
  
  bin_timesteps = midpoints_to_timesteps(energy_sivak, simulate_fn, init_trap_fn,
                                       simulation_steps_sc, key, batch_size_sc, midpoints)
  
  
  batch_grad_fn = lambda num_batches: bc_optimize.estimate_gradient_acc_rev2(
    bin_timesteps,
    num_batches,
    energy_sivak,
    init_position_fwd_sc,
    r0_init_sc,
    r0_final_sc,
    Neq,
    shift,
    simulation_steps_sc,
    dt_sc,
    temperature_sc,
    mass_sc,
    gamma_sc, beta_sc)
  
  # Optimization:
  opt_steps = 2 #test

  lr = jopt.exponential_decay(0.3, opt_steps, 0.03)
  optimizer = jopt.adam(lr)

  summaries = []
  coeffs_ = []
  losses = []
  
  
  init_state = optimizer.init_fn(init_coeffs)
  opt_state = optimizer.init_fn(init_coeffs)
  coeffs_.append((0,) + (optimizer.params_fn(opt_state),))

  grad_fn = batch_grad_fn(batch_size)
  opt_coeffs = []
  for j in tqdm.trange(opt_steps,position=1, desc="Optimize Protocol: ", leave = True):
    key, split = jax.random.split(key)
    grad, loss, gradient_estimator_total, summary_total = grad_fn(optimizer.params_fn(opt_state), split)
    opt_state = optimizer.update_fn(j, grad, opt_state)
    losses.append(loss)
    coeffs_.append((j,) + (optimizer.params_fn(opt_state),))


  losses = jax.tree_map(lambda *args: jnp.stack(args), *losses)
  opt_coeffs.append(coeffs_[-1][1])
  
  # ToDo: fix error with pickle
  #with open("./coeffs_opt.pkl", "wb") as f:
  #  pickle.dump(opt_coeffs, f)
  
  # save to txt for now
  with open('coeffs_optimized.txt', 'w') as f:
    f.write(str(opt_coeffs))

  _, ax = plt.subplots(1, 2, figsize=[24, 12])

  plot_with_stddev(losses.T, ax=ax[0])
  ax[0].set_title('Optimization')
  ax[0].set_xlabel('Number of Optimization Steps')
  ax[0].set_ylabel('Loss')

  trap_init = bc_protocol.make_trap_fxn(jnp.arange(simulation_steps_sc), init_coeffs, r0_init_sc,
                                     r0_final_sc,)
  ax[1].plot(jnp.arange(simulation_steps_sc), trap_init(jnp.arange(simulation_steps_sc)), label='Initial Guess')

  for j, coeff in coeffs_:
    if j%50 == 0 and j!=0:  
     trap_fn = bc_protocol.make_trap_fxn(jnp.arange(simulation_steps_sc),coeff,r0_init_sc,r0_final_sc)
     full_sched = trap_fn(jnp.arange(simulation_steps_sc))
     ax[1].plot(jnp.arange(simulation_steps_sc), full_sched, '-', label=f'Step {j}')

  # Plot final estimate:
  trap_fn = bc_protocol.make_trap_fxn(jnp.arange(simulation_steps_sc), coeffs_[-1][1],r0_init_sc,r0_final_sc)
  full_sched = trap_fn(jnp.arange(simulation_steps_sc))
  ax[1].plot(jnp.arange(simulation_steps_sc), full_sched, '-', label=f'Final')
  ax[1].set_title('Schedule evolution')

  ax[0].legend()
  ax[1].legend()
  plt.savefig("./optimization.png")



