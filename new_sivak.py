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

import matplotlib.pyplot as plt

def plot_with_stddev(x, label=None, n=1, axis=0, ax=plt, dt=1.):
  stddev = jnp.std(x, axis)
  mn = jnp.mean(x, axis)
  xs = jnp.arange(mn.shape[0]) * dt

  ax.fill_between(xs,
                  mn + n * stddev, mn - n * stddev, alpha=.3)
  ax.plot(xs, mn, label=label)

if __name__ == "__main__":
  
  N = 1
  dim = 1

  # ================= SIVAK & CROOKE =====================

  # Harmonic Trap Parameters S&C
  k_s_sc = 0.4 # stiffness; 
  r0_init_sc = -10. #nm; initial trap position
  r0_final_sc = 10. #nm; final trap position

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

  energy_sivak = bc_energy.V_biomolecule_sivak(kappa_l, kappa_r, x_m, delta_E, k_s_sc, beta_sc)
  
  end_time_sc = 0.01
  dt_sc = 1e-6
  simulation_steps_sc = int(end_time_sc / dt_sc)
  
  Neq = 500
  
  init_coeffs_sc = jnp.array([ 7.0646591e-02  3.3362978e+00  5.1728708e-01  1.0489219e+00
 -2.1314605e-01  4.8330837e-01  6.5749837e-03  6.6173322e-02
  1.0880868e-01  1.5066752e-01  1.2060404e-02  1.8294835e-02
 -7.4395780e-03  6.5247095e-03  3.5647728e-02 -1.1163305e-02
  6.0015433e-02 -6.1670109e-03  1.6130991e-02  3.9281033e-02
  1.6379565e-02  3.1714685e-02  1.4345808e-02  1.7843667e-02
 -1.1081177e-04]))
  trap_fn_fwd_sc = bc_protocol.make_trap_fxn(jnp.arange(simulation_steps_sc), init_coeffs_sc, r0_init_sc, r0_final_sc)
  trap_fn_rev_sc = bc_protocol.make_trap_fxn_rev(jnp.arange(simulation_steps_sc), init_coeffs_sc, r0_init_sc, r0_final_sc)
  
  _, shift = space.free() # Defines how to move a particle by small distances dR.
  # RNG
  key = random.PRNGKey(int(time.time()))
  key, split = random.split(key, 2)  
  
  simulate_sivak_fn_fwd = lambda energy_fn, keys: bc_simulate.simulate_brownian_harmonic(
    energy_fn, 
    init_position_fwd_sc, 
    trap_fn_fwd_sc,
    simulation_steps_sc, 
    Neq, 
    shift, 
    keys, 
    dt_sc,
    temperature_sc, mass_sc, gamma_sc # These parameters describe the state of the brownian system.
    )

  batch_size_sc = 1000 # Number of simulations ran. i.e. # of trajectories

  total_works, (batch_trajectories, batch_works, batch_log_probs) = bc_simulate.batch_simulate_harmonic(
    batch_size_sc, energy_sivak, simulate_sivak_fn_fwd, trap_fn_fwd_sc, simulation_steps_sc, key)

  cross_barrier_time = jnp.where(jnp.mean(batch_trajectories, axis = 0) > 0.)[0][0] * dt_sc

  simulate_sivak_fn_rev = lambda energy_fn, keys: bc_simulate.simulate_brownian_harmonic(
      energy_fn, 
      init_position_rev_sc, 
      trap_fn_rev_sc, #trap_fn_rev2_sc,
      simulation_steps_sc, 
      Neq, 
      shift, 
      keys, 
      dt_sc,
      temperature_sc, mass_sc, gamma_sc # These parameters describe the state of the brownian system.
      )

  # Where to weight the accumulated gradient the most...
  batch_size_sc = 1000 # Number of simulations ran. i.e. # of trajectories

  # Even Sampling weighted to second well 
  cb_timestep = int(cross_barrier_time / dt_sc)
  grad_sampling_timesteps_sc = jnp.arange(int((simulation_steps_sc - cb_timestep)/200))* 200 + cb_timestep

  grad_no_batch = lambda num_batches: bc_optimize.estimate_gradient_acc_rev(
      num_batches,
      energy_sivak,
      init_position_rev_sc,
      r0_init_sc,
      r0_final_sc,
      Neq,
      shift,
      simulation_steps_sc,
      dt_sc,
      temperature_sc,
      mass_sc,
      gamma_sc,
      beta_sc,
      grad_sampling_timesteps_sc)
  
  batch_size = 5 # Number of simulations/trajectories simulated. GPU optimized.
  opt_steps = 3 # Number of gradient descent steps to take.

  lr = jopt.exponential_decay(0.3, opt_steps, 0.01)
  optimizer = jopt.adam(lr)

  coeffs, summaries, losses = bc_optimize.optimize_protocol(init_coeffs_sc, grad_no_batch, optimizer, batch_size, opt_steps)

  with open("./coeffs_sivak.pkl", "wb") as f:
    pickle.dump(coeffs, f)
  
  with open("./losses_sivak.pkl", "wb") as f:
    pickle.dump(losses, f)

  def plot_with_stddev(x, label=None, n=1, axis=0, ax=plt, dt=1.):
    stddev = jnp.std(x, axis)
    mn = jnp.mean(x, axis)
    xs = jnp.arange(mn.shape[0]) * dt

    ax.fill_between(xs,
                    mn + n * stddev, mn - n * stddev, alpha=.3)
    ax.plot(xs, mn, label=label)
  _, ax = plt.subplots(1, 2, figsize=[24, 12])

  avg_loss = jnp.mean(jnp.array(losses), axis = 0)
  plot_with_stddev(avg_loss.T, ax=ax[0])

  # ax[0].set_title(f'Jarzynski Error over Optimization; Short trap; STD error sampling; {batch_size}; {opt_steps}.')
  ax[0].set_title(f'Jarzynski Error; Short Trap; End Error sampling; {batch_size}; {opt_steps}.')
  ax[0].set_xlabel('Number of Optimization Steps')
  ax[0].set_ylabel('Error')

  trap_fn = bc_protocol.make_trap_fxn(jnp.arange(simulation_steps_sc), init_coeffs_sc, r0_init_sc, r0_final_sc)
  init_sched = trap_fn(jnp.arange(simulation_steps_sc))
  ax[1].plot(jnp.arange(simulation_steps_sc), init_sched, label='Initial guess')

  for i, (_, coeff) in enumerate(coeffs):
    if i%25 == 0 and i!=0:
      trap_fn = bc_protocol.make_trap_fxn(jnp.arange(simulation_steps_sc),coeff,r0_init_sc,r0_final_sc)
      full_sched = trap_fn(jnp.arange(simulation_steps_sc))
      ax[1].plot(jnp.arange(simulation_steps_sc), full_sched, '-', label=f'Step {i}')

  # Plot final estimate:
  trap_fn = bc_protocol.make_trap_fxn(jnp.arange(simulation_steps_sc), coeffs[-1][1],r0_init_sc,r0_final_sc)
  full_sched = trap_fn(jnp.arange(simulation_steps_sc))
  ax[1].plot(jnp.arange(simulation_steps_sc), full_sched, '-', label=f'Final')


  ax[1].legend()#
  ax[1].set_title('Schedule evolution')

  plt.savefig("./evolution.png")
  
    
  batch_size_sc_rec = 1000
  bins = 4

  lin_coeffs_sc = bc_protocol.linear_chebyshev_coefficients(r0_init_sc, r0_final_sc, simulation_steps_sc, degree=24, y_intercept = 0)
  lin_trap_fn_sc = bc_protocol.make_trap_fxn(jnp.arange(simulation_steps_sc), lin_coeffs_sc, r0_init_sc, r0_final_sc)
  opt_trap_fn_sc = bc_protocol.make_trap_fxn(jnp.arange(simulation_steps_sc), coeffs[-1][1], r0_init_sc, r0_final_sc)

  total_works, (batch_trajectories, batch_works, batch_log_probs) = bc_simulate.batch_simulate_harmonic(
      batch_size_sc_rec, energy_sivak, simulate_sivak_fn_fwd, lin_trap_fn_sc, simulation_steps_sc, key)

  midpoints_lin, energies_lin = bc_landscape.energy_reconstruction(batch_works, batch_trajectories, bins, lin_trap_fn_sc, simulation_steps_sc, batch_size_sc_rec, k_s_sc, beta_sc)

  total_works, (batch_trajectories, batch_works, batch_log_probs) = bc_simulate.batch_simulate_harmonic(
      batch_size_sc_rec, energy_sivak, simulate_sivak_fn_fwd, opt_trap_fn_sc, simulation_steps_sc, key)

  midpoints_opt, energies_opt = bc_landscape.energy_reconstruction(batch_works, batch_trajectories, bins, opt_trap_fn_sc, simulation_steps_sc, batch_size_sc_rec, k_s_sc, beta_sc)
    
  plt.figure(figsize = (8,8))
  plt.plot(midpoints_lin, energies_lin, label = "Linear Protocol")
  plt.plot(midpoints_opt, energies_opt, label = "Error-Optimized Protocol")
  plt.title("Reconstructing Sivak Landscape with Different Protocols")
  plt.xlabel("Particle Position")
  plt.ylabel("Free Energy")

  energy_sivak_plot = bc_energy.V_biomolecule_sivak(kappa_l, kappa_r, x_m, delta_E, 0, beta_sc)
  sivak_E = []
  positions = jnp.linspace(r0_init_sc-10, r0_final_sc+10, num = 100)
  for i in positions:
    sivak_E.append(energy_sivak_plot([[i]], r0=0.)-37)
  plt.plot(positions, sivak_E, label = "Ground Truth")
  plt.legend()
  plt.savefig("./landscape_reconstruction.png")
    
  # max_iter = 5
  # opt_steps_landscape = 3
  # bins = 3

  # _, shift = space.free()

  # grad_no_E = lambda num_batches, energy_fn, error_samples: bc_optimize.estimate_gradient_acc_rev(
  #     num_batches, energy_fn, init_position_rev_sc, 
  #     r0_init_sc, r0_final_sc, Neq, shift, 
  #     simulation_steps_sc, dt_sc, 
  #     temperature_sc, mass_sc, gamma_sc, beta_sc,
  #     error_samples)

  # landscapes, coeffs, positions = bc_landscape.optimize_landscape(energy_sivak,
  #                     simulate_sivak_fn_rev,
  #                     trap_fn_rev_sc, 
  #                     init_coeffs_sc, 
  #                     grad_no_E,
  #                     key,
  #                     max_iter,
  #                     bins,
  #                     simulation_steps_sc,
  #                     batch_size,
  #                     opt_steps_landscape, optimizer,
  #                     r0_init_sc, r0_final_sc,
  #                     k_s_sc, beta_sc,
  #                     grad_sampling_timesteps_sc)

  # positions = jnp.array(positions)
    
  # plt.figure(figsize = (10,10))
  # true_E = []
  # pos_vec = jnp.reshape(positions, (positions.shape[0], 1, 1))
  # for j in range(positions.shape[0]):
  #   true_E.append(energy_sivak(pos_vec[j]))
  # plt.plot(positions, true_E, label = "True Landscape")

  # for num, energies in enumerate(landscapes):
  #   plt.plot(positions, energies, label = f"Iteration {num}")

  # plt.plot(positions, landscapes[-1], label = "Final Landscape")
  # plt.legend()
  # plt.xlabel("Position (x)")
  # plt.ylabel("Free Energy (G)")
  # plt.savefig("./iterate_landscape.png")
