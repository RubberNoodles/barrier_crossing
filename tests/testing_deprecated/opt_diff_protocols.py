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
  k_s_sc = 0.7 # stiffness; 
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
  dt_sc = 5e-5
  simulation_steps_sc = int(end_time_sc / dt_sc)

  Neq = 500

  init_coeffs_sc = bc_protocol.linear_chebyshev_coefficients(r0_init_sc, r0_final_sc, simulation_steps_sc, degree = 12, y_intercept = r0_init_sc)
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

  batch_size_sc = 400 # Number of simulations ran. i.e. # of trajectories

  total_works, (batch_trajectories, batch_works, batch_log_probs) = bc_simulate.batch_simulate_harmonic(
    batch_size_sc, energy_sivak, simulate_sivak_fn_fwd, simulation_steps_sc, key)
  mean_traj = jnp.mean(batch_trajectories, axis = 0)
  if int(jnp.where(mean_traj > 0.)[0].shape[0]) == 0:
    cross_barrier_time = int(simulation_steps_sc/2)
  else:
    cross_barrier_time = jnp.where(mean_traj > 0.) * dt_sc

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

  # Where do I want to sample the Jaryzynski error?
  grad_sampling_timesteps_sc = jnp.array([int(cross_barrier_time/dt_sc), int(2*simulation_steps_sc/3), simulation_steps_sc])

  batch_size = 40 # Number of simulations/trajectories simulated. GPU optimized.
  opt_steps = 3 # Number of gradient descent steps to take.

  lr = jopt.exponential_decay(0.3, opt_steps, 0.01)
  optimizer = jopt.adam(lr)

  opt_coeffs = []

  for sample_timestep in grad_sampling_timesteps_sc:
    r0_final_iter = mean_traj[sample_timestep][0][0]
    init_lin_coeffs = bc_protocol.linear_chebyshev_coefficients(r0_init_sc, r0_final_iter, simulation_steps_sc, degree = 12, y_intercept = r0_init_sc)
    grad_no_batch = lambda num_batches: bc_optimize.estimate_gradient_rev(
        num_batches,
        energy_sivak,
        init_position_rev_sc,
        r0_init_sc,
        r0_final_iter,
        Neq,
        shift,
        simulation_steps_sc,
        dt_sc,
        temperature_sc,
        mass_sc,
        gamma_sc,
        beta_sc)

    coeffs, summaries, losses = bc_optimize.optimize_protocol(init_lin_coeffs, grad_no_batch, optimizer, batch_size, opt_steps)

    opt_coeffs.append((r0_final_iter, coeffs[-1][1]))

  with open("./coeffs_opt_diff_protocol.pkl", "wb") as f:
    pickle.dump(opt_coeffs, f)

  bins = 3
  batch_size_sc_rec = 40

  _, ax = plt.subplots(1,2, figsize = (12,8))

  time_vec = jnp.arange(simulation_steps_sc)


  lin_coeffs = bc_protocol.linear_chebyshev_coefficients(r0_init_sc, r0_final_iter, simulation_steps_sc, degree = 12, y_intercept = r0_init_sc)

  energy_sivak_plot = bc_energy.V_biomolecule_sivak(kappa_l, kappa_r, x_m, delta_E, 0, beta_sc)
  sivak_E = []
  positions = jnp.linspace(r0_init_sc-10, r0_final_sc+10, num = 100)
  for i in positions:
    sivak_E.append(energy_sivak_plot([[i]], r0=0.)-45)
  ax[1].plot(positions, sivak_E, label = "Ground Truth")

  trap_fn = bc_protocol.make_trap_fxn(time_vec, lin_coeffs, r0_init_sc, r0_final_sc)
  ax[0].plot(time_vec, trap_fn(time_vec), label = f"Linear Protocol")
  for position, coef in opt_coeffs:
    trap_fn = bc_protocol.make_trap_fxn(time_vec, coef, r0_init_sc, r0_final_sc)
    ax[0].plot(time_vec, trap_fn(time_vec), label = f"Optimized around {position:.1f}")
    simulate_sivak_fn_test_trap = lambda energy_fn, keys: bc_simulate.simulate_brownian_harmonic(
      energy_fn, 
      init_position_fwd_sc, 
      trap_fn,
      simulation_steps_sc, 
      Neq, 
      shift, 
      keys, 
      dt_sc,
      temperature_sc, mass_sc, gamma_sc # These parameters describe the state of the brownian system.
      )
    
    total_works, (batch_trajectories, batch_works, batch_log_probs) = bc_simulate.batch_simulate_harmonic(
        batch_size_sc_rec, energy_sivak, simulate_sivak_fn_fwd, simulation_steps_sc, key)

    midpoints, energies = bc_landscape.energy_reconstruction(batch_works, batch_trajectories, bins, trap_fn, simulation_steps_sc, batch_size_sc_rec, k_s_sc, beta_sc)

    energy_fn = bc_energy.V_biomolecule_reconstructed(0., midpoints, energies)
    delta_F = energy_fn([[position]], r0 = 0.) - energy_fn([[r0_init_sc]], r0 = 0.)
    
    real_delta_F = energy_sivak_plot([[position]], r0 = 0.) - energy_sivak_plot([[r0_init_sc]], r0 = 0.)
    ax[1].plot(midpoints, energies, label = f"ΔF = {float(delta_F):.3f} vs. {float(real_delta_F):.3f}")

  ax[0].set_title("Protocols Optimized at Different Timesteps")
  ax[1].set_title("Reconstructions for Optimized Protocols")

  ax[1].set_xlabel("Particle Position")
  ax[1].set_ylabel("Free Energy")

  ax[0].legend()
  ax[1].legend()

  plt.savefig("./opt_protocols_reconstructions.png")
