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
_, ax = plt.subplots(1, 2, figsize=[16, 8])

if __name__ == "__main__":
  ###  LEGEND ###
  # S&C --> Parameters from Sivak and Crooke 2016
  # G&D --> Parameters from Geiger and Dellago 2010

  N = 1
  dim = 1

  _, shift = space.free() # Defines how to move a particle by small distances dR.

  key = random.PRNGKey(int(time.time()))
  key, split = random.split(key, 2)  # RNG

  # ================= SIVAK & CROOKE =====================

  # Harmonic Trap Parameters S&C
  #k_s_sc = 0.4 # stiffness; 
  k_s_sc = 0.6 # stiffness; 
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
  
  # Protcol + Simulation Parameters
  end_time_gd = 1.
  dt_gd = 0.002
  simulation_steps_gd = int(end_time_gd / dt_gd)

  end_time_sc = 0.01
  # dt_sc = 2e-8 this might be exceeding floating point precision or something..
  end_time_sc = 0.01
  dt_sc = 1e-6
  simulation_steps_sc = int(end_time_sc / dt_sc)

  end_time_custom = 1.
  dt_custom = 0.001
  simulation_steps_custom = int(end_time_custom / dt_custom)

  # Equilibration Steps; in order to correctly apply Jarzynski, the system has to 
  # be in equilibrium, which is defined by equal free energy in all degrees of freedom
  Neq = 500

  # Protocol Coefficients
  lin_coeffs_sc = bc_protocol.linear_chebyshev_coefficients(r0_init_sc, r0_final_sc, simulation_steps_sc, degree = 12, y_intercept = r0_init_sc)

  # We want to compare between three or four different protocols.
  time_vec = jnp.arange(simulation_steps_sc)

  ## Linear
  linear_trap_fn_fwd_sc = bc_protocol.make_trap_fxn(jnp.arange(simulation_steps_sc), lin_coeffs_sc, r0_init_sc, r0_final_sc)
  linear_trap_fn_rev_sc = bc_protocol.make_trap_fxn_rev(jnp.arange(simulation_steps_sc), lin_coeffs_sc, r0_init_sc, r0_final_sc)

  ## Handdrawn, custom
  test_sim_steps = 10000
  c = simulation_steps_sc / test_sim_steps
  timestep_trap_position_sc = jnp.array([[c*10.,-5.], [c*500., -2.5], [c*1600,-0.8],[c*3000, -1.0], [c*4000,-1.2],[c*5000, -0.6], [c*6000,0.], [c*7500,0.5],[c*9500,3.0], [c*10000,5.]])

  # Make a custom trap to fit a chebyshev polynomial
  custom_trap_fn_fwd = bc_protocol.make_custom_trap_fxn(jnp.arange(simulation_steps_sc), timestep_trap_position_sc, r0_init_sc, r0_final_sc)
  sampling = ((jnp.arange(100)+1)*99*c).astype(int)

  custom_coefs = onp.polynomial.Chebyshev.fit(jnp.arange(100)/100, custom_trap_fn_fwd(sampling), deg = 24, domain=[0,1]).coef

  custom_trap_fn_fwd_sc = bc_protocol.make_trap_fxn(time_vec, custom_coefs, r0_init_sc, r0_final_sc)
  custom_trap_fn_rev_sc = bc_protocol.make_trap_fxn_rev(time_vec, custom_coefs, r0_init_sc, r0_final_sc)
  
  ## Theoretical Optimal
  ### TODO ###

  batch_size = 3000 # Number of simulations/trajectories simulated. GPU optimized.
  opt_steps = 3 # Number of gradient descent steps to take.
  lr = jopt.polynomial_decay(0.3, opt_steps, 0.001)

  ### Error

  simulate_sivak_fn_rev = lambda energy_fn, keys: bc_simulate.simulate_brownian_harmonic(
      energy_fn, 
      init_position_rev_sc, 
      linear_trap_fn_rev_sc, 
      simulation_steps_sc, 
      Neq, 
      shift, 
      keys, 
      dt_sc,
      temperature_sc, mass_sc, gamma_sc # These parameters describe the state of the brownian system.
      )


  grad_rev = lambda num_batches: bc_optimize.estimate_gradient_rev(
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
      beta_sc)

  optimizer = jopt.adam(lr)

  coeffs_err, summaries_err, losses_err = bc_optimize.optimize_protocol(lin_coeffs_sc, grad_rev, optimizer, batch_size, opt_steps)

  with open("./efn_pkl/coeffs_err_sc.pkl", "wb") as f:
    pickle.dump(coeffs_err, f)
  with open("./efn_pkl/losses_err_sc.pkl", "wb") as f:
    pickle.dump(coeffs_err, f)

  _, ax = plt.subplots(1, 2, figsize=[24, 12])
  
  plot_with_stddev(jnp.array(losses_err).T, ax=ax[0])

  # ax[0].set_title(f'Jarzynski Error over Optimization; Short trap; STD error sampling; {batch_size}; {opt_steps}.')
  ax[0].set_title(f'Jarzynski Error; {batch_size} Realizations; {opt_steps} Optimization Steps.')
  ax[0].set_xlabel('Number of Optimization Steps')
  ax[0].set_ylabel('Error')

  trap_fn = bc_protocol.make_trap_fxn(jnp.arange(simulation_steps_sc), lin_coeffs_sc, r0_init_sc, r0_final_sc)
  init_sched = trap_fn(jnp.arange(simulation_steps_sc))
  ax[1].plot(jnp.arange(simulation_steps_sc), init_sched, label='Initial guess')

  for i, (_, coeff) in enumerate(coeffs_err):
    if i%100 == 0 and i!=0:
      trap_fn = bc_protocol.make_trap_fxn(jnp.arange(simulation_steps_sc),coeff,r0_init_sc,r0_final_sc)
      full_sched = trap_fn(jnp.arange(simulation_steps_sc))
      ax[1].plot(jnp.arange(simulation_steps_sc), full_sched, '-', label=f'Step {i}')

  # Plot final estimate:
  error_trap_fn_fwd_sc = bc_protocol.make_trap_fxn(jnp.arange(simulation_steps_sc), coeffs_err[-1][1],r0_init_sc,r0_final_sc)
  full_sched = error_trap_fn_fwd_sc(jnp.arange(simulation_steps_sc))
  ax[1].plot(jnp.arange(simulation_steps_sc), full_sched, '-', label=f'Final')

  ax[1].set_xlabel("Timesteps")
  ax[1].set_ylabel("Particle Position")
  ax[1].legend()#
  ax[1].set_title('Schedule evolution')

  plt.savefig("./efn_plots/error_opt.png")

  error_trap_fn_rev_sc = bc_protocol.make_trap_fxn_rev(jnp.arange(simulation_steps_sc), coeffs_err[-1][1],r0_init_sc,r0_final_sc)

  ### Work
  simulate_sivak_fn_fwd = lambda energy_fn, keys: bc_simulate.simulate_brownian_harmonic(
      energy_fn, 
      init_position_fwd_sc, 
      linear_trap_fn_fwd_sc, 
      simulation_steps_sc, 
      Neq, 
      shift, 
      keys, 
      dt_sc,
      temperature_sc, mass_sc, gamma_sc # These parameters describe the state of the brownian system.
      )


  grad_work = lambda num_batches: bc_optimize.estimate_gradient_work(
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
      gamma_sc)

  optimizer = jopt.adam(lr)

  coeffs_work, summaries_work, losses_work = bc_optimize.optimize_protocol(lin_coeffs_sc, grad_work, optimizer, batch_size, opt_steps)

  with open("./efn_pkl/coeffs_work_sc.pkl", "wb") as f:
    pickle.dump(coeffs_work, f)
  with open("./efn_pkl/losses_work_sc.pkl", "wb") as f:
    pickle.dump(coeffs_work, f)
  
  plot_with_stddev(jnp.array(losses_work).T, ax=ax[0])

  # ax[0].set_title(f'Jarzynski Error over Optimization; Short trap; STD error sampling; {batch_size}; {opt_steps}.')
  ax[0].set_title(f'Minimize Work; {batch_size} Realizations; {opt_steps} Optimization Steps.')
  ax[0].set_xlabel('Number of Optimization Steps')
  ax[0].set_ylabel('Work')

  trap_fn = bc_protocol.make_trap_fxn(jnp.arange(simulation_steps_sc), lin_coeffs_sc, r0_init_sc, r0_final_sc)
  init_sched = trap_fn(jnp.arange(simulation_steps_sc))
  ax[1].plot(jnp.arange(simulation_steps_sc), init_sched, label='Initial guess')

  for i, (_, coeff) in enumerate(coeffs_work):
    if i%100 == 0 and i!=0:
      trap_fn = bc_protocol.make_trap_fxn(jnp.arange(simulation_steps_sc),coeff,r0_init_sc,r0_final_sc)
      full_sched = trap_fn(jnp.arange(simulation_steps_sc))
      ax[1].plot(jnp.arange(simulation_steps_sc), full_sched, '-', label=f'Step {i}')

  # Plot final estimate:
  work_trap_fn_fwd_sc = bc_protocol.make_trap_fxn(jnp.arange(simulation_steps_sc), coeffs_work[-1][1],r0_init_sc,r0_final_sc)
  full_sched = work_trap_fn_fwd_sc(jnp.arange(simulation_steps_sc))
  ax[1].plot(jnp.arange(simulation_steps_sc), full_sched, '-', label=f'Final')

  ax[1].set_xlabel("Timesteps")
  ax[1].set_ylabel("Particle Position")
  ax[1].legend()#
  ax[1].set_title('Schedule evolution')

  plt.savefig("./efn_plots/work_opt.png")

  work_trap_fn_rev_sc = bc_protocol.make_trap_fxn_rev(jnp.arange(simulation_steps_sc), coeffs_work[-1][1],r0_init_sc,r0_final_sc)

  ### We have all the trap functions, now we proceed to 
  ### Perform forward simulations to see how they perform.
  trap_fns = {
      "forward": {
          "linear": linear_trap_fn_fwd_sc, 
          "custom": custom_trap_fn_fwd_sc, 
          "error": error_trap_fn_fwd_sc, 
          "work": work_trap_fn_fwd_sc
      },
      "reverse": {
          "linear": linear_trap_fn_rev_sc, 
          "custom": custom_trap_fn_rev_sc, 
          "error": error_trap_fn_rev_sc, 
          "work": work_trap_fn_rev_sc
      }
  }

  err_stds = {
      "dF_estimate": {"linear": [], "custom": [], "error": [], "work": []}, 
      "dF_std": {"linear": [], "custom": [], "error": [], "work": []}, 
      "work": {"linear": [], "custom": [], "error": [], "work": []},
      "work_std": {"linear": [], "custom": [], "error": [], "work": []},
      "jc_loss": {"linear": [], "custom": [], "error": [], "work": []}, 
      "jc_loss_std": {"linear": [], "custom": [], "error": [], "work": []},
  }

  test_batch_size = 1e4
  num_samples = 3

  for pro_type in trap_fns["forward"].keys():
      simulate_sivak_fwd = lambda energy_fn, keys: bc_simulate.simulate_brownian_harmonic(
      energy_fn, 
      init_position_fwd_sc, 
      trap_fns["forward"][pro_type],
      simulation_steps_sc, 
      Neq, 
      shift, 
      keys, 
      dt_sc,
      temperature_sc, mass_sc, gamma_sc # These parameters describe the state of the brownian system.
      )

      simulate_sivak_rev = lambda energy_fn, keys: bc_simulate.simulate_brownian_harmonic(
      energy_fn, 
      init_position_fwd_sc, 
      trap_fns["reverse"][pro_type],
      simulation_steps_sc, 
      Neq, 
      shift, 
      keys, 
      dt_sc,
      temperature_sc, mass_sc, gamma_sc # These parameters describe the state of the brownian system.
      )
      for _ in tqdm.tqdm(range(num_samples)):

          key, split = random.split(key,2)

          total_works_fwd, _ = bc_simulate.batch_simulate_harmonic(
              test_batch_size, energy_sivak, simulate_sivak_fwd, simulation_steps_sc, key)

          # Assuming Jarzynski equality holds, then we can do this to retrieve what our
          # code thinks dF is.
          dF_estimate = -1/beta_sc * jnp.log(jnp.mean(jnp.exp(-beta_sc * total_works_fwd)))
          dF_std = -1/beta_sc * jnp.log(jnp.std(jnp.exp(-beta_sc * total_works_fwd))) # Lol idk what i'm doing here.

          work = jnp.mean(total_works_fwd)
          work_std = jnp.std(total_works_fwd)

          total_works_rev, _ = bc_simulate.batch_simulate_harmonic(
              test_batch_size, energy_sivak, simulate_sivak_rev, simulation_steps_sc, key)

          jc_loss = jnp.mean(jnp.exp(beta_sc * total_works_rev))
          jc_loss_std = jnp.std(jnp.exp(beta_sc * total_works_rev))

          err_stds["dF_estimate"][pro_type].append(dF_estimate)
          err_stds["dF_std"][pro_type].append(dF_std)
          err_stds["work"][pro_type].append(work)
          err_stds["work_std"][pro_type].append(work_std)
          err_stds["jc_loss"][pro_type].append(jc_loss)
          err_stds["jc_loss_std"][pro_type].append(jc_loss_std)

  # All that remains is to make graphs histograms and graphs probably.

  protocol_types = list(trap_fns["forward"].keys())


  _, (ax_0, ax_1) = plt.subplots(1,2, figsize=(12,6))
  histograms_err = [err_stds["dF_estimate"][pro_type] for pro_type in protocol_types]
  histograms_std = [err_stds["dF_std"][pro_type] for pro_type in protocol_types]
  ax_0.hist(histograms_err, density=False, bins=20, label = protocol_types)  # density=False would make counts
  ax_0.legend()
  ax_0.set_title("Estimate for ΔF = 7.0")
  ax_0.set_ylabel('Number')
  ax_0.set_xlabel('Estimate')
  ax_1.hist(histograms_std, density=False, bins=20, label = protocol_types)  # density=False would make counts
  ax_1.legend()
  ax_1.set_title("Standard Deviaton for ΔF = 7.0")
  ax_1.set_ylabel('Number')
  ax_1.set_xlabel('Std')

  plt.savefig("./efn_plots/df_estimate_hist.png")


  _, (ax_0, ax_1) = plt.subplots(1,2, figsize=(12,6))
  histograms_err = [err_stds["work"][pro_type] for pro_type in protocol_types]
  histograms_std = [err_stds["work_std"][pro_type] for pro_type in protocol_types]
  ax_0.hist(histograms_err, density=False, bins=20, label = protocol_types)  # density=False would make counts
  ax_0.legend()
  ax_0.set_title("Dissipated Work")
  ax_0.set_ylabel('Number')
  ax_0.set_xlabel('Estimate')
  ax_1.hist(histograms_std, density=False, bins=20, label = protocol_types)  # density=False would make counts
  ax_1.legend()
  ax_1.set_title("STD of Dissipated Work")
  ax_1.set_ylabel('Number')
  ax_1.set_xlabel('Std')

  plt.savefig("./efn_plots/work_loss_hist.png")


  _, (ax_0, ax_1) = plt.subplots(1,2, figsize=(12,6))
  histograms_err = [err_stds["jc_loss"][pro_type] for pro_type in protocol_types]
  histograms_std = [err_stds["jc_loss_std"][pro_type] for pro_type in protocol_types]
  ax_0.hist(histograms_err, density=False, bins=20, label = protocol_types)  # density=False would make counts
  ax_0.legend()
  ax_0.set_title("Log Jarzynski Error (Crooks Proxy)")
  ax_0.set_ylabel('Number')
  ax_0.set_xlabel('Log Estimate')
  ax_1.hist(histograms_std, density=False, bins=20, label = protocol_types)  # density=False would make counts
  ax_1.legend()
  ax_1.set_title("Jarzynski Error Log STD")
  ax_1.set_ylabel('Number')
  ax_1.set_xlabel('Log Std')

  plt.savefig("./efn_plots/jc_loss_hist.png")
  
  with open("./efn_pkl/err_stds.pkl", "wb") as f:
    pickle.dump(err_stds, f)
  
  