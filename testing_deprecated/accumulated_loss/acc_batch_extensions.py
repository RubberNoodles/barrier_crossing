###################################################
# This code is a TEMPLATE for Sivak & Crooks
# testing. Last changed 1/30/2023. 
# WARNING: This code is NOT maintained.
###################################################

import time
import tqdm
import pickle

import csv
import sys

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
  
  path = f"./data/ext_{int(sys.argv[1])}/"
  
  ###  LEGEND ###
  # S&C --> Parameters from Sivak and Crooke 2016

  N = 1
  dim = 1

  _, shift = space.free() # Defines how to move a particle by small distances dR.

  key = random.PRNGKey(int(time.time()))
  key, split = random.split(key, 2)  # RNG

  # ================= SIVAK & CROOKE =====================

  # Harmonic Trap Parameters S&C
  #k_s_sc = 0.4 # stiffness; 
  k_s_sc = 0.1 # stiffness; 
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
  #kappa_l=21.3863/(beta_sc*x_m**2) #pN/nm #for Ebarrier = 10kT and delta_E=0, as C&S use
  kappa_l=6.38629/(beta_sc*x_m**2) #pN/nm #for Ebarrier = 2.5kT and delta_E=0, as C&S use
  #kappa_l=2.6258/(beta_sc*x_m**2)#barrier 0.625kT
  kappa_r=5*kappa_l #pN/nm; Symmetric wells.

  energy_sivak = bc_energy.V_biomolecule_sivak(kappa_l, kappa_r, x_m, delta_E, k_s_sc, beta_sc)

  end_time_sc = 0.01
  # dt_sc = 2e-8 this might be exceeding floating point precision or something..
  end_time_sc = 0.01
  dt_sc = 3e-6
  simulation_steps_sc = int(end_time_sc / dt_sc)

  # Equilibration Steps; in order to correctly apply Jarzynski, the system has to 
  # be in equilibrium, which is defined by equal free energy in all degrees of freedom
  Neq = 500

  # Protocol Coefficients
  lin_coeffs_sc = bc_protocol.linear_chebyshev_coefficients(r0_init_sc, r0_final_sc, simulation_steps_sc, degree = 12, y_intercept = r0_init_sc)

  # Trap Functions. Reverse mode trap functions are for when we compute Jarzynski error with reverse protocol trajectories.
  trap_fn_fwd_sc = bc_protocol.make_trap_fxn(jnp.arange(simulation_steps_sc), lin_coeffs_sc, r0_init_sc, r0_final_sc)
  trap_fn_rev_sc = bc_protocol.make_trap_fxn_rev(jnp.arange(simulation_steps_sc), lin_coeffs_sc, r0_init_sc, r0_final_sc)
  
  # simulate_sivak_fn_fwd = lambda energy_fn, keys: bc_simulate.simulate_brownian_harmonic(
  #   energy_fn, 
  #   init_position_fwd_sc, 
  #   trap_fn_fwd_sc,
  #   simulation_steps_sc, 
  #   Neq, 
  #   shift, 
  #   keys, 
  #   dt_sc,
  #   temperature_sc, mass_sc, gamma_sc # These parameters describe the state of the brownian system.
  #   )

  # total_works, (batch_trajectories, batch_works, batch_log_probs) = bc_simulate.batch_simulate_harmonic(
  #   1000, energy_sivak, simulate_sivak_fn_fwd, simulation_steps_sc, key)

  # midpoints_lin, energies_lin = bc_landscape.energy_reconstruction(batch_works, batch_trajectories, 100, trap_fn_fwd_sc, simulation_steps_sc, 1000, k_s_sc, beta_sc)
  # energy_sivak = bc_energy.V_biomolecule_reconstructed(k_s_sc, midpoints_lin, energies_lin) # reconstructed 
  with open("extensions.csv", 'r') as f:
    reader = csv.reader(f)
    count = 1
    for line in reader:
      if count == int(sys.argv[1]):
        extensions = [int(x) for x in line]
      count+=1
  
  grad_acc_rev = lambda num_batches: bc_optimize.estimate_gradient_acc_rev_trunc(
    extensions,
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

  batch_size = 10000 # Number of simulations/trajectories simulated. GPU optimized.
  opt_steps = 50 # Number of gradient descent steps to take.

  #lr = jopt.exponential_decay(0.3, opt_steps, 0.003)
  lr = jopt.polynomial_decay(1., opt_steps, 0.001)
  optimizer = jopt.adam(lr)

  coeffs, summaries, losses = bc_optimize.optimize_protocol(lin_coeffs_sc, grad_acc_rev, optimizer, batch_size, opt_steps, print_log=True)
  
  _, ax = plt.subplots(1, 2, figsize=[16, 8])
  #avg_loss = jnp.mean(jnp.array(losses), axis = 0)
  plot_with_stddev(losses.T, ax=ax[0])

  # ax[0].set_title(f'Jarzynski Error over Optimization; Short trap; STD error sampling; {batch_size}; {opt_steps}.')
  ax[0].set_title(f'Jarzynski Accumulated Error; {batch_size}; {opt_steps}; {simulation_steps_sc} steps; {extensions}')
  ax[0].set_xlabel('Number of Optimization Steps')
  ax[0].set_ylabel('Error')

  trap_fn = bc_protocol.make_trap_fxn(jnp.arange(simulation_steps_sc), lin_coeffs_sc, r0_init_sc, r0_final_sc)
  init_sched = trap_fn(jnp.arange(simulation_steps_sc))
  ax[1].plot(jnp.arange(simulation_steps_sc), init_sched, label='Initial guess')

  for i, (_, coeff) in enumerate(coeffs):
    if i%10 == 0 and i!=0:
      trap_fn = bc_protocol.make_trap_fxn(jnp.arange(simulation_steps_sc),coeff,r0_init_sc,r0_final_sc)
      full_sched = trap_fn(jnp.arange(simulation_steps_sc))
      ax[1].plot(jnp.arange(simulation_steps_sc), full_sched, '-', label=f'Step {i}')

  # Plot final estimate:
  trap_fn = bc_protocol.make_trap_fxn(jnp.arange(simulation_steps_sc), coeffs[-1][1],r0_init_sc,r0_final_sc)
  full_sched = trap_fn(jnp.arange(simulation_steps_sc))
  ax[1].plot(jnp.arange(simulation_steps_sc), full_sched, '-', label=f'Final')


  ax[1].legend()#
  ax[1].set_title('Schedule evolution')

  plt.savefig(path+"optimization.png")
  
  with open(path+"opt_coeffs.pkl", "wb") as f:
    pickle.dump(coeffs, f)
  
  with open(path+"opt_losses.pkl", "wb") as f:
    pickle.dump(losses, f)
    
  ### Reconstruction
  
    
  batch_size_sc_rec = 1000
  bins = 40

  lin_trap_fn_sc = bc_protocol.make_trap_fxn(jnp.arange(simulation_steps_sc), lin_coeffs_sc, r0_init_sc, r0_final_sc)
  opt_trap_fn_sc = bc_protocol.make_trap_fxn(jnp.arange(simulation_steps_sc), coeffs[-1][1], r0_init_sc, r0_final_sc)
  
  # Optimized for k_s = 0.7 (looks like cubic):
  coeffs_cubic = jnp.array([-2.4699204e-01,  6.1058893e+00, -1.7807318e-02,
                9.9026270e+00,  1.4435360e+00,  1.7319231e+01,
                1.5868250e+00,  2.0173792e+01, -5.1318378e+00,
                2.1637058e+00, -1.6113165e+01, -2.8187643e+01])
  cubic_trap_fn_sc = bc_protocol.make_trap_fxn(jnp.arange(simulation_steps_sc), coeffs_cubic, r0_init_sc, r0_final_sc)


  # Another really good protocol optimized with wrong grad for k_s = 0.1
  coeffs_old = jnp.array([ 0.92226017,  2.342257  ,  1.9017681 ,  6.224875  ,  5.144628  ,
        16.374546  ,  9.336435  , 23.34578   , 12.500979  , 23.827835  ,
        18.840345  , 31.785065  , 15.344111  ])
  old_trap_fn_sc = bc_protocol.make_trap_fxn(jnp.arange(simulation_steps_sc), coeffs_old, r0_init_sc, r0_final_sc)

  traps = {
    "Linear Protocol": (lin_trap_fn_sc, lin_coeffs_sc), 
    "Acc Optimized Protocol": (opt_trap_fn_sc, coeffs[-1][1]),
    "Cubic Protocol": (cubic_trap_fn_sc, coeffs_cubic),
    "Old Optimal Protocol": (old_trap_fn_sc, coeffs_old)
    }
  
  plot_data = {}
  
  for trap_name, (trap_fn, coeffs) in traps.items():
    key, split = random.split(key)
    simulate_sivak_fwd = lambda energy_fn, keys: bc_simulate.simulate_brownian_harmonic(
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


    total_work, (batch_trajectories, batch_works, _) = bc_simulate.batch_simulate_harmonic(
        batch_size_sc_rec, energy_sivak, simulate_sivak_fwd, simulation_steps_sc, key)
    
    mean_work = batch_works.mean()
    tail = total_work.mean() - total_work.min()

    grad, (_, summary) = grad_acc_rev(batch_size)(coeffs, split)
    # for extension in extensions:
    #   jnp.squeeze(summary[0][0]).mean(axis=0)
    # Compute the position. Don't need this right now I think.

    midpoints, energies = bc_landscape.energy_reconstruction(
        batch_works, 
        batch_trajectories, 
        bins, 
        trap_fn, 
        simulation_steps_sc, 
        batch_size_sc_rec, 
        k_s_sc, 
        beta_sc)
  
    landscape = (midpoints, energies)
    no_trap_sivak = bc_energy.V_biomolecule_sivak(kappa_l, kappa_r, x_m, delta_E, 0, beta_sc)
    
    disc = bc_landscape.landscape_discrepancies(landscape, no_trap_sivak, no_trap_sivak([[0.]]), -10., 10.)
    bias = max(disc)
    
    max_rec = bc_landscape.find_max(landscape, -10., 10.)
    difference = no_trap_sivak([[0.]]) - max_rec
    energies_aligned = []
    for energy in energies: 
      energies_aligned.append(energy + difference)
    
    no_trap_rec_fn = bc_energy.V_biomolecule_reconstructed(0, midpoints, energies_aligned)
    
    disc_samples = bc_landscape.landscape_discrepancies_samples(no_trap_rec_fn, no_trap_sivak, extensions)
    disc_samples = jnp.array(disc_samples)
    mean_disc_samples = disc_samples.mean()
    bias_samples = disc_samples.max()
    
    plot_data[trap_name] = {
      "trap": trap_fn,
      "work": total_work,
      "midpoints": midpoints,
      "energies": energies_aligned,
      "bias": bias,
      "mean_work": mean_work,
      "discrepancy": disc,
      "tail": tail,
      "loss": summary[2].mean(),
      "samples": {
        "mean_discrepancy": mean_disc_samples,
        "bias": bias_samples,
        "discrepancy": disc_samples
        }
      }
    
  
  #### PLOTTING CODE #####
  
  ## Reconstructions 
  
  plt.figure(figsize = (8,8))
  for p_name in plot_data:
    data = plot_data[p_name]
    plt.plot(data["midpoints"], data["energies"], label = p_name)
  plt.title("Reconstructing Sivak Landscape with Different Protocols")
  plt.xlabel("Particle Position")
  plt.ylabel("Free Energy")

  energy_sivak_plot = bc_energy.V_biomolecule_sivak(kappa_l, kappa_r, x_m, delta_E, 0, beta_sc)
  sivak_E = []
  positions = jnp.linspace(r0_init_sc-10, r0_final_sc+10, num = 100)
  for i in positions:
    sivak_E.append(energy_sivak_plot([[i]], r0=0.))
  plt.plot(positions, sivak_E, label = "Ground Truth", color = "b")
  plt.legend()
  plt.savefig(path+"reconstruction_all.png", transparent = False)
  
  
  plt.figure(figsize = (8,8))
  data = plot_data["Acc Optimized Protocol"]
  plt.plot(data["midpoints"], data["energies"], label = "Acc Optimized Protocol")
  plt.title("Reconstructing Sivak Landscape with Optimized Protocol")
  plt.xlabel("Particle Position")
  plt.ylabel("Free Energy")

  energy_sivak_plot = bc_energy.V_biomolecule_sivak(kappa_l, kappa_r, x_m, delta_E, 0, beta_sc)
  sivak_E = []
  positions = jnp.linspace(r0_init_sc-10, r0_final_sc+10, num = 100)
  for i in positions:
    sivak_E.append(energy_sivak_plot([[i]], r0=0.))
  plt.plot(positions, sivak_E, label = "Ground Truth", color = "b")
  plt.legend()
  plt.savefig(path+"reconstruction_opt_vs_true.png", transparent = False)
  
  ## Landscape Discrepancies
  plt.figure(figsize = (8,8))
  for p_name in plot_data:
    data = plot_data[p_name]["samples"]
    plt.plot(extensions, data["discrepancy"], '-o', label = p_name)
  plt.xlabel('Protocol/Trap Position')
  plt.ylabel('Landscape Discrepancy')
  plt.title("Discrepancy at each Extension")
  plt.legend()
  
  ## 
  
  plt.figure(figsize=[8,8])
  for p_name in plot_data:
    data = plot_data[p_name]
    plt.hist(jnp.array(data["work"])*beta_sc,20,alpha=0.5, label = f'{p_name}, mean = {data["mean_work"]:.2f}, tail length = {data["tail"]:.2f}')
  plt.xlabel("Work (kbT)")
  plt.ylabel("Counts")
  plt.legend()
  plt.title("Work Distribution for Different Protocols")
  
  plt.savefig(path + "histogram_all.png", transparent = False)


  plt.figure(figsize=[8,8])
  for p_name in ["Linear Protocol", "Acc Optimized Protocol"]:
    data = plot_data[p_name]
    plt.hist(jnp.array(data["work"])*beta_sc,20,alpha=0.7, label = f'{p_name}, mean = {data["mean_work"]:.2f}, tail length = {data["tail"]:.2f}')
  plt.xlabel("Work (kbT)")
  plt.ylabel("Counts")
  plt.legend()
  plt.title("Work Distribution for Optimized vs Linear Protocol")
  
  plt.savefig(path + "histogram_linear_vs_opt.png", transparent = False)
  

  # Table comparison of protocols --> Important graph to have! --> Adjust for different protocols etc

  # Table comparison (+ reconstruction info)
  _, (ax0, ax1) = plt.subplots(1, 2, figsize=[24, 12])
  step = jnp.arange(simulation_steps_sc)
  for p_name in plot_data:
    data = plot_data[p_name]
    ax0.plot(step, data["trap"](step), label = f'{p_name}')
  ax0.legend()
  ax0.set_title(f"Different Protocol Trajectories; {extensions}")
  columns = ('Bias', 'Mean discrepancy',"Discrepancies Samples", 'Average total work', 'Tail length', 'Loss')
  rows = []
  table_data = []
  for p_name in plot_data:
    data = plot_data[p_name]
    rows.append(p_name)
    mean_disc = float(jnp.array(data["discrepancy"]).mean())
    disc_str = "\n"
    for ch in str(data["samples"]["discrepancy"]).split(","):
      disc_str += ch + "," + "\n"

    print(disc_str)
    disc_str = "N/A"
    table_data.append([data["bias"], mean_disc, disc_str, data["mean_work"], data["tail"], data["loss"]])
  n_rows = len(table_data)
  cell_text = []
  #colors = plt.cm.BuPu(jnp.linspace(0, 0.5, len(rows)))

  for row in range(n_rows):
    y_offset = table_data[row]
    cell_text.append([x for x in y_offset])

  ax1.axis('off')

  table = plt.table(cellText=cell_text,
                        rowLabels=rows,
                        colLabels=columns,loc = 'center')
  

  table.auto_set_font_size(False)
  table.set_fontsize(12)
  table.scale(1,10)
  
  plt.savefig(path+"protocol_info.png")
