import time
import pickle
import os

import jax.numpy as jnp
import jax.random as random
import jax.example_libraries.optimizers as jopt

import matplotlib.pyplot as plt

import barrier_crossing.protocol as bc_protocol
import barrier_crossing.train as bct
import barrier_crossing.iterate_landscape as bcl
import barrier_crossing.models as bcm
from barrier_crossing.utils import parse_args
    
if __name__ == "__main__":
  args, p = parse_args()
  
  
  path = f"output_data/{args.landscape_name.replace(' ', '_').replace('.', '_').lower()}/"
  if not os.path.isdir(path):
    os.mkdir(path)
  # If triple well, we need to do something different
  # bh_index = int(sys.argv[1]) # For testing multiple landscapes
  # kappa_l_list = [ x/(p.beta*x_m**2) for x in [4., 5., 6.38629, 7., 8., 8.5, 9., 10.]]
  # kappa_l = kappa_l_list[bh_index-1]
  # kappa_r = kappa_l * 10  
  
  
  # scale_arr = [2 + i/2 for i in range(8)]
  # scale = scale_arr[bh_index - 1]
  # pos_e = [[5.4*x, 8*x**2 + 0.3 * x**3 - 6 * x ** 4 + x**6] for x in jnp.linspace(-6,6,1000)] # Triple well https://www.scielo.org.mx/pdf/rmf/v57n1/v57n1a8.pdf
  # e_positions = jnp.array(pos_e)[:,0]
  # e_energies = jnp.array(pos_e)[:,1] * scale

  # The plot on the energy function underneath will be coarse-grained due to few sample points.
  #energy_custom_plot = bc_energy.V_biomolecule_recoargs.k_s, e_positions, e_energies)
  # Protocol Coefficients
  # lin_coeffs = jnp.array([-1.14278407e-07,  3.33333325e+00,  5.03512065e-10,  2.32704592e-10,
  #       9.57856017e-10, -2.86552310e-10,  1.17483601e-09, -4.50110560e-10,
  #       1.21933308e-09, -4.65200489e-10,  7.08764991e-10, -1.05334935e-10,
  #       6.99122538e-10]) # Slightly shifted

  position_model = bcm.ScheduleModel(p.param_set, p.r0_init, p.r0_final, mode = "fwd")
  stiffness_model = bcm.ScheduleModel(p.param_set, p.ks_init, p.ks_final, mode = "fwd")

  true_simulation_fwd = lambda trap_fn, ks_fn: lambda keys: p.param_set.simulate_fn(
    trap_fn, 
    ks_fn,
    keys, 
    regime = "brownian",
    fwd = True)

  sim_no_E = lambda energy_fn: lambda trap_fn, ks_fn, keys: p.param_set.simulate_fn(
    trap_fn, 
    ks_fn,
    keys, 
    regime = "brownian",
    fwd = False,
    custom = energy_fn)

  # simulate_grad_fwd = lambda energy_fn: lambda trap_fn, keys: p.param_set.simulate_fn(
  #   trap_fn, 
  #   keys, 
  #   regime = "brownian",
  #   fwd = True,
  #   custom = energy_fn)

  max_iter = 2
  opt_steps_landscape = 5 # 1000 + 
  bins = 75
  opt_batch_size = 50 # 10k + 

  grad_no_E = lambda model, simulate_fn: lambda num_batches: loss.estimate_gradient_rev(
      num_batches,
      simulate_fn,
      model)

  # grad_no_E = lambda num_batches, energy_fn: bct.estimate_gradient_work(
  #     num_batches,
  #     simulate_grad_fwd(energy_fn),
  #     p.r0_init, p.r0_final, p.param_set.simulation_steps)
  reconstruct_fn = lambda batch_works, trajectories, position_fn, stiffness_fn, batch_size: \
    bcl.energy_reconstruction(batch_works, 
                              trajectories, 
                              bins, 
                              position_fn, 
                              stiffness_fn, 
                              p.param_set.simulation_steps,
                              batch_size, 
                              p.beta)

  lr = jopt.polynomial_decay(0.1, opt_steps_landscape, 0.001)
  optimizer = jopt.adam(lr)

  train_fn = lambda model, grad_fn, key: bct.train(model, optimizer, grad_fn, key, batch_size = opt_batch_size, num_epochs = opt_steps_landscape)


  key = random.PRNGKey(int(time.time()))
  landscapes, coeffs, losses = bcl.optimize_landscape(
    max_iter,
    args.batch_size,
    position_model,
    stiffness_model,
    true_simulation_fwd,
    sim_no_E,
    grad_no_E,
    reconstruct_fn,
    train_fn,
    key)
  # landscapes, coeffs = bcl.optimize_landscape(
  #                     true_simulation_fwd,
  #                     lin_coeffs, # First reconstruction; should one reconstruct with forward or reverse simulations? Does it matter?
  #                     grad_no_E,
  #                     key,
  #                     max_iter,
  #                     bins,
  #                     p.param_set.simulation_steps,
  #                     opt_batch_size,
  #                     args.batch_size, # number of trajectories for reconstruction
  #                     opt_steps_landscape, 
  #                     optimizer,
  #                     p.r0_init, p.r0_final,
  #                     args.k_s, p.beta,
  #                     savefig = f"{args.landscape_name}",
  #                     num_reconstructions = 500
  # )
  positions = jnp.array(landscapes[-1][0])

  plt.figure(figsize = (10,10))
  #energy_plot = bc_energy.V_biomolecule_sivak(kappa_l, kappa_r, x_m, delta_E, 0., p.beta)
  #energy_plot = bc_energy.V_biomolecule_reconstructed(0., e_positions, e_energies)
  energy_plot = p.param_set.energy_fn(k_s = 0.)
  
  true_E = []
  
  pos_vec = jnp.reshape(positions, (positions.shape[0], 1, 1))
  for j in range(positions.shape[0]):
    min_pos, _ = bcl.find_min_pos(energy_plot, -12., -8.)
    true_E.append(energy_plot(pos_vec[j])-float(energy_plot([[min_pos]])))
  plt.plot(positions, true_E, label = "True Landscape")

  for num, (positions, energies) in enumerate(landscapes):
    
    min_e = jnp.min(energies[jnp.where((positions > -12) & (positions < -8))])
    if num == 0:
      label = "Linear"
    elif num == len(landscapes)-1:
      label = "Final Landscape"
    else:
      label = f"Iteration {num}"
    plt.plot(positions, energies - min_e, label = label)
  plt.legend()
  plt.xlabel("Position (x)")
  plt.ylabel("Free Energy (G)")
  plt.title(f"Iteratively Reconstructing Landscape; {args.landscape_name}; {args.end_time}")
  plt.savefig(path + f"reconstruct_landscapes_{args.k_s}_{args.end_time}_{args.batch_size}.png")
  
  plt.figure(figsize = (8,8))
  trap_fn = position_model.protocol(position_model.coef_hist[0])[0]
  plt.plot(trap_fn(jnp.arange(p.param_set.simulation_steps-1)), label = "Linear Protocol")
  for i, coeff in enumerate(coeffs["position"]):
      trap_fn = position_model.protocol(coeff)[0]
      plt.plot(trap_fn(jnp.arange(p.param_set.simulation_steps-1)), label = f"Iteration {i}")
  plt.xlabel("Simulation Step")
  plt.ylabel("Position (x)")
  plt.title(f"Protocols Over Iteration; {args.end_time}")
  plt.legend()
  plt.savefig(path + f"opt_protocol_evolution_{args.k_s}_{args.end_time}_{args.batch_size}.png")
  
  
  with open(path + f"coeffs__{args.k_s}_{args.end_time}_{args.batch_size}.pkl", "wb") as f:
    pickle.dump(coeffs, f)

print(p.param_set)