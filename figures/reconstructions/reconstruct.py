# Given coefficients (pickled); reconstruct and determine accuracy of reconstructions.

### TODO: Compute Free energy std.
import tqdm

import barrier_crossing.energy as bce
import barrier_crossing.simulate as bcs
import barrier_crossing.iterate_landscape as bcl
import barrier_crossing.models as bcm

from barrier_crossing.utils import parse_args, make_trap_from_file, find_coeff_file

import jax.numpy as jnp
import jax.random as random

import matplotlib.pyplot as plt
import seaborn as sns
# from figures.params import * # global variables;


def plot_with_stddev(x,y, label=None, n=1, axis=0, ax=plt, dt=1., color = None):
  stddev = jnp.std(y, axis)
  mn = jnp.mean(y, axis)
  ax.fill_between(x,
                  mn + n * stddev, mn - n * stddev, alpha=.3, color = color)
  ax.plot(x, mn, label=label, color = color)
  
if __name__ == "__main__":
  args, p  = parse_args()
  
  path = args.landscape_name.replace(" ", "_").replace(".", "_").lower()
  parent_dir = f"output_data/{path}/"
  coeff_dir = parent_dir + "coeffs/"
  
  if "2_5kt" in path:
    near_eq_file = "2.5kt_lr_opt.pkl"
  elif "10kt" in path:
    near_eq_file = "10kt_lr_opt.pkl"
  else:
    near_eq_file = None
  
  model_types = {
    "Linear": "linear", 
    "Work; Position": ["position","fwd"],
    "Work; Stiffness": ["stiffness","fwd"],
    "Work; Joint": ["joint","rev"],
    "Error; Position": ["position","rev"],
    "Error; Stiffness": ["stiffness","rev"],
    # "Error; Joint": ["joint","rev"],
    # "Split; Position": ["position","fwd", "split"],
    # "Split; Stiffness": ["stiffness","fwd", "split"],
    # "Split; Joint": ["joint","fwd", "split"],
    # "Split; Position": ["position","rev", "split"],
    # "Split; Stiffness": ["stiffness","rev", "split"],
    # "Split; Joint": ["joint","rev", "split"],
    "Near Equilibrium": [near_eq_file] if near_eq_file else None
    }
  
  psm = bcm.ScheduleModel(p.param_set, p.r0_init, p.r0_final)
  ssm = bcm.ScheduleModel(p.param_set, p.ks_init, p.ks_final)
  
  plot_data = {}
  plot_color_list = ['b', 'g', 'r' ,'c', 'm', 'y', 'k', 'w'] * 2
  kde_colors = []
  plt.rc('font', size = 16)
  plt.subplots_adjust(right=0.8)
  fig_rec = plt.figure(figsize = (7,7))
  fig_pro = plt.figure(figsize = (14, 7))
  fig_hist = plt.figure(figsize = (7,7))
  fig_kde = plt.figure(figsize = (7,7))
  
  ax_reconstruct = fig_rec.add_subplot(1, 1, 1)
  ax_position = fig_pro.add_subplot(1, 2, 1)
  ax_stiffness = fig_pro.add_subplot(1, 2, 2)
  ax_hist = fig_hist.add_subplot(1,1,1)
  ax_kde = fig_kde.add_subplot(1,1,1)

  fig_pro.suptitle(f"{args.landscape_name}")
  
  ax_position.set_title("Trap Position")
  ax_stiffness.set_title("Trap Stiffness")
  ax_position.set_xlabel("Time (μs)")
  ax_stiffness.set_xlabel("Time (μs)")
  ax_position.set_ylabel("Position")
  ax_stiffness.set_ylabel("Stiffness") #TODO: add units
  ax_reconstruct.set_title(f"{args.landscape_name} Energy Reconstructions | kₛ = { args.k_s} | end_time = { args.end_time }")
  ax_reconstruct.set_title(f"{args.landscape_name} Energy Reconstructions")
  ax_reconstruct.set_xlabel("Position")
  ax_reconstruct.set_ylabel("Energy")
  ax_hist.set_title(f"{args.landscape_name} Dissipated Work Distribution | kₛ = { args.k_s} | end_time = { args.end_time }")
  ax_hist.set_xlabel("Dissipated Work")
  ax_kde.set_title(f"{args.landscape_name} Dissipated Work Distribution | kₛ = { args.k_s} | end_time = { args.end_time }")
  ax_kde.set_title(f"{args.landscape_name} Dissipated Work Distribution")
  ax_kde.set_xlabel("Dissipated Work (W)")
  ax_kde.set_ylabel("p(W)")
  kde_data = {}

  no_trap_fn = p.param_set.energy_fn(no_trap = True)
  time_vec = jnp.linspace(-20,20, 1000)
  if "triple" in args.landscape_name.lower():
    time_vec = jnp.linspace(-12,12, 1000)
  ax_reconstruct.plot(time_vec, jnp.squeeze(no_trap_fn(time_vec.reshape(1,1,1000))) - no_trap_fn(p.r0_init), label = "Original", color = 'k')

  for ind, (trap_name, model_args) in enumerate(model_types.items()):
    if model_args is None:
      continue
    dir_name, coeff_files = find_coeff_file(model_args, args)
    trap_fns = make_trap_from_file(dir_name, coeff_files, psm, ssm, p)
    
    t = jnp.arange(p.param_set.simulation_steps)

    key = random.PRNGKey(1001)
    key, split = random.split(key)

    simulate_fwd = lambda keys: p.param_set.simulate_fn(
      *trap_fns,
      keys, 
      regime = "brownian",
      fwd = True)
    
    bins = 70
    total_work, (batch_trajectories, batch_works, _) = bcs.batch_simulate_harmonic(
        args.batch_size, simulate_fwd, key)
    # work distribution data
    mean_work = batch_works.mean()
    tail = total_work.mean() - total_work.min()
    
    # Dissipated work = Work Used - Free Energy Difference
    energy_sivak = p.param_set.energy_fn()
    w_diss = total_work - (energy_sivak(p.r0_final, k_s = 0.) - energy_sivak(p.r0_init, k_s = 0.)) 
    
    
    reconstructions = 100
    es = []
    for i in tqdm.tqdm(range(reconstructions)):
        key, split = random.split(key)
        total_work, (batch_trajectories, batch_works, _) = bcs.batch_simulate_harmonic(
        args.batch_size, simulate_fwd, key)
        
        midpoints, E = bcl.energy_reconstruction(
            jnp.cumsum(batch_works, axis=1), 
            batch_trajectories, 
            bins, 
            *trap_fns,
            p.param_set.simulation_steps, 
            args.batch_size, 
            p.beta)
        es.append(E)
        
    es = jnp.array(es)
    energies = bcl.interpolate_inf(jnp.mean(es, axis = 0))
    
    landscape = (midpoints, energies)
    
    try:
      disc = bcl.landscape_discrepancies(landscape, no_trap_fn, no_trap_fn(0.), -10., 10.)
      bias = max(disc)
    except:
      print(f"FAILURE: {trap_name} simulations failed. Continuing...")
      continue
    
    no_trap_rec_fn = bce.ReconstructedLandscape(midpoints, energies).molecule_energy
    
    first_well_pos, _ = bcl.find_min_pos(no_trap_rec_fn, -10, 10)
    
    batch_size_grad = 2000
    difference = no_trap_rec_fn(first_well_pos)
    energies_aligned = energies - difference
    es_aligned = es - difference
    
    color = plot_color_list[ind]
    ax_hist.axvline(x = w_diss.mean(), color = color)
    ax_hist.hist(w_diss, weights=jnp.ones(len(w_diss)) / len(w_diss), bins = 50, label = trap_name, alpha = 0.7, color = color)
    kde_data[trap_name] = w_diss
    ax_kde.axvline(x = w_diss.mean(), color = color)
    kde_colors.append(plot_color_list[ind])
    ax_position.plot(t * p.param_set.dt / (10**(-6)), trap_fns[0](t), label = trap_name, color = color)
    ax_stiffness.plot(t * p.param_set.dt / (10**(-6)), trap_fns[1](t), label = trap_name, color = color)
    plot_with_stddev(midpoints, es_aligned, label = trap_name, ax = ax_reconstruct, color = color)
    #ax_reconstruct.plot(midpoints, energies_aligned, label = trap_name, color = color)
    print(f"Plotting complete for {trap_name}")
  sns.kdeplot(data = kde_data, palette = kde_colors, legend=True, ax = ax_kde)
  ax_hist.legend()
  ax_position.legend()
  ax_stiffness.legend()
  ax_reconstruct.legend()
  fig_hist.savefig(parent_dir + f"work_distribution_t{args.end_time}_k{args.k_s}_b{args.batch_size}.png", bbox_inches = "tight")
  fig_rec.savefig(parent_dir + f"reconstructions_t{args.end_time}_k{args.k_s}_b{args.batch_size}.png", bbox_inches = "tight")
  fig_pro.savefig(parent_dir + f"protocols_t{args.end_time}_k{args.k_s}.png", bbox_inches = "tight")
  fig_kde.savefig(parent_dir + f"kde_dist_t{args.end_time}_k{args.k_s}_b{args.batch_size}.png", bbox_inches = "tight")





# TODO: Table comparison of protocols --> Important graph to have! --> Adjust for different protocols etc

    # extensions = [-10,-5,0] # temporary
    # disc_samples = bcl.landscape_discrepancies_samples(no_trap_rec_fn, no_trap_fn, extensions)
    # disc_samples = jnp.array(disc_samples)
    # mean_disc_samples = disc_samples.mean()
    # bias_samples = disc_samples.max()

    # loss values 
    # if file_name == "split.pkl":
    #   pass
    
    #   # grad_rev = lambda num_batches: bc_optimize.estimate_gradient_rev_split(
    #   #   num_batches,
    #   #   simulate_fwd_grad,
    #   #   p.r0_init,
    #   #   p.r0_final,
    #   #   p.r0_cut,
    #   #   sim_cut_steps,
    #   #   p.param_set.simulation_steps,
    #   #   p.beta)
    #   # grad, (_, summary) = grad_rev(batch_size_grad)(*coeff, split)
      
    # else:
    #   grad_rev = lambda num_batches: bc_optimize.estimate_gradient_rev(
    #     num_batches,
    #     simulate_fwd_grad,
    #     p.r0_init,
    #     p.r0_final,
    #     p.param_set.simulation_steps,
    #     p.beta)
    #   grad, (_, summary) = grad_rev(batch_size_grad)(coeff, split)
    

    # plot_data[trap_name] = {
    #   "trap": trap_fn,
    #   "translation": difference,
    #   "work": total_work,
    #   "midpoints": midpoints,
    #   "energies": energies_aligned,
    #   "bias": bias,
    #   "mean_work": mean_work,
    #   "discrepancy": disc,
    #   "tail": tail,
    #   "work loss": summary[2].mean(), # TODO
    #   "error loss": summary[2].mean(), # TODO
    #   "accumulated loss": summary[2].mean(), # TODO
    #   "samples": {
    #     "mean_discrepancy": mean_disc_samples,
    #     "bias": bias_samples,
    #     "discrepancy": disc_samples
    #     }
    #   }
    #code.interact(local = locals())