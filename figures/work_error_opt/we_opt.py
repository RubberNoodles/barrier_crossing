# Optimize coefficients for work and error distributions.
import pickle
import os
import code

import barrier_crossing.protocol as bcp
import barrier_crossing.train as bct
import barrier_crossing.loss as bcl
import barrier_crossing.models as bcm
from barrier_crossing.utils import parse_args

import jax.numpy as jnp
import jax.random as random
import jax.example_libraries.optimizers as jopt

import matplotlib.pyplot as plt

# from figures.params import * # global variables;

def plot_with_stddev(x, label=None, n=1, axis=0, ax=plt, dt=1.):
  stddev = jnp.std(x, axis)
  mn = jnp.mean(x, axis)
  xs = jnp.arange(mn.shape[0]) * dt

  ax.fill_between(xs,
                  mn + n * stddev, mn - n * stddev, alpha=.3)
  ax.plot(xs, mn, label=label)

retrieve_coeffs = lambda model: list(zip(jnp.arange(len(model.coef_hist)), model.coef_hist))

def plot_and_save_optimization(losses, models, names, opt_steps, path, figsize = [16, 8]):
  for name_list, model in zip(names, models):
    name = "_".join(name_list)
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    plot_with_stddev(losses.T, ax=ax[0])

    # ax[0].set_title(f'Jarzynski Error over Optimization; Short trap; STD error sampling; {batch_size}; {num_epochs}.')
    ax[0].set_title(f'{name}')
    ax[0].set_xlabel('Number of Optimization Steps')
    ax[0].set_ylabel('Error')
    # I should pass in model
    trap_fn = model.protocol(model.coef_hist[0])[0]
    init_sched = trap_fn(t)
    ax[1].plot(t, init_sched, label='Initial guess')

    per_5 = opt_steps//5
    
    for i, coeff in enumerate(model.coef_hist):
      if i% per_5 == 0 and i!=0:
        trap_fn = model.protocol(coeff)[0]
        sched = trap_fn(t)
        ax[1].plot(t, sched, '-', label=f'Step {i}')

    # Plot final estimate:
    final_sched = model(t)
    ax[1].plot(t, final_sched, '-', label=f'Final')

    ax[1].legend()
    ax[1].set_title(f'{args.landscape_name} Schedule Evolution')
    file_name = name.replace(' ', '_').lower()
    fig.savefig(path + file_name + ".png")
    
    with open(path + file_name + ".pkl", "wb") as f:
      pickle.dump(model.coeffs, f)
  
if __name__ == "__main__":
  args, p = parse_args()
  key = random.PRNGKey(1001)
  
  # dir_name = [name]_t[end_time]_ks[stiffness]
  dir_name = "_".join([args.landscape_name.replace(' ', '_').replace('.', '_').lower(), f"t{args.end_time}", f"ks{args.k_s}"])
  path = f"output_data/{dir_name}/"
  if not os.path.isdir(path):
    os.mkdir(path)
  

  split_param_set = p.param_set.copy(p.param_set)
  split_param_set.end_time /= 2
  
  models = [
    {
      "name": "Position", "model": bcm.ScheduleModel(p.param_set, p.r0_init, p.r0_final, mode = "fwd")
    },
    {
      "name": "Stiffness", "model": bcm.ScheduleModel(p.param_set, p.ks_init, p.ks_final, mode = "fwd")
    },
    {
      "name": "Position", "model": bcm.ScheduleModel(p.param_set, p.r0_init, p.r0_final, mode = "rev")
    },
    {
      "name": "Stiffness", "model": bcm.ScheduleModel(p.param_set, p.ks_init, p.ks_final, mode = "rev")
    },
    {
      "name": "Position", "model": bcm.SplitModel(split_param_set, p.r0_init, (p.r0_init + p.r0_final)/2, p.r0_final, p.param_set.simulation_steps, mode="rev", num = 1)                                  
    },
    {
      "name": "Stiffness", "model":  bcm.SplitModel(split_param_set, p.ks_init, (p.ks_init + p.ks_final)/2, p.ks_final, p.param_set.simulation_steps, mode="rev", num = 1)                             
    },
    {
      "name": "Position", "model": bcm.SplitModel(split_param_set, p.r0_init, (p.r0_init + p.r0_final)/2, p.r0_final, p.param_set.simulation_steps, mode="fwd", num = 1)                                  
    },
    {
      "name": "Stiffness", "model":  bcm.SplitModel(split_param_set, p.ks_init, (p.ks_init + p.ks_final)/2, p.ks_final, p.param_set.simulation_steps, mode="fwd", num = 1)                             
    }
  ]
  
  models += [
  {
    "name": "Joint", "model": bcm.JointModel(p.param_set, models[0]["model"], models[1]["model"])
  },
  {
    "name": "Joint", "model": bcm.JointModel(p.param_set, models[2]["model"], models[3]["model"])
  },
  {
    "name": "Joint", "model": bcm.JointModel(p.param_set, models[4]["model"], models[5]["model"])
  },
  {
    "name": "Joint", "model": bcm.JointModel(p.param_set, models[6]["model"], models[7]["model"])
  }]

  batch_size = 5000 # Number of simulations/trajectories simulated. GPU optimized.
  num_epochs = 1000 # Number of gradient descent steps to take.
  lr = jopt.polynomial_decay(0.03, num_epochs, 0.0003)
  optimizer = jopt.adam(lr)
  
  train_fn = lambda model, grad_fn: bct.train(model, optimizer, grad_fn, key, batch_size = batch_size, num_epochs = num_epochs)
  
  lin_coeffs = bcp.linear_chebyshev_coefficients(p.r0_init, p.r0_final, p.param_set.simulation_steps, y_intercept = p.r0_init)
  # Trap Functions. Reverse mode trap functions are for when we compute Jarzynski error with reverse protocol trajectories.
  t = jnp.arange(p.param_set.simulation_steps)
  
  for model_dict in models:
    model = model_dict["model"]
    mode = model.mode 
    fwd = True if mode == "fwd" else False
    
    simulate_fn = None # To be defined later
    
    if mode == "fwd":
      default_trap = bcp.make_trap_fxn(t, lin_coeffs, p.r0_init, p.r0_final)
      grad_fn = lambda num_batches: bcl.estimate_gradient_work(
      num_batches,
      simulate_fn,
      model)
      
    else:
      default_trap = bcp.make_trap_fxn_rev(t, lin_coeffs, p.r0_init, p.r0_final)
      grad_fn = lambda num_batches: bcl.estimate_gradient_rev(
      num_batches,
      simulate_fn,
      model)
    
    if model_dict["name"] == "Position":
      simulate_fn = lambda trap_fn, keys: p.param_set.simulate_fn(
      trap_fn, 
      p.param_set.k_s,
      keys, 
      regime = "brownian",
      fwd = fwd)
      
    elif model_dict["name"] == "Stiffness":
      simulate_fn = lambda trap_fn, keys: p.param_set.simulate_fn(
      default_trap, 
      trap_fn,
      keys, 
      regime = "brownian",
      fwd = fwd)
    
    elif model_dict["name"] == "Joint":
      simulate_fn = lambda trap_fn, ks_fn, keys: p.param_set.simulate_fn(
      trap_fn,
      ks_fn,
      keys, 
      regime = "brownian",
      fwd = fwd)
    
    names = [[model_dict['name'], mode]]
    if isinstance(model, bcm.JointModel):
      model.pop_hist(reset = True)
      names.append(names[0] + ["stiffness"])
      names[0] += ["position"]
      
    losses = train_fn(model, grad_fn)
    
    if isinstance(model, bcm.SplitModel) or isinstance(model.models[0], bcm.SplitModel):
      split_1_names = [name + ["split_1"] for name in names]
      
      plot_and_save_optimization(losses, model.models, split_1_names, num_epochs, path)
      
      model.switch_trap()
      
      losses = train_fn(model, grad_fn)
      split_2_names = [name + ["split_2"] for name in names]
      plot_and_save_optimization(losses, model.models, split_2_names, num_epochs, path)
      
    else:
      plot_and_save_optimization(losses, model.models, names, num_epochs, path)
    
  print(p.param_set)