
import functools
import pickle
import time
import tqdm

import matplotlib.pyplot as plt

import jax
import jax_md

import jax.numpy as jnp
from jax import random
from jax.experimental import optimizers as jopt

from jax_md import quantity, space

from barrier_crossing.energy import V_biomolecule_geiger
from barrier_crossing.protocol import linear_chebyshev_coefficients, make_trap_fxn, make_trap_fxn_rev
from barrier_crossing.simulate import simulate_brownian_harmonic

def seed_stream(seed): #will 'yield' a diff. random # each time it is called
  k = jax.random.PRNGKey(seed)
  while True:
    k, y = jax.random.split(k)
    yield y

def plot_with_stddev(x, label=None, n=1, axis=0, ax=plt):
  stddev = jnp.std(x, axis)
  mn = jnp.mean(x, axis)

  ax.fill_between(jnp.arange(mn.shape[0]),
                  mn + n * stddev, mn - n * stddev, alpha=.3)
  ax.plot(mn, '-o', label=label)

def single_estimate_fwd(energy_fn, init_position, r0_init, r0_final, Neq, shift, simulation_steps, dt, temperature, mass, gamma):
  @functools.partial(jax.value_and_grad, has_aux=True) #the 'aux' is the summary
  def _single_estimate(coeffs, seed): #function only of the params to be differentiated w.r.t.
      # OLD forward direction
      trap_fn = make_trap_fxn(jnp.arange(simulation_steps), coeffs, r0_init, r0_final)
      positions, log_probs, works = simulate_brownian_harmonic(
          energy_fn, 
          init_position,
          trap_fn,
          simulation_steps,
          Neq, shift, seed, 
          dt, temperature, mass, gamma
          )
      total_work = works.sum()
      tot_log_prob = log_probs.sum()
      summary = (positions, tot_log_prob, total_work)
      
      gradient_estimator = (tot_log_prob * jax.lax.stop_gradient(total_work) + total_work) 
      
      return gradient_estimator, summary
  return _single_estimate 

def estimate_gradient_fwd(batch_size, energy_fn, init_position, r0_init, r0_final, Neq, shift, simulation_steps, dt, temperature, mass, gamma):
    mapped_estimate = jax.vmap(single_estimate_fwd(energy_fn, init_position, r0_init, r0_final, Neq, shift, simulation_steps, dt, temperature, mass, gamma), [None, 0])
    #mapped_estimate = jax.soft_pmap(lambda s: single_estimate(energy_fn, init_position, r0_init, r0_final, Neq, shift, simulation_steps, dt, temperature, mass, gamma), [None, 0])
    @jax.jit #why am I allowed to jit something whose output is a function? Thought it had to be pytree output...
    def _estimate_gradient(coeffs, seed):
      seeds = jax.random.split(seed, batch_size)
      (gradient_estimator, summary), grad = mapped_estimate(coeffs, seeds)
      return jnp.mean(grad, axis=0), (gradient_estimator, summary)
    return _estimate_gradient # < # delta ln P (W not graded) + delta W > averaged over,


def single_estimate_rev(energy_fn, init_position, r0_init, r0_final, Neq, shift, simulation_steps, dt, temperature, mass, gamma, beta):
  @functools.partial(jax.value_and_grad, has_aux = True)
  def _single_estimate(coeffs, seed):
    trap_fn = make_trap_fxn(jnp.arange(simulation_steps), coeffs, r0_init, r0_final)
    positions, log_probs, works = simulate_brownian_harmonic(
        energy_fn, 
        init_position, trap_fn,
        simulation_steps,
        Neq, shift, seed, 
        dt, temperature, mass, gamma
        )
    total_work = works.sum()
    tot_log_prob = log_probs.sum()
    summary = (positions, tot_log_prob, jnp.exp(beta*total_work))

    # NEW delta ln P * e^(beta W) (not graded) + delta W * e^(beta W) (no grad)
    gradient_estimator = (tot_log_prob) * jax.lax.stop_gradient(jnp.exp(beta * total_work)) + jax.lax.stop_gradient(beta * jnp.exp(beta*total_work)) * total_work
    return gradient_estimator, summary
  return _single_estimate 

def estimate_gradient_rev(batch_size, energy_fn, init_position, r0_init, r0_final, Neq, shift, simulation_steps, dt, temperature, mass, gamma, beta):
    """Find e^(beta ΔW) which is proportional to the error <(ΔF_(batch_size) - ΔF)> = variance of ΔF_(batch_size)(2010 Geiger and Dellago). """
    mapped_estimate = jax.vmap(single_estimate_rev(energy_fn, init_position, r0_init, r0_final, Neq, shift, simulation_steps, dt, temperature, mass, gamma, beta), [None, 0])  
    @jax.jit 
    def _estimate_gradient(coeffs, seed):
      seeds = jax.random.split(seed, batch_size)
      (gradient_estimator, summary), grad = mapped_estimate(coeffs, seeds)
      return jnp.mean(grad, axis=0), (gradient_estimator, summary)
    return _estimate_gradient # < # delta ln P (W not graded) + delta W > averaged over,


# TODO: Make a optimization loop code that can just be run as a single function
# ============== PARAMETERS ================

# Hyper parameters
# N = 1
# dim = 1
# end_time = 0.01 #s
# dt = 3e-7 #s
# simulation_steps = int((end_time)/dt)+1
# teq=0.001 #s
# Neq=int(teq/dt)

# batch_size= 4#2504 number of trajectories
# opt_steps = 2#1000 number of optimization steps
# #temperature = 4.114 #at 298K = 25C
# temperature = 4.183 #at 303K=30C
# beta=1.0/temperature #1/(pNnm)
# mass = 1e-17 #1e-17 #g
# D = 0.44*1e6 #(in nm**2/s) 
# gamma = 1./(beta*D*mass) #s^(-1)

# k_s = 0.4 #pN/nm
# epsilon = 0.5 * (1.0/beta)
# sigma = 1.0/jnp.sqrt(beta * k_s)

# #harmonic potential (I call it a "trap") parameters:
# r0_init = -0. #initial pos
# r0_final = sigma*2. #final pos
# init_position = r0_init * jnp.ones((N,dim))

# #landscape params:
# x_m=10. #nm
# delta_E=0 #pN nm
# kappa_l=21.3863/(beta*x_m**2) #pN/nm #for Ebarrier = 10kT and delta_E=0, as C&S use
# #kappa_l=6.38629/(beta*x_m**2) #pN/nm #for Ebarrier = 2.5kT and delta_E=0, as C&S use
# #kappa_l=2.6258/(beta*x_m**2)#barrier 0.625kT
# kappa_r=kappa_l #pN/nm 

# energy_fn = V_biomolecule(kappa_l, kappa_r, x_m, delta_E, k_s, beta, epsilon, sigma)
# force_fn = quantity.force(energy_fn)
# displacement_fn, shift_fn = space.free()

# save_filepath = "temp/"

# init_coeffs = linear_chebyshev_coefficients(r0_init, r0_final, simulation_steps, degree =12, y_intercept = 0.) # This implements a linear schedule
# trap_fn = make_trap_fxn(jnp.arange(simulation_steps), init_coeffs, r0_init, r0_final)

# #optimizer = jopt.adam(.1)
# #use learning rate decay to converge more accurately on the global opt value:
# lr = jopt.exponential_decay(0.1, opt_steps, 0.001)
# optimizer = jopt.adam(lr)


# grad_fxn = estimate_gradient(batch_size, energy_fn, init_position, r0_init, r0_final, Neq, shift_fn, simulation_steps, dt, temperature, mass, gamma)


def optimize_protocol(init_coeffs, batch_grad_fn, optimizer, batch_size, num_steps, save_path = None):
  summaries = []
  coeffs_ = []
  all_works = []
  
  key = jax.random.PRNGKey(int(time.time()))
  key, split = jax.random.split(key, 2)
  
  init_state = optimizer.init_fn(init_coeffs)
  opt_state = optimizer.init_fn(init_coeffs)
  coeffs_.append((0,) + (optimizer.params_fn(opt_state),))

  grad_fn = batch_grad_fn(batch_size)
  
  for j in tqdm.trange(num_steps,position=0):
    key, split = jax.random.split(key)
    grad, (_, summary) = grad_fn(optimizer.params_fn(opt_state), split)

    print(f"\n Gradient norm: {jnp.linalg.norm(grad)}")

    opt_state = optimizer.update_fn(j, grad, opt_state)
    all_works.append(summary[2])
    if j % 100 == 0:
        coeffs_.append(((j+1),) + (optimizer.params_fn(opt_state),))
    if j == (num_steps-1):
        coeffs_.append((j+1,) + (optimizer.params_fn(opt_state),))
      
  coeffs_.append((num_steps,) + (optimizer.params_fn(opt_state),))

  print("init parameters: ", optimizer.params_fn(init_state))
  print("final parameters: ", optimizer.params_fn(opt_state))

  #all_summaries = jax.tree_multimap(lambda *args: jnp.stack(args), *summaries)
  all_works = jax.tree_multimap(lambda *args: jnp.stack(args), *all_works)

  # note that you cannot pickle locally-defined functions 
  # (like the log_temp_baseline() thing), so I need to save the weights 
  # and then later recreate the schedules
  if save_path != None:
    afile = open(save_path+'coeffs.pkl', 'wb')
    pickle.dump(coeffs_, afile)
    afile.close()

    bfile = open(save_path+'works.pkl', 'wb')
    pickle.dump(all_works, bfile)
    bfile.close()
    
  return coeffs_, summaries, all_works

"""
file = open("./FORWARD_COEFFS.pkl", 'rb')
coeffs_ = pickle.load(file)
file.close()

file = open("./FORWARD_WORKS.pkl",'rb')
all_works = pickle.load(file)
file.close()"""

"""###Plots"""
def plot_fwd_opt():
  #plot work distns 
  plt.figure(figsize=[12, 12])
  for j in range(opt_steps):
    if(j%100 == 0):
      work_dist = all_works[j,:]
      plt.hist(work_dist,10,label=f'Step {j}')

  plt.legend()#

  ##### PLOT LOSS AND SCHEDULE EVOLUTION #####
  _, (ax0, ax1) = plt.subplots(1, 2, figsize=[24, 12])
  plot_with_stddev(all_works.T, ax=ax0)
  #ax0.set_ylim([0.1,1.0])
  #ax0.set_xlim([0,200])
  ax0.set_title('Total work')

  trap_fn = make_trap_fxn(jnp.arange(simulation_steps), init_coeffs, r0_init, r0_final)
  init_sched = trap_fn(jnp.arange(simulation_steps))
  ax1.plot(jnp.arange(simulation_steps), init_sched, label='initial guess')

  for j, coeffs in coeffs_:
    #if(j%50 == 0):
    trap_fn = make_trap_fxn(jnp.arange(simulation_steps),coeffs,r0_init,r0_final)
    full_sched = trap_fn(jnp.arange(simulation_steps))
    ax1.plot(jnp.arange(simulation_steps), full_sched, '-', label=f'Step {j}')

  #plot final estimate:
  trap_fn = make_trap_fxn(jnp.arange(simulation_steps),coeffs_[-1][1],r0_init,r0_final)
  full_sched = trap_fn(jnp.arange(simulation_steps))
  ax1.plot(jnp.arange(simulation_steps), full_sched, '-', label=f'Final')



  ax1.legend()#
  ax1.set_title('Schedule evolution')
  plt.savefig(save_filepath+ "forward_optimization.png")
  plt.show()

"""# Calculating Jarzynski Equality Error"""

# ============== PARAMETERS ================

# # Hyper parameters
# N = 1
# dim = 1
# end_time = 0.01 #s
# dt = 3e-7 #s
# simulation_steps = int((end_time)/dt)+1
# teq=0.001 #s
# Neq=int(teq/dt)

# batch_size= 4#2504 number of trajectories
# opt_steps = 2#1000 number of optimization steps
# #temperature = 4.114 #at 298K = 25C
# temperature = 4.183 #at 303K=30C
# beta=1.0/temperature #1/(pNnm)
# mass = 1e-17 #1e-17 #g
# D = 0.44*1e6 #(in nm**2/s) 
# gamma = 1./(beta*D*mass) #s^(-1)

# k_s = 0.4 #pN/nm
# # epsilon = 0.5 * (1.0/beta)
# sigma = 1.0/jnp.sqrt(beta * k_s)

# #harmonic potential (I call it a "trap") parameters:
# r0_init = -0. #initial pos
# r0_final = sigma*2. #final pos


# #landscape params:
# x_m=10. #nm
# delta_E=0 #pN nm
# kappa_l=21.3863/(beta*x_m**2) #pN/nm #for Ebarrier = 10kT and delta_E=0, as C&S use
# #kappa_l=6.38629/(beta*x_m**2) #pN/nm #for Ebarrier = 2.5kT and delta_E=0, as C&S use
# #kappa_l=2.6258/(beta*x_m**2)#barrier 0.625kT
# kappa_r=kappa_l #pN/nm 

# energy_fn = V_biomolecule(kappa_l, kappa_r, x_m, delta_E, k_s, beta, epsilon, sigma)
# force_fn = quantity.force(energy_fn)
# displacement_fn, shift_fn = space.free()

# # ======== REVERSE OPTIMIZATION =============

# init_coeffs = linear_chebyshev_coefficients(r0_init, r0_final, simulation_steps, degree = 12, y_intercept = 0.)

# trap_fn = make_trap_fxn_rev(jnp.arange(simulation_steps), init_coeffs, r0_init, r0_final)
# key = random.PRNGKey(int(time.time()))
# key, split = random.split(key, 2)

# # use learning rate decay (ADAMS) to converge more accurately on the global opt value:
# lr = jopt.exponential_decay(0.1, opt_steps, 0.001)
# optimizer = jopt.adam(lr)

# # rev_summaries = []
# rev_coeffs_ = []
# rev_all_works = []

# init_state = optimizer.init_fn(init_coeffs)
# opt_state = optimizer.init_fn(init_coeffs)

# rev_init_pos = r0_final * jnp.ones((N,dim))
# rev_grad_fxn = estimate_gradient_rev(
#     batch_size, energy_fn, rev_init_pos, 
#     r0_init, r0_final, Neq, shift_fn, 
#     simulation_steps, dt, 
#     temperature, mass, gamma, beta
#     )

# rev_coeffs_.append((0,) + (optimizer.params_fn(opt_state),))

# for j in tqdm.trange(opt_steps,position=0):
#   key, split = random.split(key)
#   grad, (_, summary) = rev_grad_fxn(optimizer.params_fn(opt_state), split)
#   print(grad)
#   opt_state = optimizer.update_fn(j, grad, opt_state)
#   rev_all_works.append(summary[2])
  
#   if j % 100 == 0:
#     rev_coeffs_.append(((j+1),) + (optimizer.params_fn(opt_state),))
#   if j == (opt_steps-1):
#     rev_coeffs_.append((j+1,) + (optimizer.params_fn(opt_state),))
    
# rev_coeffs_.append((opt_steps,) + (optimizer.params_fn(opt_state),))

# rev_power_works = jnp.exp(jnp.array(rev_all_works))

# print("init parameters: ", optimizer.params_fn(init_state))
# print("final parameters: ", optimizer.params_fn(opt_state))
# #all_summaries = jax.tree_multimap(lambda *args: jnp.stack(args), *summaries)
# rev_power_works = jax.tree_multimap(lambda *args: jnp.stack(args), *rev_power_works)
# # note that you cannot pickle locally-defined functions 
# # (like the log_temp_baseline() thing), so I need to save the weights 
# # and then later recreate the schedules
# afile = open(save_filepath+'rev_coeffs.pkl', 'wb')
# pickle.dump(rev_coeffs_, afile)
# afile.close()

# bfile = open(save_filepath+'rev_works.pkl', 'wb')
# pickle.dump(rev_all_works, bfile)
# bfile.close()

# ##### PLOT REVERSE WORKS #####
# _, (ax0, ax1) = plt.subplots(1, 2, figsize=[24, 12])
# plot_with_stddev(rev_power_works.T, ax=ax0)


# ax0.set_title('Error (Exponential)')

# trap_fn = make_trap_fxn_rev(jnp.arange(simulation_steps), init_coeffs, r0_init, r0_final)
# init_sched = trap_fn(jnp.arange(simulation_steps))
# ax1.plot(jnp.arange(simulation_steps) * dt, init_sched, label='initial guess')

# for j, coeffs in rev_coeffs_:
#   #if(j%50 == 0):
#   trap_fn = make_trap_fxn_rev(jnp.arange(simulation_steps),coeffs,r0_init,r0_final)
#   full_sched = trap_fn(jnp.arange(simulation_steps))
#   ax1.plot(jnp.arange(simulation_steps) * dt, full_sched, '-', label=f'Step {j}')

# #plot final estimate:
# trap_fn = make_trap_fxn_rev(jnp.arange(simulation_steps),rev_coeffs_[-1][1],r0_init,r0_final)
# full_sched = trap_fn(jnp.arange(simulation_steps))
# ax1.plot(jnp.arange(simulation_steps) * dt, full_sched, '-', label=f'Final')



# ax1.legend()#
# ax1.set_title('Schedule evolution')
# plt.savefig(save_filepath+ "backward_optimization.png")
# plt.show()

# """# Finding Average Work for Optimized Coefficients"""

