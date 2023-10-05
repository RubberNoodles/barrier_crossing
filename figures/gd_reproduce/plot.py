"""
Geiger P, Dellago C. Optimum protocol for fast-switching free-energy calculations.
Phys Rev E Stat Nonlin Soft Matter Phys. 2010 Feb;81(2 Pt 1):021127. 
doi: 10.1103/PhysRevE.81.021127. Epub 2010 Feb 23. PMID: 20365550.

Figures 10 & 11
"""

import pickle
import importlib

import jax.numpy as jnp
import jax.random as random

import matplotlib.pyplot as plt

import barrier_crossing.protocol as bc_protocol
import barrier_crossing.simulate as bc_simulate


if __name__ == "__main__":
  key = random.PRNGKey(42)
  
  work_nkt = []
  work_w = []
  error_nkt = []
  error_w = []
  
  sigma = 1.
  batch_size = 4000

  fig_energy, (ax_work, ax_nkt) = plt.subplots(1,2, figsize = (16, 8))
  fig_pro, (ax_work_opt, ax_error_opt) = plt.subplots(1,2, figsize = (16, 8))
  
  p = importlib.import_module("figures.param_set.params_geiger")
  
  sim_vec = jnp.arange(p.simulation_steps)
  lin_coeffs = bc_protocol.linear_chebyshev_coefficients(p.r0_init, p.r0_final, p.simulation_steps)
  linear_trap = bc_protocol.make_trap_fxn(sim_vec, lin_coeffs, p.r0_init, p.r0_final)
  
  for i in range(1,9):
    epsilon = i/2
    
    path = f"output_data/geiger_and_dellago_epsilon={epsilon:.1f}/".replace(".", "_")
    
    p.param_set.epsilon = epsilon
    energy_geiger = p.param_set.energy_fn()
    
    with open(path + "error_coeffs.pkl", "rb") as f:
      error_coeff = pickle.load(f)
    
    with open(path + "work_coeffs.pkl", "rb") as f:
      work_coeff = pickle.load(f)
  
      
    error_trap = bc_protocol.make_trap_fxn(sim_vec, error_coeff, p.r0_init, p.r0_final)
    work_trap = bc_protocol.make_trap_fxn(sim_vec, work_coeff, p.r0_init, p.r0_final)
    
    ax_error_opt.plot(error_trap(sim_vec), label = f"ε = {epsilon}")
    ax_work_opt.plot(work_trap(sim_vec), label = f"ε = {epsilon}")
  
    sim_error = lambda keys: p.param_set.simulate_fn(error_trap, keys, "langevin")
    sim_work = lambda keys: p.param_set.simulate_fn(work_trap, keys, "langevin")
    sim_linear = lambda keys: p.param_set.simulate_fn(linear_trap, keys, "langevin")
    
    sim_error_rev = lambda keys: p.param_set.simulate_fn(error_trap, keys, "langevin", fwd = False)
    sim_work_rev = lambda keys: p.param_set.simulate_fn(work_trap, keys, "langevin", fwd = False)
    sim_linear_rev = lambda keys: p.param_set.simulate_fn(linear_trap, keys, "langevin", fwd = False)
    
    total_er, _ = bc_simulate.batch_simulate_harmonic(batch_size, sim_error, p.simulation_steps, key)
    total_wo, _ = bc_simulate.batch_simulate_harmonic(batch_size, sim_work, p.simulation_steps, key)
    total_li, _ = bc_simulate.batch_simulate_harmonic(batch_size, sim_linear, p.simulation_steps, key)
    
    total_rev_er, _ = bc_simulate.batch_simulate_harmonic(batch_size, sim_error_rev, p.simulation_steps, key)
    total_rev_wo, _ = bc_simulate.batch_simulate_harmonic(batch_size, sim_work_rev, p.simulation_steps, key)
    total_rev_li, _ = bc_simulate.batch_simulate_harmonic(batch_size, sim_linear_rev, p.simulation_steps, key)
    
    work_err = total_er.mean()
    work_wo = total_wo.mean()
    work_lin = total_li.mean()
    
    work_w.append(work_wo/work_lin)
    error_w.append(work_err/work_lin)
    
    nkt_err = jnp.exp( p.beta * total_rev_er).mean()
    nkt_wo = jnp.exp( p.beta * total_rev_wo).mean()
    nkt_lin = jnp.exp( p.beta * total_rev_li).mean()
    
    work_nkt.append(nkt_wo/nkt_lin)
    error_nkt.append(nkt_err/nkt_lin)
  
  epsilons = [i/2 for i in range(1, 9)]
  
  ax_work.plot(epsilons, work_w, "-o", label = "Work Optimized")
  ax_work.plot(epsilons, error_w, "-o", label = "Error Optimized")
  ax_work.set_title("Work")
  ax_work.set_xlabel("ε")
  ax_work.legend()
  
  ax_nkt.plot(epsilons, work_nkt, "-o", label = "Work Optimized")
  ax_nkt.plot(epsilons, error_nkt, "-o", label = "Error Optimized")
  ax_nkt.set_title("Nkt")
  ax_nkt.set_xlabel("ε")
  ax_nkt.legend()
  
  ax_error_opt.legend()
  ax_error_opt.set_title("Error Optimized Protocols")
  ax_work_opt.legend()
  ax_work_opt.set_title("Error Optimized Protocols")
    
  fig_energy.savefig("compare_linear.png")
  fig_pro.savefig("protocols.png")
    
    
    