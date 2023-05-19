
import time
import tqdm
import pickle

import jax

import jax.numpy as jnp
import numpy as onp

import jax.random as random

import jax.example_libraries.optimizers as jopt

from jax_md import space

import barrier_crossing.energy as bc_energy
import barrier_crossing.protocol as bc_protocol
import barrier_crossing.simulate as bc_simulate
import barrier_crossing.optimize as bc_optimize
import barrier_crossing.iterate_landscape as bc_landscape

import matplotlib.pyplot as plt

def plot_with_stddev(x, label=None, n=1, axis=0, ax=plt, dt=1.):
  stddev = jnp.std(x, axis)
  mn = jnp.mean(x, axis)
  xs = jnp.arange(mn.shape[0]) * dt

  ax.fill_between(xs,
                  mn + n * stddev, mn - n * stddev, alpha=.3)
  ax.plot(xs, mn, label=label)

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
k_s_sc = 0.4 # stiffness; 
#k_s_sc = 0.6 # stiffness; 
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
#delta_E=7.0 #pN nm
delta_E=1. #pN nm
#kappa_l=21.3863/(beta_sc*x_m**2) #pN/nm #for Ebarrier = 10kT and delta_E=0, as C&S use
#kappa_l=6.38629/(beta_sc*x_m**2) #pN/nm #for Ebarrier = 2.5kT and delta_E=0, as C&S use
kappa_l=2.6258/(beta_sc*x_m**2)#barrier 0.625kT
kappa_r=kappa_l #pN/nm; Symmetric wells.

# ================= GEIGER & DELLAGO =====================

# Brownian Environment G&D
temperature_gd = 1.0
beta_gd = 1.0
gamma_gd = 1.0

# G&D Landscape Parameters
k_s_gd = 1.0 # stiffness
epsilon = 2.
sigma = 1.0/jnp.sqrt(beta_gd * k_s_gd)

# Harmonic Trap Parameters G&D
r0_init_gd = -0. #nm; initial trap position.
r0_final_gd = 2. * sigma #nm; final trap position

# Particle Parameters G&D
mass_gd = 1.0 # G&D
init_position_fwd_gd = r0_init_gd*jnp.ones((N,dim)) #nm
init_position_rev_gd = r0_final_gd*jnp.ones((N,dim))

energy_geiger_plot = bc_energy.V_biomolecule_geiger(0, epsilon, sigma)
energy_sivak_plot = bc_energy.V_biomolecule_sivak(kappa_l, kappa_r, x_m, delta_E, 0, beta_sc)

pos_e = [[x, 0.002 * (x**2 - 1) * (x**2 - 49)] for x in jnp.linspace(-10,10,100)]
positions = jnp.array(pos_e)[:,0]
energies = jnp.array(pos_e)[:,1]

# The plot on the energy function underneath will be coarse-grained due to few sample points.
energy_custom_plot = bc_energy.V_biomolecule_reconstructed(0, positions, energies)


r0_init_custom = -7
r0_final_custom = 7
k_s_custom = 0.4

# (energy_function, (interval_start, interval_end), name_of_energy)
e_fns = [
    (energy_geiger_plot, (r0_init_gd-1, r0_final_gd+1), "Geiger & Dellago"), 
    (energy_sivak_plot, (r0_init_sc-10, r0_final_sc+10),"Sivak & Crooks"), 
    (energy_custom_plot, (r0_init_custom-8, r0_final_custom+8), "Custom Landscape")
     ]

# Define energies with non-zero trap coefficients
energy_geiger = bc_energy.V_biomolecule_geiger(k_s_gd, epsilon, sigma)
energy_sivak = bc_energy.V_biomolecule_sivak(kappa_l, kappa_r, x_m, delta_E, k_s_sc, beta_sc)
energy_custom = bc_energy.V_biomolecule_reconstructed(k_s_custom, positions, energies)


# Protcol + Simulation Parameters
end_time_gd = 1.
dt_gd = 0.002
simulation_steps_gd = int(end_time_gd / dt_gd)

#end_time_sc = 0.01
# dt_sc = 2e-8 this might be exceeding floating point precision or something..
end_time_sc = 10.
dt_sc = 3e-6
simulation_steps_sc = int(end_time_sc / dt_sc)

end_time_custom = 1.
dt_custom = 0.001
simulation_steps_custom = int(end_time_custom / dt_custom)

# Equilibration Steps; in order to correctly apply Jarzynski, the system has to 
# be in equilibrium, which is defined by equal free energy in all degrees of freedom
Neq = 500

# Protocol Coefficients
lin_coeffs_gd = bc_protocol.linear_chebyshev_coefficients(r0_init_gd, r0_final_gd, simulation_steps_gd, degree = 12, y_intercept = r0_init_gd)
lin_coeffs_sc = bc_protocol.linear_chebyshev_coefficients(r0_init_sc, r0_final_sc, simulation_steps_sc, degree = 12, y_intercept = r0_init_sc)
lin_coeffs_custom = bc_protocol.linear_chebyshev_coefficients(r0_init_custom, r0_final_custom, simulation_steps_custom, degree = 12, y_intercept = r0_init_custom)

# Trap Functions. Reverse mode trap functions are for when we compute Jarzynski error with reverse protocol trajectories.
trap_fn_fwd_gd = bc_protocol.make_trap_fxn(jnp.arange(simulation_steps_gd), lin_coeffs_gd, r0_init_gd, r0_final_gd)
trap_fn_fwd_sc = bc_protocol.make_trap_fxn(jnp.arange(simulation_steps_sc), lin_coeffs_sc, r0_init_sc, r0_final_sc)
trap_fn_fwd_custom = bc_protocol.make_trap_fxn(jnp.arange(simulation_steps_custom), lin_coeffs_custom, r0_init_custom, r0_final_custom)

trap_fn_rev_gd = bc_protocol.make_trap_fxn_rev(jnp.arange(simulation_steps_gd), lin_coeffs_gd, r0_init_gd, r0_final_gd)
trap_fn_rev_sc = bc_protocol.make_trap_fxn_rev(jnp.arange(simulation_steps_sc), lin_coeffs_sc, r0_init_sc, r0_final_sc)
trap_fn_rev_custom = bc_protocol.make_trap_fxn_rev(jnp.arange(simulation_steps_custom), lin_coeffs_custom, r0_init_custom, r0_final_custom)

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

total_works, (batch_trajectories, batch_works, batch_log_probs) = bc_simulate.batch_simulate_harmonic(
    250, energy_sivak, simulate_sivak_fn_fwd, simulation_steps_sc, key)

midpoints_lin, energies_lin = bc_landscape.energy_reconstruction(batch_works, batch_trajectories, 100, trap_fn_fwd_sc, simulation_steps_sc, 250, k_s_sc, beta_sc)
energy_sivak = bc_energy.V_biomolecule_reconstructed(k_s_sc, jnp.array(midpoints_lin), jnp.array(energies_lin)) # reconstructed 

top = float(max(energies_lin[30:70]))

plt.figure(figsize = (8,8))
plt.plot(midpoints_lin, energies_lin - top, label = "Linear Protocol")
# plt.plot(midpoints_opt, energies_opt, label = "Error-Optimized Protocol")
plt.title(f"Reconstructing Sivak Landscape with Linear Protocol; {end_time_sc}")
plt.xlabel("Particle Position")
plt.ylabel("Free Energy")

energy_sivak_plot = bc_energy.V_biomolecule_sivak(kappa_l, kappa_r, x_m, delta_E, 0, beta_sc)
sivak_E = []
positions = jnp.linspace(r0_init_sc-10, r0_final_sc+10, num = 100)
for i in positions:
  sivak_E.append(energy_sivak_plot([[i]], r0=0.) - float(energy_sivak_plot([[0.]], r0=0.)))
plt.plot(positions, sivak_E, label = "Ground Truth", color = "green")
plt.legend()

plt.savefig("temp.png", transparent = False)
