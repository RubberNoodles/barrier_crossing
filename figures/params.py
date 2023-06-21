############################################
### GLOBAL parameters for all figures.   ###
############################################
# Style: https://docs.python.org/3/faq/programming.html#how-do-i-share-global-variables-across-modules

# Symmetric.

import jax.numpy as jnp
import jax.random as random
from jax_md import space

import barrier_crossing.energy as bc_energy

N = 1
dim = 1

_, shift = space.free() # Defines how to move a particle by small distances dR.

# ================= SIVAK & CROOKE =====================

# Harmonic Trap Parameters S&C
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

# Symmetric wells.
#kappa_l=21.3863/(beta_sc*x_m**2) #pN/nm #for Ebarrier = 10kT and delta_E=0, as C&S use
#kappa_l=6.38629/(beta_sc*x_m**2) # Ebarrier ~ 13 kT
kappa_l=2.6258/(beta_sc*x_m**2)#barrier 0.625kT
kappa_r=kappa_l #pN/nm; 

# Asymmetric wells.
# kappa_l=6.38629/(beta*x_m**2)
# kappa_r= 5 * kappa_l #pN/nm; Asymmetric wells

end_time_sc = 1e-5
dt_sc = 3e-9
simulation_steps_sc = int(end_time_sc / dt_sc)

Neq = 500

energy_sivak = bc_energy.V_biomolecule_sivak(kappa_l, kappa_r, x_m, delta_E, k_s_sc, beta_sc)

