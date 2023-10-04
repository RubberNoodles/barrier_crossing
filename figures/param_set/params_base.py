############################################
### GLOBAL parameters for all figures.   ###
############################################
# Style: https://docs.python.org/3/faq/programming.html#how-do-i-share-global-variables-across-modules

# Symmetric.

import jax.numpy as jnp
import jax.random as random
from jax_md import space

import barrier_crossing.energy as bc_energy
import barrier_crossing.param as bc_params

N = 1
dim = 1

_, shift = space.free() # Defines how to move a particle by small distances dR.

# ================= SIVAK & CROOKE =====================

# Harmonic Trap Parameters S&C
r0_init = -10. #nm; initial trap position
r0_final = 10. #nm; final trap position

# Particle Parameters S&C
mass = 1e-17 # g
init_position_fwd = r0_init*jnp.ones((N,dim)) #nm
init_position_rev = r0_final*jnp.ones((N,dim))

# Brownian Environment S&C
temperature = 4.183 #at 303K=30C S&C
beta=1.0/temperature #1/(pNnm)
D = 0.44*1e6 #(in nm**2/s) 
gamma = 1./(beta*D*mass) #s^(-1)


# S&C Energy landscape params:
x_m=10. #nm
delta_E= 0. #pN nm

Neq = 500

end_time = 3e-6
dt = 3e-9
simulation_steps = int(end_time / dt)
kappa_l=6.38629/(beta*x_m**2) # Ebarrier ~ 2,5 kT
kappa_r=10*kappa_l #pN/nm; 

k_s = 0.1 # stiffness; 

sc_params = bc_params.SCParameters( N = N,
                                    dim = dim,
                                    shift = shift,
                                    k_s = k_s,
                                    mass = mass,
                                    init_position_fwd = init_position_fwd,
                                    init_position_rev = init_position_rev,
                                    temperature = temperature,
                                    beta = temperature,
                                    D = D, #(in nm**2/s) 
                                    gamma = gamma,
                                    x_m = x_m,
                                    delta_E = delta_E,
                                    kappa_l = kappa_l,
                                    kappa_r = kappa_r,
)

