############################################
### GLOBAL parameters for all figures.   ###
############################################
# Style: https://docs.python.org/3/faq/programming.html#how-do-i-share-global-variables-across-modules

# Symmetric.

import jax.numpy as jnp
from jax_md import space

import barrier_crossing.utils as bcu

N = 1
dim = 1

_, shift = space.free() # Defines how to move a particle by small distances dR.

# ================= SIVAK & CROOKE =====================

# Megan's is 0.4
#k_s = 1.0 # stiffness; 
k_s = 1.0 # stiffness; for 10kt

# Harmonic Trap Parameters S&C
r0_init = -10. #nm; initial trap position
r0_final = 10. #nm; final trap position
r0_cut = 0. #nm; where to split protocol
ks_init = k_s
ks_final = k_s
ks_cut = (ks_init + ks_final)/2

# Particle Parameters S&C
mass = 1e-17 # g
init_position_fwd = r0_init*jnp.ones((N,dim)) #nm
init_position_rev = r0_final*jnp.ones((N,dim))

# Brownian Environment S&C
temperature = 4.183 #at 303K=30C S&C
# temperature = temperature/4
beta=1.0/temperature #1/(pNnm)
D = 0.44*1e6 #(in nm**2/s) 
gamma = 1./(beta*D*mass) #s^(-1)


# S&C Energy landscape params:
x_m=10. #nm
delta_E= 0. #pN nm
kappa_l=6.38629/(beta*x_m**2) # Ebarrier ~ 2,5 kT
kappa_r=kappa_l #pN/nm; 
Neq = 1000

# end_time = 4e-6 # Langevin
# dt = 2e-9
end_time = 1e-4 # Brownian
dt = 2e-7 # needs to be 1e-7 to avoid drift...

sc_params = bcu.SCParameters( N = N,
                                    dim = dim,
                                    shift = shift,
                                    k_s = k_s,
                                    mass = mass,
                                    init_position_fwd = init_position_fwd,
                                    init_position_rev = init_position_rev,
                                    temperature = temperature,
                                    beta = beta,
                                    D = D, #(in nm**2/s) 
                                    gamma = gamma,
                                    x_m = x_m,
                                    delta_E = delta_E,
                                    kappa_l = kappa_l,
                                    kappa_r = kappa_r,
                                    dt = dt,
                                    end_time = end_time,
                                    Neq = Neq,
)
