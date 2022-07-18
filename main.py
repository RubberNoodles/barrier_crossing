import time
import sys

import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from jax import random 

from jax_md import quantity, space

from barrier_crossing.energy import V_biomolecule, V_simple_spring, run_brownian_opt
from barrier_crossing.protocol import linear_chebyshev_coefficients, make_trap_fxn, make_trap_fxn_rev
"""#Simulations

##Brownian particle simulation

###Parameters:
"""

#the parameters as I have set them are meant to be similar to an experiment where DNA hairpins are unfolded by laser tweezers
save_filepath = "temp/"
N = 1 # number of particles
dim = 1 # number of dimensions
end_time = 0.001 
dt = 3e-7 #integration time step
simulation_steps = int((end_time)/dt)+1
teq=0.001 #how long to let the system equilibrate in the initial potentials before you start applying the protocol 
Neq=int(teq/dt) #number of equilibration timesteps

batch_size=5000 # how many simulations/trajectories to perform
beta=1.0/4.114 #inverse temperature kT (Boltzmann factor times temperature)
mass = 1e-17 #particle mass 
D = 0.44*1e6 #diffusion constant (tells you how fast the particle is able to move around (friction slows it down)) 
gamma = 1./(beta*D*mass) #friction coefficient

k_s = 0.4 #stiffness
epsilon = int(sys.argv[1])/2 * (1.0/beta) # PRESIDENTIAL EPSILON
print(f"Epsilon is: {int(sys.argv[1])/2} * 1/beta = {epsilon}")
sigma = 1.0/jnp.sqrt(beta * k_s)

#harmonic potential (I call it a "trap") parameters:
r0_init = -0. #initial pos
r0_final = sigma*2. #final pos

#landscape params (see the 2016 Sivak and Crooks paper for what the parameters mean in the mathematical expression)
x_m=10. #How far the energy barrier is from the left-most well 
delta_E=0 #how separated the two minima are from each other -- take them to be at the same height
kappa_l=6.38629/(beta*x_m**2)#for a barrier height 2.5kT
kappa_r=kappa_l #set wells to have equal curvature (not necessary for us, but 2016 Crooks paper does this)

"""###Run Simulation:"""
trap_coeffs = linear_chebyshev_coefficients(r0_init, r0_final, simulation_steps, degree = 12, y_intercept = 0.) # what shape the trap protocol is 
trap_fn = make_trap_fxn(jnp.arange(simulation_steps), trap_coeffs, r0_init, r0_final)
