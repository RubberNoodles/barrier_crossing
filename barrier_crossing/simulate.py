import time
import sys

import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from jax import random 

from jax_md import quantity, space

from energy import V_biomolecule, V_simple_spring, run_brownian_opt
from protocol import linear_chebyshev_coefficients, make_trap_fxn, make_trap_fxn_rev
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


def simulate_protocol(trap_coeffs):  
  #generate random keys for simulation:

  key = random.PRNGKey(int(time.time()))
  key, split = random.split(key, 2)
  init_position=-0.*jnp.ones((N,dim)) #initial particle location

  #see JAX-MD documentation for details on how these energy/force/displacement functions work:
  energy_fn = V_biomolecule(kappa_l, kappa_r, x_m, delta_E, k_s, beta, epsilon, sigma)
  force_fn = quantity.force(energy_fn)
  displacement_fn, shift_fn = space.free()

  tot_works = []

  # To generate a bunch of samples, we 'map' across seeds.
  mapped_sim = jax.soft_pmap(lambda keys : run_brownian_opt(energy_fn, trap_coeffs, init_position, r0_init, r0_final, Neq, shift_fn, keys, simulation_steps, dt, 1./beta, mass, gamma))
  seeds = jax.random.split(split, batch_size)
  trajectories, _, works = mapped_sim(seeds) #seed is array with diff seed for each run. I'm discarding the log prob data, here

  tot_works = jnp.sum(works, 1)
  
  return tot_works, (trajectories, works)


tot_works, (trajectories, works) = simulate_protocol(trap_coeffs)
print("average work done in moving the particle: ",jnp.mean(tot_works))

"""###plots"""

#what the underlying 'molecular' potential looks like:

x=jnp.linspace(-4,10,200)
xvec=jnp.reshape(jnp.linspace(-4,10,200), [200,1,1])
k_splot = 0.
Vfn = V_biomolecule(0, 0, 0, 0, k_splot, beta, epsilon, sigma) # returns in pN nm
V = []
for j in range(len(xvec)):
  V.append(Vfn(xvec[j], r0=0.))
plt.figure(figsize=(10,10))
plt.plot(x,V,'-o')
plt.savefig(save_filepath+ "potential.png")
plt.show()
####PLOT RESULTS FORWARD#####

_, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=[30, 12])
ax0.plot(dt*1000*jnp.arange(simulation_steps),trap_fn(jnp.arange(simulation_steps)), '-o')
ax0.set(xlabel="time (ms)")
ax0.set_title('Initial trap schedule')



for j in range(batch_size):
    if j % 1000 == 0:
      ax1.plot(dt*1000*jnp.arange(simulation_steps), trajectories[j][:,0,0])
#ax1.legend()#
ax1.set(xlabel="time (ms)")
ax1.set_title('Particle positions')


for j in range(batch_size):
  if j % 1000 == 0:
    ax2.plot(dt*1000*jnp.arange(simulation_steps)+1, works[j], '-o')
#ax2.plot(dt*1000*jnp.arange(simulation_steps)+1, summary[1], '-o')
ax2.set(xlabel="time (ms)")
ax2.set_title('Energy increments')
plt.savefig(save_filepath+ "forward_sim.png")
plt.show()

####PLOT RESULTS BACKWARD#####
back_sim_steps = jnp.flip(jnp.arange(simulation_steps))
_, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=[30, 12])
ax0.plot(dt*1000*back_sim_steps,trap_fn(jnp.flip(back_sim_steps)), '-o')
ax0.set(xlabel="time (ms)")
ax0.set_title('Initial backwards trap schedule')



for j in range(batch_size):
    if j % 1000 == 0:
      ax1.plot(dt*1000*back_sim_steps, trajectories[j][:,0,0])
#ax1.legend()#
ax1.set(xlabel="time (ms)")
ax1.set_title('Backward Particle positions')


for j in range(batch_size):
  if j % 1000 == 0:
    ax2.plot(dt*1000*back_sim_steps+1, works[j], '-o')
#ax2.plot(dt*1000*jnp.arange(simulation_steps)+1, summary[1], '-o')
ax2.set(xlabel="time (ms)")
ax2.set_title('Backward Energy increments')
plt.savefig(save_filepath+ "backward_sim.png")
plt.show()

##### PLOT WORK DISTRIBUTION #####
plt.figure(figsize=[12, 12])
# plt.hist(jnp.array(tot_works)*beta,20,alpha=1.0,color='g')

plt.xlabel("Work (kbT)")
plt.ylabel("counts")
plt.legend()
plt.savefig(save_filepath+ "USELESSwork_distribution.png")
plt.show()
print("forward mean:", jnp.mean(jnp.array(tot_works)*beta), "kbT")

"""## Optimization of trap protocol

### Parameters
"""