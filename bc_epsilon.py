# -*- coding: utf-8 -*-

import time 
import collections
import functools
import os
import pickle 
import pandas as pd

import jax
import jax.numpy as jnp

from jax.scipy import ndimage
from scipy import integrate

from jax import jit
from jax import grad
from jax import vmap
from jax import value_and_grad

from jax import random
from jax import lax
from jax import ops

from jax.experimental import stax
from jax.experimental import optimizers as jopt

from jax_md import space
from jax_md import minimize
from jax_md import simulate
from jax_md import space
from jax_md import energy
from jax_md import quantity
from jax_md import smap
import jax_md as jmd

f32 = jnp.float32
f64 = jnp.float64

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import ArtistAnimation
from matplotlib import rc
from matplotlib import colors
import seaborn as sns
rc('animation', html='jshtml')
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import tqdm
import time
import typing

import sys

import scipy as sp
import scipy.special as sps

from numpy import polynomial
import numpy as onp 

#import jax.profiler
#server = jax.profiler.start_server(port=1234)

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

from barrier_crossing import energy as bc_energy
from barrier_crossing import protocol as pc
from barrier_crossing.simulate import simulate_protocol
"""#Function definitions

##Miscellaneous
"""

# SAVE FILE PATH
save_filepath = "./bc_epsilon_"+sys.argv[1]+"/"



# Hyper parameters
N = 1
dim = 1
end_time = 0.01 #s
dt = 3e-7 #s
simulation_steps = int((end_time)/dt)+1
teq=0.001 #s
Neq=int(teq/dt)

batch_size= 3600#2504 number of trajectories
opt_steps = 500#1000 number of optimization steps
#temperature = 4.114 #at 298K = 25C
temperature = 4.183 #at 303K=30C
beta=1.0/temperature #1/(pNnm)
mass = 1e-17 #1e-17 #g
D = 0.44*1e6 #(in nm**2/s) 
gamma = 1./(beta*D*mass) #s^(-1)
init_position=-0.*jnp.ones((N,dim)) #nm

k_s = 0.4 #stiffness
# epsilon = 0.5 * (1.0/beta)
sigma = 1.0/jnp.sqrt(beta * k_s)

#harmonic potential (I call it a "trap") parameters:
r0_init = -0. #initial pos
r0_final = sigma*2. #final pos

#landscape params:
x_m=10. #nm
delta_E=0 #pN nm
kappa_l=21.3863/(beta*x_m**2) #pN/nm #for Ebarrier = 10kT and delta_E=0, as C&S use
#kappa_l=6.38629/(beta*x_m**2) #pN/nm #for Ebarrier = 2.5kT and delta_E=0, as C&S use
#kappa_l=2.6258/(beta*x_m**2)#barrier 0.625kT
kappa_r=kappa_l #pN/nm 

energy_fn = bc_energy.V_biomolecule(kappa_l, kappa_r, x_m, delta_E, k_s, beta, epsilon, sigma)
force_fn = quantity.force(energy_fn)
displacement_fn, shift_fn = space.free()





"""# Final Plots"""

fig, ax = plt.subplots(figsize=(12,12))

ax.set_title('Optimal Protocols for Work vs. Error')

error_trap_fn = pc.make_trap_fxn(jnp.arange(simulation_steps),rev_coeffs_[-1][1],r0_init,r0_final)
error_sched = error_trap_fn(jnp.arange(simulation_steps))
ax.plot(jnp.arange(simulation_steps), error_sched, '-', label=f'Error Optimized')

work_trap_fn = pc.make_trap_fxn(jnp.arange(simulation_steps),coeffs_[-1][1],r0_init,r0_final)
work_sched = work_trap_fn(jnp.arange(simulation_steps))
ax.plot(jnp.arange(simulation_steps), work_sched, '-', label=f'Work Optimized')

ax.set_xlabel("Time (ns)")
ax.set_ylabel("Position (x)")
ax.legend()#
plt.savefig(save_filepath + "finalprotocols.png")
plt.show()

# Fig. 11 (epsilon = 0.5 k_B T), Geiger et al. 2010

linear_works, _ = simulate_protocol(coeffs_[0][1])
forward_optimized_works, _ = simulate_protocol(coeffs_[-1][1])
backward_optimized_works, _ = simulate_protocol(rev_coeffs_[-1][1])
lin_avg = jnp.mean(linear_works)
forward_avg = jnp.mean(forward_optimized_works)
backward_avg = jnp.mean(backward_optimized_works)

print(f"Linear protocol; work used: {lin_avg}")
print(f"Forward-direction Work-optimized protocol; work used: {forward_avg}")
print(f"Backward-direction Error-optimized protocol; work used: {backward_avg}")
print("Ratios:")
print(f"Forward/Linear: {forward_avg/lin_avg}")
print(f"Backward/Linear: {backward_avg/lin_avg}")

df = pd.DataFrame([
                   {"Name": "Linear Average", "Value": lin_avg}, 
                   {"Name": "Forward Average", "Value": forward_avg}, 
                   {"Name": "Backward Average", "Value": backward_avg}, 
                   {"Name": "Forward/Linear Ratio", "Value": forward_avg/lin_avg},
                   {"Name": "Backward/Linear Ratio", "Value": backward_avg/lin_avg}])
df.set_index("Name").to_csv(save_filepath + "work_outputs.csv")



