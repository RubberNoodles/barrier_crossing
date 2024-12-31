import figures.param_set.params_base as p

Neq = p.Neq
dt = p.dt
r0_init = p.r0_init
r0_final = p.r0_final
r0_cut = p.r0_cut
beta = p.beta
ks_init = p.ks_init
ks_final = p.ks_final
ks_cut = p.ks_cut
beta = p.beta

param_set = p.sc_params

factor = 2.4618721485137938
    
import jax.numpy as jnp
import barrier_crossing.energy as bce
pos_e = [[5.5*x, 8*x**2 + 0.3 * x**3 - 6 * x ** 4 + x**6] for x in jnp.linspace(-6,6,1000)] # https://www.scielo.org.mx/pdf/rmf/v57n1/v57n1a8.pdf
e_positions = jnp.array(pos_e)[:,0]
e_energies = jnp.array(pos_e)[:,1] * factor


energy_triple_well = bce.ReconstructedLandscape(e_positions, e_energies)
param_set.set_energy_fn(energy_triple_well)