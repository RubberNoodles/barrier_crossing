import figures.param_set.params_base as p

import jax.numpy as jnp
import barrier_crossing.energy as bc_energy

simulation_steps = p.simulation_steps
sim_cut_steps = p.sim_cut_steps
Neq = p.Neq
dt = p.dt
r0_init = p.r0_init
r0_final = p.r0_final
r0_cut = p.r0_cut
beta = p.beta

param_set = p.sc_params

pos_e = [[5.5*x, 8*x**2 + 0.3 * x**3 - 6 * x ** 4 + x**6] for x in jnp.linspace(-6,6,1000)] # https://www.scielo.org.mx/pdf/rmf/v57n1/v57n1a8.pdf
e_positions = jnp.array(pos_e)[:,0]
e_energies = jnp.array(pos_e)[:,1] * 3.5


energy_triple_well = bc_energy.V_biomolecule_reconstructed(p.k_s, e_positions, e_energies)
param_set.set_energy_fn(energy_triple_well)
