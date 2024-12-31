import figures.param_set.params_base as p
import barrier_crossing.utils as bc_params

import jax.numpy as jnp

epsilon = 0.5
sigma = 1.0
simulation_steps = 500
sim_cut_steps = simulation_steps//2
Neq = p.Neq
dt = 2e-3
r0_init = 0.
r0_final = 2. * sigma
r0_cut = p.r0_cut
beta = p.beta

init_position_fwd = r0_init*jnp.ones((p.N, p.dim)) #nm
init_position_rev = r0_final*jnp.ones((p.N, p.dim))


param_set = bc_params.GDParameters( N = p.N,
                                    dim = p.dim,
                                    shift = p.shift,
                                    k_s = 1.0,
                                    mass = 1.0,
                                    init_position_fwd = init_position_fwd,
                                    init_position_rev = init_position_rev,
                                    temperature = 1.0,
                                    beta = 1.0,
                                    gamma = 1.0,
                                    dt = dt,
                                    end_time = 1.0,
                                    Neq = p.Neq,
                                    # simulation_steps = simulation_steps,
                                    epsilon = epsilon,
                                    sigma = sigma
)
