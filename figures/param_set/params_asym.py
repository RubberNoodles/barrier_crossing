import figures.param_set.params_base as p

simulation_steps = p.simulation_steps
Neq = p.Neq
dt = p.dt
r0_init = p.r0_init
r0_final = p.r0_final
beta = p.beta

param_set = p.sc_params

param_set.kappa_l = 6.38629/(p.beta* p.x_m**2)
param_set.kappa_r = 5 * param_set.kappa_l

energy_sivak = param_set.energy_fn(p.k_s)