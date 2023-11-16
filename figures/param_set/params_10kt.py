import figures.param_set.params_base as p

simulation_steps = p.simulation_steps
sim_cut_steps = p.sim_cut_steps
Neq = p.Neq
dt = p.dt
r0_init = p.r0_init
r0_final = p.r0_final
r0_cut = p.r0_cut
beta = p.beta

param_set = p.sc_params

param_set.kappa_l= 6.38629/(beta* p.x_m **2)
param_set.kappa_r = param_set.kappa_l

energy_sivak = param_set.energy_fn(p.k_s)