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

factor = 1.438113007545471
param_set.kappa_l = factor * 2.6258/(p.beta* p.x_m **2)
param_set.kappa_r = param_set.kappa_l