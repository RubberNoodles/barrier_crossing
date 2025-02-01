import params_base as p
import jax.numpy as jnp
import os
import jax
import tqdm
import barrier_crossing.energy as bce

BASE_STRING = """import copy
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

param_set = copy.deepcopy(p.sc_params)

"""

barrier_heights = [2.5, 5, 10, 15, 20, 25, 30, 35, 40]
base_param_set = p.sc_params

X_VEC = jnp.linspace(-10,10, 1000)
def make_double_well(factor):
    base_param_set.kappa_l = factor * 2.6258/(p.beta* p.x_m **2)
    base_param_set.kappa_r = base_param_set.kappa_l
    
    out_string = f"""factor = {factor}
param_set.kappa_l = factor * 2.6258/(p.beta* p.x_m **2)
param_set.kappa_r = param_set.kappa_l"""
    
    return base_param_set.energy_fn(no_trap = True), out_string

def make_double_well_asym(factor):
    base_param_set.kappa_l = factor * 2.6258/(p.beta* p.x_m **2)
    base_param_set.kappa_r = 10 * base_param_set.kappa_l
    
    out_string = f"""factor = {factor}
param_set.kappa_l = factor * 2.6258/(p.beta* p.x_m **2)
param_set.kappa_r = 10 * param_set.kappa_l"""
    return base_param_set.energy_fn(no_trap = True), out_string

def make_triple_well(factor):
    
    pos_e = [[5.5*x, 8*x**2 + 0.3 * x**3 - 6 * x ** 4 + x**6] for x in jnp.linspace(-6,6,1000)] # https://www.scielo.org.mx/pdf/rmf/v57n1/v57n1a8.pdf
    e_positions = jnp.array(pos_e)[:,0]
    e_energies = jnp.array(pos_e)[:,1] * factor


    energy_triple_well = bce.ReconstructedLandscape(e_positions, e_energies)
    base_param_set.set_energy_fn(energy_triple_well)
    
    out_string = f"""factor = {factor}
    
import jax.numpy as jnp
import barrier_crossing.energy as bce
pos_e = [[5.5*x, 8*x**2 + 0.3 * x**3 - 6 * x ** 4 + x**6] for x in jnp.linspace(-6,6,1000)] # https://www.scielo.org.mx/pdf/rmf/v57n1/v57n1a8.pdf
e_positions = jnp.array(pos_e)[:,0]
e_energies = jnp.array(pos_e)[:,1] * factor


energy_triple_well = bce.ReconstructedLandscape(e_positions, e_energies)
param_set.set_energy_fn(energy_triple_well)"""
    return base_param_set.energy_fn(no_trap=True), out_string

factors = jnp.linspace(0.5,100, 30000)

energy_files = {"double_well": make_double_well, "asymmetric": make_double_well_asym, "triple_well": make_triple_well}
for landscape_type, energy_test_fn in energy_files.items():
    
    for barrier_height in tqdm.tqdm(barrier_heights):
        last_string = None
        
        # Some binary search
        diff = barrier_height
        left = 0.01
        right = 1000
        factor = (left + right) / 2
        while diff > 0.001:
            energy_fn, out_string = energy_test_fn(factor)
            vmap_e = jax.vmap(energy_fn)
            energies = jnp.array(list(vmap_e(X_VEC)))
            energies -= energies[0]
            
            guess_barrier = jnp.max(energies)
            diff = jnp.abs(guess_barrier - barrier_height)
            
            if guess_barrier > barrier_height:
                right = factor
            else:
                left = factor
            
            factor = (left + right)/2
            
            last_string = out_string
        
        if last_string is not None:
            out_string = last_string
        
        os.makedirs(landscape_type, exist_ok= True)
        with open(f"{landscape_type}/params_{str(barrier_height).replace('.', '_')}kt.py", "w") as f:
            f.write(BASE_STRING + out_string)
            