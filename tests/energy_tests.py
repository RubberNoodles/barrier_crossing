"""Tests for barrier_crossing.energy, defining energy and environments of MD simulations
Forked from jax_md testing"""

from barrier_crossing.energy import brownian, nvt_langevin

import matplotlib.pyplot as plt

import jax.numpy as jnp
import numpy as onp

import jax.random as random

from jax_md import test_util
from jax_md import space

from absl.testing import absltest
from absl.testing import parameterized

C = 3 # Empirical constant for stochastic Gaussian convergence (CLT limit)
mass = 1e-17
temperature = 4.183
beta = 1.0/temperature #1/(pNnm)
D = 0.44*1e6 #(in nm**2/s)
gamma = 1./(beta*D*mass) 

_, shift = space.free() # Defines how to move a particle by small distances dR.

PARTICLE_NUMS = [1000,3000,10000]
SCALES = [0.01, 1.]
class UnitTests(test_util.JAXMDTestCase):
  
  @parameterized.named_parameters(test_util.cases_from_list({
     'testcase_name': f"N={N} alpha={alpha}",
     'N': N,
     'alpha': alpha
    } for N in PARTICLE_NUMS for alpha in SCALES))
  def test_brownian(self, N, alpha):
    variances = []
    means = []
    steps = 1000
    alpha = 1.0
    E = lambda x, t: jnp.sum( 0.5 * alpha * x ** 2)

    N = 10000
    X = jnp.ones((N,1,1))

    dt = 1e-7
    init, apply = brownian(E, shift, dt=dt, T_schedule=temperature, gamma=gamma)

    key = random.PRNGKey(0)


    state = init(key, X, mass=mass)

    for _ in range(steps):
      state = apply(state)
      variances.append(jnp.var(jnp.squeeze(state.position)))
      means.append(jnp.mean(jnp.squeeze(state.position)))
    
    true_var = temperature/alpha
    
    fig, ax = plt.subplots(1,2, figsize = (16,8))
    ax[0].hist(jnp.squeeze(state.position), bins = 100)
    ax[0].set_title("Brownian")
    ax[1].plot(variances)
    ax[1].set_title(f"Variance over steps; dt = {dt}")
    ax[1].axhline(true_var, color = 'r', label = f"True Variance = {true_var}")
    ax[1].legend()
    fig.savefig(f"brownian_test_{N}_{alpha}.png")
    
    s_mean = onp.mean(means[-steps//10:])
    s_var = onp.mean(variances[-steps//10:])
  
    assert s_mean <  C / (onp.sqrt(N) * alpha), f"Mean Position Drift: {s_mean}"
    assert onp.abs(s_var - true_var)/true_var < 2*C/onp.sqrt(N), f"Variance Drift: {onp.abs(s_var - true_var)/true_var}"
  
  
  @parameterized.named_parameters(test_util.cases_from_list({
     'testcase_name': f"N={N} alpha={alpha}",
     'N': N,
     'alpha': alpha
    } for N in PARTICLE_NUMS for alpha in SCALES))
  def test_langevin(self, N, alpha):
    steps = 1000
    

    dt = 3e-9
    variances = []
    means = []
    
    E = lambda x: jnp.sum( 0.5 * alpha * x ** 2)
    X = jnp.zeros((N,1,1))

    init, apply = nvt_langevin(E, shift, dt=dt, kT=temperature, gamma=gamma)

    key = random.PRNGKey(0)


    state = init(key, X, mass=mass)

    for _ in range(steps):
      state = apply(state)
      variances.append(jnp.var(jnp.squeeze(state.position)))
      means.append(jnp.mean(jnp.squeeze(state.position)))
      
    true_var = temperature/alpha
    
    fig, ax = plt.subplots(1,2, figsize = (16,8))
    ax[0].hist(jnp.squeeze(state.position), bins = 100)
    ax[0].set_title("Langevin")
    ax[1].plot(variances)
    ax[1].set_title(f"Variance over steps; dt = {dt}")
    ax[1].axhline(true_var, color = 'r', label = f"True Variance = {true_var}")
    ax[1].legend()
    fig.savefig(f"langevin_test_{N}_{alpha}.png")
    
    s_mean = onp.mean(means[-steps//10:])
    s_var = onp.mean(variances[-steps//10:])
    
    assert s_mean <  C / (onp.sqrt(N) * alpha), f"Mean Position Drift: {s_mean}"
    assert onp.abs(s_var - true_var)/true_var < C/onp.sqrt(N), f"Variance Drift: {onp.abs(s_var - true_var)/true_var}"
  
if __name__ == '__main__':
  absltest.main()