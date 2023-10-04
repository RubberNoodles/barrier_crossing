"""Tests for barrier_crossing.protocol, functions for defining velocities at which to move a harmonic trap"""

from barrier_crossing import protocol

import jax.numpy as jnp
from jax_md import test_util

from absl.testing import absltest
from absl.testing import parameterized

class UnitTests(test_util.JAXMDTestCase):
  @parameterized.parameters(
    (-10., 10.),
    (-2., 0.)
  )
  def test_make_trap(self, r0_init, r0_final):
    coeff_arr = [0. if i!=1 else 3. for i in range(13) ]
    coeffs = jnp.array(coeff_arr)
    sim_steps = 2000
    trap_fn = protocol.make_trap_fxn(jnp.arange(sim_steps), coeffs, r0_init, r0_final)
    
    
    test_steps = jnp.array([500,1000,1500])
    true = [scale * 3. for scale in [-0.5,0.,0.5]]
    
    (r0_final - r0_init)
    self.assertAllClose(jnp.array((trap_fn(0), trap_fn(sim_steps))), jnp.array((r0_init, r0_final)))
    self.assertAllClose(trap_fn(sim_steps+1),r0_final)
    self.assertAllClose(trap_fn(test_steps), jnp.array(true), rtol = 0.01, atol = 0.01)
    
if __name__ == '__main__':
  absltest.main()