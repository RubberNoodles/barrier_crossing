"""Tests for barrier_crossing.energy, defining energy and environments of MD simulations"""

from barrier_crossing import energy

import jax.numpy as jnp
from jax_md import test_util

from absl.testing import absltest
from absl.testing import parameterized

class UnitTests(test_util.JAXMDTestCase):
  def test_brownian(self):
    # TODO
    pass
  
  def test_langevin(self):
    # TODO
    pass
  
  
if __name__ == '__main__':
  absltest.main()