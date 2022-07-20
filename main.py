import time
import sys

import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from jax import random 

from jax_md import quantity, space

from barrier_crossing.energy import V_biomolecule_geiger
from barrier_crossing.simulate import simulate_brownian_harmonic, batch_simulate_harmonic
from barrier_crossing.protocol import linear_chebyshev_coefficients, make_trap_fxn, make_trap_fxn_rev

from main_tests import test_geiger_simulate, test_fwd_opt
from main_geiger import geiger_error_opt, geiger_work_opt, get_linear_works

if __name__ == "__main__":
  # test_geiger_simulate()
  # test_fwd_opt()
  batch_size = 1e5
  opt_steps = 5
  task_num = int(sys.argv[1])
  geiger_work_opt(task_num, batch_size, opt_steps)
  geiger_error_opt(task_num, batch_size, opt_steps)
  get_linear_works(task_num, batch_size)
