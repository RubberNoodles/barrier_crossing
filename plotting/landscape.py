
import time
import tqdm
import pickle

import csv
import sys

import barrier_crossing.energy as bc_energy
import barrier_crossing.protocol as bc_protocol
import barrier_crossing.simulate as bc_simulate
import barrier_crossing.optimize as bc_optimize
import barrier_crossing.iterate_landscape as bc_landscape

import jax

import jax.numpy as jnp
import numpy as onp

import jax.random as random

import jax.example_libraries.optimizers as jopt

from jax_md import space

import matplotlib.pyplot as plt


def optimization_plot(losses, coeffs):
  return NotImplementedError

def reconstruction_plot(...):
  return NotImplementedError

def protocol_statistics_plot(...):
  return NotImplementedError

