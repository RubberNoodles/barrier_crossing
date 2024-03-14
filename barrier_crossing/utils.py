### Create datastructures to collect molecular dynamics simulation parameters
### for different simulation regimes

from jax_md import space, util

from dataclasses import dataclass

import barrier_crossing.energy as bc_energy
import barrier_crossing.simulate as bc_simulate
import barrier_crossing.protocol as bc_protocol

import jax.numpy as jnp



import importlib

from argparse import ArgumentParser
import os
import shutil
import pickle
import copy

ShiftFn = space.ShiftFn
Array = util.Array


@dataclass
class MDParameters:
  N: int
  dim: int
  shift: ShiftFn # Defines how to move a particle by small distances dR.
  mass: float
  temperature: float
  gamma: float
  beta: float
  init_position_fwd: Array
  init_position_rev: Array
  k_s: float
  end_time: float
  dt: float
  
  
  Neq: int
    
  def energy_fn(self, k_s, custom = None):
    # Energy function output to be overridden by subclasses.
    pass
  
  @property
  def simulation_steps(self):
    return int(self.end_time/self.dt)
  
  def set_energy_fn(self, custom):
    self.custom = custom
  
  def simulate_fn(self, trap_fn, key, regime = "langevin", fwd = True, custom = None, **kwargs):
    init_pos = self.init_position_fwd if fwd else self.init_position_rev
    energy_fn = custom if custom else self.energy_fn() # Primarily for plotting/iterative procedure
    
    self.__dict__.update(kwargs)
    
    
    if regime.strip().lower() == "langevin":
      return bc_simulate.simulate_langevin_harmonic( energy_fn,
                                                     init_pos,
                                                     trap_fn,
                                                     self.simulation_steps,
                                                     self.Neq,
                                                     self.shift,
                                                     key,
                                                     self.dt,
                                                     self.temperature,
                                                     self.mass,
                                                     self.gamma
                                                     )
      
    elif regime.strip().lower() == "brownian":
      return bc_simulate.simulate_brownian_harmonic( energy_fn,
                                                     init_pos,
                                                     trap_fn,
                                                     self.simulation_steps,
                                                     self.Neq,
                                                     self.shift,
                                                     key,
                                                     self.dt,
                                                     self.temperature,
                                                     self.mass,
                                                     self.gamma
                                                     )
    else:
      raise TypeError(f"'{regime}' not found. Currently available regimes are 'langevin' and 'brownian'.")
      
  def __str__(self):
    output_str = "Parameter Set: \n"
    for param in self.__dict__.keys():
      output_str += f"{param}: {self.__dict__[param]}\n"
    return output_str.strip()

  def __getstate__(self):
    return_dict = copy.deepcopy(self.__dict__)
    if return_dict["fn"]:
      print("WARNING: Shift function cannot be pickled. Continuing...")
      return_dict["fn"] = None
    return return_dict
  
  def __setstate__(self, d):
    if d["fn"] == None:
      print("WARNING: Shift function inferred as space.free(). Manually change if this was unintended.")
      d["fn"] = space.free()
    
    self.__dict__ == d

@dataclass
class SCParameters(MDParameters):
  D: float
  x_m: float
  delta_E: float
  kappa_l: float
  kappa_r: float
  
  def energy_fn(self, k_s = None):
    if "custom" in self.__dict__.keys():
      return self.custom
    if k_s or k_s == 0.:
      return bc_energy.V_biomolecule_sivak(self.kappa_l, self.kappa_r, self.x_m, self.delta_E, k_s, self.beta)
    else:
      return bc_energy.V_biomolecule_sivak(self.kappa_l, self.kappa_r, self.x_m, self.delta_E, self.k_s, self.beta)
  
  
  
@dataclass
class GDParameters(MDParameters):
  epsilon: float
  sigma: float
  
  def energy_fn(self, k_s = None):
    if "custom" in self.__dict__.keys():
      return self.custom
    if k_s or k_s == 0.:
      return bc_energy.V_biomolecule_geiger(k_s, self.epsilon, self.sigma)
    else:
      return bc_energy.V_biomolecule_geiger(self.k_s, self.epsilon, self.sigma)


        
def parse_args():
  """Parse arguments for running various simulations

  Returns:
      args: Namespace of arguments parsed by ArgumentParser()
      p: Parameter set loaded at param_set.params_`param_name`.py
  """
  parser = ArgumentParser()
  parser.add_argument("--landscape_name", type = str, help = "Name of underlying landscape (e.g. 25KT Sivak & Crooks).")
  parser.add_argument("--param_suffix", type = str, help = "Suffix of parameter set inside figures/param_set directory.")
  parser.add_argument('--end_time', type=float, default = None, help='Length of the simulation in seconds.')
  parser.add_argument('--k_s', type = float, default = None, help = "Trap stiffness k_s.")
  parser.add_argument('--batch_size', type = int, default = None, help = "Number of trajectories to simulate in parallel for RECONSTRUCTION.")
  args = parser.parse_args()
  
  param_name = args.param_suffix
  p = importlib.import_module(f"figures.param_set.params_{param_name}")
  if args.k_s:
    p.param_set.k_s = args.k_s
  if args.end_time:
    p.param_set.end_time = args.end_time
  return args, p
  
def copy_dir_coeffs(parent_dir, path, coeff_dir, coeff_files):
  if not os.path.isdir(parent_dir):
    os.mkdir(parent_dir)
  if not os.path.isdir(parent_dir+"plotting"):
    os.mkdir(parent_dir + "plotting")

  if not os.path.isdir(parent_dir+"coeffs"):
    os.mkdir(parent_dir + "coeffs")
    
  if len(os.listdir(parent_dir + "coeffs")) == 0:
    # Only make a new 
    for coeff_type in coeff_files.values():
      
      if ".pkl" in coeff_type and not "lr" in coeff_type: # Linear response optima are in the correct dir
        try:
          shutil.copy(f"../work_error_opt/output_data/{path}/{coeff_type[:-4]}_coeffs.pkl", coeff_dir + coeff_type)
        except FileNotFoundError as err:
          print(f"{err} occurred. Coefficients {coeff_type} not copied. Continuing...")



def make_trap_from_file(file_name, coeff_dir, p: MDParameters):
  if file_name == "linear":
      coeff = bc_protocol.linear_chebyshev_coefficients(p.r0_init, p.r0_final, p.param_set.simulation_steps, degree = 12, y_intercept = p.r0_init)
  else:
    if "lr" in file_name:
      path = file_name
    else:
      path = coeff_dir + file_name
    try:
      with open(path, "rb") as f:
        coeff = jnp.array(pickle.load(f))
    except FileNotFoundError:
      print(f"In order to run this code, you need a file of coefficients called {coeff_dir+file_name}")
      raise
  time_vec = jnp.arange(p.param_set.simulation_steps)
  if file_name == "split.pkl":
    sim_cut_steps = p.param_set.simulation_steps // 2
    # We are going to only look at gradient values for the second set of coefficients
    _a = bc_protocol.make_trap_fxn( jnp.arange(sim_cut_steps), coeff[0], p.r0_init, p.r0_cut)
    _b = bc_protocol.make_trap_fxn( jnp.arange(p.param_set.simulation_steps - sim_cut_steps), coeff[1], p.r0_cut, p.r0_final)
    trap_fn = bc_protocol.trap_sum(p.param_set.simulation_steps, sim_cut_steps, _a, _b)
    
  elif "lr" in file_name:
    time = jnp.linspace(0, p.param_set.simulation_steps, coeff.shape[0])
    positions = jnp.vstack((time, coeff)).T
    trap_fn = bc_protocol.make_custom_trap_fxn(time_vec, positions, p.r0_init, p.r0_final)  
  else:
    trap_fn = bc_protocol.make_trap_fxn(time_vec, coeff, p.r0_init, p.r0_final)
  
  return trap_fn, coeff