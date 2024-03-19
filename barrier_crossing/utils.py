"""Create datastructures to collect molecular dynamics simulation parameters for different simulation regimes.
Also provide helper functions such as `find_coeff_files` for IO."""
from jax_md import space, util

from dataclasses import dataclass

import barrier_crossing.energy as bce
import barrier_crossing.simulate as bcs
import barrier_crossing.protocol as bcp
import barrier_crossing.models as bcm

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


  @property
  def landscape(self):
    # Landscape output to be overridden by subclasses.
    pass
  
  @property
  def simulation_steps(self):
    return int(self.end_time/self.dt)
  
  def energy_fn(self, no_trap = False):
    if "custom" in self.__dict__.keys():
      return self.custom.total_energy
    else:
      if no_trap:
        return lambda position, **kwargs: self.landscape.total_energy(position, k_s = 0., r0 = 0., **kwargs)
      else:
        return self.landscape.total_energy
  
  def set_energy_fn(self, custom):
    self.custom = custom
  
  def simulate_fn(self, trap_fn, ks_trap_fn,  key, regime = "brownian", fwd = True, custom = None, **kwargs):
    init_pos = self.init_position_fwd if fwd else self.init_position_rev
    energy_fn = custom if custom else self.energy_fn() # Primarily for plotting/iterative procedure
    
    self.__dict__.update(kwargs)

    if regime.strip().lower() == "langevin":
      return bcs.simulate_langevin_harmonic( energy_fn,
                                                     init_pos,
                                                     trap_fn,
                                                     ks_trap_fn,
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
      return bcs.simulate_brownian_harmonic( energy_fn,
                                                     init_pos,
                                                     trap_fn,
                                                     ks_trap_fn,
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
    if "shift" in return_dict.keys():
      if return_dict["shift"]:
        print("WARNING: Shift function cannot be pickled. Continuing...")
        return_dict["shift"] = None
    return return_dict
  
  def __setstate__(self, d):
    if "shift" in d.keys():
      if d["shift"] == None:
        print("WARNING: Shift function inferred as space.free(). Manually change if this was unintended.")
        d["shift"] = space.free()
      
    self.__dict__ == d

  @classmethod
  def copy(cls, obj):
    """Manual copy"""
    return cls(**obj.__dict__)
    
    
@dataclass
class SCParameters(MDParameters):
  D: float
  x_m: float
  delta_E: float
  kappa_l: float
  kappa_r: float
  
  @property
  def landscape(self):
    return bce.SivakLandscape(self.kappa_l, self.kappa_r, self.x_m, self.delta_E, self.beta)

@dataclass
class GDParameters(MDParameters):
  epsilon: float
  sigma: float
  
  @property
  def landscape(self):
    return bce.GeigerLandscape(self.epsilon, self.sigma)
        
### IO HELPER FUNCTIONS ###

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


def find_coeff_file(model_info, args):
  """Helper function for reconstruct.py"""
  if model_info == "linear":
    return None, None
  dir_name = "../work_error_opt/output_data/" + "_".join([args.landscape_name.replace(' ', '_').replace('.', '_').lower(), f"t{args.end_time}", f"ks{args.k_s}"]) +"/"
  #dir_name = "figures/work_error_opt/output_data/" + "_".join([args.landscape_name.replace(' ', '_').replace('.', '_').lower(), f"t{args.end_time}", f"ks{args.k_s}"]) +"/"
  
  model_type = model_info[0]
  if "lr" in model_type:
    dir_name = ""
    file_name = model_info[0]
    return dir_name, {"position": [file_name]}
  mode = model_info[1]
  
  
  if model_type == "joint":
    coeff_searches = {
      "position":[f"joint_{mode}_position"], 
      "stiffness": [f"joint_{mode}_stiffness"]
      }
  
  else:
    coeff_searches = {f"{mode}":[f"{model_type}_{mode}"]}
  if "split" in model_info:
    for key, search in coeff_searches.items():
      coeff_searches[key] = [search[0] + "_split_1", search[0] + "_split_2"]
  
  coeff_searches = { key: [search + ".pkl" for search in searches] for key, searches in coeff_searches.items()}
  return dir_name, coeff_searches
  
def make_constant_trap(model, val):
  _tmp_1 = model.init_pos
  _tmp_2 = model.final_pos
  model.init_pos = val
  model.final_pos = val
  constant_trap = model.protocol(jnp.concatenate([jnp.array([val]), jnp.zeros(12)]))
  model.init_pos = _tmp_1
  model.final_post = _tmp_2
  
  return constant_trap

def make_trap_from_file(dir_name, file_names, position_protocol_maker, stiffness_protocol_maker, p):
  """Helper function to output list containing [position_trap, stiffness_trap], in that order"""
  params = position_protocol_maker.params
  
  constant_stiffness_schedule = make_constant_trap(stiffness_protocol_maker, params.k_s)
  traps = [position_protocol_maker.protocol(position_protocol_maker.coeffs), constant_stiffness_schedule]
  sim_cut_steps = params.simulation_steps//2
  
  if file_names is None: # For linear trap
    return traps
      
  if "position" in file_names.keys():
    for file_name in file_names["position"]:
      
      path = dir_name + file_name
      try:
        with open(path, "rb") as f:
          coeff = jnp.array(pickle.load(f))    
      except FileNotFoundError as e:
        # figures/work_error_opt/output_data/double_well_10kt_barrier_brownian_tNone_ksNone/joint_rev_position.pkl
        print(f"In order to run this code, you need a file of coefficients at {dir_name+file_name}")
        raise e
      if "lr" in file_name:
        even = jnp.linspace(0, params.simulation_steps, coeff.shape[0])
        traps[0] = bcp.make_custom_trap_fxn(jnp.arange(params.simulation_steps), jnp.vstack([even, coeff]).T, p.r0_init, p.r0_final)
      elif "split_1" in file_name:
        traps[0] = [bcp.make_trap_fxn( jnp.arange(sim_cut_steps), coeff, p.r0_init, p.r0_cut)]
      elif "split_2" in file_name:
        traps[0].append(bcp.make_trap_fxn( jnp.arange(params.simulation_steps - sim_cut_steps), coeff, p.r0_cut, p.r0_final))
      else:
        traps[0] = position_protocol_maker.protocol(coeff)
  
  if "stiffness" in file_names.keys():
    for file_name in file_names["stiffness"]:
      path = dir_name + file_name
      try:
        with open(path, "rb") as f:
            coeff = jnp.array(pickle.load(f))    
      except FileNotFoundError as e:
        print(f"In order to run this code, you need a file of coefficients at `{dir_name+file_name}`.")
        raise e
      
      if "split_1" in file_name:
        traps[1] = [bcp.make_trap_fxn( jnp.arange(sim_cut_steps), coeff, p.ks_init, p.ks_cut)]
      elif "split_2" in file_name:
        traps[1].append(bcp.make_trap_fxn( jnp.arange(params.simulation_steps - sim_cut_steps), coeff, p.ks_cut, p.ks_final))
      else:
        traps[1] = stiffness_protocol_maker.protocol(coeff)
          
  for i in range(0,2):
    if isinstance(traps[i], list):
      traps[i] = bcp.trap_sum(params.simulation_steps, sim_cut_steps, traps[i][0], traps[i][1])
  
  return traps
