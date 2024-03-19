"""
This module contains models derived from `protocol.py`. Derived objects originate from `ScheduleModel`,
providing functionality for gradient descent JAX optimization.
"""
import jax.numpy as jnp
import copy

import barrier_crossing.protocol as bcp
from barrier_crossing.utils import MDParameters

class ScheduleModel:
  """
  Model designed for JAX optimization. Stores coefficients of schedule as optimizeable weights.
  To retrieve protocol, one can call the model directly (backwards compatibility) or receive a protocol
  as an object using `model.protocol(coeffs)`
  """
  def __init__(self, p: MDParameters, init_pos, final_pos, coeffs = None, mode = "fwd"):
    if coeffs is None:
      # Initalize at linear 
      coeffs = bcp.linear_chebyshev_coefficients(init_pos, final_pos, p.simulation_steps, y_intercept = init_pos)
    
    self._params = p
    
    self._coeffs = coeffs
    self.coef_hist = [coeffs]
    self.models = [self] # Cross compatibiltiy with JointModel
    
    self.init_pos = init_pos
    self.final_pos = final_pos
    self.can_grad = True  
    self.mode = mode
    self.__check_mode()
    
  
  @classmethod
  def from_positions(cls, p: MDParameters,  init_pos, final_pos, positions, mode = "fwd"):
    """Alternative constructor for creating protocols interpolating between `positions`"""
    print("WARNING: Models created from positions are not saveable")
    t = jnp.arange(p.simulation_steps)
    new_obj = cls(p, init_pos, final_pos, mode)
    new_obj.init_pos = init_pos
    new_obj.final_pos = final_pos
    even = jnp.linspace(0, p.simulation_steps, positions.shape[0])
    ttp = jnp.vstack([even, positions]).T
    new_obj._protocol_fwd = bcp.make_custom_trap_fxn(t, ttp, init_pos, final_pos)
    new_obj._protocol_rev = bcp.make_custom_trap_fxn_rev(t, ttp, init_pos, final_pos)
    new_obj.mode = mode
    new_obj.__check_mode()
    new_obj.can_grad = False
    
    return new_obj
  
  
  def __check_mode(self):
    if self.mode not in ("rev", "fwd"):
      raise TypeError(f"Expected \'fwd\' or \'rev\' mode protocols, got {self.mode}")
  
  def plot_protocol(self, checkpoints = None):
    t = jnp.arange(self._params.simulation_steps)
    schedules = []
    if checkpoints is not None:
      for checkpoint in checkpoints:
        coeffs = bcp.self.coef_hist[checkpoint]
        if self.mode == "fwd":
          c_protocol = bcp.make_trap_fxn(t, coeffs, self.init_pos, self.final_pos)
        elif self.mode == "rev":
          c_protocol = bcp.make_trap_fxn_rev(t, coeffs, self.init_pos, self.final_pos)
        else:
          self.__check_mode()
        
        schedules.append(c_protocol(t))
    else:
      schedules = [self(t)]
    
    return t, schedules
  
  @property
  def coeffs(self):
    if self.can_grad:
      return self._coeffs
    else:
      return None
  
  @coeffs.setter
  def coeffs(self, new):
    self._coeffs = new
    self.coef_hist.append(new)
  
  def pop_hist(self, reset = True):
    history = copy.deepcopy(self.coef_hist)
    if reset: 
      self.coef_hist = []
      self.coeffs = bcp.linear_chebyshev_coefficients(self.init_pos, 
                                                      self.final_pos, 
                                                      self.params.simulation_steps, 
                                                      y_intercept = self.init_pos)
    return history
  
  @property
  def params(self):
    return self._params
  
  @params.setter
  def params(self, new: MDParameters):
    if not isinstance(new, MDParameters):
      raise TypeError(f"Expected class of type or inherited from type MDParamters, got {type(new)}")
    self._params = new
  
  def protocol(self, coeffs, train = False):
    """Returns Callable: timestep -> trap_position"""
    if not self.can_grad:
      return [self._protocol_fwd] if self.mode == "fwd" else [self._protocol_rev]
    
    t = jnp.arange(self._params.simulation_steps)
    if self.mode == "fwd":
      trap_fn = bcp.make_trap_fxn(t, coeffs, self.init_pos, self.final_pos)
    elif self.mode == "rev":
      trap_fn =  bcp.make_trap_fxn_rev(t, coeffs, self.init_pos, self.final_pos)
    
    if train:
      return [trap_fn]
    else:
      return trap_fn
  
  def __call__(self, timestep):
    if self.can_grad:
      return self.protocol(self.coeffs)(timestep)
    else:
      return self._protocol_fwd(timestep) if self.mode == "fwd" else self._protocol_rev(timestep)
  
  
class JointModel(ScheduleModel):
  """
  Model for optimizing several models jointly. Gradient descent steps happen simultaneously.
  All derived methods return parent output placed in list.
  """
  def __init__(self, p: MDParameters, *models):
    self._params = p
    self.models = models
    self._coeffs = [model.coeffs for model in self.models]
    self.coef_hist = [self._coeffs]
    
    self.can_grad = True
  
  @property
  def mode(self):
    return self.models[0].mode
  
  @mode.setter
  def mode(self, new):
    for model in self.models:
      model.mode = new
      model.__check_mode()
    self.mode = new
      
  @property
  def coeffs(self):
    if self.can_grad:
      return jnp.concatenate(self._coeffs)
    else:
      return None
  
  @coeffs.setter
  def coeffs(self, new):
    len_array = list(map(len, self._coeffs))
    if new.shape[0] != sum(len_array):
      raise ValueError(f"Incorrect shapes, supposed to be ({sum(len_array)},), got {new.shape}")
    else:
      split = 0
      for i, model in enumerate(self.models):
        c = new[split: split+len_array[i]]
        split += len_array[i]
        model.coeffs = c
        self._coeffs[i] = c
        
    self.coef_hist.append(self._coeffs)
  
  @property
  def params(self):
    return self._params
  
  @params.setter
  def params(self, new: MDParameters):
    if not isinstance(new, MDParameters):
      raise TypeError(f"Expected class of type or inherited from type MDParamters, got {type(new)}")
    for model in self.models:
      model.params = new
    self._params = new
  
  def protocol(self, coeffs, train = False):
    """Returns tuple containing protocols of interest"""
    if train is True:
      assert self.can_grad
    
    len_array = list(map(len, self._coeffs))
    protocol_arr = []
    if coeffs.shape[0] != sum(len_array):
      raise ValueError(f"Incorrect shapes, supposed to be ({sum(len_array)},), got {coeffs.shape}")
    else:
      split = 0
      for i, model in enumerate(self.models):
        c = coeffs[split: split+len_array[i]]
        split += len_array[i]
        protocol_arr.append(model.protocol(c))
        
    return protocol_arr

  def pop_hist(self, reset = True):
    hist = []
    self.coef_hist = []
    for model in self.models:
      hist.append(model.pop_hist(reset = reset))
    
    return hist
  
  def switch_trap(self, reset = True):
    hist = []
    if reset:
      self.coef_hist = []
    for model in self.models:
      if isinstance(model, SplitModel):
        hist.append(model.switch_trap())
    
    return hist
  
  def plot_protocols(self, checkpoints = None):
    raise NotImplementedError("Use plot_protocol on individual models instead.")
  
  def __call__(self, timestep):
    return [model(timestep) for model in self.models]
  
class SplitModel(ScheduleModel):
  """
  WIP Model for optimizing a portion of the protocol while keeping a different part constant.

  TODO: Allow functionality for more than two portions.
  """
  def __init__(self, p: MDParameters, init_pos, cut_pos, final_pos, total_sim_steps, coeffs = None, mode = "fwd", num = 1):
    if num == 1:
      super().__init__(p, init_pos, cut_pos, coeffs, mode)
      self.lin_bound = final_pos
    else:
      super().__init__(p, cut_pos, final_pos, coeffs, mode)
      self.lin_bound = init_pos
    
    self.cut_pos = cut_pos
    self.num = num
    self.total_sim_steps = total_sim_steps
    self.rem_steps = total_sim_steps - p.simulation_steps
    t = jnp.arange(self.rem_steps)
    
    
    if self.mode == "rev":
      make_lin = bcp.make_trap_fxn_rev
    elif self.mode == "fwd":
      # print("WARNING: For plotting uses only.")
      make_lin = bcp.make_trap_fxn 
    if num == 1:
      lin_c = bcp.linear_chebyshev_coefficients(self.cut_pos, self.lin_bound, self.rem_steps, y_intercept = self.cut_pos)
      self.lin_trap = make_lin(t, lin_c, self.cut_pos, self.lin_bound)
    else:
      lin_c = bcp.linear_chebyshev_coefficients(self.lin_bound, self.cut_pos, self.rem_steps, y_intercept = self.lin_bound)
      self.lin_trap = make_lin(t, lin_c, self.lin_bound, self.cut_pos)
    
  def switch_trap(self):
    """Switch which trap will be optimized. History of optimization will be returned from self.pop_hist"""
    new_num = 3-self.num
    self.num = new_num
    
    history = self.pop_hist()
    if self.mode == "rev":
      make_lin = bcp.make_trap_fxn_rev
    elif self.mode == "fwd":
      # print("WARNING: For plotting uses only.")
      make_lin = bcp.make_trap_fxn 
    
    self._params.end_time = self._params.dt * self.total_sim_steps - self._params.end_time
    self.rem_steps = self.total_sim_steps - self._params.simulation_steps
    
    t = jnp.arange(self.rem_steps)
    if new_num == 1:
      _tmp = self.lin_bound
      self.lin_bound = self.final_pos
      self.final_pos = self.init_pos
      self.init_pos = _tmp
      
      lin_c = bcp.linear_chebyshev_coefficients(self.cut_pos, self.lin_bound, self.rem_steps, y_intercept = self.cut_pos)
      new_c = bcp.linear_chebyshev_coefficients(self.init_pos, self.final_pos, self.rem_steps, y_intercept = self.init_pos)
      self.lin_trap = make_lin(t, lin_c, self.cut_pos, self.lin_bound)
      
    else:
      _tmp = self.final_pos
      self.final_pos = self.lin_bound
      self.lin_bound = self.init_pos
      self.init_pos = _tmp
      lin_c = bcp.linear_chebyshev_coefficients(self.lin_bound, self.cut_pos, self.rem_steps, y_intercept = self.lin_bound)
      new_c = bcp.linear_chebyshev_coefficients(self.init_pos, self.final_pos, self.rem_steps, y_intercept = self.init_pos)
      self.lin_trap = make_lin(t, lin_c, self.lin_bound, self.cut_pos)
    
    
    self.coeffs = new_c
    self.coef_hist.pop(0)
    return history
  
  def protocol(self, coeffs):
    if self.mode == "rev":
      trap_sum = bcp.trap_sum_rev
    elif self.mode == "fwd":
      # print("WARNING: For plotting uses only.")
      trap_sum = bcp.trap_sum
    if self.num == 1:
      total_trap = trap_sum(self.total_sim_steps, self._params.simulation_steps, super().protocol(coeffs), self.lin_trap)
    else:
      total_trap = trap_sum(self.total_sim_steps,self.total_sim_steps - self._params.simulation_steps, self.lin_trap, super().protocol(coeffs))
      
    return [total_trap]
  
  def single_protocol(self, coeffs):
    return super().protocol(coeffs)
  
  def __getstate__(self):
    return_dict = copy.deepcopy(self.__dict__)
    if return_dict["lin_trap"]:
      print("WARNING: lin_trap function cannot be pickled. Continuing...")
      return_dict["lin_trap"] = None
    return return_dict
  
  def __setstate__(self, new_dict):
    d = copy.deepcopy(new_dict)
    if d["lin_trap"] == None:
      t = jnp.arange(d["_params"].simulation_steps)
    
      if d["mode"] == "rev":
        make_lin = bcp.make_trap_fxn_rev
      elif d["mode"] == "fwd":
        make_lin = bcp.make_trap_fxn
      if d["num"] == 1:
        lin_c = bcp.linear_chebyshev_coefficients(d["init_pos"], d["cut_pos"], d["_params"].simulation_steps, y_intercept = d["init_pos"])
        d["lin_trap"] = make_lin(t, lin_c, d["init_pos"], d["cut_pos"])
      else:
        lin_c = bcp.linear_chebyshev_coefficients(d["cut_pos"], d["final_pos"], d["_params"].simulation_steps, y_intercept = d["init_pos"])
        d["lin_trap"] = make_lin(t, lin_c, d["cut_pos"], d["final_pos"])
      print("WARNING: lin_trap function inferred as linear trap. Manually change if this was unintended.")