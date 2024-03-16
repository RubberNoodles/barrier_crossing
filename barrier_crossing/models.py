import jax.numpy as jnp

import barrier_crossing.protocol as bcp
from barrier_crossing.utils import MDParameters

class ScheduleModel:
  def __init__(self, p: MDParameters, init_pos, final_pos, coeffs = None, mode = "fwd"):
    t = jnp.arange(p.simulation_steps)
    
    if coeffs is None:
      # Initalize at linear 
      coeffs = bcp.linear_chebyshev_coefficients(init_pos, final_pos, p.simulation_steps, y_intercept = init_pos)
    
    self._params = p
    
    self._coeffs = coeffs
    self.init_pos = init_pos
    self.final_pos = final_pos
    self.protocol_fwd = bcp.make_trap_fxn(t, coeffs, init_pos, final_pos)
    self.protocol_rev = bcp.make_trap_fxn_rev(t, coeffs, init_pos, final_pos)  
    self.can_grad = True  
    self.mode = mode
    self.__check_mode()
    
  
  @classmethod
  def from_positions(cls, p: MDParameters,  init_pos, final_pos, positions, mode = "fwd"):
    """Alternative constructor for creating protocols interpolating between `positions`"""
    t = jnp.arange(p.simulation_steps)
    new_obj = cls(p, init_pos, final_pos, mode)
    new_obj.init_pos = init_pos
    new_obj.final_pos = final_pos
    new_obj.protocol_fwd = bcp.make_custom_trap_fxn(t, positions, init_pos, final_pos)
    new_obj.protocol_fwd = bcp.make_custom_trap_fxn_rev(t, positions, init_pos, final_pos)
    
    new_obj.mode = mode
    new_obj.__check_mode()
    new_obj.can_grad = False
    
    return new_obj
  
  
  def __check_mode(self):
    if self.mode not in ("rev", "fwd"):
      raise TypeError(f"Expected \'fwd\' or \'rev\' mode protocols, got {self.mode}")
  
  @property
  def coeffs(self):
    if self.can_grad:
      return self._coeffs
    else:
      return None
  
  @coeffs.setter
  def coeffs(self, new):
    t = jnp.arange(self._params.simulation_steps)
    self.protocol_fwd = bcp.make_trap_fxn(t, new, self.init_pos, self.final_pos)
    self.protocol_rev = bcp.make_trap_fxn_rev(t, new, self.init_pos, self.final_pos)  
    self._coeffs = new
  
  @property
  def params(self):
    return self._params
  
  @params.setter
  def params(self, new: MDParameters):
    if not isinstance(new, MDParameters):
      raise TypeError(f"Expected class of type or inherited from type MDParamters, got {type(new)}")
    self._params = new
    
  @property
  def protocol(self):
    """Returns Callable: timestep -> trap_position"""
    if self.mode == "fwd":
      trap_fn = self.protocol_fwd
    elif self.mode == "rev":
      trap_fn = self.protocol_rev
    return trap_fn
  
  def __call__(self, timestep):
    if self.mode == "fwd":
      trap_fn = self.protocol_fwd
    elif self.mode == "rev":
      trap_fn = self.protocol_rev
    return trap_fn(timestep)
  
  
class JointModel(ScheduleModel):
  def __init__(self, p: MDParameters, *models):
    self._params = p
    self.models = models
    self._coeffs = [model.coeffs for model in self.models]
    
    
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
      return self._coeffs
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
        print(c)
        split += len_array[i]
        model.coeffs = c
        self._coeffs[i] = c
        
    self._coeffs = new
  
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
  
  @property
  def protocol(self):
    """Returns tuple containing protocols of interest"""
    return [model.protocol for model in self.models]
  
  def __call__(self, timestep):
    return [model(timestep) for model in self.models]