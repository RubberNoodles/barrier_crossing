from barrier_crossing.utils import MDParameters
import barrier_crossing.simulate as bcs
import barrier_crossing.models as bcm

import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS

import numpy as np
import jax.numpy as jnp
import jax.random as random

from typing import Iterable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import seaborn as sns
import pandas as pd

def _plot_energy_fn(param_set):
  
  pure_energy_fn = param_set.energy_fn(no_trap = True)
  fwd_pos = param_set.init_position_fwd[0][0]
  rev_pos = param_set.init_position_rev[0][0]
  x_vec = jnp.linspace(fwd_pos - 10, rev_pos + 10, 1000)
  
  
  # Only plot values within 50% of barrier height
  barrier_height = -jnp.inf
  for e in jnp.linspace(fwd_pos, rev_pos, 1000):
    if barrier_height < pure_energy_fn(e):
      barrier_height = pure_energy_fn(e)
    
  barrier_height -= pure_energy_fn(fwd_pos)
  
  plotting_es = []
  plotting_xs = []
  for x in x_vec:
    energy = pure_energy_fn(x) - pure_energy_fn(fwd_pos)
    if energy < 1.5 * barrier_height:
      plotting_xs.append(x)
      plotting_es.append(energy)
  
  return plotting_xs, plotting_es


def add_energy_fn_plot(param_set: MDParameters, fig = plt, **kw):
  width = kw.get("width", 0.3)
  height = kw.get("height", 0.2)
  loc = kw.get("loc", "ul") # Not used at the moment
  
  
  if loc == "ul":
    inset = fig.add_axes([0.15, 1 - height - 0.15, width, height])
  elif loc == "ur":
    inset = fig.add_axes([1-width - 0.15, 1 - height - 0.15, width, height])
  else:
    raise NotImplementedError
      
  inset.plot(*_plot_energy_fn(param_set),  color = "k")
  inset.axis("off")
  
  return inset


def plot_protocols():
  return NotImplementedError
  
def plot_evolving(ax = plt, label = None, **kw):
  model = kw.pop("model", None)
  if model is not None:
    if isinstance(model, bcm.JointModel):
      ax.axis(False)
      
      pos_model, stiff_model = model.models
      
      pos_axis = inset_axes(ax, width="85%", height="40%", loc="upper center")
      plot_evolving(ax = pos_axis, label = label, model = pos_model, **kw)
      pos_axis.set_title("Position")
      
      stiff_axis = inset_axes(ax, width="85%", height="40%", loc="lower center")
      plot_evolving(ax = stiff_axis, label = label, model = stiff_model, **kw)
      stiff_axis.set_title("Stiffness")
      
      return
      
  data = kw.pop("data", None)
  coeffs = kw.pop("coeffs", None)
  N = kw.pop("num", 7)
  col = kw.pop("color", "k")
  
  if model is not None:
    assert data is None
    if coeffs is not None:
      schedules = []
      assert N == len(coeffs)
      for coeff in coeffs:
        x_vec, schedule = model.plot_protocol(coeffs = coeff)
        schedules += schedule
        
    else:
      T = len(model.coef_hist) - 1
      checkpoints = [(i * T)//N for i in range(N)]
      x_vec, schedules = model.plot_protocol(checkpoints = checkpoints)
    
    x_vec = (model.params.dt * x_vec) * 1e6
    
    evolving_data = jnp.vstack(schedules)
    
  elif data is not None:
    T = data.shape[0] - 1
    evolving_data = jnp.vstack([data[(i*T)//N] for i in range(N)])
    x_vec = kw.pop("xs", evolving_data.shape[1])
  else:
    raise ValueError("No plotting arguments given.")
  
  def _alpha_fn(i):
    if i >= N-1:
      return 1
    else:
      return 0.1 + (0.3 * (i/N))
  
  for i in range(N):
    if label and i == N-1:
        plot = ax.plot(x_vec, evolving_data[i], label = label, color = col, alpha = _alpha_fn(i), **kw)
    else:
      plot = ax.plot(x_vec, evolving_data[i], color = col, alpha = _alpha_fn(i), **kw)
      
  return plot

def plot_with_stdev(x, label=None, n=1, axis=0, ax=plt, **kw):
  xs = kw.get("xs", jnp.arange(x.shape[1-axis]))
  stddev = jnp.std(x, axis)
  mn = jnp.mean(x, axis)

  ax.fill_between(xs,
                  mn + n * stddev, mn - n * stddev, alpha=.3)
  ax.plot(xs, mn, label=label) 
  
  
def plot_landscapes(param_set, landscapes, labels, ax = plt, **kw):
  """Center all landscapes and plot """
  assert len(landscapes) == len(labels), "Number of landscapes and labels must be the same"
  
  
  N = len(landscapes)
  
  colors = kw.get("colors", None)
  if colors:
    assert len(colors) == N
  else:
    colors = list(TABLEAU_COLORS.values())[:N]
  
  iterated_landscape_index = kw.get("iterated_index", None)
  iterated_landscapes = None
  if iterated_landscape_index is not None:
    assert isinstance(landscapes[iterated_landscape_index], Iterable)
    iterated_landscapes = landscapes.pop(iterated_landscape_index)
    iterated_label = labels.pop(iterated_landscape_index)
    iterated_color = colors.pop(iterated_landscape_index)
  
  ax.plot(*_plot_energy_fn(param_set), label = "True Landscape", color = "k")
  # First plot true landscape
  
  for (midpoints, energies), label, color in zip(landscapes, labels, colors):
    midpoints = jnp.array(midpoints)
    first_well_index = jnp.argmin(jnp.abs(midpoints - param_set.init_position_fwd[0][0]))
    energies = jnp.array(energies) - energies[first_well_index] # Set first well = 0
    
    ax.plot(midpoints, energies, label = label, color = color, linestyle = "--")
  
  if iterated_landscapes is not None:
    energies = jnp.array([landscape[1] for landscape in iterated_landscapes])
    midpoints = iterated_landscapes[0][0]
    first_well_index = jnp.argmin(jnp.abs(midpoints - param_set.init_position_fwd[0][0]))
    energies -= energies[0, :][first_well_index]
    plot_evolving(ax = ax, label = iterated_label, color = iterated_color, xs = midpoints, data = energies, linestyle = "--")
    
  
  return 0
    
   
def add_axes(plot_type, ax = plt):
  if ax == plt:
    ax = ax.gca()
  
  if "schedule" in plot_type:
    plot_type, schedule_type  = plot_type.split("_")
    
  match plot_type:
    case "landscape":
      ax.set_xlabel("Position (nM)")
      ax.set_ylabel("Energy (kT)")
    case "schedule":
      ax.set_xlabel("Time (us)")
      if schedule_type == "position":
        ax.set_ylabel("Position (nM)")
      elif schedule_type == "stiffness":
        ax.set_ylabel("Trap Stiffness (pN/nM²)")
    case "hist":
      ax.set_xlabel("Dissipated Work (kT)")
  

def plot_hists(models, labels, ax = plt):
  assert len(models) == len(labels)
  colors = list(TABLEAU_COLORS.values())[:len(models)]
  
  for model, label, color in zip(models, labels, colors):
    batch_key = random.PRNGKey(1001)
    if isinstance(model, bcm.JointModel):
        simulate_fn = lambda key: model.params.simulate_fn(*model.models, key = key)
    else:
        if isinstance(model, bcm.ScheduleModel):
            is_stiffness = (model.init_pos == model.final_pos)
        else:
            raise NotImplementedError
        if is_stiffness:
            simulate_fn = lambda key: model.params.simulate_fn(trap_fn = False, ks_trap_fn = model, key = key)
        else:
            simulate_fn = lambda key: model.params.simulate_fn(trap_fn = model, ks_trap_fn = model.params.k_s, key = key)
      
    total_works, _ = bcs.batch_simulate_harmonic(10000, simulate_fn, batch_key)
    
    counts, bin_edges, _ = ax.hist(total_works, color = color, bins = 50, density = True, alpha = 0.1)

    outline_x = np.repeat(bin_edges, 2)[1:-1]  
    outline_y = np.repeat(counts, 2)  
    ax.plot(outline_x, outline_y, color=color, lw = 2, label = label)

  