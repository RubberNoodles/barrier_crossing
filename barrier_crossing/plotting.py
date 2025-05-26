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
  elif loc == "uc":
    inset = fig.add_axes([0.6 - width - 0.15, 1 - height - 0.15, width, height])
  else:
    raise NotImplementedError
      
  inset.plot(*_plot_energy_fn(param_set),  color = "k")
  inset.axis("off")
  
  return inset


def plot_protocols():
  return NotImplementedError
  
def plot_evolving(ax = plt, label = None, **kw):
  """Plot how a protocol or other data evolves over iterations.
  
  This function visualizes how a protocol (position or stiffness schedule) changes over multiple iterations.
  It can show either the evolution of coefficients from a model's history or from provided coefficients.
  
  Parameters
  ----------
  ax : matplotlib.axes.Axes, optional
      The axis to plot on. If not provided, uses the current axis.
  label : str, optional
      Label for the legend. Only shown for the final iteration.
  **kw : dict
      Additional keyword arguments:
      - model : ScheduleModel or JointModel
          The model containing the protocol to plot
      - data : array-like
          Alternative to model: directly provide the data to plot
      - coeffs : list of arrays
          Alternative to model: provide coefficients to plot
      - num : int, default=7
          Number of iterations to show
      - color : str, default='k'
          Color for the lines
      
  Returns
  -------
  list
      List of matplotlib Line2D objects representing the plotted lines
      
  Examples
  --------
  Basic usage with a model:
  >>> fig, ax = plt.subplots()
  >>> plot_evolving(ax=ax, model=position_model, num=5)
  
  Plot with custom coefficients:
  >>> coeffs = [model.coef_hist[i] for i in range(5)]
  >>> plot_evolving(ax=ax, model=model, coeffs=coeffs, num=5)
  
  Plot with custom styling:
  >>> plot_evolving(ax=ax, model=model, num=7, color="blue", label="My Protocol")
  """
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
  """Plot energy landscapes including true and reconstructed landscapes.
  
  This function visualizes energy landscapes, showing both the true landscape and any
  reconstructed landscapes. It can optionally show how a landscape evolves over iterations.
  
  Parameters
  ----------
  param_set : MDParameters
      The parameter set containing the true landscape information
  landscapes : list of tuples
      List of (midpoints, energies) tuples for each landscape to plot
  labels : list of str
      Labels for each landscape in the legend
  ax : matplotlib.axes.Axes, optional
      The axis to plot on. If not provided, uses the current axis.
  **kw : dict
      Additional keyword arguments:
      - colors : list of str, optional
          Colors for each landscape. If not provided, uses tableau colors
      - iterated_index : int, optional
          Index of the landscape to show as evolving over iterations
      
  Returns
  -------
  int
      0 on successful completion
      
  Examples
  --------
  Basic usage with true and reconstructed landscapes:
  >>> fig, ax = plt.subplots()
  >>> landscapes = [(midpoints1, energies1), (midpoints2, energies2)]
  >>> labels = ["True Landscape", "Reconstructed Landscape"]
  >>> plot_landscapes(param_set, landscapes, labels, ax=ax)
  
  Plot with custom colors:
  >>> colors = ["red", "blue"]
  >>> plot_landscapes(param_set, landscapes, labels, ax=ax, colors=colors)
  
  Plot with an evolving landscape:
  >>> # First landscape will be shown evolving
  >>> plot_landscapes(param_set, landscapes, labels, ax=ax, iterated_index=0)
  """
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
    
    # Starting value
    first_well_e = energies[first_well_index] 
    for pos, e in zip(midpoints, energies):
        if jnp.abs(pos - midpoints[first_well_index]) > 3:
          continue
        if first_well_e > e:
          first_well_e = e
      
    energies = jnp.array(energies) - first_well_e
    
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
    
    # print(label, " ", jnp.mean(jnp.exp(-model.params.beta * total_works)))


def print_jarzynski_free_energy_difference(models, labels, num_samples, batch_key = random.PRNGKey(1001)):
  dgs = []
  print("Jarzynski Free Energy Differences")
  for model, label in zip(models, labels):
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
      
    total_works, _ = bcs.batch_simulate_harmonic(num_samples, simulate_fn, batch_key)
    b = model.params.beta
    dg = -(1/b) * jnp.log(jnp.mean(jnp.exp(-b * total_works)))
    print(f"{label}: {dg}")
    dgs.append(dg)

  return dgs