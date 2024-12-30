import functools
import matplotlib.pyplot as plt

import jax

import jax.numpy as jnp
from barrier_crossing.protocol import make_trap_fxn_rev, trap_sum_rev
import barrier_crossing.models as bcm

def single_estimate_work(simulate_fn, model: bcm.ScheduleModel, reg):
  assert model.mode == "fwd", f"Forward gradients must have `fwd` mode, found {model.mode}"
  
  @functools.partial(jax.value_and_grad, has_aux=True) #the 'aux' is the summary
  def _single_estimate(coeffs, seed): 
    
    # Forward direction to compute the gradient of the work used.
    positions, log_probs, works = simulate_fn(*model.protocol(coeffs, train = True), seed)
    total_work = works.sum()
    tot_log_prob = log_probs.sum()
    
    diff = jnp.mean(jnp.square(coeffs))
    
    summary = (positions, tot_log_prob, total_work + reg * diff)
    
    
    gradient_estimator = (tot_log_prob * jax.lax.stop_gradient(total_work) + total_work) + reg * diff
    
    return gradient_estimator, summary
  return _single_estimate 

def estimate_gradient_work(batch_size,
                          simulate_fn, model: bcm.ScheduleModel, reg = 0.1):
  """Compute the total gradient with forward trajectories and loss based on work used.
  
    Args:
      batch_size: Number of trajectories to simulate simultaneously.
      simulate_fn: Callable (trap_fn, keys) -> positions, log_probs, works.
        Simulator function governing particle dynamics.
      r0_init: float describing where trap begins
      r0_final float describing where trap ends
      simulation_steps: number of steps to run simulation for.
    Returns:
      Callable with inputs ``coeffs`` as the Chebyshev coefficients that specify the protocol, and
      ``seed`` to set rng."""
  
  mapped_estimate = jax.vmap(single_estimate_work(simulate_fn, model, reg = 0.1), [None, 0])

  @jax.jit
  def _estimate_gradient(coeffs, seed):
    seeds = jax.random.split(seed, batch_size)
    (gradient_estimator, summary), grad = mapped_estimate(coeffs, seeds)
    return jnp.mean(grad, axis=0), (gradient_estimator, summary)
  return _estimate_gradient


def single_estimate_rev(simulate_fn, model: bcm.ScheduleModel, reg):
  assert model.mode == "rev", f"Reverse gradients must have `rev` mode, found {model.mode}"
  beta = model.params.beta
  
  @functools.partial(jax.value_and_grad, has_aux = True)
  def _single_estimate(coeffs, seed):
    positions, log_probs, works = simulate_fn(*model.protocol(coeffs, train = True), seed)
    
    total_work = works.sum()
    tot_log_prob = log_probs.sum()
    
    diff = jnp.mean(jnp.square(coeffs)) # L2 norm regularization
    
    summary = (positions, tot_log_prob, jnp.exp(beta*total_work) + reg * diff)
    gradient_estimator = reg * diff + (tot_log_prob) * jax.lax.stop_gradient(jnp.exp(beta * total_work)) + jax.lax.stop_gradient(beta * jnp.exp(beta*total_work)) * total_work
    return gradient_estimator, summary
  return _single_estimate 

def estimate_gradient_rev(batch_size,
                          simulate_fn,
                          model: bcm.ScheduleModel,
                          reg = 0.1):
  """
  Find e^(beta ΔW) which is proportional to the error <(ΔF_(batch_size) - ΔF)> = variance of ΔF_(batch_size) (2010 Geiger and Dellago). 
  Compute the total gradient with forward trajectories and loss based on work used.
    Args:
      batch_size: Number of trajectories to simulate simultaneously.
      simulate_fn: Callable (schedule_fn, keys) -> positions, log_probs, works.
        Simulator function governing particle dynamics.
      model: ScheduleModel class containing protocol/schedule information
    Returns:
      Callable with inputs ``coeffs`` as the Chebyshev coefficients that specify the protocol, and
      ``seed`` to set rng."""
  mapped_estimate = jax.vmap(single_estimate_rev(simulate_fn,
                          model, reg), [None, 0])  
  @jax.jit 
  def _estimate_gradient(coeffs, seed):
    seeds = jax.random.split(seed, batch_size)
    (gradient_estimator, summary), grad = mapped_estimate(coeffs, seeds)
    return jnp.mean(grad, axis=0), (gradient_estimator, summary)
  return _estimate_gradient