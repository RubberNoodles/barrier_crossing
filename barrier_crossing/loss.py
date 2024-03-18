import functools
import matplotlib.pyplot as plt

import jax

import jax.numpy as jnp
from barrier_crossing.protocol import make_trap_fxn_rev, trap_sum_rev
import barrier_crossing.models as bcm

def single_estimate_work(simulate_fn, model: bcm.ScheduleModel):
  assert model.mode == "fwd", f"Reverse gradients must have `fwd` mode, found {model.mode}"
  
  @functools.partial(jax.value_and_grad, has_aux=True) #the 'aux' is the summary
  def _single_estimate(coeffs, seed): 
    
    # Forward direction to compute the gradient of the work used.
    positions, log_probs, works = simulate_fn(*model.protocol(coeffs), seed)
    total_work = works.sum()
    tot_log_prob = log_probs.sum()
    summary = (positions, tot_log_prob, total_work)
    
    gradient_estimator = (tot_log_prob * jax.lax.stop_gradient(total_work) + total_work) 
    
    return gradient_estimator, summary
  return _single_estimate 

def estimate_gradient_work(batch_size,
                          simulate_fn, model: bcm.ScheduleModel):
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
  
  mapped_estimate = jax.vmap(single_estimate_work(simulate_fn, model), [None, 0])

  @jax.jit
  def _estimate_gradient(coeffs, seed):
    seeds = jax.random.split(seed, batch_size)
    (gradient_estimator, summary), grad = mapped_estimate(coeffs, seeds)
    return jnp.mean(grad, axis=0), (gradient_estimator, summary)
  return _estimate_gradient


def single_estimate_rev(simulate_fn, model: bcm.ScheduleModel):
  assert model.mode == "rev", f"Reverse gradients must have `rev` mode, found {model.mode}"
  beta = model.params.beta
  
  @functools.partial(jax.value_and_grad, has_aux = True)
  def _single_estimate(coeffs, seed):
    positions, log_probs, works = simulate_fn(*model.protocol(coeffs), seed)
    
    total_work = works.sum()
    tot_log_prob = log_probs.sum()
    summary = (positions, tot_log_prob, jnp.exp(beta*total_work))

    gradient_estimator = (tot_log_prob) * jax.lax.stop_gradient(jnp.exp(beta * total_work)) + jax.lax.stop_gradient(beta * jnp.exp(beta*total_work)) * total_work
    return gradient_estimator, summary
  return _single_estimate 

def estimate_gradient_rev(batch_size,
                          simulate_fn,
                          model: bcm.ScheduleModel):
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
                          model), [None, 0])  
  @jax.jit 
  def _estimate_gradient(coeffs, seed):
    seeds = jax.random.split(seed, batch_size)
    (gradient_estimator, summary), grad = mapped_estimate(coeffs, seeds)
    return jnp.mean(grad, axis=0), (gradient_estimator, summary)
  return _estimate_gradient


# def single_estimate_rev(simulate_fn,
#                         r0_init, r0_final, simulation_steps, 
#                         beta):
#   @functools.partial(jax.value_and_grad, has_aux = True)
#   def _single_estimate(coeffs, seed):
#     trap_fn = make_trap_fxn_rev(jnp.arange(simulation_steps), coeffs, r0_init, r0_final)
#     positions, log_probs, works = simulate_fn(trap_fn, seed)
#     total_work = works.sum()
#     tot_log_prob = log_probs.sum()
#     summary = (positions, tot_log_prob, jnp.exp(beta*total_work))

#     gradient_estimator = (tot_log_prob) * jax.lax.stop_gradient(jnp.exp(beta * total_work)) + jax.lax.stop_gradient(beta * jnp.exp(beta*total_work)) * total_work
#     return gradient_estimator, summary
#   return _single_estimate 

# def estimate_gradient_rev(batch_size,
#                           simulate_fn,
#                           r0_init, r0_final, simulation_steps, 
#                           beta):
#   """
#   Find e^(beta ΔW) which is proportional to the error <(ΔF_(batch_size) - ΔF)> = variance of ΔF_(batch_size) (2010 Geiger and Dellago). 
#   Compute the total gradient with forward trajectories and loss based on work used.
#     Args:
#       batch_size: Number of trajectories to simulate simultaneously.
#       simulate_fn: Callable (trap_fn, keys) -> positions, log_probs, works.
#         Simulator function governing particle dynamics.
#       r0_init: float describing where trap begins
#       r0_final float describing where trap ends
#       simulation_steps: number of steps to run simulation for.
#       beta: float describing inverse of temperature in kT^-1
#     Returns:
#       Callable with inputs ``coeffs`` as the Chebyshev coefficients that specify the protocol, and
#       ``seed`` to set rng."""
#   mapped_estimate = jax.vmap(single_estimate_rev(simulate_fn,
#                           r0_init, r0_final, simulation_steps, 
#                           beta), [None, 0])  
#   @jax.jit 
#   def _estimate_gradient(coeffs, seed):
#     seeds = jax.random.split(seed, batch_size)
#     (gradient_estimator, summary), grad = mapped_estimate(coeffs, seeds)
#     return jnp.mean(grad, axis=0), (gradient_estimator, summary)
#   return _estimate_gradient


def single_estimate_rev_split(simulate_fn_no_trap, r0_init, r0_final, r_cut,step_cut,
                        simulation_steps, beta):
  """
  New gradient for the optimization that splits the original 
  trap functions into two parts. 
  """
  @functools.partial(jax.value_and_grad, has_aux = True)
  def _single_estimate(coeffs_for_opt, coeffs_leave,seed):
    trap_leave = make_trap_fxn_rev(jnp.arange(step_cut), coeffs_leave, r0_init, r_cut)
    trap_opt = make_trap_fxn_rev(jnp.arange(simulation_steps - step_cut), coeffs_for_opt, r_cut, r0_final)
    trap_fn = trap_sum_rev(jnp.arange(simulation_steps),simulation_steps, step_cut,trap_leave, trap_opt)
    positions, log_probs, works = simulate_fn_no_trap(trap_fn, seed)
    total_work = works.sum()
    tot_log_prob = log_probs.sum()
    summary = (positions, tot_log_prob, jnp.exp(beta*total_work))

    gradient_estimator = (tot_log_prob) * jax.lax.stop_gradient(jnp.exp(beta * total_work)) + jax.lax.stop_gradient(beta * jnp.exp(beta*total_work)) * total_work
    return gradient_estimator, summary
  return _single_estimate

def estimate_gradient_rev_split(batch_size,
                          simulate_fn_no_trap,
                          r0_init, r0_final, r_cut, step_cut,
                          simulation_steps,beta):
  mapped_estimate = jax.vmap(single_estimate_rev_split(simulate_fn_no_trap, r0_init, r0_final, r_cut,step_cut,
                        simulation_steps, beta), [None,None, 0])
  @jax.jit
  def _estimate_gradient(coeffs_for_opt,coeffs_leave, seed):
    seeds = jax.random.split(seed, batch_size)
    (gradient_estimator, summary), grad = mapped_estimate(coeffs_for_opt,coeffs_leave, seeds)
    return jnp.mean(grad, axis=0), (gradient_estimator, summary)
  return _estimate_gradient

