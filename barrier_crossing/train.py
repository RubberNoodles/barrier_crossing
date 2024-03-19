"""Training loop for optimizing JAX models. TODO: Generalize for multi-dimension simulations."""

import tqdm
import jax
import jax.numpy as jnp
import barrier_crossing.models as bcm

def train(model: bcm.ScheduleModel, optimizer, batch_grad_fn, key, batch_size = 3000, num_epochs = 500): 
  
  state = optimizer.init_fn(model.coeffs)
  losses = []
  grad_fn = batch_grad_fn(batch_size)
  
  for j in tqdm.trange(num_epochs, desc = "Optimize Protocol: "):
    key, split = jax.random.split(key)
    grad, (_, summary) = grad_fn(optimizer.params_fn(state), split)
    
    state = optimizer.update_fn(j, grad, state)
    coeffs = optimizer.params_fn(state)
    
    model.coeffs = coeffs
    
    loss = summary[2]
    losses.append(loss)
  
  return jax.tree_util.tree_map(lambda *args: jnp.stack(args), *losses)


optimze_protocol = train # For backwards compatibility