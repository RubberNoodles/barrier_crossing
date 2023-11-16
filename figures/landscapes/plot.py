import matplotlib.pyplot as plt
import importlib

import jax.numpy as jnp

if __name__ == "__main__":
  landscapes = {"Double Well 10kT Barrier":"10kt",
              "Asymetric Double Well":"asym",
              "Triple Well":"triple_well",
              "Geiger Dellago":"geiger",
                        }
  
  fig, ax = plt.subplots(2,2, figsize = [8,8])
  
  num_samples = 100
  
  for i, (name, ext) in enumerate(landscapes.items()):
    p = importlib.import_module(f"figures.param_set.params_{ext}")

    length = p.r0_final - p.r0_init
    if ext == "triple_well":
      interval = (p.r0_init - length/16, p.r0_final + length/16)
    else:
      interval = (p.r0_init - length/2, p.r0_final + length/2)
    
    e_fn = p.param_set.energy_fn(0.)
    
    x = jnp.linspace(*interval, num_samples)
    e_val = []
    for val in x.reshape(num_samples,1,1):
      e_val.append(e_fn(val, 0))

    ax[i//2, i%2].plot(x, e_val, "-o")
    ax[i//2, i%2].set_title(name)
    # ax[i//2, i%2].set_xlabel("Position (x)")

  

  fig.legend()
  plt.savefig("landscapes.png", transparent = False)