import pickle
import jax.numpy as jnp

import matplotlib.pyplot as plt

import barrier_crossing.protocol as bc_protocol 

from figures.params import * # global variables;



if __name__ == "__main__":

  data_path = "output_data/"
  with open(data_path + "work_coeffs.pkl", "rb") as f:
    coeffs_work = pickle.load(f)
  with open(data_path + "error_coeffs.pkl", "rb") as f:
    coeffs_err = pickle.load(f)
  
  work_trap = bc_protocol.make_trap_fxn(jnp.arange(simulation_steps_sc), coeffs_work, r0_init_sc, r0_final_sc)
  err_trap = bc_protocol.make_trap_fxn(jnp.arange(simulation_steps_sc), coeffs_err, r0_init_sc, r0_final_sc)

  plt.figure(figsize=[8,8])

  time_vec = jnp.arange(simulation_steps_sc)

  plt.title("Optimized Protocols")
  plt.xlabel("Time (s)")
  plt.ylabel("Trap Position")
  plt.plot(time_vec * dt_sc, work_trap(time_vec), label = "Work Optimized")
  plt.plot(time_vec * dt_sc, err_trap(time_vec), label = "Error Optimized")

  plt.legend()
  plt.savefig("plots/opt_protocols.png")



  
