import jax.numpy as jnp
import pickle
import matplotlib.pyplot as plt

# epsilon: "0.5", data: ""
data_path = "./geiger_output/"
file_arr = ["forward_coeffs", "forward_works", "reverse_coeffs", "reverse_works", "works_forward_opt", "works_linear", "works_reverse_opt"]

geiger_output_arr = [] # deprecated

error_ratio = []
work_ratio = []

for eps_2 in range(1,9):
  epsilon = eps_2/2
  print(f"Code is running for epsilon: {epsilon:1f}")
  final_path = data_path + f"epsilon_{epsilon:.1f}/"

  forward_work = None
  reverse_work = None
  linear_work = None

  for file_name in file_arr:
    f = open(final_path + file_name + ".pkl", "rb")

    pickle_value = pickle.load(f)
    if file_name == "works_forward_opt":
      forward_work = pickle_value
    elif file_name == "works_reverse_opt":
      reverse_work = pickle_value
    elif file_name == "works_linear":
      linear_work = pickle_value

    geiger_output_arr.append({"epsilon": epsilon, "data": pickle_value, "file_name":file_name}) # Unnecessary code.
    f.close() 
  
  forward_avg_work = jnp.mean(forward_work) 
  reverse_avg_work = jnp.mean(reverse_work) 
  linear_avg_work = jnp.mean(linear_work) 

  work_ratio.append(forward_avg_work/linear_avg_work)
  error_ratio.append(reverse_avg_work/linear_avg_work)

plt.plot(jnp.arange(0.5,4.5,0.5), jnp.array(work_ratio), label="Work Minimized Ratio")
plt.plot(jnp.arange(0.5,4.5,0.5), jnp.array(error_ratio), label="Error Minimized Ratio")
plt.legend()
plt.title("Work Advantage for Error vs Work Minimized Protocols")
plt.xlabel("Epsilon")
plt.ylabel("Work, Optimized Protocol/Work, Linear protocol")
plt.savefig("./geiger_output/final_fig11.png")
#plt.show()
