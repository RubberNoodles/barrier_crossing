# Barrier Crossing
_By: Megan Engel, Oliver Cheng, Zosia Adamska_

## Reconstructing Free Energy Landscapes using Jarzynski Identity

Made with Python 3.9.10

### Instructions

Clone the repository: 
``` 
# using https
git clone https://github.com/RubberNoodles/barrier_crossing.git 

# using SSH (for FASRC cluster/secure file transfer uses)
git clone git@github.com:RubberNoodles/barrier_crossing.git

cd barrier_crossing
```

Download all the general Python dependencies. Building with virtual environment highly recommended: `python3 -m venv bc_env;source bc_env/bin/activate`:
```
pip3 install --upgrade pip
pip3 install -r requirements.txt
```
Install the [JAX library with GPU](https://github.com/google/jax#installation), which may require you to install CUDA + CUDnn if you have a working NVIDIA GPU.

**NOTE:** For users without a GPU, install JAX with CPU instead.

main.py: NOT implemented.
energy.py: brownian state, energy functions
protocol.py: chebyshev coefficient stuff protocol generation with harmonic trap
simulate.py: forward or backward simulations
optimize.py: given landscape, find optimial protocol
