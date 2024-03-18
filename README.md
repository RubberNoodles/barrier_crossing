# Barrier Crossing
_By: Megan Engel, Oliver Cheng, Zofia Adamska_

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

Download all the general Python dependencies.
```
python3 -m venv bc_env
source bc_env/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt
```
Install the [JAX library with GPU](https://github.com/google/jax#installation), which may require you to install CUDA + CUDnn if you have a working NVIDIA GPU.

**NOTE:** For users without a GPU, install JAX with CPU instead.

## Package Overview

This sections describes the individual files that are a part of this package contained in the `barrier_crossing` folder.

## Figures

All figures found in the paper were produced using the codes available in the `figures` directory. See the corresponding `README.md` for more information.

### `energy.py`
Implements the potential energy functions for molecular dynamics simulations. Currently contains `brownian` and `langevin` corresponding to their respective
dynamics. In addition, classes for (Geiger & Dellago 2010) and (Sivak & Crooks 2016) landscapes are available.

### `protocol.py`
Functions to create a protocol/schedule described by Chebyshev polynomials. 

### `models.py`
Building off of `protocol.py`, creates models designed to be optimized under JAX transformations and AD. Provides functionality for single protocol, joint protocol, and split protocol optimizations.

### `simulate.py`
Using `brownian` and `langevin` simulators, creates simple functions for invoking large batches of simulations, analagous to running large batches of experiments to sample rare trajectories

### `loss.py`
Loss functions according to modern statistical mechanics theory. For example: dissipated work, reverse Jarzynski Loss, loss sampled across reaction coordinates. Uses JAX transformations (jit/vmap) for performance.

### `train.py`
Training loop to optimize models from `models.py`.

### `iterate_landscape.py`
Contains vectorized code to run reconstructions in O(`simulation_steps` * `bins`).

Also includes code to iteratively attempt to reconstruct a black box landscape by looping over two tasks: based on a protocol for pulling the molecule, take a large number of simulations to reconstruct an energy landscape. Using this energy landscape, optimize a protocol to use to once again simulate and reconstruct.

### `utils.py`

Utility functions for the rest of the package, including code to help parse IO for figures.
