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

Download all the general Python dependencies. Building with virtual environment highly recommended: `python3 -m venv bc_env;source bc_env/bin/activate`:
```
pip3 install --upgrade pip
pip3 install -r requirements.txt
```
Install the [JAX library with GPU](https://github.com/google/jax#installation), which may require you to install CUDA + CUDnn if you have a working NVIDIA GPU.

**NOTE:** For users without a GPU, install JAX with CPU instead.

## Package Overview

This sections describes the individual files that are a part of this package contained in the `barrier_crossing` folder.

### `energy.py`
Implements the potential energy functions for molecular dynamics simulations. Currently contains `brownian(...)` to describe Brownian motion, as well as
code for two double well landscapes (Geiger & Dellago 2010) and (Sivak & Crooks 2016).

### `protocol.py`
Functions to create a protocol/schedule described by Chebyshev polynomials. 

### `simulate.py`
Contains `simulate_brownian_harmonic` and `batch_simulate_harmonic` functions in order to simulate a Brownian particle moving over a given free energy landscape dragged by a harmonic trap with schedule specified by Chebyshev polynomials. Use `batch_simulate_harmonic` in order to run a simulating functing such as `simulate_brownian_harmonic` in large batches to replicate running large batches of experiments to find different trajectories.

### `optimize.py`
Several different loss functions that one could choose from. The main ones coming from 

Training loop using Jarzynski equality error for a single energy difference or multiple energy differences across the landscape (Geiger & Dellago 2010, Engel 2022, Jarzynski 1997), or work used to drag the particle over the landscape as a loss function.

### `iterate_landscape.py`
Iteratively attempt to reconstruct a black box landscape by looping over two tasks: based on a protocol for pulling the molecule, take a large number of simulations to reconstruct an energy landscape. Using this energy landscape, optimize a protocol to use to once again simulate and reconstruct.

## Testing

There is a variety of tests that can be found in `tests`.