# Results and Figures


_These codes were made for `barrier_crossing == 0.16.2`._

This directory contains Python codes and FASRC `.sh` scripts that we used to produce each figure. 
Each directory is a separate test. They can be run through the cluster with the appropriate SLURM 
commands, or with native Python. The following is a description of each directory/file. All `.pkl` or 
other outputs can be found in the `/output_data` directory of each folder. `old` contains old outputs that are saved and 
may be helpful reference.

All tests will take the parameters in `params.py` in order to maintain consistency. Change these values at your own peril.

Before running SLURM batch scripts, one should must install the following dependencies:
```
pip3 install -e "../"
pip3 install -r "../requirements.txt"
pip3 install --upgrade "jax[cuda12_pip]==0.4.16" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
`work_error_opt`: Optimize a protocol using dissipative work vs. Jarzynski error on Sivak & Crooks double well landscape.

`reconstructions`: Using protocols optimized through various regimes, reconstruct the landscape with forward simulations according to Engel 2020 and plot the reconstructions.

`accumulated_grad`: Optimizes a protocol using a loss function that accumulates loss from an array of different protocol starting points. `extensions.csv` is a list of comma-separated extensions that `acc_batch_extensions.py` will find optimal protocols for. We then take a batch of forward simulations with this optimized protocol to reconstruct the landscape using Megan's method and to plot a work distribution histogram.

`iterative`: Using the iterative landscape reconstruction procedure, iterate on the landscape. The procedure consists of forward optimizations on a true landscape to obtain dissipitave work values at each position. Using this data and Engel 2020's reconstrution procedure, we can reconstruct a landscape that we can sample from and perform gradient descent to learn an optimal protocol where the landscape is one of the elements of our computational graph.




