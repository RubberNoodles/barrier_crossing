# Results and Figures


_These codes were made for `barrier_crossing == 0.16.2`._

This directory contains Python codes and FASRC `.sh` scripts that we used to produce each figure. 
Each directory is a separate test. They can be run through the cluster with the appropriate SLURM 
commands, or with native Python. The following is a description of each directory/file. All outputs 
can be found in the `/data` directory of each folder. `old` contains old outputs that are saved and 
may be helpful reference.

All tests will take the parameters in `params.py` in order to maintain consistency. Change these values at your own peril.

**Codes using Sivak & Crooks Landscape**

`optimize_work_err`: Optimize a protocol using Geiger and Dellago form for the Jarzynski error, and find that error optimized protocols narrow the work distribution. # TODO

`accumulated_loss`: Optimizes a protocol using a loss function that accumulates loss from an array of different protocol starting points. `extensions.csv` is a list of comma-separated extensions that `acc_batch_extensions.py` will find optimal protocols for. We then take a batch of forward simulations with this optimized protocol to reconstruct the landscape using Megan's method and to plot a work distribution histogram.

`comparing_error_fn`: Optimize protocol with loss functions that track Jarzynski error or dissipative work. Compare the work distributions of the optimized protocols.

`iterating_landscape`: Using the iterative landscape reconstruction procedure, iterate on the landscape.

`template_sc.py`: This contains all imports and initial parameters that we use for the Sivak & Crooks symmetric double-well landscape. It is recommended to use this file as the beginning of any new tests.


TODO: Create a file which contains standard parameters that can be modified as opposed to modifying test functions.

**Codes Using Geiger & Dellago Landscape**

`slow_reconstruction`: We find that a perfect reconstruction can be made by moving the simulation sufficiently slowly. # TODO




