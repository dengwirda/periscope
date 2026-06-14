
Various benchmark configurations for `PERISCOPE`.

Runs are controlled by the following environment variables:
````
export BINDIR=path/to/periscope
export MSHDIR=path/to/mesh+data
export NUMCPU=num-cpu-threads
````
which should be setup based on `PERISCOPE`'s installation on a given machine.

To run each benchmark:
````
./<benchmark-dir>/ics_xyz.sh  # build initial conditions
./<benchmark-dir>/run_xyz.sh  # launch run
````
Output is saved to an `out_xyz.nc` file, which can be opened for visualisation 
using e.g. paraview.

