## `PERISCOPE`

`PERISCOPE` is a numerical toolbox for the simulation of geophysical flows using 
variable resolution unstructured meshes. 

<p align="center"> <img src = "../main/img/merger_b.jpg"> </p>

Presently, a nonlinear rotating shallow water model is available &mdash; solved on 
spherical domains (either the full sphere or regional pieces thereof), with 
support for various boundary conditions, drag laws and forcing types. A staggered 
mesh mimetic finite volume discretisation is employed that maintains the energy 
(and to a lesser extent enstrophy) balances associated with geophysical flows.

`PERISCOPE` is implemented using a mix of `Python` and `Cython` and must first be 
compiled:

    python3 setup.py build_ext --inplace
    
While not currently intended for massively parallel HPC workflows, `PERISCOPE` is
written to exploit thread-based parallelism via `OpenMP`, suitable for single-node
systems.

To run the shallow-water solver (see `swe.py --help`):

    python3 swe.py \
    --mesh-file="path+name-to-mpas-mesh+init-file" \
    --numthread=cores \
    --num-steps=number-of-time-steps \
    --time-step=delta_t \
    --save-freq=output-freq-th-steps

Solver output is saved to an MPAS-'ish' NetCDF file that can be visualised via
e.g. paraview.

Input files for various cases can be built from MPAS-format mesh files using the 
utilities provided. A set of example meshes can be downloaded from
[releases](https://github.com/dengwirda/periscope/releases) of this repository. 

The barotopic jet of [Galewesky et al](https://doi.org/10.1111/j.1600-0870.2004.00071.x):

    python3 ICs/jet.py \
    --mesh-file="path+name-to-mpas-mesh-file" \
    --init-file="path+name-to-mpas-init-file" \
    --with-pert=True \
    --radius=6371220.
    
Various [Williamson et al](https://doi.org/10.1016/S0021-9991(05)80016-6) SWE configurations:

    python3 ICs/wtc.py \
    --mesh-file="path+name-to-mpas-mesh-file" \
    --init-file="path+name-to-mpas-init-file" \
    --radius=6371220. \
    --test-case=N
    
Vortex-driven configurations of [Roullet & Gaillard](https://doi.org/10.1029/2021MS002663):

    python3 ICs/vtx.py \
    --mesh-file="path+name-to-mpas-mesh-file" \
    --init-file="path+name-to-mpas-init-file" \
    --test-case=N

As well as a range of other flows &mdash; see the cases in `ICs` for details.

For example, to build + run the barotropic jet test case using the CVT-optimised 
'level-7' icosahedral mesh provided in `PERISCOPE`'s
[releases](https://github.com/dengwirda/periscope/releases):

    python3 ICs/jet.py \
    --mesh-file="mesh_w_elev_cvt_7.nc" \
    --init-file="bjet_cvt_7.nc" \
    --with-pert=True \
    --radius=6371220.

    python3 swe.py \
    --mpas-file="bjet_cvt_7.nc" \
    --numthread=cores \
    --num-steps=2592 \
    --time-step=200. \
    --save-freq=432 \
    --stat-freq=108 \
    --integrate="RK32-FB"

Output is saved to the `out_bjet_cvt_7.nc` file, which can be opened for visualisation 
in e.g. paraview.

