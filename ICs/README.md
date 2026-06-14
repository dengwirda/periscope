## `Pre-baked test cases`

A number of test cases are available to run out-of-the-box:

### `Geostrophic balance`

See [Williamson et al](https://doi.org/10.1016/S0021-9991(05)80016-6).

    python ICs/wtc.py \
    --mesh-file="mesh_w_elev_cvt_7.nc" \
    --init-file="tc2_cvt_7.nc" \
    --radius=6371220. \
    --test-case=2

    python swe.py \
    --mesh-file="tc2_cvt_7.nc" \
    --time-span="12d" --save-time="1d" --stat-time="1d" \
    --numthread=cores

### `Flow over topography`

See [Williamson et al](https://doi.org/10.1016/S0021-9991(05)80016-6).

    python ICs/wtc.py \
    --mesh-file="mesh_w_elev_cvt_7.nc" \
    --init-file="tc5_cvt_7.nc" \
    --radius=6371220. \
    --test-case=5

    python swe.py \
    --mesh-file="tc5_cvt_7.nc" \
    --time-span="50d" --save-time="1d" --stat-time="1d" \
    --numthread=cores

### `Rossby-Haurwitz wave`

See [Williamson et al](https://doi.org/10.1016/S0021-9991(05)80016-6).

    python ICs/wtc.py \
    --mesh-file="mesh_w_elev_cvt_7.nc" \
    --init-file="tc6_cvt_7.nc" \
    --radius=6371220. \
    --test-case=6

    python swe.py \
    --mesh-file="tc6_cvt_7.nc" \
    --time-span="10d" --save-time="1d" --stat-time="1d" \
    --numthread=cores

### `Barotropic jet roll-up`

See [Galewesky et al](https://doi.org/10.1111/j.1600-0870.2004.00071.x).

    python ICs/jet.py \
    --mesh-file="mesh_w_elev_cvt_7.nc" \
    --init-file="jet_cvt_7.nc" \
    --radius=6371220. \
    --with-pert=TRUE

    python swe.py \
    --mesh-file="jet_cvt_7.nc" \
    --time-span="6d" --save-time="1d" --stat-time="1d" \
    --numthread=cores
    
### `Tohoku tsunami wave`

See [McDugald, Mohan, Engwirda, et al](https://doi.org/10.1029/2025GL115345).

    python ICs/wav.py \
    --mesh-file="mesh_w_elev_cvt_8.nc" \
    --init-file="tsu_cvt_8.nc" \
    --radius=6371220. \
    --test-case=4 \
    --xydz-file="fujii.txydz"
    
    python swe.py \ 
    --mesh-file="tsu_cvt_8.nc" \
    --time-span="8h" --save-time="30m" --stat-time="30m" \
    --wetdry-h0=1.E-03 \
    --loglaw-z0=0.0100 --loglaw-lo=0.0025 --loglaw-hi=1. \
    --uu-visc-4=1.E+11 --hh-diff-2=1.E+02 \
    --numthread=cores

### `Merging vortex pair`

See [Roullet & Gaillard](https://doi.org/10.1029/2021MS002663).

    python ICs/vtx.py \
    --mesh-file="mesh_disk_3.nc" \
    --init-file="merger_3_fs.nc" \
    --test-case=1 \
    --wall-slip=1.

    python swe.py \
    --mesh-file="merger_3_fs.nc" \
    --num-steps=10000 \
    --time-step=0.001 --save-freq=250 --stat-freq=250 \
    --numthread=cores

### `Dipole-wall interaction`

See [Roullet & Gaillard](https://doi.org/10.1029/2021MS002663).

    python ICs/vtx.py \
    --mesh-file="mesh_disk_3.nc" \
    --init-file="dipole_3_ns.nc" \
    --test-case=2 \
    --wall-slip=0.

    python swe.py \
    --mesh-file="dipole_3_ns.nc" \
    --num-steps=10000 \
    --time-step=0.001 --save-freq=250 --stat-freq=250 \
    --pv-upwind=0.6667 \
    --uu-visc-4=1.E+08 \
    --leith-chi=1.E+00 --leith-max=1.E+02 \
    --numthread=cores
        
### `Vortex-shedding wake`

    python ICs/obc.py \
    --mesh-file="mesh_disk_3.nc" \
    --init-file="wake_3.nc" \
    --radius=50000. \
    --test-case=1

    python swe.py \
    --mesh-file="wake_3.nc" \
    --forc-file="frc_wake_3.nc" \
    --num-steps=2500 --save-freq=250 --stat-freq=250 \
    --wetdry-h0=1.E-03 \
    --loglaw-z0=0.0100 --loglaw-lo=0.0025 --loglaw-hi=1. \
    --pv-upwind=0.6667 \
    --uu-visc-4=1.E+11 --hh-diff-2=1.E+02 \
    --leith-chi=1.E+00 --leith-max=1.E+04 \
    --numthread=cores

