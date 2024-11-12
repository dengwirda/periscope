## `Pre-baked test cases`

A number of test cases are available to run out-of-the-box:

### `Geostrophic balance`

    python3 ICs/wtc.py \
    --mesh-file="mesh_w_elev_cvt_7.nc" \
    --init-file="wtc2_cvt_7.nc" \
    --radius=6371220. \
    --test-case=2

    python3 swe.py \
    --mesh-file="wtc2_cvt_7.nc" \
    --num-steps=3456 \
    --time-step=300. \
    --save-freq=288 --stat-freq=144 \
    --integrate="RK32-FB" \
    --numthread=cores

    python3 swe.py \
    --mesh-file="wtc2_cvt_7.nc" \
    --num-steps=5184 \
    --time-step=200. \
    --save-freq=432 --stat-freq=216 \
    --integrate="RK22-FB" \
    --numthread=cores

### `Flow over a mountain`

    python3 ICs/wtc.py \
    --mesh-file="mesh_w_elev_cvt_7.nc" \
    --init-file="wtc5_cvt_7.nc" \
    --radius=6371220. \
    --test-case=5

    python3 swe.py \
    --mesh-file="wtc5_cvt_7.nc" \
    --num-steps=36000 \
    --time-step=120. \
    --save-freq=720 --stat-freq=144 \
    --integrate="RK22-FB" \
    --numthread=cores

    python3 swe.py \
    --mesh-file="wtc5_cvt_7.nc" \
    --num-steps=24000 \
    --time-step=180. \
    --save-freq=480 --stat-freq=120 \
    --integrate="RK32-FB" \
    --numthread=cores

### `Rossby-Haurwitz wave`

    python3 ICs/wtc.py \
    --mesh-file="mesh_w_elev_cvt_7.nc" \
    --init-file="wtc6_cvt_7.nc" \
    --radius=6371220. \
    --test-case=6

    python3 swe.py \
    --mesh-file="wtc6_cvt_7.nc" \
    --num-steps=4608 \
    --time-step=150. \
    --save-freq=576 --stat-freq=144 \
    --integrate="RK32-FB" \
    --numthread=cores

### `Barotropic jet roll-up`

    python3 ICs/jet.py \
    --mesh-file="mesh_w_elev_cvt_7.nc" \
    --init-file="bjet_cvt_7.nc" \
    --radius=6371220. \
    --with-pert=TRUE

    python3 swe.py \
    --mesh-file="bjet_cvt_7.nc" \
    --num-steps=2592 \
    --time-step=200. \
    --save-freq=432 --stat-freq=108 \
    --integrate="RK32-FB"  \
    --numthread=cores

    python3 swe.py \
    --mesh-file="bjet_cvt_7.nc" \
    --num-steps=3840 \
    --time-step=135. \
    --save-freq=640 --stat-freq=160 \
    --integrate="RK22-FB" \
    --numthread=cores

### `Linear gravity wave`

    python3 ICs/wav.py \
    --mesh-file="mesh_w_elev_cvt_7.nc" \
    --init-file="lgw1_cvt_7.nc" \
    --radius=6371220. \
    --test-case=1

    python3 swe.py \
    --mesh-file="lgw1_cvt_7.nc" \
    --num-steps=384 \
    --time-step=900. \
    --save-freq=16 --stat-freq=16 \
    --integrate="RK32-FB" \
    --no-advect=TRUE \
    --no-rotate=TRUE \
    --numthread=cores
    
### `Tohoku tsunami wave`

    python3 ICs/wav.py \
    --mesh-file="mesh_w_elev_cvt_8.nc" \
    --init-file="tsu_cvt_8.nc" \
    --radius=6371220. \
    --test-case=4 \
    --xydz-file="fujii.txydz"
    
    python3 swe.py \ 
    --mesh-file="tsu_cvt_8.nc" \
    --num-steps=1920 \
    --time-step=15. \
    --save-freq=40 --stat-freq=40 \
    --hh-scheme="upwind" \
    --wetdry-h0=1.E-03 \
    --loglaw-z0=0.0100 --loglaw-lo=0.0025 --loglaw-hi=1. \
    --uu-visc-2=1.E+02 --uu-visc-4=1.E+11 \
    --hh-diff-2=1.E+02 --hh-diff-4=1.E+11 \
    --numthread=cores

### `Merging vortex pair`

    python3 ICs/vtx.py \
    --mesh-file="mesh_disk_3.nc" \
    --init-file="merger_3.nc" \
    --test-case=1 \
    --wall-slip=1.

    python3 swe.py \
    --mesh-file="merger_3.nc" \
    --num-steps=10000 \
    --time-step=0.001 \
    --save-freq=250 --stat-freq=250 \
    --numthread=cores

### `Dipole-wall interaction`

    python3 ICs/vtx.py \
    --mesh-file="mesh_disk_3.nc" \
    --init-file="dipole_3_fs.nc" \
    --test-case=2 \
    --wall-slip=1.

    python3 swe.py \
    --mesh-file="dipole_3.nc" \
    --num-steps=10000 \
    --time-step=0.001 \
    --save-freq=250 --stat-freq=250 \
    --numthread=cores
    
    python3 ICs/vtx.py \
    --mesh-file="mesh_disk_3.nc" \
    --init-file="dipole_3_ns.nc" \
    --test-case=2 \
    --wall-slip=0.

    python3 swe.py \
    --mesh-file="dipole_3.nc" \
    --num-steps=10000 \
    --time-step=0.001 \
    --save-freq=250 --stat-freq=250 \
    --uu-visc-2=1.E+00 --uu-visc-4=1.E+08 \
    --leith-chi=0.3875 --leith-max=1.E+02 \
    --numthread=cores
        
### `Vortex-shedding wake`

    python3 ICs/obc.py \
    --mesh-file="mesh_disk_3.nc" \
    --init-file="wake_3.nc" \
    --radius=50000. \
    --test-case=1

    python3 swe.py \
    --mesh-file="wake_3.nc" \
    --forc-file="frc_wake_3.nc" \
    --time-step=0.0625 \
    --num-steps=1440 \
    --save-freq=144 --stat-freq=144 \
    --hh-scheme="upwind" \
    --wetdry-h0=1.E-03 \
    --loglaw-z0=0.0100 --loglaw-lo=0.0025 --loglaw-hi=1. \
    --uu-visc-2=1.E+02 --uu-visc-4=1.E+11 \
    --hh-diff-2=1.E+02 --hh-diff-4=1.E+11 \
    --leith-chi=0.3875 --leith-max=1.E+03 \
    --numthread=cores

### `Wet-dry shore run-up`

    python3 ICs/obc.py \
    --mesh-file="mesh_disk_3.nc" \
    --init-file="hill_3.nc" \
    --radius=50000. \
    --test-case=2

    python3 swe.py \
    --mesh-file="hill_3.nc" \
    --forc-file="frc_hill_3.nc" \
    --time-step=0.1 \
    --num-steps=2500 \
    --save-freq=50 --stat-freq=50 \
    --hh-scheme="upwind" \
    --wetdry-h0=1.E-03 \
    --loglaw-z0=0.0100 --loglaw-lo=0.0025 --loglaw-hi=1. \
    --uu-visc-2=1.E+02 --uu-visc-4=1.E+11 \
    --hh-diff-2=1.E+02 --hh-diff-4=1.E+11 \
    --numthread=cores

### `A wind driven gyre`

    python3 ICs/wdg.py \
    --mesh-file="mesh_20to38N_30km.nc" \
    --init-file="gyre_fs_30km.nc" \
    --radius=6371220. --wall-slip=1.

    python3 swe.py \
    --mesh-file="gyre_fs_30km.nc" \
    --forc-file="gyre_fs_30km.nc" \
    --soln-file="gyre_fs_30km_00to06years.nc" \
    --numthread=4 \
    --time-step=5400. \
    --num-steps=34560 \
    --save-freq=480 --stat-freq=480 \
    --linlaw-cd=1.E-07 \
    --uu-visc-2=5.E+02 --ref-scale=-1 --msh-fixes=0

    python3 ICs/wdg.py \
    --mesh-file="mesh_20to38N_30km.nc" \
    --init-file="gyre_ns_30km.nc" \
    --radius=6371220. --wall-slip=0.

    python3 swe.py \
    --mesh-file="gyre_ns_30km.nc" \
    --forc-file="gyre_ns_30km.nc" \
    --soln-file="gyre_ns_30km_00to06years.nc" \
    --numthread=4 \
    --time-step=5400. \
    --num-steps=34560 \
    --save-freq=480 --stat-freq=480 \
    --linlaw-cd=1.E-07 \
    --uu-visc-2=5.E+02 --ref-scale=-1 --msh-fixes=0

