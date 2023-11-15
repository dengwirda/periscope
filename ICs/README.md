## `Pre-baked test cases`

A number of test cases are available to run out-of-the-box:

### `Geostrophic balance`

    python3 wtc.py \
    --mesh-file="mesh_w_elev_cvt_7.nc" \
    --init-file="wtc2_cvt_7.nc" \
    --radius=6371220. \
    --test-case=2

    python3 swe.py \
    --mesh-file="wtc2_cvt_7.nc" \
    --num-steps=3456 \
    --time-step=300. \
    --save-freq=288 \
    --stat-freq=144 \
    --integrate="RK32-FB" \
    --numthread=cores

    python3 swe.py \
    --mesh-file="wtc2_cvt_7.nc" \
    --num-steps=5184 \
    --time-step=200. \
    --save-freq=432 \
    --stat-freq=216 \
    --integrate="RK22-FB" \
    --numthread=cores

### `Flow along mountain`

    python3 wtc.py \
    --mesh-file="mesh_w_elev_cvt_7.nc" \
    --init-file="wtc5_cvt_7.nc" \
    --radius=6371220. \
    --test-case=5

    python3 swe.py \
    --mesh-file="wtc5_cvt_7.nc" \
    --num-steps=36000 \
    --time-step=120. \
    --save-freq=720 \
    --stat-freq=144 \
    --integrate="RK22-FB" \
    --numthread=cores

    python3 swe.py \
    --mesh-file="wtc5_cvt_7.nc" \
    --num-steps=24000 \
    --time-step=180. \
    --save-freq=480 \
    --stat-freq=120 \
    --integrate="RK32-FB" \
    --numthread=cores

### `RossbyHaurwitz wave`

    python3 wtc.py \
    --mesh-file="mesh_w_elev_cvt_7.nc" \
    --init-file="wtc6_cvt_7.nc" \
    --radius=6371220. \
    --test-case=6

    python3 swe.py \
    --mesh-file="wtc6_cvt_7.nc" \
    --num-steps=4608 \
    --time-step=150. \
    --save-freq=576 \
    --stat-freq=144 \
    --integrate="RK32-FB" \
    --numthread=cores

### `Barotropic jet roll-up`

    python3 jet.py \
    --mesh-file="mesh_w_elev_cvt_7.nc" \
    --init-file="bjet_cvt_7.nc" \
    --radius=6371220. \
    --with-pert=TRUE

    python3 swe.py \
    --mesh-file="bjet_cvt_7.nc" \
    --num-steps=2592 \
    --time-step=200. \
    --save-freq=432 \
    --stat-freq=108 \
    --integrate="RK32-FB"  \
    --numthread=cores

    python3 swe.py \
    --mesh-file="bjet_cvt_7.nc" \
    --num-steps=3840 \
    --time-step=135. \
    --save-freq=640 \
    --stat-freq=160 \
    --integrate="RK22-FB" \
    --numthread=cores

### `Linear gravity wave`

    python3 wav.py \
    --mesh-file="mesh_w_elev_cvt_7.nc" \
    --init-file="lgw1_cvt_7.nc" \
    --radius=6371220. \
    --test-case=1

    python3 swe.py \
    --mesh-file="lgw1_cvt_7.nc" \
    --num-steps=384 \
    --time-step=900. \
    --save-freq=16 \
    --stat-freq=16 \
    --integrate="RK32-FB" \
    --no-advect=TRUE \
    --no-rotate=TRUE \
    --numthread=cores

### `Merging vortex pair`

    python3 vtx.py \
    --mesh-file="mesh_disk_3.nc" \
    --init-file="merger_3.nc" \
    --test-case=1

    python3 swe.py \
    --mesh-file="merger_3.nc" \
    --num-steps=10000 \
    --time-step=0.001 \
    --save-freq=250 \
    --stat-freq=125 \
    --integrate="RK32-FB" \
    --wall-slip=1.0 \
    --numthread=cores

### `Dipole-wall interaction`

    python3 vtx.py \
    --mesh-file="mesh_disk_3.nc" \
    --init-file="dipole_3.nc" \
    --test-case=2

    python3 swe.py \
    --mesh-file="dipole_3.nc" \
    --num-steps=10000 \
    --time-step=0.001 \
    --save-freq=250 \
    --stat-freq=125 \
    --integrate="RK32-FB" \
    --wall-slip=0.0 \
    --numthread=cores

