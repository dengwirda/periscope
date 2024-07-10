
PYTHON="python3"
MAXCPU=6
BINDIR="periscope"
MSHDIR="swe_cases"
RUNDIR="swe_cases"

#---------------------------------------------------------------- 30.0km
NUMCPU=$((4 <= MAXCPU ? 4 : MAXCPU))
echo "Using " $NUMCPU "cores"

${PYTHON} ${BINDIR}/ICs/wdg.py \
--mesh-file=${MSHDIR}"/mesh_20to38N_30km.nc" \
--init-file=${RUNDIR}"/gyre_fs_30km.nc" \
--radius=6371220. --wall-slip=1.

${PYTHON} ${BINDIR}/swe.py \
--mesh-file=${RUNDIR}"/gyre_fs_30km.nc" \
--forc-file=${RUNDIR}"/gyre_fs_30km.nc" \
--soln-file=${RUNDIR}"/gyre_fs_30km_00to09years.nc" \
--numthread=${NUMCPU} \
--time-step=5400. \
--num-steps=51840 --save-freq=480 --stat-freq=480 \
--linlaw-cd=1.E-07 \
--uu-visc-2=5.E+02 --ref-scale=-1 --msh-fixes=0

${PYTHON} ${BINDIR}/swe.py \
--mesh-file=${RUNDIR}"/gyre_fs_30km_00to09years.nc" \
--forc-file=${RUNDIR}"/gyre_fs_30km.nc" \
--soln-file=${RUNDIR}"/gyre_fs_30km_09to15years.nc" \
--numthread=${NUMCPU} \
--time-step=5400. \
--num-steps=34560 --save-freq=480 --stat-freq=480 \
--linlaw-cd=1.E-07 \
--uu-visc-2=5.E+02 --ref-scale=-1 --msh-fixes=0

${PYTHON} ${BINDIR}/ICs/wdg.py \
--mesh-file=${MSHDIR}"/mesh_20to38N_30km.nc" \
--init-file=${RUNDIR}"/gyre_ns_30km.nc" \
--radius=6371220. --wall-slip=0.

${PYTHON} ${BINDIR}/swe.py \
--mesh-file=${RUNDIR}"/gyre_ns_30km.nc" \
--forc-file=${RUNDIR}"/gyre_ns_30km.nc" \
--soln-file=${RUNDIR}"/gyre_ns_30km_00to09years.nc" \
--numthread=${NUMCPU} \
--time-step=5400. \
--num-steps=51840 --save-freq=480 --stat-freq=480 \
--linlaw-cd=1.E-07 \
--uu-visc-2=5.E+02 --ref-scale=-1 --msh-fixes=0

${PYTHON} ${BINDIR}/swe.py \
--mesh-file=${RUNDIR}"/gyre_ns_30km_00to09years.nc" \
--forc-file=${RUNDIR}"/gyre_ns_30km.nc" \
--soln-file=${RUNDIR}"/gyre_ns_30km_09to15years.nc" \
--numthread=${NUMCPU} \
--time-step=5400. \
--num-steps=34560 --save-freq=480 --stat-freq=480 \
--linlaw-cd=1.E-07 \
--uu-visc-2=5.E+02 --ref-scale=-1 --msh-fixes=0

#---------------------------------------------------------------- 15.0km
NUMCPU=$((8 <= MAXCPU ? 8 : MAXCPU))
echo "Using " $NUMCPU "cores"

${PYTHON} ${BINDIR}/ICs/wdg.py \
--mesh-file=${MSHDIR}"/mesh_20to38N_15km.nc" \
--init-file=${RUNDIR}"/gyre_fs_15km.nc" \
--radius=6371220. --wall-slip=1.

${PYTHON} ${BINDIR}/swe.py \
--mesh-file=${RUNDIR}"/gyre_fs_15km.nc" \
--forc-file=${RUNDIR}"/gyre_fs_15km.nc" \
--soln-file=${RUNDIR}"/gyre_fs_15km_00to09years.nc" \
--numthread=${NUMCPU} \
--time-step=2700. \
--num-steps=103680 --save-freq=960 --stat-freq=960 \
--linlaw-cd=1.E-07 \
--uu-visc-2=5.E+02 --ref-scale=-1 --msh-fixes=0

${PYTHON} ${BINDIR}/swe.py \
--mesh-file=${RUNDIR}"/gyre_fs_15km_00to09years.nc" \
--forc-file=${RUNDIR}"/gyre_fs_15km.nc" \
--soln-file=${RUNDIR}"/gyre_fs_15km_09to15years.nc" \
--numthread=${NUMCPU} \
--time-step=2700. \
--num-steps=69120 --save-freq=960 --stat-freq=960 \
--linlaw-cd=1.E-07 \
--uu-visc-2=5.E+02 --ref-scale=-1 --msh-fixes=0

${PYTHON} ${BINDIR}/ICs/wdg.py \
--mesh-file=${MSHDIR}"/mesh_20to38N_15km.nc" \
--init-file=${RUNDIR}"/gyre_ns_15km.nc" \
--radius=6371220. --wall-slip=0.

${PYTHON} ${BINDIR}/swe.py \
--mesh-file=${RUNDIR}"/gyre_ns_15km.nc" \
--forc-file=${RUNDIR}"/gyre_ns_15km.nc" \
--soln-file=${RUNDIR}"/gyre_ns_15km_00to09years.nc" \
--numthread=${NUMCPU} \
--time-step=2700. \
--num-steps=103680 --save-freq=960 --stat-freq=960 \
--linlaw-cd=1.E-07 \
--uu-visc-2=5.E+02 --ref-scale=-1 --msh-fixes=0

${PYTHON} ${BINDIR}/swe.py \
--mesh-file=${RUNDIR}"/gyre_ns_15km_00to09years.nc" \
--forc-file=${RUNDIR}"/gyre_ns_15km.nc" \
--soln-file=${RUNDIR}"/gyre_ns_15km_09to15years.nc" \
--numthread=${NUMCPU} \
--time-step=2700. \
--num-steps=69120 --save-freq=960 --stat-freq=960 \
--linlaw-cd=1.E-07 \
--uu-visc-2=5.E+02 --ref-scale=-1 --msh-fixes=0

#---------------------------------------------------------------- 7.50km
NUMCPU=$((16<= MAXCPU ? 16 : MAXCPU))
echo "Using " $NUMCPU "cores"

${PYTHON} ${BINDIR}/ICs/wdg.py \
--mesh-file=${MSHDIR}"/mesh_20to38N_7p5km.nc" \
--init-file=${RUNDIR}"/gyre_fs_7p5km.nc" \
--radius=6371220. --wall-slip=1.

${PYTHON} ${BINDIR}/swe.py \
--mesh-file=${RUNDIR}"/gyre_fs_7p5km.nc" \
--forc-file=${RUNDIR}"/gyre_fs_7p5km.nc" \
--soln-file=${RUNDIR}"/gyre_fs_7p5km_00to09years.nc" \
--numthread=${NUMCPU} \
--time-step=1350. \
--num-steps=207360 --save-freq=1920 --stat-freq=1920 \
--linlaw-cd=1.E-07 \
--uu-visc-2=5.E+02 --ref-scale=-1 --msh-fixes=0

${PYTHON} ${BINDIR}/swe.py \
--mesh-file=${RUNDIR}"/gyre_fs_7p5km_00to09years.nc" \
--forc-file=${RUNDIR}"/gyre_fs_7p5km.nc" \
--soln-file=${RUNDIR}"/gyre_fs_7p5km_09to15years.nc" \
--numthread=${NUMCPU} \
--time-step=1350. \
--num-steps=138240 --save-freq=1920 --stat-freq=1920 \
--linlaw-cd=1.E-07 \
--uu-visc-2=5.E+02 --ref-scale=-1 --msh-fixes=0

${PYTHON} ${BINDIR}/ICs/wdg.py \
--mesh-file=${MSHDIR}"/mesh_20to38N_7p5km.nc" \
--init-file=${RUNDIR}"/gyre_ns_7p5km.nc" \
--radius=6371220. --wall-slip=0.

${PYTHON} ${BINDIR}/swe.py \
--mesh-file=${RUNDIR}"/gyre_ns_7p5km.nc" \
--forc-file=${RUNDIR}"/gyre_ns_7p5km.nc" \
--soln-file=${RUNDIR}"/gyre_ns_7p5km_00to09years.nc" \
--numthread=${NUMCPU} \
--time-step=1350. \
--num-steps=207360 --save-freq=1920 --stat-freq=1920 \
--linlaw-cd=1.E-07 \
--uu-visc-2=5.E+02 --ref-scale=-1 --msh-fixes=0

${PYTHON} ${BINDIR}/swe.py \
--mesh-file=${RUNDIR}"/gyre_ns_7p5km_00to09years.nc" \
--forc-file=${RUNDIR}"/gyre_ns_7p5km.nc" \
--soln-file=${RUNDIR}"/gyre_ns_7p5km_09to15years.nc" \
--numthread=${NUMCPU} \
--time-step=1350. \
--num-steps=138240 --save-freq=1920 --stat-freq=1920 \
--linlaw-cd=1.E-07 \
--uu-visc-2=5.E+02 --ref-scale=-1 --msh-fixes=0

#---------------------------------------------------------------- 3.75km
NUMCPU=$((32<= MAXCPU ? 32 : MAXCPU))
echo "Using " $NUMCPU "cores"

${PYTHON} ${BINDIR}/ICs/wdg.py \
--mesh-file=${MSHDIR}"/mesh_20to38N_3p75km.nc" \
--init-file=${RUNDIR}"/gyre_fs_3p75km.nc" \
--radius=6371220. --wall-slip=1.

${PYTHON} ${BINDIR}/swe.py \
--mesh-file=${RUNDIR}"/gyre_fs_3p75km.nc" \
--forc-file=${RUNDIR}"/gyre_fs_3p75km.nc" \
--soln-file=${RUNDIR}"/gyre_fs_3p75km_00to09years.nc" \
--numthread=${NUMCPU} \
--time-step=675. \
--num-steps=414720 --save-freq=3840 --stat-freq=3840 \
--linlaw-cd=1.E-07 \
--uu-visc-2=5.E+02 --ref-scale=-1 --msh-fixes=0

${PYTHON} ${BINDIR}/swe.py \
--mesh-file=${RUNDIR}"/gyre_fs_3p75km_00to09years.nc" \
--forc-file=${RUNDIR}"/gyre_fs_3p75km.nc" \
--soln-file=${RUNDIR}"/gyre_fs_3p75km_09to15years.nc" \
--numthread=${NUMCPU} \
--time-step=675. \
--num-steps=276480 --save-freq=3840 --stat-freq=3840 \
--linlaw-cd=1.E-07 \
--uu-visc-2=5.E+02 --ref-scale=-1 --msh-fixes=0

${PYTHON} ${BINDIR}/ICs/wdg.py \
--mesh-file=${MSHDIR}"/mesh_20to38N_3p75km.nc" \
--init-file=${RUNDIR}"/gyre_ns_3p75km.nc" \
--radius=6371220. --wall-slip=0.

${PYTHON} ${BINDIR}/swe.py \
--mesh-file=${RUNDIR}"/gyre_ns_3p75km.nc" \
--forc-file=${RUNDIR}"/gyre_ns_3p75km.nc" \
--soln-file=${RUNDIR}"/gyre_ns_3p75km_00to09years.nc" \
--numthread=${NUMCPU} \
--time-step=675. \
--num-steps=414720 --save-freq=3840 --stat-freq=3840 \
--linlaw-cd=1.E-07 \
--uu-visc-2=5.E+02 --ref-scale=-1 --msh-fixes=0

${PYTHON} ${BINDIR}/swe.py \
--mesh-file=${RUNDIR}"/gyre_ns_3p75km_00to09years.nc" \
--forc-file=${RUNDIR}"/gyre_ns_3p75km.nc" \
--soln-file=${RUNDIR}"/gyre_ns_3p75km_09to15years.nc" \
--numthread=${NUMCPU} \
--time-step=675. \
--num-steps=276480 --save-freq=3840 --stat-freq=3840 \
--linlaw-cd=1.E-07 \
--uu-visc-2=5.E+02 --ref-scale=-1 --msh-fixes=0

