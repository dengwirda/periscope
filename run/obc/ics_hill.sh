
if [ ! -n "${BINDIR+z}" ]; then
  echo "BINDIR=path/to/periscope is required"; exit 1
fi

if [ ! -n "${MSHDIR+z}" ]; then
  echo "MSHDIR=path/to/mesh-dirs is required"; exit 1
fi

if [ ! -n "${PYTHON+z}" ]; then PYTHON="python3" ; fi

if [ -f ${BINDIR}/swe.py ]
then
  
  opts=(
    --mesh-file=${MSHDIR}/"mesh_disk_3.nc"
    --init-file=${MSHDIR}/"hill_3.nc"
    --test-case=2
    --radius=50000.0
  )

  ${PYTHON} ${BINDIR}/ICs/obc.py "${opts[@]}"

else
  echo "PERISCOPE not found"
fi

