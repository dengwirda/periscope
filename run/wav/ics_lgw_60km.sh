
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
    --mesh-file=${MSHDIR}/"mesh_w_elev_cvt_7.nc"
    --init-file=${MSHDIR}/"lgw_cvt_7.nc"
    --test-case=1
    --radius=6371220.0
  )

  ${PYTHON} ${BINDIR}/ICs/wav.py "${opts[@]}"

else
  echo "PERISCOPE not found"
fi

