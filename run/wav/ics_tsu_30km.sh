
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
    --mesh-file=${MSHDIR}/"mesh_w_elev_cvt_8.nc"
    --init-file=${MSHDIR}/"tsu_cvt_8.nc"
    --test-case=4
    --radius=6371220.0
    --xydz-file=${MSHDIR}/"fujii.txydz"
  )

  ${PYTHON} ${BINDIR}/ICs/wav.py "${opts[@]}"

else
  echo "PERISCOPE not found"
fi

