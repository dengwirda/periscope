
import numpy as np

# A simple interface to Akima's interpolation schemes 

# Authors: Darren Engwirda

def interp1d(xd, fd, xi):

    nxd = xd.size
    nip = xi.size
    
    fi = np.zeros(nip, dtype=np.float64)

    ier = int(+0)

    interp1d_(nxd, xd, fd, nip, xi, fi, ier)

    return fi


def interp2d(xd, yd, fd, xi, yi):

    nxd = xd.size
    nyd = yd.size
    nip = xi.size
    nwk = nxd * nyd * 3

    wk = np.zeros(nwk, dtype=np.float64)
    fi = np.zeros(nip, dtype=np.float64)

    ier = int(+0)

    interp2d_(nxd, nyd, xd, yd, fd, nip, xi, yi, fi, ier, wk)

    return fi


try:
    # load cython kernels, if compiled
    from akima_ import interp1d_
    from akima_ import interp2d_

except ImportError:
    raise RuntimeError("Cython back-end not found")

