
import time
import sys
import os
import numpy as np
import netCDF4 as nc

import argparse

# Modify the underlying DEM variables in a mesh

if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--mesh-file", dest="mesh_file", type=str,
        required=True, help="Path to user mesh file.")

    parser.add_argument(
        "--cell-list", dest="cell_list", type=int,
        required=True,
        nargs="*", help="List of cell IDs to modify.")

    parser.add_argument(
        "--cell-elev", dest="cell_elev", type=float,
        required=True, help="New elevation on cells.")

    args = parser.parse_args()

    mesh = nc.Dataset(args.mesh_file, "r+")

    try:
        elev = np.asarray(mesh["bed_elevation"][:], 
                          dtype=np.float32)
        ocnh = np.asarray(mesh["ocn_thickness"][:], 
                          dtype=np.float32)
    except:
        raise ValueError("Bathymetry information not found")

    try:
        ocov = np.asarray(mesh["ocn_cover"][:], 
                          dtype=np.float32)
    except:
        ocov = None

    if (elev is not None):
        dlev = args.cell_elev - elev[args.cell_list]

        elev[args.cell_list]+= dlev

        mesh["bed_elevation"][:] = elev

        ocnh[args.cell_list] = \
            np.maximum(0.0, -elev[args.cell_list])

        mesh["ocn_thickness"][:] = ocnh

    if (ocov is not None):
        ocov[args.cell_list] = (args.cell_elev < 0.)

        mesh["ocn_cover"][:] = ocov

    mesh.close()
