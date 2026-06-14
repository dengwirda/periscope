
import time
import sys
import os
import numpy as np
import netCDF4 as nc
from scipy import spatial

import argparse

# land-fill TPXO10 data for shoreline continuity

HERE = os.path.abspath(os.path.dirname(__file__))

if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--tpxo-file", dest="tpxo_file", type=str,
        required=True, help="Path to TPXO data file.")

    args = parser.parse_args()

    print("Loading the TPXO file...")

    tpxo = nc.Dataset(args.tpxo_file, "r+")

    ncon = int(tpxo.dimensions["nc"].size)
    xlon = np.asarray(tpxo["lon_z"][:], dtype=np.float64)
    ylat = np.asarray(tpxo["lat_z"][:], dtype=np.float64)

    xlon = xlon[:, 0] * np.pi / 180.
    ylat = ylat[0, :] * np.pi / 180.

    for icon in range(ncon):
        con_amp = np.asarray(
            tpxo["ha"][ icon], dtype=np.float32)
        con_phs = np.asarray(
            tpxo["hp"][ icon], dtype=np.float32)

        con_amp = con_amp.T
        con_phs = con_phs.T

        XLON, YLAT = np.meshgrid(xlon, ylat)

        xpos = np.cos(XLON) * np.cos(YLAT)
        ypos = np.sin(XLON) * np.cos(YLAT)
        zpos = np.sin(YLAT)

        ppos = np.vstack( (
            xpos.ravel(), ypos.ravel(), zpos.ravel()
            ) ).T

        land = con_amp <= 0.
        seas = con_amp >  0.

        sidx = np.argwhere(seas.ravel()>0)

        lpos = ppos [land.ravel(), :]
        opos = ppos [seas.ravel(), :]

        tree = spatial.cKDTree(opos, leafsize=32)

        try:  # ridiculous argument renaming...
            __, cell = tree.query(lpos, n_jobs=-1)
        except:
            __, cell = tree.query(lpos, workers=-1)

        vec_amp = con_amp.ravel()
        vec_phs = con_phs.ravel()

        cell = np.squeeze (sidx[cell])
        con_amp[land] = vec_amp[cell]
        con_phs[land] = vec_phs[cell]

        con_amp = con_amp.T
        con_phs = con_phs.T

        tpxo["ha"][icon, :] = con_amp
        tpxo["hp"][icon, :] = con_phs

    tpxo.close()

