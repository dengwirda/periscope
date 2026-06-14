
import sys
import os
import numpy as np
import xarray
import argparse

sys.path.insert(
    1, os.path.join(sys.path[0], "..", ".."))

from msh import load_mesh
from ops import operators

def smooth(mesh, mats, df_cell, smth):
    for scan in range(smth):
        df_dual = mats.dual_kite_sums * df_cell
        df_dual/= mesh.vert.area
        df_cell = mats.cell_kite_sums * df_dual
        df_cell/= mesh.cell.area
    return df_cell

if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--mesh-file", dest="mesh_file", type=str,
        required=True, help="Path to user mesh file.")

    parser.add_argument(
        "--sol1-file", dest="sol1_file", type=str,
        required=True, help="Path to user soln file.")

    parser.add_argument(
        "--sol2-file", dest="sol2_file", type=str,
        required=True, help="Path to user soln file.")

    parser.add_argument(
        "--diff-file", dest="diff_file", type=str,
        required=True, help="Path to soln diff file.")

    args = parser.parse_args()

#------------------------------------ load an MPAS mesh file

    print("Loading the mesh file...")

    mesh = load_mesh(args.mesh_file)
    
#------------------------------------ build TRSK matrix op's

    print("Building coefficients...")

    mats = operators(mesh)

    dat1 = xarray.open_dataset(args.sol1_file)
    dat2 = xarray.open_dataset(args.sol2_file) 

    h1_min_ = np.squeeze(np.asarray(dat1["hh_min_"][:]))
    h1_max_ = np.squeeze(np.asarray(dat1["hh_max_"][:]))
    z1_rms_ = np.squeeze(np.asarray(dat1["zt_rms_"][:]))    
    k1_rms_ = np.squeeze(np.asarray(dat1["ke_rms_"][:]))
    k1_max_ = np.squeeze(np.asarray(dat1["ke_max_"][:]))

    h2_min_ = np.squeeze(np.asarray(dat2["hh_min_"][:]))
    h2_max_ = np.squeeze(np.asarray(dat2["hh_max_"][:]))
    z2_rms_ = np.squeeze(np.asarray(dat2["zt_rms_"][:]))    
    k2_rms_ = np.squeeze(np.asarray(dat2["ke_rms_"][:]))
    k2_max_ = np.squeeze(np.asarray(dat2["ke_max_"][:]))

    shallow = 4.0  # neglect delta if too shallow

    # change in tidal range
    dh_max_ = 1. * (h2_max_ - h2_min_) - \
              1. * (h1_max_ - h1_min_)

    dh_max_*= np.minimum(
        1., np.minimum(h1_min_, h2_min_) / shallow)

    dh_max_ = smooth(mesh, mats, dh_max_, 1)

    dz_rms_ = 2. * (z2_rms_ - z1_rms_)

    dz_rms_*= np.minimum(
        1., np.minimum(h1_min_, h2_min_) / shallow)

    dz_rms_ = smooth(mesh, mats, dz_rms_, 1)
    
    # change in tidal flows
    du_rms_ = np.sqrt(2. * k2_rms_) - np.sqrt(2. * k1_rms_)

    du_rms_*= np.minimum(
        1., np.minimum(h1_min_, h2_min_) / shallow)

    du_rms_ = smooth(mesh, mats, du_rms_, 1)

    du_max_ = np.sqrt(2. * k2_max_) - np.sqrt(2. * k1_max_)

    du_max_*= np.minimum(
        1., np.minimum(h1_min_, h2_min_) / shallow) 

    du_max_ = smooth(mesh, mats, du_max_, 1)

    data = xarray.open_dataset(args.mesh_file)
    data["dh_max_"] = (("nCells", "nVertLevels"), 
        np.reshape(dh_max_, (mesh.cell.size, 1)))
    data["dz_rms_"] = (("nCells", "nVertLevels"), 
        np.reshape(dz_rms_, (mesh.cell.size, 1)))    
    data["du_rms_"] = (("nCells", "nVertLevels"), 
        np.reshape(du_rms_, (mesh.cell.size, 1)))
    data["du_max_"] = (("nCells", "nVertLevels"), 
        np.reshape(du_max_, (mesh.cell.size, 1)))
    
    data.to_netcdf(args.diff_file)

