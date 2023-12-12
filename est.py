
import os
import numpy as np
import netCDF4 as nc
import argparse

""" Util. to set time-step estimates via CFL approximations.
"""
#-- Part of the PERISCOPE solver
#-- Darren Engwirda
#-- d.engwirda@gmail.com
#-- https://github.com/dengwirda/

def est(opts, mesh):

#-- find max. depth for each cell; to be safe, take the max 
#-- of the self depth and the adj. edge-midpoint depths

    ncel = mesh.dimensions["nCells"].size

    zt_cell = np.zeros((ncel), dtype=np.float32)
    zb_cell = np.zeros((ncel), dtype=np.float32)

    if ("ssh" in mesh.variables.keys()):
        zt_cell = np.asarray(
            mesh["ssh"][:], dtype=np.float32)

    if ("zb_cell" in mesh.variables.keys()):
        hh_cell = np.asarray(
            mesh["zb_cell"][:], dtype=np.float32)

    if ("bottomDepth" in mesh.variables.keys()):
        zb_cell = -1 * np.asarray(
            mesh["bottomDepth"][:], dtype=np.float32)
            
    if ("bed_elevation" in mesh.variables.keys()):
        zb_cell = np.asarray(
            mesh["bed_elevation"][:], dtype=np.float32)

    hh_cell = zt_cell - zb_cell

    if ("hh_cell" in mesh.variables.keys()):
        hh_cell = np.asarray(
            mesh["hh_cell"][:], dtype=np.float32)

    if ("ocn_thickness" in mesh.variables.keys()):
        hh_cell = np.asarray(
            mesh["ocn_thickness"][:], dtype=np.float32)

    hh_cell = np.maximum(0., hh_cell)

    class base: pass
    cell = base()
    cell.cell = np.asarray(
        mesh["cellsOnCell"][:, :], dtype=np.int32)
    cell.topo = np.asarray(
        mesh["nEdgesOnCell"][:], dtype=np.int32)
    cell.area = np.asarray(
        mesh["areaCell"][:], dtype=np.float32)

    hh_keep = hh_cell[:]
    for epos in range(np.max(cell.topo)):

        mask = cell.topo > epos

        icel = np.argwhere(mask).ravel()

        jcel = cell.cell[mask, epos] - 1

        mask = jcel > -1
        icel = icel[mask]
        jcel = jcel[mask]

        hh_keep[icel] = \
            np.maximum(hh_keep[icel], 
                0.5 * (hh_cell[icel] + hh_cell[jcel]))

#-- estimate per-cell time-step using a combination of BTR
#-- phase-speed and Lagranian velocity thresholds:
#-- DT-BTR <= sqrt(AREA) / [U-SAFE + sqrt(G*(H + H-SAFE))]

    return np.sqrt(cell.area) / (opts.u_safety +
        np.sqrt(opts.gravity * (hh_keep + opts.h_safety)))


if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "--mesh-file", dest="mesh_file", type=str,
        required=True, 
        help="Path to user INITIAL conditions file.")

    parser.add_argument(
        "--u-safety", dest="u_safety", type=float,
        default=0.0,
        required=False, 
        help="Estimate of max. Lagrangian velocity.")

    parser.add_argument(
        "--h-safety", dest="h_safety", type=float,
        default=0.0,
        required=False, 
        help="Estimate of max. deflection of surface.")
        
    parser.add_argument(
        "--gravity", dest="gravity", type=float,
        default=9.8062,
        required=False,
        help="Acceleration due to gravity {g=9.8062}.")

    opts = parser.parse_args()

    mesh = nc.Dataset(opts.mpas_file, "r+")

    step = est(opts, mesh)

    if ("dt_btr_est" not in mesh.variables.keys()):
        mesh.createVariable("dt_btr_est", "f4", ("nCells"))

    mesh["dt_btr_est"][:] = step
    
    mesh.close()


