
import time
import sys
import os
import numpy as np
from scipy.sparse.linalg import gcrotmk
from scipy.interpolate import RectBivariateSpline

import xarray
import argparse

sys.path.insert(
    1, os.path.join(sys.path[0], ".."))

from stb import strtobool

from msh import load_mesh, cell_quad, dual_quad
from ops import operators

#-- BTR model config. using ERA5 forcing
#-- Authors: Darren Engwirda

def init(name, save, era5, back, ramp, deep, rsph=0.E+0):

#------------------------------------ load an MPAS mesh file

    print("Loading the mesh file...")

    if (rsph <= 0.): rsph= None

    mesh = load_mesh(name, rsph)
    rsph = mesh.rsph
    
#------------------------------------ build TRSK matrix op's

    print("Building coefficients...")

    mats = operators(mesh)

    grav = 9.81 * (1. - 0.08)   # gravity + SAL
    erot = 7.292E-05            # Earth's omega
    orho = 1027.                # seawater density
    arho = 1.225                # atmosph. density
    
    base = xarray.open_dataset(name)

    uu_edge = np.zeros(
        mesh.edge.size, dtype=np.float64)
    
    zb_cell = np.asarray(
        base["bed_elevation"][:], dtype=np.float32)
    
    zb_cell = np.minimum(2.0, zb_cell)
    
    # smooth at grid-scale
    zb_dual = mats.dual_kite_sums * zb_cell
    zb_dual/= mesh.vert.area
    z2_cell = mats.cell_kite_sums * zb_dual
    z2_cell/= mesh.cell.area
    zb_cell = +0.5 * (zb_cell + z2_cell)
   
    # set limits on depths 
    zb_cell = np.maximum(-deep, zb_cell)

    zt_cell = +0.0 * zb_cell
    hh_cell = np.maximum(+0.00, zt_cell - zb_cell)

    # free slip boundaries
    bc_slip = 0.875 * \
        np.ones((mesh.edge.size), dtype=np.float32)

    print("Build long term means...")

    fdat = xarray.open_dataset(back)
    pbar = np.asarray(fdat[ "sp"][:])
    ubar = np.asarray(fdat["u10"][:])
    vbar = np.asarray(fdat["v10"][:])

    # time averages, careful with fp. truncation
    pbar = np.asarray(np.mean(
        pbar, axis=0, dtype=np.float64), dtype=np.float32)
    ubar = np.asarray(np.mean(
        ubar, axis=0, dtype=np.float64), dtype=np.float32)
    vbar = np.asarray(np.mean(
        vbar, axis=0, dtype=np.float64), dtype=np.float32)

    print("Interpolating forcing...")

    fdat = xarray.open_dataset(era5)
    xlon = np.asarray(fdat["longitude"][:]) * np.pi / 180.
    ylat = np.asarray(fdat[ "latitude"][:]) * np.pi / 180.
    patm = np.asarray(fdat[ "sp"][:])
    ux10 = np.asarray(fdat["u10"][:])
    uy10 = np.asarray(fdat["v10"][:])

    ylat = np.flip(ylat, axis=0)
    patm = np.flip(patm, axis=1)
    pbar = np.flip(pbar, axis=0)
    ux10 = np.flip(ux10, axis=1)
    uy10 = np.flip(uy10, axis=1)

    patm-= pbar  # subtract long-term mean p_atm

    nfrc = patm.shape[0]
    xx_time = np.linspace(0., nfrc * 3600., nfrc)  # ERA5 hr

    # initial linear ramp
    rfac = np.minimum(
        1., xx_time / ramp / 24. / 60. / 60.)

    xi_cell = np.zeros(
        (nfrc, mesh.cell.size), dtype=np.float32)
    Tu_edge = np.zeros(
        (nfrc, mesh.edge.size), dtype=np.float32)
    Tu_curl = np.zeros(
        (nfrc, mesh.vert.size), dtype=np.float32)

    for step in range(nfrc):
        # interp. p_atm
        ifun = RectBivariateSpline(
            ylat, xlon, patm[step, :, :])
        ps_cell = \
            ifun.ev(mesh.cell.ylat, mesh.cell.xlon)
        ps_vert = \
            ifun.ev(mesh.vert.ylat, mesh.vert.xlon)

        # 1. / rho_ocn * (p_atm - p_atm_mean)
        xi_cell[step, :] = \
            cell_quad(mesh, ps_cell, ps_vert) / orho
  
        # interp. u_10m
        ifun = RectBivariateSpline(
            ylat, xlon, ux10[step, :, :])
        ux_edge = \
            ifun.ev(mesh.edge.ylat, mesh.edge.xlon)
        ux_vert = \
            ifun.ev(mesh.vert.ylat, mesh.vert.xlon)

        ifun = RectBivariateSpline(
            ylat, xlon, uy10[step, :, :])
        uy_edge = \
            ifun.ev(mesh.edge.ylat, mesh.edge.xlon)
        uy_vert = \
            ifun.ev(mesh.vert.ylat, mesh.vert.xlon)

        # Simpson's rule
        ux_edge = 4. / 6. * ux_edge + \
            1. / 6. * mats.edge_vert_sums * ux_vert
        uy_edge = 4. / 6. * uy_edge + \
            1. / 6. * mats.edge_vert_sums * uy_vert

        # rho_atm / rho_ocn * cw * |w| * w_i
        uu_norm = \
            np.sqrt(ux_edge ** 2 + uy_edge ** 2)

        # Garratt's form
        cw_edge = 1. / 1000. * \
            (0.75 + 0.067 * uu_norm) * arho / orho

        Tx_edge = cw_edge * uu_norm * ux_edge
        Ty_edge = cw_edge * uu_norm * uy_edge

        Tu_edge[step, :] = mesh.edge.cos_ * Tx_edge \
                         + mesh.edge.sin_ * Ty_edge

        xi_cell[step, :]*= rfac[step]
        Tu_edge[step, :]*= rfac[step]

        Tu_curl[step, :] = \
            mats.dual_curl_sums * Tu_edge[step, :]
        Tu_curl[step, :]/= mesh.vert.area

#-- inject mesh with IC.'s and write to MPAS-ish NetCDF file

    print("Output written to:", save)

    init = xarray.open_dataset(name)
    init.attrs.update({"sphere_radius": mesh.rsph})
    init.attrs.update({"config_gravity": grav})
    init["xCell"] = (("nCells"), mesh.cell.xpos)
    init["yCell"] = (("nCells"), mesh.cell.ypos)
    init["zCell"] = (("nCells"), mesh.cell.zpos)
    init["areaCell"] = (("nCells"), mesh.cell.area)

    init["xEdge"] = (("nEdges"), mesh.edge.xpos)
    init["yEdge"] = (("nEdges"), mesh.edge.ypos)
    init["zEdge"] = (("nEdges"), mesh.edge.zpos)
    init["dvEdge"] = (("nEdges"), mesh.edge.vlen)
    init["dcEdge"] = (("nEdges"), mesh.edge.clen)

    init["xVertex"] = (("nVertices"), mesh.vert.xpos)
    init["yVertex"] = (("nVertices"), mesh.vert.ypos)
    init["zVertex"] = (("nVertices"), mesh.vert.zpos)
    init["areaTriangle"] = (("nVertices"), mesh.vert.area)
    init["kiteAreasOnVertex"] = (
        ("nVertices", "vertexDegree"), mesh.vert.kite)

    init["bc_slip"] = (("nEdges"), bc_slip)

    init["hh_cell"] = (
        ("Time", "nCells", "nVertLevels"),
        np.reshape(hh_cell, (1, mesh.cell.size, 1)))
    
    init["zb_cell"] = (("nCells"), zb_cell)

    init["uu_edge"] = (
        ("Time", "nEdges", "nVertLevels"),
        np.reshape(uu_edge, (1, mesh.edge.size, 1)))
    
    init["ff_cell"] = (("nCells"),
        2.00E+00 * erot * np.sin(mesh.cell.ylat))
    init["ff_edge"] = (("nEdges"),
        2.00E+00 * erot * np.sin(mesh.edge.ylat))
    init["ff_vert"] = (("nVertices"),
        2.00E+00 * erot * np.sin(mesh.vert.ylat))

    print(init)

    init.to_netcdf(save, format="NETCDF4")

    path, file = os.path.split(save)
    sfrc = os.path.join(path, "frc_" + file)
    
    forc = xarray.open_dataset(name)
    forc.attrs.update({"sphere_radius": mesh.rsph})
    forc.attrs.update({"config_gravity": grav})
    forc["xCell"] = (("nCells"), mesh.cell.xpos)
    forc["yCell"] = (("nCells"), mesh.cell.ypos)
    forc["zCell"] = (("nCells"), mesh.cell.zpos)
    forc["areaCell"] = (("nCells"), mesh.cell.area)

    forc["xEdge"] = (("nEdges"), mesh.edge.xpos)
    forc["yEdge"] = (("nEdges"), mesh.edge.ypos)
    forc["zEdge"] = (("nEdges"), mesh.edge.zpos)
    forc["dvEdge"] = (("nEdges"), mesh.edge.vlen)
    forc["dcEdge"] = (("nEdges"), mesh.edge.clen)

    forc["xVertex"] = (("nVertices"), mesh.vert.xpos)
    forc["yVertex"] = (("nVertices"), mesh.vert.ypos)
    forc["zVertex"] = (("nVertices"), mesh.vert.zpos)
    forc["areaTriangle"] = (("nVertices"), mesh.vert.area)
    forc["kiteAreasOnVertex"] = (
        ("nVertices", "vertexDegree"), mesh.vert.kite)

    forc["xx_time"] = ("Time", xx_time)

    forc["Xi_cell"] = (
        ("Time", "nCells", "nVertLevels"),
        np.reshape(xi_cell, (nfrc, mesh.cell.size, 1)))

    forc["Tu_edge"] = (
        ("Time", "nEdges", "nVertLevels"),
        np.reshape(Tu_edge, (nfrc, mesh.edge.size, 1)))
    forc["Tu_curl"] = (
        ("Time", "nVertices", "nVertLevels"),
        np.reshape(Tu_curl, (nfrc, mesh.vert.size, 1)))

    print(forc)

    forc.to_netcdf(sfrc, format="NETCDF4")

    return
    
if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "--mesh-file", dest="mesh_file", type=str,
        required=True, help="Path to user mesh file.")

    parser.add_argument(
        "--init-file", dest="init_file", type=str,
        required=True, help="IC's filename to write.")

    parser.add_argument(
        "--era5-file", dest="era5_file", type=str,
        required=True, help="ERA5 forcing data file.")

    parser.add_argument(
        "--mean-file", dest="mean_file", type=str,
        required=True, help="ERA5 history data file.")

    parser.add_argument(
        "--ramp-days", dest="ramp_days", type=float,
        required=True, help="Length of initial ramp.")

    parser.add_argument(
        "--max-depth", dest="max_depth", type=float,
        required=True, help="Limit on max ocn depth.")

    parser.add_argument(
        "--radius", dest="radius", type=float,
        default=0., required=False, 
        help="Value of sphere_radius; zero to use mesh data.")

    args = parser.parse_args()

    init(name=args.mesh_file,
         save=args.init_file,
         era5=args.era5_file,
         back=args.mean_file,
         ramp=args.ramp_days, 
         deep=args.max_depth,
         rsph=args.radius)

