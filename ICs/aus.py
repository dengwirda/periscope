
import time
import sys
import os
import numpy as np
from scipy.sparse.linalg import gcrotmk
from scipy.integrate import quadrature

import xarray
import argparse

sys.path.insert(
    1, os.path.join(sys.path[0], ".."))

from stb import strtobool

from msh import load_mesh, cell_quad, dual_quad
from ops import operators

#-- BTR model for King Sound
#-- Authors: Darren Engwirda

def init(name, save, rsph=0.E+0):

#------------------------------------ load an MPAS mesh file

    print("Loading the mesh file...")

    if (rsph <= 0.): rsph= None

    mesh = load_mesh(name, rsph)
    rsph = mesh.rsph
    
#------------------------------------ build TRSK matrix op's

    print("Building coefficients...")

    mats = operators(mesh)


    grav = 9.81                 # gravity
    erot = 7.292E-05            # Earth's omega
    u0 = 2.0                    # velocity
    N = 24 * 3

    base = xarray.open_dataset(name)

    uu_edge = u0 * np.ones(
        (mesh.edge.size), dtype=np.float64)
    uu_edge*= mesh.edge.cos_
    
    zb_cell = np.asarray(
        base["bed_elevation"][:], dtype=np.float32)
    
    zb_cell = np.minimum(+10., +1 * zb_cell)
    
   #zb_cell+= 5.  # just to test wet/dry
    
    # smooth at grid-scale
    zb_dual = mats.dual_kite_sums * zb_cell
    zb_dual/= mesh.vert.area
    zb_cell = mats.cell_kite_sums * zb_dual
    zb_cell/= mesh.cell.area
    
    zb_dual = mats.dual_kite_sums * zb_cell
    zb_dual/= mesh.vert.area
    zb_cell = mats.cell_kite_sums * zb_dual
    zb_cell/= mesh.cell.area
    
    hh_cell = np.maximum(+0.0, -1 * zb_cell)
    
    # external signal at OBCs
    uI_edge = u0 * np.ones(
        (mesh.edge.size), dtype=np.float64)
    uI_edge*= mesh.edge.cos_
    
    hI_edge = mats.edge_cell_sums * hh_cell

    # 24 hrs in sec
    xx_time = np.linspace(0., 24. * 60. * 60., N)

    uE_edge = np.zeros(
        (N, mesh.edge.size), dtype=np.float32)
    hE_edge = np.zeros(
        (N, mesh.edge.size), dtype=np.float32)
    for it in range(N):
        T = 12. * 60. * 60.  # period
        hE_edge[it, :] = hI_edge
        uE_edge[it, :] = \
            np.sin(2. * np.pi / T * xx_time[it]) * uI_edge
        
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
        
    # slip boundary condition?
    bc_slip = 1. * np.ones(mesh.edge.size, dtype=np.float64)
        
    init["bc_slip"] = (("nEdges"), bc_slip)

    print(init)

    init.to_netcdf(save, format="NETCDF4")
    
    path, file = os.path.split(save)
    save = os.path.join(path, "frc_" + file)
    
    forc = xarray.Dataset()
    forc["xx_time"] = ("Time", xx_time)
    
    forc["hE_edge"] = (
        ("Time", "nEdges", "nVertLevels"),
        np.reshape(hE_edge, (N, mesh.edge.size, 1)))
        
    forc["uE_edge"] = (
        ("Time", "nEdges", "nVertLevels"),
        np.reshape(uE_edge, (N, mesh.edge.size, 1)))

    print(forc)

    forc.to_netcdf(save, format="NETCDF4")

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
        "--radius", dest="radius", type=float,
        default=0., required=False, 
        help="Value of sphere_radius; zero to use mesh data.")

    args = parser.parse_args()

    init(name=args.mesh_file,
         save=args.init_file,
         rsph=args.radius)

