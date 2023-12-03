
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
from ops import trsk_mats

#-- Open BCs and wakes around obstructions
#-- Authors: Darren Engwirda

def init(name, save, rsph=0.E+0):

#------------------------------------ load an MPAS mesh file

    print("Loading the mesh file...")

    if (rsph <= 0.): rsph= None

    mesh = load_mesh(name, rsph)
    rsph = mesh.rsph
    
#------------------------------------ build TRSK matrix op's

    print("Building coefficients...")

    trsk = trsk_mats(mesh)

#-- requires rsph = 50000. to scale disk to 1000m

    grav = 9.81                 # gravity
    f = 1.0E-04                 # coriolis
    u0 = 3.0                    # velocity
    h0 = 15.                    # depth
    z0 = 0.0                    # hill position
    y0 =-250.

    uu_edge = u0 * np.ones(
        (mesh.edge.size), dtype=np.float64)
    uu_edge*= mesh.edge.cos_
    
    hh_cell = h0 * np.ones(
        (mesh.cell.size), dtype=np.float64)
    
    zb_cell = np.zeros(hh_cell.shape, dtype=np.float64)
    
    zb_cell+= 0.95 * h0 * np.exp(
        -.00025 * (mesh.cell.ypos - y0)**2 
        -.00025 * (mesh.cell.zpos - z0)**2
            )
 
    hh_cell = hh_cell - zb_cell
    
    # external signal at OBCs
    uE_edge = u0 * np.ones(
        (mesh.edge.size), dtype=np.float64)
    uE_edge*= mesh.edge.cos_
    
    hE_edge = h0 * np.ones(
        (mesh.edge.size), dtype=np.float64)
    
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
        f * np.ones(mesh.cell.size))
    init["ff_edge"] = (("nEdges"),
        f * np.ones(mesh.edge.size))
    init["ff_vert"] = (("nVertices"),
        f * np.ones(mesh.vert.size))

    init["is_open"] = (("nEdges"), mesh.edge.mask>0)

    print(init)

    init.to_netcdf(save, format="NETCDF4")
    
    path, file = os.path.split(save)
    save = os.path.join(path, "frc_" + file)
    
    forc = xarray.Dataset()
    forc["hE_edge"] = (
        ("Time", "nEdges", "nVertLevels"),
        np.reshape(hE_edge, (1, mesh.edge.size, 1)))
        
    forc["uE_edge"] = (
        ("Time", "nEdges", "nVertLevels"),
        np.reshape(uE_edge, (1, mesh.edge.size, 1)))

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

