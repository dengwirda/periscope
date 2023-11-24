
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

#-- Investigation of wakes around obstructions
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
    f = 1.0E-04

    Ls = 50.
    u0 = 5.
    h0 = 25.

    uu_edge = u0 * np.ones(
        (mesh.edge.size), dtype=np.float64)
    uu_edge*= mesh.edge.cos_

    us_edge = u0 * np.ones(
        (mesh.edge.size), dtype=np.float64)
    us_edge*= mesh.edge.cos_

    hh_cell = h0 * np.ones(
        (mesh.cell.size), dtype=np.float64)
        
    zs_cell = 0. * np.ones(
        (mesh.cell.size), dtype=np.float64)
    
    zb_cell = np.zeros(hh_cell.shape, dtype=np.float64)
   
    rr_edge = np.sqrt(
        mesh.edge.ypos ** 2 + mesh.edge.zpos ** 2)
    rr_cell = np.sqrt(
        mesh.cell.ypos ** 2 + mesh.cell.zpos ** 2)
    
    t_relax = Ls / np.sqrt(grav * h0)
    
    ur_edge = np.minimum(1., 
              np.maximum(0., 
              rr_edge - 500. - .5 * Ls) / Ls) / t_relax
        
    hr_cell = np.minimum(1., 
              np.maximum(0., 
              rr_cell - 500. - .5 * Ls) / Ls) / t_relax
    
    
    # we need to have real tendencies on the bnd edges here...
    # so apply sponge after the BCs have been masked?
    
    # we also need to turn the other tendencies off in the sponge?
    # or have some way to specify an open BC? which would be a pain!
    # - maybe not that bad: is_open[nCells]?
    # could have the ramps on [0,1] and compute a local sqrt(g*h) factor that multiples the ramp?
    # have to just try and see?
    
    
    
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
   
    init["zs_cell"] = (
        ("Time", "nCells", "nVertLevels"),
        np.reshape(zs_cell, (1, mesh.cell.size, 1)))
    init["hr_cell"] = (
        ("Time", "nCells", "nVertLevels"),
        np.reshape(hr_cell, (1, mesh.cell.size, 1)))
        
    init["us_edge"] = (
        ("Time", "nEdges", "nVertLevels"),
        np.reshape(us_edge, (1, mesh.edge.size, 1)))
    init["ur_edge"] = (
        ("Time", "nEdges", "nVertLevels"),
        np.reshape(ur_edge, (1, mesh.edge.size, 1)))

    init["ff_cell"] = (("nCells"), 
        f * np.ones(mesh.cell.size))
    init["ff_edge"] = (("nEdges"),
        f * np.ones(mesh.edge.size))
    init["ff_vert"] = (("nVertices"),
        f * np.ones(mesh.vert.size))

    print(init)

    init.to_netcdf(save, format="NETCDF4")

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

