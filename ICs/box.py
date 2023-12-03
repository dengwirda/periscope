
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

#-- Wind-driven gyre test-case: doi.org/10.1029/2022MS003594
#-- Authors: Darren Engwirda

def init(name, save, rsph=0.0, slip=1.0):

#------------------------------------ load an MPAS mesh file

    print("Loading the mesh file...")

    if (rsph <= 0.): rsph= None

    mesh = load_mesh(name, rsph)
    rsph = mesh.rsph
    
#------------------------------------ build TRSK matrix op's

    print("Building coefficients...")

    trsk = trsk_mats(mesh)

#-- build a stream-function, velocity field + thickness IC's

    erot = 7.292E-05            # Earth's omega
    grav = 0.02                 # reduced gravity

    Tau0 = -0.2
    rho0 = 1000.
    phi0 = 20. * np.pi / 180.
    dphi = 18. * np.pi / 180.

    uu_edge = np.zeros(
        (mesh.edge.size), dtype=np.float64)

    hh_cell = 500. * np.ones(
        (mesh.cell.size), dtype=np.float64)
    
    zb_cell = np.zeros(hh_cell.shape, dtype=np.float64)
    
    bc_slip = np.zeros(uu_edge.shape, dtype=np.float64)
    bc_slip[mesh.edge.mask] = slip
    
    Tu_edge = mesh.edge.cos_ * Tau0 / rho0 * \
        np.cos(np.pi * (mesh.edge.ylat - phi0) / dphi)
    
    Tv_edge = trsk.edge_lsqr_perp * Tu_edge
    
    Tm_edge = 0.5 * (Tu_edge ** 2 + Tv_edge ** 2)
    Tm_cell = trsk.cell_wing_sums * Tm_edge
    Tm_cell/= mesh.cell.area
    Tm_cell = np.sqrt(2. * Tm_cell)

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
        
    init["Tu_edge"] = (
        ("Time", "nEdges", "nVertLevels"),
        np.reshape(Tu_edge, (1, mesh.edge.size, 1)))
        
    init["Tu_norm"] = (
        ("Time", "nCells", "nVertLevels"),
        np.reshape(Tm_cell, (1, mesh.cell.size, 1)))

    init["bc_slip"] = (("nEdges"), bc_slip)

    init["ff_cell"] = (("nCells"),
        2.00E+00 * erot * np.sin(mesh.cell.ylat))
    init["ff_edge"] = (("nEdges"),
        2.00E+00 * erot * np.sin(mesh.edge.ylat))
    init["ff_vert"] = (("nVertices"),
        2.00E+00 * erot * np.sin(mesh.vert.ylat))
    
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
        
    parser.add_argument(
        "--wall-slip", dest="wall_slip", type=float,
        default=1., required=False, 
        help="Wall no-slip vs free-slip coefficient.")

    args = parser.parse_args()

    init(name=args.mesh_file,
         save=args.init_file,
         rsph=args.radius,
         slip=args.wall_slip)
