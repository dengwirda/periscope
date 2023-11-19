
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

#-- Idealised vortex-pair test-cases
#-- Authors: Darren Engwirda

def init(name, save, rsph, case):

#------------------------------------ load an MPAS mesh file

    print("Loading the mesh file...")

    if (rsph <= 0.): rsph= None

    mesh = load_mesh(name, rsph)   
    rsph = mesh.rsph
    
#------------------------------------ build TRSK matrix op's

    print("Building coefficients...")

    trsk = trsk_mats(mesh)

#------------------------------------ compute test-case IC's

    if (case <= 0):
        ValueError("Unsupported test-case.")

    if (case == 1): 
        vtx1(name, save, rsph, mesh, trsk)
        
    if (case == 2): 
        vtx2(name, save, rsph, mesh, trsk)
        
    if (case >= 3): 
        ValueError("Unsupported test-case.")

    return


def vtx1(name, save, rsph, mesh, trsk):

#-- Merging (submesoscale) vortices:
#-- G. Roullet & T. Gaillard (2022): A Fast Monotone Discretization 
#-- of the Rotating Shallow Water Equations; JAMES.
#-- https://doi.org/10.1029/2021MS002663 

    f = 5.
    grav = 1.
    H = 1.
    h0 = 0.2
    sigma = 0.07
    d = 1.4 * sigma
    
    print("Computing flow thickness...")
    
    hh_cell = H + h0 * np.exp(
        -((mesh.cell.ypos-d) ** 2 + 
            mesh.cell.zpos ** 2) / (2. * sigma ** 2)) + \
                + h0 * np.exp(
        -((mesh.cell.ypos+d) ** 2 + 
            mesh.cell.zpos ** 2) / (2. * sigma ** 2))
            
    hh_vert = H + h0 * np.exp(
        -((mesh.vert.ypos-d) ** 2 + 
            mesh.vert.zpos ** 2) / (2. * sigma ** 2)) + \
                + h0 * np.exp(
        -((mesh.vert.ypos+d) ** 2 + 
            mesh.vert.zpos ** 2) / (2. * sigma ** 2))
    
    zb_cell = np.zeros(hh_cell.shape, dtype=np.float64)
        
    print("Computing velocity field...")

    uu_edge = -(grav / f) * trsk.edge_grad_perp * hh_vert

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

    init["rv_dual"] = (
        ("nVertices"),
        (trsk.dual_curl_sums * uu_edge) / mesh.vert.area)

    init["ff_cell"] = (("nCells"),
        f * np.ones(mesh.cell.size, dtype=np.float64))
    init["ff_edge"] = (("nEdges"),
        f * np.ones(mesh.edge.size, dtype=np.float64))
    init["ff_vert"] = (("nVertices"),
        f * np.ones(mesh.vert.size, dtype=np.float64))

    print(init)

    init.to_netcdf(save, format="NETCDF4")

    return


def vtx2(name, save, rsph, mesh, trsk):

#-- Dipole//wall interaction:
#-- G. Roullet & T. Gaillard (2022): A Fast Monotone Discretization 
#-- of the Rotating Shallow Water Equations; JAMES.
#-- https://doi.org/10.1029/2021MS002663 

    f = 5.
    grav = 1.
    H = 1.
    h0 = 0.15
    sigma = 0.1
    d = 1.1 * sigma
    
    print("Computing flow thickness...")
    
    hh_cell = H + h0 * np.exp(
        -((mesh.cell.ypos-d) ** 2 + 
            mesh.cell.zpos ** 2) / (2. * sigma ** 2)) + \
                - h0 * np.exp(
        -((mesh.cell.ypos+d) ** 2 + 
            mesh.cell.zpos ** 2) / (2. * sigma ** 2))
            
    hh_vert = H + h0 * np.exp(
        -((mesh.vert.ypos-d) ** 2 + 
            mesh.vert.zpos ** 2) / (2. * sigma ** 2)) + \
                - h0 * np.exp(
        -((mesh.vert.ypos+d) ** 2 + 
            mesh.vert.zpos ** 2) / (2. * sigma ** 2))
    
    zb_cell = np.zeros(hh_cell.shape, dtype=np.float64)
        
    print("Computing velocity field...")

    uu_edge = -(grav / f) * trsk.edge_grad_perp * hh_vert

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

    init["rv_dual"] = (
        ("nVertices"),
        (trsk.dual_curl_sums * uu_edge) / mesh.vert.area)

    init["ff_cell"] = (("nCells"),
        f * np.ones(mesh.cell.size, dtype=np.float64))
    init["ff_edge"] = (("nEdges"),
        f * np.ones(mesh.edge.size, dtype=np.float64))
    init["ff_vert"] = (("nVertices"),
        f * np.ones(mesh.vert.size, dtype=np.float64))

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
        "--test-case", dest="test_case", type=int,
        required=True, help="Test case number (1-2).")

    parser.add_argument(
        "--radius", dest="radius", type=float,
        default=0., required=False, 
        help="Value of sphere_radius; zero to use mesh data.")

    args = parser.parse_args()

    init(name=args.mesh_file,
         save=args.init_file,
         rsph=args.radius,
         case=args.test_case)
