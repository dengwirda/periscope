
import time
import sys
import os
import numpy as np
from scipy.sparse.linalg import gcrotmk
from scipy.integrate import quadrature

import xarray
import argparse

sys.path.insert(
    1, os.path.join(sys.path[0], '..'))

from stb import strtobool

from msh import load_mesh, cell_quad, dual_quad
from ops import trsk_mats

#-- Spin-down from random initial vorticity 
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

#-- build a stream-function, velocity field + thickness IC's

    erot = 7.292E-05            # Earth's omega
    grav = 9.80616              # gravity
   
    grav = grav / 100.          # reduced gravity

    print("Computing streamfunction...")

    rv_dual = np.random.rand(mesh.vert.size)
    
    # smooth at grid-scale a little
    rv_cell = trsk.cell_kite_sums * rv_dual
    rv_cell/= mesh.cell.area
    
    rv_dual = trsk.dual_kite_sums * rv_cell
    rv_dual/= mesh.vert.area
    
    rv_cell = trsk.cell_kite_sums * rv_dual
    rv_cell/= mesh.cell.area
    
    rv_dual = trsk.dual_kite_sums * rv_cell
    rv_dual/= mesh.vert.area
    
    """
    actually... doing it this way induces a grid-scale mode!
    
    rv_cell = trsk.cell_kite_sums * rv_dual
    rv_cell/= mesh.cell.area
    
    rv_cell*= 2.5E+06 * (1. / 2.) ** 2
    rv_cell-= np.mean(rv_cell)
    
    sf_cell, info = gcrotmk(
        trsk.cell_flux_sums * 
        trsk.edge_grad_norm, rv_cell, 
            tol=1.E-08, atol=1.E-08, m=50, k=25)
            
    sf_vert = trsk.dual_kite_sums * sf_cell
    sf_vert/= mesh.vert.area
    """

    rv_dual*= 2.5E+06 * (1. / 2.) ** 2
    rv_dual-= np.mean(rv_dual)

    sf_vert, info = gcrotmk(
        trsk.dual_flux_sums * 
        trsk.edge_grad_perp, rv_dual, 
            tol=1.E-08, atol=1.E-08, m=50, k=25)
            
    sf_cell = trsk.cell_kite_sums * sf_vert
    sf_cell/= mesh.cell.area

    print("Computing velocity field...")

    uu_edge = trsk.edge_grad_perp * sf_vert * -1.
    vv_edge = trsk.edge_grad_norm * sf_cell * -1.

    hh_cell = 5000. * np.ones(
        (mesh.cell.size), dtype=np.float64)
    
    zb_cell = np.zeros(hh_cell.shape, dtype=np.float64)

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

    init["streamfunction"] = (("nVertices"), sf_vert)
    init["rv_dual"] = (
        ("nVertices"),
        (trsk.dual_curl_sums * uu_edge) / mesh.vert.area)

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

    args = parser.parse_args()

    init(name=args.mesh_file,
         save=args.init_file,
         rsph=args.radius)
