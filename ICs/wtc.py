
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

# SWE test cases due to Williamson et al
# Authors: Darren Engwirda

def init(name, save, rsph, case):

#-- Williamson, D. L., et al. (1992) A Standard Test Set for
#-- Numerical Approximations to the Shallow Water Equations
#-- in Spherical Geometry, J. Comp. Phys., 102, pp. 211--224

#------------------------------------ load an MPAS mesh file

    print("Loading the mesh file...")

    if (rsph <= 0.): rsph= None

    mesh = load_mesh(name, rsph)   
    rsph = mesh.rsph
    
#------------------------------------ build TRSK matrix op's

    print("Building coefficients...")

    trsk = trsk_mats(mesh)

#------------------------------------ compute test-case IC's

    if (case <= 1):
        ValueError("Unsupported test-case.")

    if (case == 2): 
        wtc2(name, save, rsph, mesh, trsk)

    if (case == 22): 
        wtcb(name, save, rsph, mesh, trsk)
        
    if (case == 3):
        ValueError("Unsupported test-case.")
    
    if (case == 4): 
        ValueError("Unsupported test-case.")

    if (case == 5): 
        wtc5(name, save, rsph, mesh, trsk)
        
    if (case == 55): 
        wtcd(name, save, rsph, mesh, trsk)
        
    if (case == 6): 
        wtc6(name, save, rsph, mesh, trsk)
        
    if (case >= 7): 
        ValueError("Unsupported test-case.")

    return


def wtc2(name, save, rsph, mesh, trsk):

#-- build a stream-function, velocity field + thickness IC's

    erot = 7.292E-05            # Earth's omega
    grav = 9.80616              # gravity

    umag = 2.0 * np.pi * rsph / (12.0 * 86400.0)
    g_h0 = 29400.0

    sf_vert = rsph * umag * np.sin(mesh.vert.ylat) * -1.

    uu_edge = trsk.edge_grad_perp * sf_vert * -1.
    vv_edge = trsk.edge_lsqr_perp * uu_edge
    
    ke_edge = 0.5 * (uu_edge ** 2 + vv_edge ** 2)
   #ke_edge = uu_edge ** 2
    ke_cell = trsk.cell_wing_sums * ke_edge
    ke_cell/= mesh.cell.area

    fh_vert = (g_h0 - (
        rsph * erot * umag + 0.5 * umag ** 2) * \
        np.sin(mesh.vert.ylat) ** 2) / (grav)

    fh_cell = (g_h0 - (
        rsph * erot * umag + 0.5 * umag ** 2) * \
        np.sin(mesh.cell.ylat) ** 2) / (grav)

    hh_cell = cell_quad(mesh, fh_cell, fh_vert)

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

    init["ke_cell"] = (
        ("Time", "nCells", "nVertLevels"),
        np.reshape(ke_cell, (1, mesh.cell.size, 1)))

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


def wtcb(name, save, rsph, mesh, trsk):

#-- build a stream-function, velocity field + thickness IC's
#-- TC2 "thin", as per Peixoto

    erot = 7.292E-05            # Earth's omega
    grav = 9.80616              # gravity

    umag = 2.0 * np.pi * rsph / (12.0 * 86400.0)
    g_h0 = 29400.0

    sf_vert = rsph * umag * np.sin(mesh.vert.ylat) * -1.

    uu_edge = trsk.edge_grad_perp * (sf_vert) * -1.

    fh_vert = (g_h0 - (
        rsph * erot * umag + 0.5 * umag ** 2) * \
        np.sin(mesh.vert.ylat) ** 2) / (grav)

    fh_cell = (g_h0 - (
        rsph * erot * umag + 0.5 * umag ** 2) * \
        np.sin(mesh.cell.ylat) ** 2) / (grav)

    zb_cell = cell_quad(mesh, fh_cell, fh_vert)

    hh_cell = np.ones(zb_cell.shape, dtype=np.float64)

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


def wtc5(name, save, rsph, mesh, trsk):

#-- build a stream-function, velocity field + thickness IC's

    erot = 7.292E-05            # Earth's omega
    grav = 9.80616              # gravity
   
    umag = 20.0                 # m/s flow
    g_h0 = 5960.0 * grav

    xmid = 3.0 * np.pi / 2.0
    ymid = 1.0 * np.pi / 6.0
    rrad = 1.0 * np.pi / 9.0
    hs_0 = 2000.0

    sf_vert = rsph * umag * np.sin(mesh.vert.ylat) * -1.

    uu_edge = trsk.edge_grad_perp * (sf_vert) * -1.

    fh_vert = (g_h0 - (
        rsph * erot * umag + 0.5 * umag ** 2) * \
        np.sin(mesh.vert.ylat) ** 2) / (grav)

    fh_cell = (g_h0 - (
        rsph * erot * umag + 0.5 * umag ** 2) * \
        np.sin(mesh.cell.ylat) ** 2) / (grav)

    hh_cell = cell_quad(mesh, fh_cell, fh_vert)

    rr_vert = (mesh.vert.xlon - xmid) ** 2 + \
              (mesh.vert.ylat - ymid) ** 2
    rr_vert = np.sqrt(np.minimum(rrad ** 2, rr_vert))
    fz_vert = hs_0 * (1.0 - rr_vert / rrad)

    rr_cell = (mesh.cell.xlon - xmid) ** 2 + \
              (mesh.cell.ylat - ymid) ** 2
    rr_cell = np.sqrt(np.minimum(rrad ** 2, rr_cell))
    fz_cell = hs_0 * (1.0 - rr_cell / rrad) 

    zb_cell = cell_quad(mesh, fz_cell, fz_vert)
    
    hh_cell = hh_cell - zb_cell

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
    
    
def wtcd(name, save, rsph, mesh, trsk):

#-- build a stream-function, velocity field + thickness IC's

#-- reduced gravity version

    erot = 7.292E-05            # Earth's omega
    grav = 9.80616 / 100.       # gravity
   
    umag = 1.0                  # m/s flow
    g_h0 = 5960.0 * grav

    xmid = 3.0 * np.pi / 2.0
    ymid = 1.0 * np.pi / 6.0
    rrad = 1.0 * np.pi / 9.0
    hs_0 = 2000.0

    sf_vert = rsph * umag * np.sin(mesh.vert.ylat) * -1.

    uu_edge = trsk.edge_grad_perp * (sf_vert) * -1.

    fh_vert = (g_h0 - (
        rsph * erot * umag + 0.5 * umag ** 2) * \
        np.sin(mesh.vert.ylat) ** 2) / (grav)

    fh_cell = (g_h0 - (
        rsph * erot * umag + 0.5 * umag ** 2) * \
        np.sin(mesh.cell.ylat) ** 2) / (grav)

    hh_cell = cell_quad(mesh, fh_cell, fh_vert)

    rr_vert = (mesh.vert.xlon - xmid) ** 2 + \
              (mesh.vert.ylat - ymid) ** 2
    rr_vert = np.sqrt(np.minimum(rrad ** 2, rr_vert))
    fz_vert = hs_0 * (1.0 - rr_vert / rrad)

    rr_cell = (mesh.cell.xlon - xmid) ** 2 + \
              (mesh.cell.ylat - ymid) ** 2
    rr_cell = np.sqrt(np.minimum(rrad ** 2, rr_cell))
    fz_cell = hs_0 * (1.0 - rr_cell / rrad) 

    zb_cell = cell_quad(mesh, fz_cell, fz_vert)
    
    hh_cell = hh_cell - zb_cell

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
    
    
def wtc6(name, save, rsph, mesh, trsk):

#-- build a stream-function, velocity field + thickness IC's

    erot = 7.292E-05            # Earth's omega
    grav = 9.80616              # gravity

    g_h0 = 8000. * grav

    Rfac = 4                    # wave-number
    wfac = 7.848E-06
    Kfac = 7.848E-06

    print("Computing streamfunction...")

    sf_vert = rsph ** 2 * (
        - wfac * np.sin(mesh.vert.ylat) +
        + Kfac * np.cos(mesh.vert.ylat) ** Rfac *
                 np.sin(mesh.vert.ylat) *
                 np.cos(mesh.vert.xlon * Rfac)
        )
              
    print("--> done: vert!")

    sf_cell = rsph ** 2 * (
        - wfac * np.sin(mesh.cell.ylat) +
        + Kfac * np.cos(mesh.cell.ylat) ** Rfac *
                 np.sin(mesh.cell.ylat) *
                 np.cos(mesh.cell.xlon * Rfac)
        )

    print("--> done: cell!")

    print("Computing velocity field...")

    uu_edge = trsk.edge_grad_perp * sf_vert * -1.

   #vv_edge = trsk.edge_flux_perp * uu_edge * -1.
    vv_edge = trsk.edge_grad_norm * sf_cell * -1.

    du_cell = trsk.cell_flux_sums * uu_edge

    print("--> max(abs(unrm)):", np.max(uu_edge))
    print("--> sum(div(unrm)):", np.sum(du_cell))

#-- invert du/dt = 0 condition for layer thickness:
#-- -g * del^2 h = div[ (curl(u)+f) * u_perp + grad(K) ]
#-- leads to a h which is in discrete balance

    print("Computing flow thickness...")

    ff_edge = 2.0 * erot * np.sin(mesh.edge.ylat)
    
    ke_edge = 0.5 * (uu_edge ** 2 + 
                     vv_edge ** 2 )
    ke_cell = trsk.cell_wing_sums * ke_edge
    ke_cell/= mesh.cell.area
    
    rv_edge = trsk.quad_curl_sums * uu_edge
    rv_edge/= mesh.quad.area
    av_edge = ff_edge + rv_edge  # curl(u) + f
    
    rh_edge = trsk.cell_flux_sums * (
        av_edge * vv_edge + trsk.edge_grad_norm * ke_cell)
    rh_edge = rh_edge * -1.0 / grav

    rh_edge = rh_edge - np.mean(rh_edge) # INT rhs must be 0
    
    ttic = time.time()
    hh_cell, info = gcrotmk(
        trsk.cell_flux_sums * 
        trsk.edge_grad_norm, rh_edge, 
            tol=1.E-08, atol=1.E-08, m=50, k=25)
    ttoc = time.time()
   #print(ttoc - ttic)    

    if (info != +0): raise Exception("Did not converge!")

    hh_cell-= np.min(hh_cell)
    
    hh_cell = g_h0 / grav + hh_cell
    
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
        "--test-case", dest="test_case", type=int,
        required=True, help="Test case number (2-6).")

    parser.add_argument(
        "--radius", dest="radius", type=float,
        default=0., required=False, 
        help="Value of sphere_radius; zero to use mesh data.")

    args = parser.parse_args()

    init(name=args.mesh_file,
         save=args.init_file,
         rsph=args.radius,
         case=args.test_case)
         
