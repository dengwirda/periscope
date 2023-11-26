
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

from map import maptocell

# SWE test cases for linear wave problems
# Authors: Darren Engwirda

def init(name, save, rsph, _ics, case, xmid, ymid, hmag):

#------------------------------------ load an MPAS mesh file

    print("Loading the mesh file...")

    if (rsph <= 0.): rsph= None

    mesh = load_mesh(name, rsph)   
    rsph = mesh.rsph

#------------------------------------ build TRSK matrix op's

    print("Forming coefficients...")

    trsk = trsk_mats(mesh)

#------------------------------------ compute test-case IC's

    xmid = xmid * np.pi / 180.0
    ymid = ymid * np.pi / 180.0

    if (case <= 0):
        ValueError("Unsupported test-case.")

    if (case == 1):
        wav1(name, save, rsph, mesh, trsk, xmid, ymid, hmag)
        
    if (case == 2):
        wav2(name, save, rsph, mesh, trsk, xmid, ymid, hmag)

    if (case == 3):
        tsu1(name, save, rsph, mesh, trsk, xmid, ymid, hmag)
        
    if (case == 4):
        tsu2(name, save, rsph, mesh, trsk, _ics)

    if (case >= 5):
        ValueError("Unsupported test-case.")

    return


def wav1(name, save, rsph, mesh, trsk, xmid, ymid, hmag):

#-- simple isolated gravity-wave test-case

    erot = 7.292E-05            # Earth's omega
    grav = 9.80616              # gravity

    uu_edge = np.zeros(mesh.edge.size, dtype=np.float64)

    hh_cell = hmag * np.exp(
            - 100. * (mesh.cell.xlon - xmid) ** 2 + \
            - 100. * (mesh.cell.ylat - ymid) ** 2 ) \
            + 500.0  # + 0.1

    zb_cell = np.zeros(mesh.cell.size, dtype=np.float64)

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

    hh_dual = trsk.dual_kite_sums * hh_cell
    hh_dual/= mesh.vert.area

    init["hh_dual"] = (
        ("Time", "nVertices", "nVertLevels"),
        np.reshape(hh_dual, (1, mesh.vert.size, 1)))

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

    return
    
    
def wav2(name, save, rsph, mesh, trsk, xmid, ymid, hmag):

#-- simple isolated gravity-wave test-case

#-- reduced gravity version

    erot = 7.292E-05            # Earth's omega
    grav = 9.80616              # gravity
    
    grav = grav / 100.          # reduced gravity

    uu_edge = np.zeros(mesh.edge.size, dtype=np.float64)

    hh_cell = hmag * np.exp(
            - 100. * (mesh.cell.xlon - xmid) ** 2 + \
            - 100. * (mesh.cell.ylat - ymid) ** 2 ) \
            + 500.0

    zb_cell = np.zeros(mesh.cell.size, dtype=np.float64)

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

    hh_dual = trsk.dual_kite_sums * hh_cell
    hh_dual/= mesh.vert.area

    init["hh_dual"] = (
        ("Time", "nVertices", "nVertLevels"),
        np.reshape(hh_dual, (1, mesh.vert.size, 1)))

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

    return


def tsu1(name, save, rsph, mesh, trsk, xmid, ymid, hmag):

#-- earth-topography tsunami-wave test-case

    erot = 7.292E-05            # Earth's omega
    grav = 9.80616              # gravity
    
    data = xarray.open_dataset(name)

    if ("bed_elevation" not in data.variables.keys()):
        raise ValueError("Elevation data not found.")

    if ("ocn_cover" not in data.variables.keys()):
        raise ValueError("Ocn.-mask data not found.")

    if ("ice_thickness" not in data.variables.keys()):
        raise ValueError("Elevation data not found.")

    oc_mask = np.asarray(
        data["ocn_cover"][:], dtype=np.float32)

    zb_cell = np.asarray(
        data["bed_elevation"][:], dtype=np.float64)
    zb_cell+= np.asarray(
        data["ice_thickness"][:], dtype=np.float64)

    zb_cell[oc_mask >= 0.375] = \
        np.minimum(-1.0, zb_cell[oc_mask >= 0.375])

    uu_edge = np.zeros(mesh.edge.size, dtype=np.float64)

    """
    hh_cell = -zb_cell + 1.00 * np.exp( -1. * (
            100.0 * (mesh.cell.xlon - xmid) ** 2 + \
            100.0 * (mesh.cell.ylat - ymid) ** 2
            ) ** 1 )
    """

    """
    hh_cell = -zb_cell + hmag * np.exp( -1. * (
            250.0 * (mesh.cell.xlon - xmid) ** 2 + \
            250.0 * (mesh.cell.ylat - ymid) ** 2
            ) ** 4 )
    """

    """
    hh_cell = -zb_cell + hmag * np.exp( -1. * (
            25000 * (mesh.cell.xlon - xmid) ** 2 + \
            250.0 * (mesh.cell.ylat - ymid) ** 2
            ) ** 4 )
    """

    hh_cell = -zb_cell + hmag * np.exp( -1. * (
            250.0 * (mesh.cell.xlon - xmid) ** 2 + \
            50000 * (mesh.cell.ylat - ymid) ** 2
            ) ** 4 )
    
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

    init["is_mask"] = (("nCells"), oc_mask<0.375)

    print(init)

    init.to_netcdf(save, format="NETCDF4")

    return
    
    
def tsu2(name, save, rsph, mesh, trsk, _ics):

#-- earth-topography tsunami-wave test-case w obs 

    erot = 7.292E-05            # Earth's omega
    grav = 9.80616              # gravity
    
    data = xarray.open_dataset(name)

    if ("bed_elevation" not in data.variables.keys()):
        raise ValueError("Elevation data not found.")

    if ("ice_thickness" not in data.variables.keys()):
        raise ValueError("Elevation data not found.")

    zb_cell = np.asarray(
        data["bed_elevation"][:], dtype=np.float64)
    zb_cell+= np.asarray(
        data["ice_thickness"][:], dtype=np.float64)

    zb_cell = np.minimum(-1., zb_cell)
    
    uu_edge = np.zeros(mesh.edge.size, dtype=np.float64)

    dz_data = np.loadtxt(_ics)

    mask = dz_data[:, 0] > 0.
    xlon = dz_data[mask, 1] * np.pi / 180.
    ylat = dz_data[mask, 2] * np.pi / 180.
    dlev = dz_data[mask, 3]

    zt_cell = maptocell(mesh, xlon, ylat, dlev)

    hh_cell = np.maximum(1., zt_cell - zb_cell)
    
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
        required=True, help="Test case number (1-3).")

    parser.add_argument(
        "--xydz-file", dest="xydz_file", type=str,
        required=False, help="Path to user _dz file.")

    parser.add_argument(
        "--radius", dest="radius", type=float,
        default=0., required=False, 
        help="Value of sphere_radius; zero to use mesh data.")

    parser.add_argument(
        "--wave-xmid", dest="wave_xmid", type=float,
        default=+180.,
        required=False,
        help="Centre of wave in lon. direction [deg].")

    parser.add_argument(
        "--wave-ymid", dest="wave_ymid", type=float,
        default=+0.00,
        required=False,
        help="Centre of wave in lat. direction [deg].")

    parser.add_argument(
        "--wave-size", dest="wave_size", type=float,
        default=+5.00,
        required=False,
        help="Magnitude of fluid surface deflection [m].")

    args = parser.parse_args()

    init(name=args.mesh_file,
         save=args.init_file,
         rsph=args.radius,
         _ics=args.xydz_file,
         case=args.test_case,
         xmid=args.wave_xmid,
         ymid=args.wave_ymid,
         hmag=args.wave_size)
