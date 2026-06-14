
import numpy as np
import netCDF4 as nc

import argparse

from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import csr_matrix

from msh import load_mesh, cell_quad

# (Quasi) remap a "pixel" DEM onto an MPAS-like mesh

def remap_var(args, mesh, var):
#-- remap a gridded variable to cells in the mesh

    print("*inject-var:", var)

    data = nc.Dataset(args.elev_file, "r")

    xlon = np.asarray(data["lon"][:])
    ylat = np.asarray(data["lat"][:])
    fdat = np.asarray(data[ var ][:])

    xmid = 0.5 * (xlon[:-1:] + xlon[1::])
    ymid = 0.5 * (ylat[:-1:] + ylat[1::])

    ffun = RegularGridInterpolator(
        (ymid, xmid), fdat,
        bounds_error=False, fill_value=None)

    xcel = mesh.cell.xlon
    ycel = mesh.cell.ylat
    xcel[xcel > np.pi] -= 2.0 * np.pi

    xvrt = mesh.vert.xlon
    yvrt = mesh.vert.ylat
    xvrt[xvrt > np.pi] -= 2.0 * np.pi

    fv_cell = ffun((ycel * 180.0 / np.pi,
                    xcel * 180.0 / np.pi))

    fv_dual = ffun((yvrt * 180.0 / np.pi,
                    xvrt * 180.0 / np.pi))

    return cell_quad(mesh, fv_cell, fv_dual)


if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "--mesh-file", dest="mesh_file", type=str,
        required=True,
        help="Path to MPAS-like mesh file.")

    parser.add_argument(
        "--elev-file", dest="elev_file", type=str,
        required=True,
        help="Path to DEM pixel file to remap.")

    args = parser.parse_args()

    mesh = load_mesh(args.mesh_file)

    zb_cell = remap_var(args, mesh, "bed_elevation")
    hh_cell = remap_var(args, mesh, "ocn_thickness")
    ih_cell = remap_var(args, mesh, "ice_thickness")
    ds_cell = remap_var(args, mesh, "bed_slope")
    dx_cell = remap_var(args, mesh, "bed_dz_dx")
    dy_cell = remap_var(args, mesh, "bed_dz_dy")

    mesh = nc.Dataset(args.mesh_file, "r+")

    if ("bed_elevation" not in mesh.variables.keys()):
        mesh.createVariable("bed_elevation", "f4", ("nCells"))
    if ("ocn_thickness" not in mesh.variables.keys()):
        mesh.createVariable("ocn_thickness", "f4", ("nCells"))
    if ("ice_thickness" not in mesh.variables.keys()):
        mesh.createVariable("ice_thickness", "f4", ("nCells"))

    mesh["bed_elevation"][:] = zb_cell
    mesh["bed_elevation"].long_name = "elevation of bed"
    mesh["ocn_thickness"][:] = hh_cell
    mesh["ocn_thickness"].long_name = "thickness of ocn"
    mesh["ice_thickness"][:] = ih_cell
    mesh["ice_thickness"].long_name = "thickness of ice"

    if ("bed_slope" not in mesh.variables.keys()):
        mesh.createVariable("bed_slope", "f4", ("nCells"))
    if ("bed_dz_dx" not in mesh.variables.keys()):
        mesh.createVariable("bed_dz_dx", "f4", ("nCells"))
    if ("bed_dz_dy" not in mesh.variables.keys()):
        mesh.createVariable("bed_dz_dy", "f4", ("nCells"))

    mesh["bed_slope"][:] = ds_cell
    mesh["bed_slope"].long_name = "RMS magnitude of bed slopes"
    mesh["bed_dz_dx"][:] = dx_cell
    mesh["bed_dz_dx"].long_name = \
        "derivative of bed elevation along lon.-axis"
    mesh["bed_dz_dy"][:] = dy_cell
    mesh["bed_dz_dy"].long_name = \
        "derivative of bed elevation along lat.-axis"

    mesh.close()

