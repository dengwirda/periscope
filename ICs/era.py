
import time
import sys
import os
import numpy as np
from scipy.sparse.linalg import gcrotmk
from scipy.interpolate import RectBivariateSpline
from akima import interp2d

import xarray
import argparse

sys.path.insert(
    1, os.path.join(sys.path[0], ".."))

from stb import strtobool

from msh import load_mesh, cell_quad, dual_quad
from ops import operators

#-- General BTR model config. based on ERA5 forcing
#-- Authors: Darren Engwirda

def coarsen(data, down):
#-- coarsen onto lower freq. temporal spacing 
    shap = list(data.shape); shap[0] //= down

    xtmp = np.zeros((shap), dtype=np.float64)
    for step in range(down):
        xtmp += data[step::down, :, :] / down

    return np.asarray(xtmp, dtype=np.float32)


def init(name, save, nsqr, 
         era5, back, ramp, down, deep, rsph=0.):

#------------------------------------ load an MPAS mesh file

    print("Loading the mesh file...")

    if (rsph <= 0.): rsph= None

    mesh = load_mesh(name, rsph)
    rsph = mesh.rsph
    
#------------------------------------ build TRSK matrix op's

    print("Building coefficients...")

    mats = operators(mesh)

    grav = 9.80616              # gravity
    erot = 7.292E-05            # Earth's omega
    orho = 1035.                # seawater density
    arho = 1.225                # atmosph. density
    irho = 1000.                # iceshelf density
    
    base = xarray.open_dataset(name)

    uu_edge = np.zeros(
        mesh.edge.size, dtype=np.float64)
    
    zb_cell = np.asarray(
        base["bed_elevation"][:], dtype=np.float32)

    ih_cell = np.asarray(
        base["ice_thickness"][:], dtype=np.float32)

    # offset bathymetry to represent ice-shelf
    zb_cell+= ih_cell * irho / orho

    zb_cell = np.minimum(1.0, zb_cell)

    ih_edge = mats.edge_wing_sums * ih_cell
    ih_edge/= mesh.edge.area
    
    # smooth at grid-scale
    zb_dual = mats.dual_kite_sums * zb_cell
    zb_dual/= mesh.vert.area
    z2_cell = mats.cell_kite_sums * zb_dual
    z2_cell/= mesh.cell.area
    zb_cell = 2./3. * zb_cell + 1./3. * z2_cell
   
    # set limits on depths 
    zb_cell = np.maximum(-deep, zb_cell)

    zt_cell = +0.0 * zb_cell
    hh_cell = np.maximum(+0.00, zt_cell - zb_cell)

    # free slip boundaries
    bc_slip = 0.875 * \
        np.ones((mesh.edge.size), dtype=np.float32)

    print("Calculating wave-drag...")

    dzdx = np.asarray(
        base["bed_dz_dx"][:], dtype=np.float32)
    dzdy = np.asarray(
        base["bed_dz_dy"][:], dtype=np.float32)

    fdat = xarray.open_dataset(nsqr)
    Nbar = np.asarray(
        fdat[  "Nsq_bar"][:], dtype=np.float32)
    Nbot = np.asarray(
        fdat[  "Nsq_bot"][:], dtype=np.float32)

    xlon = np.linspace(
        -1. * np.pi, +1. * np.pi, Nbar.shape[1])
    ylat = np.linspace(
        -.5 * np.pi, +.5 * np.pi, Nbar.shape[0])

    xcpi = 1. * mesh.cell.xlon
    xcpi[xcpi > np.pi] -= 2. * np.pi
    xvpi = 1. * mesh.vert.xlon
    xvpi[xvpi > np.pi] -= 2. * np.pi

    # interp. N**2 onto mesh via (quasi-)remap
    """    
    ifun = RectBivariateSpline(ylat, xlon, Nbar, 
                               kx=1, ky=1)
    n2_cell = ifun.ev(mesh.cell.ylat, xcpi)
    n2_vert = ifun.ev(mesh.vert.ylat, xvpi)
    """

    n2_cell = interp2d(
        ylat, xlon, Nbar, mesh.cell.ylat, xcpi)
    n2_vert = interp2d(
        ylat, xlon, Nbar, mesh.vert.ylat, xvpi)

    Nm_cell = cell_quad(mesh, n2_cell, n2_vert)
    Nm_cell+= 1.E-12

    """
    ifun = RectBivariateSpline(ylat, xlon, Nbot, 
                               kx=1, ky=1)
    n2_cell = ifun.ev(mesh.cell.ylat, xcpi)
    n2_vert = ifun.ev(mesh.vert.ylat, xvpi)
    """

    n2_cell = interp2d(
        ylat, xlon, Nbot, mesh.cell.ylat, xcpi)
    n2_vert = interp2d(
        ylat, xlon, Nbot, mesh.vert.ylat, xvpi)

    Nb_cell = cell_quad(mesh, n2_cell, n2_vert)
    Nb_cell+= 1.E-12

    ff_cell = 2 * erot * np.sin(mesh.cell.ylat)
    f2_cell = ff_cell ** 2
    
    # use ADCIRC style "local-generation" scheme
    """    
    omegaM2 = 2. * np.pi / 24.00 / 60.0 / 60.0  #!!M2?
    c1_cell = 1. / omegaM2 * np.sqrt(
        np.maximum(0., Nm_cell - omegaM2 ** 2) *
        np.maximum(0., Nb_cell - omegaM2 ** 2) )

    # limit dissipation at super-critical slopes
    alpha = np.sqrt(np.maximum(1.E-12,
        (omegaM2 ** 2 - f2_cell) / 
        (Nb_cell - omegaM2 ** 2) ) )

    gamma = np.sqrt(dzdx ** 2 + dzdy ** 2) / alpha
    gamma[gamma<=1.] = 1.
    c1_cell*= 1. / gamma  #!! super-critical limiter?
    c1_cell*= np.sqrt(dzdx ** 2 + dzdy ** 2)
    """

    # take composition of 24, 12, & 06hr constituents
    # 24hr:
    omega24 = 2. * np.pi / 24.00 / 60.0 / 60.0
    c1_24hr = 1. / omega24 * np.sqrt(
        np.maximum(0., Nm_cell - omega24 ** 2) *
        np.maximum(0., Nb_cell - omega24 ** 2) )

    # limit dissipation at super-critical slopes
    alpha = np.sqrt(np.maximum(1.E-12,
        (omega24 ** 2 - f2_cell) / 
        (Nb_cell - omega24 ** 2) ) )

    gamma = np.sqrt(dzdx ** 2 + dzdy ** 2) / alpha
    gamma[gamma<=1.] = 1.
    c1_24hr*= 1. / gamma ** 2
    c1_24hr*= np.sqrt(dzdx ** 2 + dzdy ** 2)

    # 12hr:
    omega12 = 2. * np.pi / 12.00 / 60.0 / 60.0
    c1_12hr = 1. / omega12 * np.sqrt(
        np.maximum(0., Nm_cell - omega12 ** 2) *
        np.maximum(0., Nb_cell - omega12 ** 2) )

    # limit dissipation at super-critical slopes
    alpha = np.sqrt(np.maximum(1.E-12,
        (omega12 ** 2 - f2_cell) / 
        (Nb_cell - omega12 ** 2) ) )

    gamma = np.sqrt(dzdx ** 2 + dzdy ** 2) / alpha
    gamma[gamma<=1.] = 1.
    c1_12hr*= 1. / gamma ** 2
    c1_12hr*= np.sqrt(dzdx ** 2 + dzdy ** 2)

    # 06hr:
    omega06 = 2. * np.pi / +6.00 / 60.0 / 60.0
    c1_06hr = 1. / omega06 * np.sqrt(
        np.maximum(0., Nm_cell - omega06 ** 2) *
        np.maximum(0., Nb_cell - omega06 ** 2) )

    # limit dissipation at super-critical slopes
    alpha = np.sqrt(np.maximum(1.E-12,
        (omega06 ** 2 - f2_cell) / 
        (Nb_cell - omega06 ** 2) ) )

    gamma = np.sqrt(dzdx ** 2 + dzdy ** 2) / alpha
    gamma[gamma<=1.] = 1.
    c1_06hr*= 1. / gamma ** 2
    c1_06hr*= np.sqrt(dzdx ** 2 + dzdy ** 2)

    #!! how to choose weights? energy per constituent?
    c1_cell = 3. / 12. * c1_24hr + \
              8. / 12. * c1_12hr + \
              1. / 12. * c1_06hr

    # depth, background and upper scalings on cd
    c1_cell*= np.sqrt(np.maximum(
        0., hh_cell - 64.)) / 256. / np.pi ** 2
    c1_cell+= 1.E-08  # extra global dissipation
    c1_cell = np.minimum(c1_cell, 0.001)  # max bound

    # add extra top surface drag for ice-shelves
    c2_cell = np.zeros((mesh.cell.size), dtype=np.float32)

    c1_cell[ih_cell>0.] = (      # ensure twd in fisc
        np.maximum(1.E-06, c1_cell[ih_cell>0.]))

    c2_cell[ih_cell>0.] = .0025  # extra drag in fisc
   
    # map drag coefficients, from cells to edges
    c1_vert = mats.dual_kite_sums *  c1_cell  # simpson's
    c1_vert/= mesh.vert.area
    c1_edge = mats.edge_wing_sums *  c1_cell
    c1_edge/= mesh.edge.area
    c1_ends = mats.edge_tail_sums *  c1_vert
    c1_ends/= mesh.edge.area
    c1_edge = 4. / 6. * c1_edge + 2. / 6. * c1_ends

    c2_vert = mats.dual_kite_sums *  c2_cell
    c2_vert/= mesh.vert.area
    c2_edge = mats.edge_wing_sums *  c2_cell
    c2_edge/= mesh.edge.area
    c2_ends = mats.edge_tail_sums *  c2_vert
    c2_ends/= mesh.edge.area
    c2_edge = 4. / 6. * c2_edge + 2. / 6. * c2_ends

    print("Preprocessing forcing...")

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

    # average over cycles
    patm = coarsen(patm, down=down)
    ux10 = coarsen(ux10, down=down)
    uy10 = coarsen(uy10, down=down)

    nfrc = patm.shape[0] + 1  # care w start/end
    xx_time = np.linspace(
        0., (nfrc - 1) * 60. * 60. * down, nfrc)  # ERA5 hr

    # initial linear ramp
    rfac = np.minimum(
        1., xx_time / ramp / 24. / 60. / 60.)

    xi_cell = np.zeros(
        (nfrc, mesh.cell.size), dtype=np.float32)
    Tu_edge = np.zeros(
        (nfrc, mesh.edge.size), dtype=np.float32)
    Tu_curl = np.zeros(
        (nfrc, mesh.vert.size), dtype=np.float32)

    for step in range(1, 8): #nfrc):
        # interp. p_atm
        """
        ifun = RectBivariateSpline(
            ylat, xlon, patm[step - 1, :, :])
        ps_cell = \
            ifun.ev(mesh.cell.ylat, mesh.cell.xlon)
        ps_vert = \
            ifun.ev(mesh.vert.ylat, mesh.vert.xlon)
        """

        ps_temp = interp2d(ylat, xlon, 
            patm[step - 1, :, :], 
        np.hstack((mesh.cell.ylat, mesh.vert.ylat)),
        np.hstack((mesh.cell.xlon, mesh.vert.xlon)))

        ps_cell = ps_temp[:mesh.cell.size]
        ps_vert = ps_temp[ mesh.cell.size:
                           mesh.vert.size+
                           mesh.cell.size]
    
        # 1. / rho_ocn * (p_atm - p_atm_mean)
        xi_cell[step, :] = \
            cell_quad(mesh, ps_cell, ps_vert) / orho

        # interp. u_10m
        """        
        ifun = RectBivariateSpline(
            ylat, xlon, ux10[step - 1, :, :])
        ux_edge = \
            ifun.ev(mesh.edge.ylat, mesh.edge.xlon)
        ux_vert = \
            ifun.ev(mesh.vert.ylat, mesh.vert.xlon)

        ifun = RectBivariateSpline(
            ylat, xlon, uy10[step - 1, :, :])
        uy_edge = \
            ifun.ev(mesh.edge.ylat, mesh.edge.xlon)
        uy_vert = \
            ifun.ev(mesh.vert.ylat, mesh.vert.xlon)
        """

        ux_temp = interp2d(ylat, xlon, 
            ux10[step - 1, :, :], 
        np.hstack((mesh.edge.ylat, mesh.vert.ylat)),
        np.hstack((mesh.edge.xlon, mesh.vert.xlon)))

        ux_edge = ux_temp[:mesh.edge.size]
        ux_vert = ux_temp[ mesh.edge.size:
                           mesh.vert.size+
                           mesh.edge.size]

        uy_temp = interp2d(ylat, xlon, 
            uy10[step - 1, :, :], 
        np.hstack((mesh.edge.ylat, mesh.vert.ylat)),
        np.hstack((mesh.edge.xlon, mesh.vert.xlon)))

        uy_edge = uy_temp[:mesh.edge.size]
        uy_vert = uy_temp[ mesh.edge.size:
                           mesh.vert.size+
                           mesh.edge.size]

        # Simpson's rule
        ux_edge = 4. / 6. * ux_edge + \
            1. / 6. * mats.edge_vert_sums * ux_vert
        uy_edge = 4. / 6. * uy_edge + \
            1. / 6. * mats.edge_vert_sums * uy_vert

        # rho_atm / rho_ocn * cw * |w| * w_i
        uu_norm = \
            np.sqrt(ux_edge ** 2 + uy_edge ** 2)

        # Garratt's form, with saturation
        uu_scal = np.minimum(33.3333, uu_norm)
        cw_edge = 1. / 1000. * \
            (0.75 + 0.067 * uu_scal) * arho / orho

        Tx_edge = cw_edge * uu_norm * ux_edge
        Ty_edge = cw_edge * uu_norm * uy_edge

        Tu_edge[step, :] = mesh.edge.cos_ * Tx_edge \
                         + mesh.edge.sin_ * Ty_edge

        xi_cell[step, :]*= rfac[step]
        Tu_edge[step, :]*= rfac[step]

        # no wind stress under ice-shelves...
        Tu_edge[step, ih_edge > 0.] = 0.

        # output curl of wind stress for viz.
        Tu_curl[step, :] = \
            mats.dual_curl_sums * Tu_edge[step, :]
        Tu_curl[step, :]/= mesh.vert.area

    # piecewise const. extrap. to start
    xi_cell[0, :] = rfac[0] * xi_cell[1, :]
    Tu_edge[0, :] = rfac[0] * Tu_edge[1, :]
    Tu_curl[0, :] = rfac[0] * Tu_curl[1, :]

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

    init["c1_edge"] = (("nEdges"), c1_edge)
    init["cd_wave"] = (("nCells"), c1_cell)

    init["c2_edge"] = (("nEdges"), c2_edge)
    init["cd_surf"] = (("nCells"), c2_cell)
    
    init["N2_mean"] = (("nCells"), Nm_cell)
    init["N2_deep"] = (("nCells"), Nb_cell)

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
        "--nsqr-file", dest="nsqr_file", type=str,
        required=True, help="N**2 density data file.")

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
        "--averaging", dest="averaging", type=int,
        required=True, help="Num. cycles to average.")

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
         nsqr=args.nsqr_file,         
         era5=args.era5_file,
         back=args.mean_file,
         ramp=args.ramp_days,
         down=args.averaging, 
         deep=args.max_depth,
         rsph=args.radius)

