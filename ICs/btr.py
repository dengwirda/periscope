
import time
import sys
import os
import numpy as np
from scipy.sparse.linalg import gcrotmk

import xarray
import argparse

sys.path.insert(
    1, os.path.join(sys.path[0], ".."))

from stb import strtobool

from msh import load_mesh, load_flow, cell_quad, dual_quad
from map import flatten
from map import idw_remap, interp2d
from ops import operators

#-- General BTR model config: tides, OBCs, ERA5 forcing 

grav = 9.7963               # gravity (mean at ssh; Griffies)
erot = 7.292E-05            # Earth's omega
orho = 1027.                # seawater density
arho = 1.225                # atmosph. density
# A. Shumskiy, (1960): Density of glacier ice, J. Glaciology
irho = 885.0                # iceshelf density

def coarsen(data, down):
#-- coarsen onto lower freq. temporal spacing 
    shap = list(data.shape); shap[0] //= down
    xtmp = np.zeros((shap), dtype=np.float64)
    for step in range(down):
        xtmp += data[step::down, :, :] / down

    return np.asarray(xtmp, dtype=np.float32)


def drag(mesh, mats, name, 
         nsqr, flat, zb_cell, hh_cell, ih_cell):

#-- apply various wave and top/bottom drag parameterisations

    print("Calculating wave-drag...")

    c1_edge = None; c1_cell = None 
    c2_edge = None; c2_cell = None
    Nm_cell = None; Nb_cell = None

    try:
        base = xarray.open_dataset(name)
        dzdx = np.asarray(
            base["bed_dz_dx"][:], dtype=np.float32)
        dzdy = np.asarray(
            base["bed_dz_dy"][:], dtype=np.float32)
       #grad = np.asarray(
       #base[ "bed_gradient"][:], dtype=np.float32)
    except:
        print("No elev grad data found")
        print("Skipping drag scheme")
        return c1_edge, c1_cell, \
               c2_edge, c2_cell, Nm_cell, Nb_cell

    try:
        fdat = xarray.open_dataset(nsqr)
        Nbar = np.asarray(
            fdat[  "Nsq_bar"][:], dtype=np.float32)
        Nbot = np.asarray(
            fdat[  "Nsq_bot"][:], dtype=np.float32)
    except:
        print("No N-squared data found")
        print("Skipping drag scheme")
        return c1_edge, c1_cell, \
               c2_edge, c2_cell, Nm_cell, Nb_cell

    xlon = np.linspace(
        -1. * np.pi, +1. * np.pi, Nbar.shape[1])
    ylat = np.linspace(
        -.5 * np.pi, +.5 * np.pi, Nbar.shape[0])

    ylat = flatten(ylat, flat)

    xcpi = 1. * mesh.cell.xlon
    xcpi[xcpi > np.pi] -= 2. * np.pi
    xvpi = 1. * mesh.vert.xlon
    xvpi[xvpi > np.pi] -= 2. * np.pi
    
   #Nbot = np.maximum(Nbot, Nbar / 100.)

    # interp. N**2 onto mesh via (quasi-)remap
    n2_cell = interp2d(
        ylat, xlon, Nbar, mesh.cell.ylat, xcpi)
    n2_vert = interp2d(
        ylat, xlon, Nbar, mesh.vert.ylat, xvpi)

    Nm_cell = cell_quad(mesh, n2_cell, n2_vert)

    n2_cell = interp2d(
        ylat, xlon, Nbot, mesh.cell.ylat, xcpi)
    n2_vert = interp2d(
        ylat, xlon, Nbot, mesh.vert.ylat, xvpi)

    Nb_cell = cell_quad(mesh, n2_cell, n2_vert)

    ff_cell = 2 * erot * np.sin(mesh.cell.ylat)
    f2_cell = ff_cell ** 2
    
    # use ADCIRC style "local-generation" scheme

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
    c1_cell = 2. / 11. * c1_24hr + \
              8. / 11. * c1_12hr + \
              1. / 11. * c1_06hr

   #c1_cell+= 1.E-08  # extra global dissipation 
    c1_cell = np.maximum(c1_cell, 5.E-06)  # min val.
    c1_cell = np.minimum(c1_cell, 1.E+00)  # max val.

    # depth, background and upper scalings on cd
    c1_cell*= np.sqrt(np.maximum(
        0., hh_cell - 300.)) / 300. / np.pi ** 2
   
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
    
    """
    c1_vert = mats.dual_kite_sums *  (c1_cell ** 2)
    c1_vert/= mesh.vert.area
    c1_edge = mats.edge_wing_sums *  (c1_cell ** 2)
    c1_edge/= mesh.edge.area
    c1_ends = mats.edge_tail_sums *  (c1_vert ** 2)
    c1_ends/= mesh.edge.area
    c1_edge = np.sqrt(
        4. / 6. * c1_edge + 2. / 6. * c1_ends)
    
    c2_vert = mats.dual_kite_sums *  (c2_cell ** 2)
    c2_vert/= mesh.vert.area
    c2_edge = mats.edge_wing_sums *  (c2_cell ** 2)
    c2_edge/= mesh.edge.area
    c2_ends = mats.edge_tail_sums *  (c2_vert ** 2)
    c2_ends/= mesh.edge.area
    c2_edge = np.sqrt(
        4. / 6. * c2_edge + 2. / 6. * c2_ends)
    """

    return c1_edge, c1_cell, \
           c2_edge, c2_cell, Nm_cell, Nb_cell


def obcs(mesh, mats, name, nest, zb_cell, hh_cell):

#-- remap open boundary condition data from parent model

    print("Preprocessing BC data...")

    xx_time = None; uE_edge = None; hE_edge = None

    try:
        base = xarray.open_dataset(name)
        ebnd = np.asarray(
            base["is_open"][:], dtype=np.int8)
        ebnd = (ebnd != 0)
    except:
        print("No OBCs found")
        print("Skipping OBCs scheme")
        return xx_time, uE_edge, hE_edge

    try:
        soln = xarray.open_dataset(nest)
        zb_soln = np.asarray(
            soln["zb_cell"][:], dtype=np.float32)
        hh_soln = np.asarray(
            soln["hh_cell"][720:1440], dtype=np.float32)
        hh_soln = np.squeeze(hh_soln)
        uu_soln = np.asarray(
            soln["uu_edge"][720:1440], dtype=np.float32)
        uu_soln = np.squeeze(uu_soln)
    except:
        print("No outer solution found")
        print("Skipping OBCs scheme")
        return xx_time, uE_edge, hE_edge

    print("Loading the nest file...")

    MESH = load_mesh(nest, rsph=None)

    print("Building coefficients...")

    MATS = operators(MESH)

    zb_edge = mats.edge_wing_sums * zb_cell
    zb_edge/= mesh.edge.area

    # add external OBC signals

    print("Interpolating BC data...")

    ppos = np.zeros(
        (MESH.cell.size, 3), dtype=np.float64)
    ppos[:, 0] = MESH.cell.xpos
    ppos[:, 1] = MESH.cell.ypos
    ppos[:, 2] = MESH.cell.zpos

    vbnd = np.zeros(mesh.vert.size, dtype=bool)
    vidx = np.unique(mesh.edge.vert[ebnd, :] - 1)
    vbnd[vidx] = True

    epos = np.zeros((
        np.count_nonzero(ebnd), 3), dtype=np.float64)
    epos[:, 0] = mesh.edge.xpos[ebnd]
    epos[:, 1] = mesh.edge.ypos[ebnd]
    epos[:, 2] = mesh.edge.zpos[ebnd]

    # to interp onto OBC edges
    emat, __ = idw_remap(ppos, epos, halo=8, dpow=2)

    vpos = np.zeros((
        np.count_nonzero(vbnd), 3), dtype=np.float64)
    vpos[:, 0] = mesh.vert.xpos[vbnd]
    vpos[:, 1] = mesh.vert.ypos[vbnd]
    vpos[:, 2] = mesh.vert.zpos[vbnd]

    # to interp onto OBC verts
    vmat, __ = idw_remap(ppos, vpos, halo=8, dpow=2)
    
    # load solution to nest in
    N = hh_soln.shape[0]

    xx_time = soln.timestart + np.linspace(0., 
        (N - 1) * soln.save_freq * soln.time_step, N)

    uE_edge = np.zeros(
        (N, mesh.edge.size), dtype=np.float32)
    hE_edge = np.zeros(
        (N, mesh.edge.size), dtype=np.float32)

    for step in range(N):

        zt_data = hh_soln[step, :] + zb_soln

        zt_ebnd = np.zeros(
            mesh.edge.size, dtype=np.float64)
        zt_ebnd[ebnd] = emat * zt_data

        zt_vbnd = np.zeros(
            mesh.vert.size, dtype=np.float64)
        zt_vbnd[vbnd] = vmat * zt_data

        # quadrature via simpson's
        zt_bnds = 4. / 6. * zt_ebnd + \
            1. / 6. * mats.edge_vert_sums * zt_vbnd

        hE_bnds = zt_bnds - zb_edge
        hE_bnds = np.maximum(0.0, hE_bnds)
        hE_bnds[np.logical_not(ebnd)] = 0.
        hE_edge [step, :] = hE_bnds
  
        he_soln = \
            MATS.edge_wing_sums * hh_soln[step, :]
        he_soln/= MESH.edge.area

        uu_data = uu_soln  [step, :] * he_soln
        vv_data =-MATS.edge_lsqr_perp* uu_data

        # transform local to lon-lat
        uW_edge = MESH.edge.cos_ * uu_data + \
                  MESH.edge.sin_ * vv_data
        uN_edge =-MESH.edge.sin_ * uu_data + \
                  MESH.edge.cos_ * vv_data

        uW_cell = MATS.cell_wing_sums* uW_edge
        uW_cell/= MESH.cell.area
        uN_cell = MATS.cell_wing_sums* uN_edge
        uN_cell/= MESH.cell.area

        uW_ebnd = np.zeros(
            mesh.edge.size, dtype=np.float64)
        uN_ebnd = np.zeros(
            mesh.edge.size, dtype=np.float64)
        uW_ebnd[ebnd] = emat * uW_cell
        uN_ebnd[ebnd] = emat * uN_cell

        uW_vbnd = np.zeros(
            mesh.vert.size, dtype=np.float64)
        uN_vbnd = np.zeros(
            mesh.vert.size, dtype=np.float64)
        uW_vbnd[vbnd] = vmat * uW_cell
        uN_vbnd[vbnd] = vmat * uN_cell

        # quadrature via simpson's
        uW_bnds = 4. / 6. * uW_ebnd + \
            1. / 6. * mats.edge_vert_sums * uW_vbnd
        uN_bnds = 4. / 6. * uN_ebnd + \
            1. / 6. * mats.edge_vert_sums * uN_vbnd

        # transform lon-lat to local
        uE_bnds = mesh.edge.cos_ * uW_bnds - \
                  mesh.edge.sin_ * uN_bnds
        uE_bnds[np.logical_not(ebnd)] = 0.
        uE_edge [step, :] = \
            uE_bnds / np.maximum(0.1, hE_bnds)

    return xx_time, uE_edge, hE_edge


def surf(mesh, mats, era5, hist, 
         ramp, down, zb_cell, hh_cell, ih_cell):

#-- remap ERA5 surface winds + pressure from gridded set

    print("Preprocessing forcing...")

    xx_time = None; xi_cell = None
    Tu_edge = None; Tu_curl = None 
    uW_cell = None; uN_cell = None

    try:
        fdat = xarray.open_dataset(hist)
        pbar = np.asarray(fdat[ "sp"][:])
        """
        ubar = np.asarray(fdat["u10"][:])
        vbar = np.asarray(fdat["v10"][:])

        """

        # time means, careful with fp. truncation
        pbar = np.asarray(np.mean(
            pbar, axis=0, dtype=np.float64), 
                          dtype=np.float32)
        """
        ubar = np.asarray(np.mean(
            ubar, axis=0, dtype=np.float64), 
                          dtype=np.float32)
        vbar = np.asarray(np.mean(
            vbar, axis=0, dtype=np.float64), 
                          dtype=np.float32)
        """

    except:
        pbar = 0.; ubar = 0.; vbar = 0.;
        print("No forcing offset found")
       #print("Skipping frc. scheme")
       #return xx_time, xi_cell, \
       #       Tu_edge, Tu_curl, uW_cell, uN_cell

    print("Interpolating forcing...")

    try:
        fdat = xarray.open_dataset(era5)
        xlon = np.asarray(
            fdat["longitude"][:]) * np.pi / 180.
        ylat = np.asarray(
            fdat[ "latitude"][:]) * np.pi / 180.
        patm = np.asarray(fdat[ "sp"][:])
        ux10 = np.asarray(fdat["u10"][:])
        uy10 = np.asarray(fdat["v10"][:])
    except:
        print("No atmos. forcing found")
        print("Skipping frc. scheme")
        return xx_time, xi_cell, \
               Tu_edge, Tu_curl, uW_cell, uN_cell

    ylat = np.flip(ylat, axis=0)
    ylat = flatten(ylat, flat)
    patm = np.flip(patm, axis=1)
    pbar = np.flip(pbar, axis=0)
    ux10 = np.flip(ux10, axis=1)
    uy10 = np.flip(uy10, axis=1)

    patm-= pbar  # subtract long-term mean p_atm
   #ux10-= ubar
   #uy10-= vbar

    # average over cycles
    patm = coarsen(patm, down=down)
    ux10 = coarsen(ux10, down=down)
    uy10 = coarsen(uy10, down=down)

    nfrc = patm.shape[0] + 1  # care w start/end
    xx_time = np.linspace(
        0., (nfrc - 1) * 60. * 60. * down, nfrc)

    ih_edge = mats.edge_wing_sums * ih_cell
    ih_edge/= mesh.edge.area

    # initial linear ramp
    if (ramp > 0):
        rfac = np.minimum(
            1., xx_time / ramp / 24. / 60. / 60.)
    else:
        rfac = np.ones(xx_time.size, dtype=float)

    xi_cell = np.zeros(
        (nfrc, mesh.cell.size), dtype=np.float32)

    Tu_edge = np.zeros(
        (nfrc, mesh.edge.size), dtype=np.float32)
    Tu_curl = np.zeros(
        (nfrc, mesh.vert.size), dtype=np.float32)

    uW_cell = np.zeros(
        (nfrc, mesh.cell.size), dtype=np.float32)
    uN_cell = np.zeros(
        (nfrc, mesh.cell.size), dtype=np.float32)

    for step in range(1, nfrc):
        # interp. p_atm
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

        ux_cell = mats.cell_kite_sums * ux_vert
        ux_cell/= mesh.cell.area
        uy_cell = mats.cell_kite_sums * uy_vert
        uy_cell/= mesh.cell.area

        # output surface wind vector for viz.
        uW_cell[step, :] = ux_cell
        uN_cell[step, :] = uy_cell

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
                         - mesh.edge.sin_ * Ty_edge

        xi_cell[step, :]*= rfac[step]
        Tu_edge[step, :]*= rfac[step]

        uW_cell[step, :]*= rfac[step]
        uN_cell[step, :]*= rfac[step]

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
    uW_cell[0, :] = rfac[0] * uW_cell[1, :]
    uN_cell[0, :] = rfac[0] * uN_cell[1, :]

    return xx_time, xi_cell, \
           Tu_edge, Tu_curl, uW_cell, uN_cell


def slr_(mesh, mats, mslr, zb_cell):

#-- remap mean sea-level rise function from gridded data

    print("Preprocessing SLR set...")

    try:
        fdat = xarray.open_dataset(mslr)
        xlon = np.asarray(
            fdat["lon"][:]) * np.pi / 180.
        ylat = np.asarray(
            fdat["lat"][:]) * np.pi / 180.
        zslr = np.asarray(fdat["slr"][:])
    except:
        print("No sea-level rise data found")
        print("Skipping mSLR scheme")
        return np.zeros(
            zb_cell.shape, dtype=zb_cell.dtype)

    ylat = flatten(ylat, flat)

    xpos = \
        np.hstack((mesh.cell.xlon, mesh.vert.xlon))
    xpos[xpos > np.pi]-= 2. * np.pi

    dz_temp = interp2d(ylat, xlon, zslr, 
        np.hstack((mesh.cell.ylat, mesh.vert.ylat)),
        xpos)

    dz_cell = dz_temp[:mesh.cell.size]
    dz_vert = dz_temp[ mesh.cell.size:
                       mesh.vert.size+
                       mesh.cell.size]

    return cell_quad(mesh, dz_cell, dz_vert)


def init(name, save, 
         mslr, slrf, iscf, nest, nsqr, 
         era5, hist, ramp, down, zmin, zmax, slip, 
         rsph=0., invf=-1.):

#------------------------------------ load an MPAS mesh file

    print("Loading the mesh file...")

    if (rsph <= 0.): rsph= None

    mesh = load_mesh(name, rsph)
    rsph = mesh.rsph
    
    flat = 0.
    if (invf > 0.): flat = 1. / invf

#------------------------------------ build TRSK matrix op's

    print("Building coefficients...")

    mats = operators(mesh)

    base = xarray.open_dataset(name)
    try:
        zb_cell = np.asarray(
            base["bed_elevation"][:], dtype=np.float32)
        ih_cell = np.asarray(
            base["ice_thickness"][:], dtype=np.float32)
    except:
        raise ValueError("No bathymetry found")

    try:
        zb_drag =-np.asarray(
            base["hrm_thickness"][:], dtype=np.float32)
    except:
        zb_drag = zb_cell[:]

    uu_edge = np.zeros(
        mesh.edge.size, dtype=np.float32)

    # scales ice-shelf thickness wrt. scenario
    ih_cell*= iscf

    # offset bathymetry to represent ice-shelf
    zb_cell+= ih_cell * irho / orho
    zb_drag[ih_cell>0.] = zb_cell[ih_cell>0.]

    # offset bathymetry to represent sea-level
    zt_mslr = slr_(mesh, mats, mslr, zb_cell)
    zt_mslr*= slrf
    zb_cell-= zt_mslr
    zb_drag-= zt_mslr

    print("mSLR:", np.sum(mesh.cell.area * zt_mslr) 
                 / np.sum(mesh.cell.area))

    # smooth at grid-scale
    zb_dual = mats.dual_kite_sums * zb_cell
    zb_dual/= mesh.vert.area
    z2_cell = mats.cell_kite_sums * zb_dual
    z2_cell/= mesh.cell.area
    zb_cell =(7.0 * zb_cell + 1.0 * z2_cell) / 8.0

    zb_dual = mats.dual_kite_sums * zb_drag
    zb_dual/= mesh.vert.area
    z2_cell = mats.cell_kite_sums * zb_dual
    z2_cell/= mesh.cell.area
    zb_drag =(7.0 * zb_drag + 1.0 * z2_cell) / 8.0
   
    # set limits on depths
    zb_cell = np.minimum(-zmin, zb_cell)    
    zb_cell = np.maximum(-zmax, zb_cell)
    zb_drag = np.minimum(-zmin, zb_drag)    
    zb_drag = np.maximum(-zmax, zb_drag)

    zt_cell = +0.0 * zb_cell
    hh_cell = np.maximum(+0.00, zt_cell - zb_cell)

    # free slip boundaries
    bc_slip = slip * np.ones(
        (mesh.edge.size), dtype=np.float32)

    bc_slip[np.logical_and.reduce((
        mesh.edge.ylat >= +60. * np.pi / 180.,
        mesh.edge.ylat <= -60. * np.pi / 180.,
        )) ] = 0.333  # enhanced dissipation = landfast ice

    xlon = mesh.edge.xlon[:]
    xlon[xlon>np.pi]-= 2. * np.pi

    # should this be based on the length of coastline 
    # per cell compared to the cell's perimeter?
    bc_slip[np.logical_and.reduce((
        mesh.edge.ylat >= +30. * np.pi / 180.,
        mesh.edge.ylat <= +65. * np.pi / 180.,
        xlon >=-155. * np.pi / 180.,
        xlon <=-115. * np.pi / 180.,
        )) ] = 0.100  # enhanced dissipation = US northwest

#-- inject mesh with IC.'s and write to MPAS-ish NetCDF file

    c1_edge, c1_cell, c2_edge, c2_cell, Nm_cell, Nb_cell = \
        drag(mesh, mats, name, 
             nsqr, flat, zb_cell, hh_cell, ih_cell)

    print("Output written to:", save)

    init = xarray.open_dataset(name)
    init.attrs.update({"sphere_radius": mesh.rsph})
    init.attrs.update({"sphere_flatten": flat})
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

    if (c1_edge is not None):
        init["c1_edge"] = (("nEdges"), c1_edge)
    if (c1_cell is not None):
        init["cd_wave"] = (("nCells"), c1_cell)

    if (c2_edge is not None):
        init["c2_edge"] = (("nEdges"), c2_edge)
    if (c2_cell is not None):
        init["cd_surf"] = (("nCells"), c2_cell)
    
    if (Nm_cell is not None):
        init["N2_mean"] = (("nCells"), Nm_cell)
    if (Nb_cell is not None):
        init["N2_deep"] = (("nCells"), Nb_cell)

    init["hh_cell"] = (
        ("Time", "nCells", "nVertLevels"),
        np.reshape(hh_cell, (1, mesh.cell.size, 1)))
    
    init["zt_mslr"] = (("nCells"), zt_mslr)
    init["zb_cell"] = (("nCells"), zb_cell)
    init["zb_drag"] = (("nCells"), zb_drag)

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

#-- inject mesh + frc. data and write to MPAS-ish NetCDF file

    print("")

    xx_time, xi_cell, Tu_edge, Tu_curl, uW_cell, uN_cell = \
        surf(mesh, mats, era5, hist, 
             ramp, down, zb_cell, hh_cell, ih_cell)

    path, file = os.path.split(save)
    sfrc = os.path.join(path, "frc_" + file)
    
    frc_ = xarray.open_dataset(name)
    frc_.attrs.update({"sphere_radius": mesh.rsph})
    frc_.attrs.update({"sphere_flatten": flat})
    frc_.attrs.update({"config_gravity": grav})
    frc_["xCell"] = (("nCells"), mesh.cell.xpos)
    frc_["yCell"] = (("nCells"), mesh.cell.ypos)
    frc_["zCell"] = (("nCells"), mesh.cell.zpos)
    frc_["areaCell"] = (("nCells"), mesh.cell.area)

    frc_["xEdge"] = (("nEdges"), mesh.edge.xpos)
    frc_["yEdge"] = (("nEdges"), mesh.edge.ypos)
    frc_["zEdge"] = (("nEdges"), mesh.edge.zpos)
    frc_["dvEdge"] = (("nEdges"), mesh.edge.vlen)
    frc_["dcEdge"] = (("nEdges"), mesh.edge.clen)

    frc_["xVertex"] = (("nVertices"), mesh.vert.xpos)
    frc_["yVertex"] = (("nVertices"), mesh.vert.ypos)
    frc_["zVertex"] = (("nVertices"), mesh.vert.zpos)
    frc_["areaTriangle"] = (("nVertices"), mesh.vert.area)
    frc_["kiteAreasOnVertex"] = (
        ("nVertices", "vertexDegree"), mesh.vert.kite)

    if (xx_time is not None):
        frc_["xx_time"] = ("Time", xx_time)

    if (xi_cell is not None):
        N = xx_time.shape[0]
        frc_["Xi_cell"] = (
            ("Time", "nCells", "nVertLevels"),
        np.reshape(xi_cell, (N, mesh.cell.size, 1)))

    if (Tu_edge is not None):
        N = xx_time.shape[0]
        frc_["Tu_edge"] = (
            ("Time", "nEdges", "nVertLevels"),
        np.reshape(Tu_edge, (N, mesh.edge.size, 1)))

    if (Tu_curl is not None):
        N = xx_time.shape[0]
        frc_["Tu_curl"] = (
            ("Time", "nVertices", "nVertLevels"),
        np.reshape(Tu_curl, (N, mesh.vert.size, 1)))

    if (uW_cell is not None):
        N = xx_time.shape[0]
        frc_["uW_cell"] = (
            ("Time", "nCells", "nVertLevels"),
        np.reshape(uW_cell, (N, mesh.cell.size, 1)))

    if (uN_cell is not None):
        N = xx_time.shape[0]
        frc_["uN_cell"] = (
            ("Time", "nCells", "nVertLevels"),
        np.reshape(uN_cell, (N, mesh.cell.size, 1)))

    print(frc_)

    frc_.to_netcdf(sfrc, format="NETCDF4")

    del xx_time, xi_cell
    del Tu_edge, Tu_curl, uW_cell, uN_cell

#-- inject mesh + OBCs data and write to MPAS-ish NetCDF file

    print("")

    xx_time, uE_edge, hE_edge = \
        obcs(mesh, mats, name, nest, zb_cell, hh_cell)

    path, file = os.path.split(save)
    sobc = os.path.join(path, "obc_" + file)
    
    obc_ = xarray.open_dataset(name)
    obc_.attrs.update({"sphere_radius": mesh.rsph})
    obc_.attrs.update({"sphere_flatten": flat})
    obc_.attrs.update({"config_gravity": grav})
    obc_["xCell"] = (("nCells"), mesh.cell.xpos)
    obc_["yCell"] = (("nCells"), mesh.cell.ypos)
    obc_["zCell"] = (("nCells"), mesh.cell.zpos)
    obc_["areaCell"] = (("nCells"), mesh.cell.area)

    obc_["xEdge"] = (("nEdges"), mesh.edge.xpos)
    obc_["yEdge"] = (("nEdges"), mesh.edge.ypos)
    obc_["zEdge"] = (("nEdges"), mesh.edge.zpos)
    obc_["dvEdge"] = (("nEdges"), mesh.edge.vlen)
    obc_["dcEdge"] = (("nEdges"), mesh.edge.clen)

    obc_["xVertex"] = (("nVertices"), mesh.vert.xpos)
    obc_["yVertex"] = (("nVertices"), mesh.vert.ypos)
    obc_["zVertex"] = (("nVertices"), mesh.vert.zpos)
    obc_["areaTriangle"] = (("nVertices"), mesh.vert.area)
    obc_["kiteAreasOnVertex"] = (
        ("nVertices", "vertexDegree"), mesh.vert.kite)

    if (xx_time is not None):
        obc_["xx_time"] = ("Time", xx_time)

    if (uE_edge is not None):
        N = xx_time.shape[0]
        obc_["uE_edge"] = (
            ("Time", "nEdges", "nVertLevels"),
        np.reshape(uE_edge, (N, mesh.edge.size, 1)))

    if (hE_edge is not None):
        N = xx_time.shape[0]
        obc_["hE_edge"] = (
            ("Time", "nEdges", "nVertLevels"),
        np.reshape(hE_edge, (N, mesh.edge.size, 1)))

    print(obc_)

    obc_.to_netcdf(sobc, format="NETCDF4")

    del xx_time, uE_edge, hE_edge


if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--mesh-file", dest="mesh_file", type=str,
        required=True, help="Path to user mesh file.")

    parser.add_argument(
        "--init-file", dest="init_file", type=str,
        required=True, help="IC's filename to write.")

    parser.add_argument(
        "--isc-scale", dest="isc_scale", type=float,
        default=1.,
        required=False, help="Ice-shelf thick. scale.")

    parser.add_argument(
        "--slr-scale", dest="slr_scale", type=float,
        default=1.,
        required=False, help="Sea-level height scale.")

    parser.add_argument(
        "--mslr-file", dest="mslr_file", type=str,
        default="",
        required=False, help="Sea-level rise dataset.")

    parser.add_argument(
        "--nest-file", dest="nest_file", type=str,
        default="",
        required=False, help="Soln to nest inside of.")

    parser.add_argument(
        "--nsqr-file", dest="nsqr_file", type=str,
        default="",
        required=False, help="N**2 density data file.")

    parser.add_argument(
        "--era5-file", dest="era5_file", type=str,
        default="",
        required=False, help="ERA5 forcing data file.")

    parser.add_argument(
        "--hist-file", dest="hist_file", type=str,
        default="",
        required=False, help="ERA5 history data file.")

    parser.add_argument(
        "--ramp-days", dest="ramp_days", type=float,
        default=0.,
        required=False, help="Length of initial ramp.")

    parser.add_argument(
        "--downscale", dest="downscale", type=int,
        default=+1,
        required=False, help="Num. cycles to average.")

    parser.add_argument(
        "--min-depth", dest="min_depth", type=float,
        default=-1.E+6,
        required=False, help="Limit on min ocn depth.")

    parser.add_argument(
        "--max-depth", dest="max_depth", type=float,
        default=+1.E+6,
        required=False, help="Limit on max ocn depth.")

    parser.add_argument(
        "--wall-slip", dest="wall_slip", type=float,
        default=0.9375,
        required=False, 
        help="Wall slip BCs; 0=no slip, 1=free slip.")

    parser.add_argument(
        "--flux-hmin", dest="flux_hmin", type=float,
        default=.1,
        required=False, 
        help="Minimum thickness for nested flux adj.")

    parser.add_argument(
        "--radius", dest="radius", type=float,
        default=0., required=False, 
        help="Value of sphere_radius; zero to use mesh data.")

    parser.add_argument(
        "--inv-flatten", dest="inv_flatten", type=float,
        default=-1, required=False,
        help="Inv. flattening of spheroidal coord. system")

    args = parser.parse_args()

    init(name=args.mesh_file, save=args.init_file,
         mslr=args.mslr_file, slrf=args.slr_scale, 
         iscf=args.isc_scale,
         nest=args.nest_file, nsqr=args.nsqr_file,
         era5=args.era5_file, hist=args.hist_file,
         ramp=args.ramp_days, down=args.downscale,
         zmin=args.min_depth, zmax=args.max_depth, 
         slip=args.wall_slip, 
         rsph=args.radius, 
         invf=args.inv_flatten)

