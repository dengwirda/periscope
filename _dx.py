
import time
import math
import numpy as np

""" SWE spatial discretisation using TRSK-like operators
"""
#-- Darren Engwirda

from _fp import flt32_t, flt64_t
from _fp import reals_t, index_t

from log import tcpu

from mem import variables

HH_TINY        = 1.0E-04
UU_TINY        = 1.0E-16
PV_TINY        = 1.0E-16

def hrmn_mean(xone, xtwo):

#-- harmonic mean of two vectors (ie. biased toward lesser)

    return +2.0 * xone * xtwo / (xone + xtwo)


def scalingVk(mesh, trsk, cnfg):

#-- local gridsize scaling on div^k and del^k operators

    # diam. of equiv. circle
    dx_cell = 2. * np.sqrt(mesh.cell.area / np.pi)

    # smooth near grid-scale
    dx_edge = trsk.edge_wing_sums * dx_cell
    dx_edge/= mesh.edge.area
    dx_cell = trsk.cell_wing_sums * dx_edge
    dx_cell/= mesh.cell.area
    
    dx_edge = trsk.edge_wing_sums * dx_cell
    dx_edge/= mesh.edge.area
    dx_cell = trsk.cell_wing_sums * dx_edge
    dx_cell/= mesh.cell.area
    
    dx_edge = trsk.edge_wing_sums * dx_cell
    dx_edge/= mesh.edge.area
    dx_cell = trsk.cell_wing_sums * dx_edge
    dx_cell/= mesh.cell.area
    
    if (cnfg.ref_scale > 0.0):
        s2_cell = (dx_cell / cnfg.ref_scale) ** 1
        s4_cell = (dx_cell / cnfg.ref_scale) ** 3
    else:
        s2_cell = np.ones(
            (mesh.cell.size), dtype=reals_t)
        s4_cell = np.ones(
            (mesh.cell.size), dtype=reals_t)
    
    dx_edge = trsk.edge_wing_sums * dx_cell
    dx_edge/= mesh.edge.area

    if (cnfg.ref_scale > 0.0):
        s2_edge = (dx_edge / cnfg.ref_scale) ** 1
        s4_edge = (dx_edge / cnfg.ref_scale) ** 3
    else:
        s2_edge = np.ones(
            (mesh.edge.size), dtype=reals_t)
        s4_edge = np.ones(
            (mesh.edge.size), dtype=reals_t)

    return s2_edge, s4_edge, \
           s2_cell, s4_cell


def diag_vars(mesh, trsk, flow, cnfg, hh_cell, uu_edge):

#-- compute diagnostic variables from the current state

    ff_dual = flow.ff_vert
    ff_edge = flow.ff_edge
    ff_cell = flow.ff_cell
    
    hE_edge = flow.hE_edge
    uE_edge = flow.uE_edge
    
    zb_cell = flow.zb_cell 
    gg_cell = flow.gravity

    hh_dual, hh_edge, h2_edge, hh_bias = compute_H(
        mesh, trsk, cnfg, hh_cell, uu_edge)

    hh_edge, uu_edge = computeBC(
        mesh, trsk, cnfg, 
        hh_edge, uu_edge, 
        gg_cell, hE_edge, uE_edge)
        
    vv_edge = computeVV(
        mesh, trsk, cnfg, uu_edge)

    ke_cell, ke_bias = computeKE(
        mesh, trsk, cnfg, 
        hh_cell, h2_edge, hh_dual, 
        uu_edge, vv_edge,
        +1. / 2. * cnfg.time_step)

    rv_dual, pv_dual, \
    rv_cell, pv_cell, \
    pv_edge, pv_bias = computePV(
        mesh, trsk, cnfg, 
        hh_cell, h2_edge, hh_dual, uu_edge, vv_edge,
        ff_dual, ff_edge, ff_cell, 
        +1. / 2. * cnfg.time_step)

    return hh_edge, hh_dual, hh_bias, \
           ke_cell, ke_bias, \
           rv_cell, pv_cell, \
           rv_dual, pv_dual, pv_edge, pv_bias, \
           vv_edge


def invariant(mesh, trsk, flow, cnfg, hh_cell, uu_edge):

#-- compute the discrete energy and enstrophy invariants

    ff_dual = flow.ff_vert
    ff_edge = flow.ff_edge
    ff_cell = flow.ff_cell

    hE_edge = flow.hE_edge
    uE_edge = flow.uE_edge

    zb_cell = flow.zb_cell
    gg_cell = flow.gravity

    hh_dual, hh_edge, h2_edge, hh_bias = compute_H(
        mesh, trsk, cnfg, hh_cell, uu_edge)

    hh_edge, uu_edge = computeBC(
        mesh, trsk, cnfg, 
        hh_edge, uu_edge, 
        gg_cell, hE_edge, uE_edge)
        
    vv_edge = computeVV(
        mesh, trsk, cnfg, uu_edge)

    ke_edge = uu_edge ** 2
    ke_edge*= hh_edge * mesh.edge.area
    
    pe_cell = flow.gravity * (
        hh_cell * 0.5 + zb_cell - np.min(zb_cell))

    pe_cell*= hh_cell * mesh.cell.area

    kk_sums = math.fsum(ke_edge) \
            + math.fsum(pe_cell)

    rv_dual, pv_dual, rv_cell, pv_cell, \
    pv_edge, pv_bias = computePV(
        mesh, trsk, cnfg, 
        hh_cell, h2_edge, hh_dual, uu_edge, vv_edge,
        ff_dual, ff_edge, ff_cell, 
        +1. / 2. * cnfg.time_step)

   #pv_sums = 0.5 * math.fsum(
   #    mesh.edge.area * hh_edge * pv_edge ** 2)

    pv_sums = 0.5 * math.fsum(
        mesh.vert.area * hh_dual * pv_dual ** 2)

    return kk_sums, pv_sums


def computeBC(mesh, trsk, cnfg,
        hh_edge, uu_edge, gg_cell, hE_edge, uE_edge):
        
#-- setup open bnd. conditions
   
    if (hE_edge is None): return hh_edge, uu_edge
    if (uE_edge is None): return hh_edge, uu_edge
   
    ttic = time.time()
        
    hh_edge, uu_edge = _computeBC(
        mesh, trsk, cnfg, 
        hh_edge, uu_edge, gg_cell, hE_edge, uE_edge)
        
    ttoc = time.time()
    tcpu.computeBC = tcpu.computeBC + (ttoc - ttic)
        
    return hh_edge, uu_edge


def upwinding(mesh, trsk, cnfg, 
        sw_dual, ss_dual, ss_cell, uu_edge, vv_edge, 
        ss_edge, up_bias,
        delta_t, sv_tiny, uu_tiny,
        up_kind, up_min_, up_max_):

#-- streamline upwind eval.'s

    ttic = time.time()

    ss_edge, up_bias = _upwinding(
        mesh, trsk, cnfg, 
        sw_dual, ss_dual, ss_cell, uu_edge, vv_edge, 
        ss_edge, up_bias, 
        delta_t, sv_tiny, uu_tiny, 
        up_kind, up_min_, up_max_)
    
    ttoc = time.time()
    tcpu.upwinding = tcpu.upwinding + (ttoc - ttic)

    return ss_edge, up_bias


def compute_H(mesh, trsk, cnfg, hh_cell, uu_edge):

#-- compute discrete thickness

    ttic = time.time()
    
    hh_dual, hh_edge, h2_edge, hh_bias = \
        _computeHH(
            mesh, trsk, cnfg, hh_cell, uu_edge)
    
    ttoc = time.time()
    tcpu.compute_H = tcpu.compute_H + (ttoc - ttic)

    return hh_dual, hh_edge, h2_edge, hh_bias
    

def computeKE(mesh, trsk, cnfg, 
        hh_cell, hh_edge, hh_dual, uu_edge, vv_edge,
        delta_t):

#-- reconstruct kinetic energy

    ttic = time.time()

    up_edge = variables.ke_bias
 
    ke_cell = _computeKE(
        mesh, trsk, cnfg, uu_edge, vv_edge)
        
    ttoc = time.time()
    tcpu.computeKE = tcpu.computeKE + (ttoc - ttic)

    return ke_cell, up_edge


def _build_PV(mesh, trsk, cnfg, 
        hh_cell, hh_edge, hh_dual, uu_edge, vv_edge, 
        ff_dual, ff_edge, ff_cell,
        delta_t):
           
#-- compute discrete vorticity
              
    ttic = time.time()
              
    rv_dual, pv_dual, p2_dual, \
    rv_cell, pv_cell, \
    rv_edge, pv_edge = _computePV(
        mesh, trsk, cnfg, 
        hh_cell, hh_edge, hh_dual, uu_edge, vv_edge, 
        ff_dual, ff_edge, ff_cell)
    
    ttoc = time.time()
    tcpu.computePV = tcpu.computePV + (ttoc - ttic)

    return rv_dual, pv_dual, p2_dual, \
           rv_cell, pv_cell, \
           rv_edge, pv_edge
              
              
def computePV(mesh, trsk, cnfg, 
        hh_cell, hh_edge, hh_dual, uu_edge, vv_edge, 
        ff_dual, ff_edge, ff_cell,
        delta_t):
  
#-- compute discrete vorticity
  
    rv_dual, pv_dual, p2_dual, \
    rv_cell, pv_cell, \
    rv_edge, pv_edge = _build_PV(
        mesh, trsk, cnfg, 
        hh_cell, hh_edge, hh_dual, uu_edge, vv_edge, 
        ff_dual, ff_edge, ff_cell, 
        delta_t)
            
    up_edge = variables.pv_bias
            
    pv_edge, up_edge = upwinding(
        mesh, trsk, cnfg, 
        p2_dual, pv_dual, pv_cell, uu_edge, vv_edge, 
        pv_edge, up_edge,
        delta_t, PV_TINY, UU_TINY, 
        cnfg.pv_upwind, 
        cnfg.pv_min_up, cnfg.pv_max_up)
          
    return rv_dual, pv_dual, rv_cell, pv_cell, \
           pv_edge, up_edge
              
              
def computeVV(mesh, trsk, cnfg, uu_edge):

#-- get tangential velocity

    ttic = time.time()

    vv_edge = _computeVV(mesh, trsk, cnfg, uu_edge)

    ttoc = time.time()
    tcpu.computeVV = tcpu.computeVV + (ttoc - ttic)

    return vv_edge
              
              
def addtendUH(mesh, trsk, cnfg, hh_edge, uu_edge, 
                                hh_tend):

#-- div. for thickness flux

    ttic = time.time()

    hh_tend = _advect_UH(
        mesh, trsk, cnfg, 
            hh_edge, uu_edge, hh_tend)

    ttoc = time.time()
    tcpu.advect_UH = tcpu.advect_UH + (ttoc - ttic)

    return hh_tend
    
              
def addtendUV(mesh, trsk, cnfg, hh_edge, uu_edge,
                                pv_edge, ke_cell,
                                uu_tend):

#-- energy-neutral UV. flux

    ttic = time.time()

    uu_tend = _advect_UV(
        mesh, trsk, cnfg, hh_edge, 
            uu_edge, pv_edge, ke_cell, uu_tend)

    ttoc = time.time()
    tcpu.advect_UV = tcpu.advect_UV + (ttoc - ttic)

    return uu_tend
    
    
def addtendGZ(mesh, trsk, cnfg, hh_cell, zb_cell, 
                                gg_cell, uu_tend):

#-- get z pressure gradient

    ttic = time.time()

    uu_tend = _computeGZ(
        mesh, trsk, cnfg, 
            hh_cell, zb_cell, gg_cell, uu_tend)
        
    ttoc = time.time()
    tcpu.computeGZ = tcpu.computeGZ + (ttoc - ttic)

    return uu_tend
    

def addtendDU(mesh, trsk, cnfg, uu_edge, uu_tend):

#-- damping div^k operators

    if (cnfg.du_visc_k == 0): return uu_tend

    ttic = time.time()
    
    uu_tend = _computeDU(
        mesh, trsk, cnfg, uu_edge, uu_tend)

    ttoc = time.time()
    tcpu.computeDU = tcpu.computeDU + (ttoc - ttic)

    return uu_tend


def addtendVU(mesh, trsk, cnfg, uu_edge, uu_tend):

#-- viscous del^k operators

    if (cnfg.uu_visc_k == 0): return uu_tend

    ttic = time.time()
            
    uu_tend = _computeVU(
        mesh, trsk, cnfg, uu_edge, uu_tend)

    ttoc = time.time()
    tcpu.computeVU = tcpu.computeVU + (ttoc - ttic)

    return uu_tend
    
    
def addtendVH(mesh, trsk, cnfg, hh_cell, zb_cell, 
                                gg_cell, hh_tend):

#-- diffusive del^k operators

    if (cnfg.hh_diff_k == 0): return hh_tend

    ttic = time.time()

    hh_tend = _computeVH(
        mesh, trsk, cnfg, 
            hh_cell, zb_cell, gg_cell, hh_tend)
    
    ttoc = time.time()
    tcpu.computeVH = tcpu.computeVH + (ttoc - ttic)

    return hh_tend
    
    
def addtendTU(mesh, trsk, cnfg, Tu_edge, hh_edge, 
                                uu_tend):

#-- forcing from external tau

    if (Tu_edge is None): return uu_tend

    ttic = time.time()

    uu_tend = _computeTU(
        mesh, trsk, cnfg, 
            Tu_edge, hh_edge, uu_tend)
    
    ttoc = time.time()
    tcpu.computeTU = tcpu.computeTU + (ttoc - ttic)

    return uu_tend


def computeCd(mesh, trsk, cnfg, hh_cell, uu_edge):

#-- loglaw bottom drag term

    ttic = time.time()

    vv_edge = computeVV(
            mesh, trsk, cnfg, uu_edge)
            
    cd_edge = _computeCd(
        mesh, trsk, cnfg, 
            HH_TINY, hh_cell, uu_edge, vv_edge)
            
    ttoc = time.time()
    tcpu.computeCd = tcpu.computeCd + (ttoc - ttic)

    return cd_edge


try:
    # load cython kernels, if compiled
    from _kx import _computeBC
    from _kx import _upwinding
    from _kx import _computeHH
    from _kx import _computeKE
    from _kx import _computePV
    from _kx import _computeVV
    from _kx import _advect_UH
    from _kx import _advect_UV
    from _kx import _computeGZ
    from _kx import _computeDU
    from _kx import _computeVU
    from _kx import _computeVH
    from _kx import _computeTU
    from _kx import _computeCd

except ImportError:
    raise RuntimeError("Cython back-end not found")


