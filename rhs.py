
import numpy as np

""" SWE rhs. evaluations for various Runge-Kutta methods 
"""
#-- Darren Engwirda

from _fp import flt32_t, flt64_t
from _fp import reals_t, index_t

from log import tcpu

from _dx import computeBC, compute_H, addtendUH, \
                computeKE, computePV, addtendUV, \
                computeVV, addtendGZ, \
                addtendDU, addtendVU, addtendVH, \
                addtendTU

def rhs_slw_h(mesh, trsk, flow, cnfg, hh_cell, uu_edge, hh_tend):

#-- evaluate slow tendencies dH/dt = RHS(t,U,H)

    return hh_tend


def rhs_fst_h(mesh, trsk, flow, cnfg, hh_cell, uu_edge, hh_tend):

#-- evaluate fast tendencies dH/dt = RHS(t,U,H)

    if cnfg.no_h_tend: return hh_tend

    zb_cell = flow.zb_cell; gg_cell = flow.gravity
    
    hE_edge = flow.hE_edge
    uE_edge = flow.uE_edge

    hh_dual, hh_edge, h2_edge, hh_bias = \
              compute_H(mesh, trsk, cnfg, hh_cell, uu_edge)

    hh_edge, uu_edge = computeBC(
        mesh, trsk, cnfg, 
        hh_edge, uu_edge, 
        gg_cell, hE_edge, uE_edge)
  
    hh_tend = addtendUH(mesh, trsk, cnfg, hh_edge, uu_edge, 
                                          hh_tend)
    
    hh_tend = addtendVH(mesh, trsk, cnfg, hh_cell, zb_cell, 
                                          gg_cell, hh_tend)

    hh_tend[mesh.cell.mask] = reals_t(0.0)

    return hh_tend


def rhs_all_h(mesh, trsk, flow, cnfg, hh_cell, uu_edge, hh_tend):
    
#-- evaluate full tendencies dH/dt = RHS(t,U,H)
    
    if (cnfg.calc_fast):
        hh_tend = rhs_fst_h(
            mesh, trsk, flow, cnfg, hh_cell, uu_edge, hh_tend)
        
    if (cnfg.calc_slow):
        hh_tend = rhs_slw_h(
            mesh, trsk, flow, cnfg, hh_cell, uu_edge, hh_tend)
        
    return hh_tend


def rhs_slw_u(mesh, trsk, flow, cnfg, hh_cell, uu_edge, uu_tend):
    
#-- evaluate slow tendencies dU/dt = RHS(t,U,H)
    
    if cnfg.no_u_tend: return uu_tend
    
    gg_cell = flow.gravity
    
    ff_cell = flow.ff_cell
    ff_edge = flow.ff_edge
    ff_dual = flow.ff_vert
    
    Tu_edge = flow.Tu_edge
    hE_edge = flow.hE_edge
    uE_edge = flow.uE_edge

    hh_dual, hh_edge, h2_edge, hh_bias = \
              compute_H(mesh, trsk, cnfg, hh_cell, uu_edge)
              
    hh_edge, uu_edge = computeBC(
        mesh, trsk, cnfg, 
        hh_edge, uu_edge, 
        gg_cell, hE_edge, uE_edge)

    vv_edge = computeVV(
        mesh, trsk, cnfg, uu_edge)

    ke_cell, ke_bias = computeKE(
        mesh, trsk, cnfg, 
        hh_cell, hh_edge, hh_dual, 
        uu_edge, vv_edge,
        +1. / 2. * cnfg.time_step)

    rv_dual, pv_dual, rv_cell, pv_cell, \
    pv_edge, pv_bias = computePV(
        mesh, trsk, cnfg, 
        hh_cell, h2_edge, hh_dual, 
        uu_edge, vv_edge, 
        ff_dual, ff_edge, ff_cell, 
        +1. / 2. * cnfg.time_step)

    uu_tend = addtendUV(mesh, trsk, cnfg, hh_edge, uu_edge,
                                          pv_edge, ke_cell, 
                                          uu_tend)

    uu_tend = addtendDU(mesh, trsk, cnfg, uu_edge, uu_tend)
    
    uu_tend = addtendVU(mesh, trsk, cnfg, uu_edge, uu_tend)
    
    uu_tend = addtendTU(mesh, trsk, cnfg, Tu_edge, h2_edge,
                                          uu_tend)
    
    uu_tend[mesh.edge.mask] = reals_t(0.0)
    
    return uu_tend


def rhs_fst_u(mesh, trsk, flow, cnfg, hh_cell, uu_edge, uu_tend):

#-- evaluate fast tendencies dU/dt = RHS(t,U,H)

    if cnfg.no_u_tend: return uu_tend

    zb_cell = flow.zb_cell; gg_cell = flow.gravity

    uu_tend = addtendGZ(mesh, trsk, cnfg, hh_cell, zb_cell, 
                                          gg_cell, uu_tend)
    
    uu_tend[mesh.edge.mask] = reals_t(0.0)
    
    return uu_tend


def rhs_all_u(mesh, trsk, flow, cnfg, hh_cell, uu_edge, uu_tend):

#-- evaluate full tendencies dU/dt = RHS(t,U,H)

    if (cnfg.calc_fast):
        uu_tend = rhs_fst_u(
            mesh, trsk, flow, cnfg, hh_cell, uu_edge, uu_tend)

    if (cnfg.calc_slow):
        uu_tend = rhs_slw_u(
            mesh, trsk, flow, cnfg, hh_cell, uu_edge, uu_tend)

    return uu_tend


