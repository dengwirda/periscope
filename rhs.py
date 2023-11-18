
import numpy as np

""" SWE rhs. evaluations for various Runge-Kutta methods 
"""
#-- Darren Engwirda

from _fp import flt32_t, flt64_t
from _fp import reals_t, index_t

from log import tcpu

from _dx import compute_H, addtendUH, \
                computeKE, computePV, addtendUV, \
                computeVV, addtendGZ, \
                addtendDU, addtendVU, addtendVH

def rhs_slw_h(mesh, trsk, flow, cnfg, hh_cell, uu_edge, rh_cell):

#-- evaluate slow tendencies dH/dt = RHS(t,U,H)

    return rh_cell


def rhs_fst_h(mesh, trsk, flow, cnfg, hh_cell, uu_edge, rh_cell):

#-- evaluate fast tendencies dH/dt = RHS(t,U,H)

    if cnfg.no_h_tend: return rh_cell

    zb_cell = flow.zb_cell; gg_cell = flow.grav

    hh_dual, hh_edge, h2_edge = \
              compute_H(mesh, trsk, cnfg, hh_cell, uu_edge)

    rh_cell = addtendUH(mesh, trsk, cnfg, hh_edge, uu_edge, 
                                          rh_cell)
    
    rh_cell = addtendVH(mesh, trsk, cnfg, hh_cell, zb_cell, 
                                          gg_cell, rh_cell)

    rh_cell[mesh.cell.mask] = reals_t(0.0)

    return rh_cell


def rhs_all_h(mesh, trsk, flow, cnfg, hh_cell, uu_edge, rh_cell):
    
#-- evaluate full tendencies dH/dt = RHS(t,U,H)
    
    if (cnfg.calc_fast):
        rh_cell = rhs_fst_h(
            mesh, trsk, 
            flow, cnfg, hh_cell, uu_edge, rh_cell)
        
    if (cnfg.calc_slow):
        rh_cell = rhs_slw_h(
            mesh, trsk, 
            flow, cnfg, hh_cell, uu_edge, rh_cell)
        
    return rh_cell


def rhs_slw_u(mesh, trsk, flow, cnfg, hh_cell, uu_edge, ru_edge):
    
#-- evaluate slow tendencies dU/dt = RHS(t,U,H)
    
    if cnfg.no_u_tend: return ru_edge
    
    ff_cell = flow.ff_cell
    ff_edge = flow.ff_edge
    ff_dual = flow.ff_vert

    vv_edge = computeVV(mesh, trsk, cnfg, uu_edge)

    hh_dual, hh_edge, h2_edge = \
              compute_H(mesh, trsk, cnfg, hh_cell, uu_edge)

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

    ru_edge = addtendUV(
        mesh, trsk, cnfg, 
        hh_edge, uu_edge, 
        pv_edge, ke_cell, ru_edge)

    ru_edge = addtendDU(mesh, trsk, cnfg, uu_edge, ru_edge)
    
    ru_edge = addtendVU(mesh, trsk, cnfg, uu_edge, ru_edge)
    
    
    #!!
    ru_edge-= flow.Tu_edge[0, :, 0] / hh_edge
    
    
    ru_edge[mesh.edge.mask] = reals_t(0.0)
    
    return ru_edge


def rhs_fst_u(mesh, trsk, flow, cnfg, hh_cell, uu_edge, ru_edge):

#-- evaluate fast tendencies dU/dt = RHS(t,U,H)

    if cnfg.no_u_tend: return ru_edge

    zb_cell = flow.zb_cell; gg_cell = flow.grav

    ru_edge = addtendGZ(mesh, trsk, cnfg, hh_cell, zb_cell, 
                                          gg_cell, ru_edge)
    
    ru_edge[mesh.edge.mask] = reals_t(0.0)
    
    return ru_edge


def rhs_all_u(mesh, trsk, flow, cnfg, hh_cell, uu_edge, ru_edge):

#-- evaluate full tendencies dU/dt = RHS(t,U,H)

    if (cnfg.calc_fast):
        ru_edge = rhs_fst_u(
            mesh, trsk, 
            flow, cnfg, hh_cell, uu_edge, ru_edge)

    if (cnfg.calc_slow):
        ru_edge = rhs_slw_u(
            mesh, trsk, 
            flow, cnfg, hh_cell, uu_edge, ru_edge)

    return ru_edge


