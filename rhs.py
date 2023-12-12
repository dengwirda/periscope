
import numpy as np

""" SWE rhs. evaluations for various Runge-Kutta methods 
"""
#-- Part of the PERISCOPE solver
#-- Darren Engwirda
#-- d.engwirda@gmail.com
#-- https://github.com/dengwirda/

from _fp import flt32_t, flt64_t
from _fp import reals_t, index_t

from log import tcpu

from _dx import computeBC, \
                compute_H, addtendUH, addtendVH, \
                computeKE, computePV, addtendUV, \
                computeVV, addtendGZ, \
                computeNu, addtendDU, addtendVU, \
                addtendTU

def rhs_slw_h(mesh, mats, flow, cnfg, hh_cell, uu_edge, hh_tend):

#-- evaluate slow tendencies dH/dt = RHS(t,U,H)

    return hh_tend


def rhs_fst_h(mesh, mats, flow, cnfg, hh_cell, uu_edge, hh_tend):

#-- evaluate fast tendencies dH/dt = RHS(t,U,H)

    if cnfg.no_h_tend: return hh_tend

    zb_cell = flow.zb_cell; gg_cell = flow.gravity
    
    hE_edge = flow.hE_edge
    uE_edge = flow.uE_edge

    hh_dual, hh_edge, h2_edge, hh_bias = \
              compute_H(mesh, mats, cnfg, hh_cell, uu_edge)

    hh_edge, uu_edge = computeBC(
        mesh, mats, cnfg, 
        hh_edge, uu_edge, 
        gg_cell, hE_edge, uE_edge)
  
    # thickness advection
    hh_tend = addtendUH(mesh, mats, cnfg, hh_edge, uu_edge, 
                                          hh_tend)
    
    # del^k dissipation
    hh_tend = addtendVH(mesh, mats, cnfg, hh_cell, zb_cell, 
                                          gg_cell, hh_tend)

    hh_tend[mesh.cell.mask] = reals_t(0.0)

    return hh_tend


def rhs_all_h(mesh, mats, flow, cnfg, hh_cell, uu_edge, hh_tend):
    
#-- evaluate full tendencies dH/dt = RHS(t,U,H)
    
    if (cnfg.calc_fast):
        hh_tend = rhs_fst_h(
            mesh, mats, flow, cnfg, hh_cell, uu_edge, hh_tend)
        
    if (cnfg.calc_slow):
        hh_tend = rhs_slw_h(
            mesh, mats, flow, cnfg, hh_cell, uu_edge, hh_tend)
        
    return hh_tend


def rhs_slw_u(mesh, mats, flow, cnfg, hh_cell, uu_edge, uu_tend):
    
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
              compute_H(mesh, mats, cnfg, hh_cell, uu_edge)
              
    hh_edge, uu_edge = computeBC(
        mesh, mats, cnfg, 
        hh_edge, uu_edge, 
        gg_cell, hE_edge, uE_edge)

    vv_edge = computeVV(
        mesh, mats, cnfg, uu_edge)

    ke_cell, ke_bias = computeKE(
        mesh, mats, cnfg, 
        hh_cell, hh_edge, hh_dual, 
        uu_edge, vv_edge,
        +1. / 2. * cnfg.time_step)

    rv_dual, pv_dual, r2_dual, p2_dual, \
    rv_cell, pv_cell, \
    pv_edge, pv_bias = computePV(
        mesh, mats, cnfg, 
        hh_cell, h2_edge, hh_dual, 
        uu_edge, vv_edge, 
        ff_dual, ff_edge, ff_cell, 
        +1. / 2. * cnfg.time_step)

    # nonlinear advection
    uu_tend = addtendUV(mesh, mats, cnfg, hh_edge, uu_edge,
                                          pv_edge, ke_cell, 
                                          uu_tend)

    # leith sub-grid
    nu_edge = computeNu(mesh, mats, cnfg, r2_dual, rv_cell)

    # div^k dissipation
    uu_tend = addtendDU(mesh, mats, cnfg, uu_edge, uu_tend)
    
    # del^k dissipation
    uu_tend = addtendVU(mesh, mats, cnfg, uu_edge, nu_edge,
                                          uu_tend)
    
    # external stresses
    uu_tend = addtendTU(mesh, mats, cnfg, Tu_edge, h2_edge,
                                          uu_tend)
    
    uu_tend[mesh.edge.mask] = reals_t(0.0)
    
    return uu_tend


def rhs_fst_u(mesh, mats, flow, cnfg, hh_cell, uu_edge, uu_tend):

#-- evaluate fast tendencies dU/dt = RHS(t,U,H)

    if cnfg.no_u_tend: return uu_tend

    zb_cell = flow.zb_cell; gg_cell = flow.gravity

    # pressure gradient
    uu_tend = addtendGZ(mesh, mats, cnfg, hh_cell, zb_cell, 
                                          gg_cell, uu_tend)
    
    uu_tend[mesh.edge.mask] = reals_t(0.0)
    
    return uu_tend


def rhs_all_u(mesh, mats, flow, cnfg, hh_cell, uu_edge, uu_tend):

#-- evaluate full tendencies dU/dt = RHS(t,U,H)

    if (cnfg.calc_fast):
        uu_tend = rhs_fst_u(
            mesh, mats, flow, cnfg, hh_cell, uu_edge, uu_tend)

    if (cnfg.calc_slow):
        uu_tend = rhs_slw_u(
            mesh, mats, flow, cnfg, hh_cell, uu_edge, uu_tend)

    return uu_tend


