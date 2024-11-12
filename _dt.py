
import time
import numpy as np

""" SWE time integration via various Runge-Kutta methods
"""
#-- Part of the PERISCOPE solver
#-- Darren Engwirda, Jeremy Lilly
#-- d.engwirda@gmail.com
#-- https://github.com/dengwirda/

from _fp import flt32_t, flt64_t
from _fp import reals_t, index_t

from log import tcpu

from mem import variables
from mem import get_vec_v, get_vec_e, get_vec_c, \
                put_vec_v, put_vec_e, put_vec_c

from _dx import computeCd

from rhs import rhs_all_u, rhs_slw_u, rhs_fst_u, \
                rhs_all_h, rhs_slw_h, rhs_fst_h

def mark_time(cnfg, flow, time):

#-- Update simulation time and interp. on forc. tendencies

    if (flow.xx_time is not None):
    
    #-- linear interp. between prev and next
        prev = flow.xx_time[flow.step - 1]
        next = flow.xx_time[flow.step - 0]
        cnfg.timeisnow = time
        cnfg.frc_blend = min (
            1.0, (time - prev) / (next - prev))
        
    else:
    
    #-- piecewise const. data, so do nothing
        cnfg.timeisnow = time
        cnfg.frc_blend = reals_t(0.0)
    
    return cnfg
    

def step_eqns(mesh, mats, flow, cnfg, 
              hh_cell, uu_edge,     # state
              ch_cell, cu_edge):    # compensators

#-- A single time-step - via user-defined method of choice

    if   ("RK22" in cnfg.integrate):

        hh_cell, uu_edge, \
        ch_cell, cu_edge = step_RK22(
            mesh, mats, flow, cnfg, 
            hh_cell, uu_edge, ch_cell, cu_edge)

    elif ("RK32" in cnfg.integrate):

        hh_cell, uu_edge, \
        ch_cell, cu_edge = step_RK32(
            mesh, mats, flow, cnfg, 
            hh_cell, uu_edge, ch_cell, cu_edge)

    return hh_cell, uu_edge, ch_cell, cu_edge
    
    
def step_bnds(mesh, mats, flow, cnfg, 
              hh_cell, uu_edge,     # state
              hh_min_, hh_max_,     # up/lo bounds
              uu_min_, uu_max_):
              
#-- Expand the min./max. status for each degree of freedom
              
    hh_min_, hh_max_ = \
        bnd_x_vec(cnfg, hh_cell, hh_min_, hh_max_)

    uu_min_, uu_max_ = \
        bnd_x_vec(cnfg, uu_edge, uu_min_, uu_max_)
    
    return hh_min_, hh_max_, uu_min_, uu_max_
    
    
def step_RK22(mesh, mats, flow, cnfg, 
              hh_cell, uu_edge,     # state
              ch_cell, cu_edge):    # compensators

#-- A 2-stage RK + FB scheme, a'la ROMS:
#-- A.F. Shchepetkin, J.C. McWilliams (2005): The regional oceanic 
#-- modeling system (ROMS): a split-explicit, free-surface, 
#-- topography-following-coordinate oceanic model
#-- doi.org/10.1016/j.ocemod.2004.08.002

#-- drag included via 2nd-order IMEX scheme

#-- low-precision support via compensated summation

    start_t = cnfg.timeisnow

    k1_step = 1.0 / 1.0 * cnfg.time_step
    k2_step = 1.0 / 1.0 * cnfg.time_step
    
    isFB = 1.0 * ("FB" in cnfg.integrate)

    gravity = flow.gravity
    c1_edge = flow.c1_edge
    c2_edge = flow.c2_edge
    z0_edge = flow.z0_edge
    n0_edge = flow.n0_edge

    h1_cell = variables.h1_cell
    u1_edge = variables.u1_edge
    h2_cell = variables.h2_cell
    u2_edge = variables.u2_edge

    hm_cell = variables.h3_cell
    um_edge = variables.u3_edge
      
    hb_cell = variables.hb_cell

    rh_cell = variables.hh_tend
    ru_edge = variables.uu_tend
    
#-- 1st RK + FB stage

    cnfg = mark_time(
        cnfg, flow, start_t + 0. / 1. * cnfg.time_step)

    ttic = time.time()

    if cnfg.fb_weight:
        BETA = cnfg.fb_weight[0] * isFB
    else:
        BETA = 0.333333333333333 * isFB

    rh_cell = rhs_all_h(
        mesh, mats, flow, cnfg, hh_cell, uu_edge, rh_cell)

    h1_cell = adv_x_fst(
        cnfg, hh_cell, k1_step, rh_cell, ch_cell, h1_cell)

    ttoc = time.time()
    tcpu.thickness+= (ttoc - ttic)

    ttic = time.time()

    hb_cell = sum_2_way(
        cnfg, hb_cell, 0.0 + 1.0 * BETA, h1_cell,
                       1.0 - 1.0 * BETA, hh_cell)

    ru_edge = rhs_all_u(
        mesh, mats, flow, cnfg, hb_cell, uu_edge, ru_edge)

    u1_edge = adv_x_fst(
        cnfg, uu_edge, k1_step, ru_edge, cu_edge, u1_edge)

    if (cnfg.calc_drag and cnfg.anylaw_cd > 0.):
        cd_edge = computeCd(
            mesh, mats, cnfg, gravity, u1_edge,
            c1_edge, c2_edge, 
            z0_edge, n0_edge)

    #-- euler scheme implicit solve
        u1_edge = inv_x_1st(
            cnfg, u1_edge, 1. * k1_step, cd_edge, uu_edge)

    ttoc = time.time()
    tcpu.momentum_+= (ttoc - ttic)

#-- 2nd RK + FB stage

    cnfg = mark_time(
        cnfg, flow, start_t + 1. / 1. * cnfg.time_step)

    ttic = time.time()

    if cnfg.fb_weight:
        BETA = cnfg.fb_weight[1] * isFB
    else:
        BETA = 0.666666666666667 * isFB

    hm_cell = sum_2_way(
        cnfg, hm_cell, 0.5, hh_cell, 
                       0.5, h1_cell)
    um_edge = sum_2_way(
        cnfg, um_edge, 0.5, uu_edge, 
                       0.5, u1_edge)

    rh_cell = rhs_all_h(
        mesh, mats, flow, cnfg, hm_cell, um_edge, rh_cell)

    # compensation for fp round-off
    h2_cell, ch_cell = adv_x_cmp(
        cnfg, hh_cell, k2_step, rh_cell, ch_cell, h2_cell)

    ttoc = time.time()
    tcpu.thickness+= (ttoc - ttic)

    ttic = time.time()

    hb_cell = sum_3_way(
        cnfg, hb_cell, 0.0 + 0.5 * BETA, h2_cell,
                       0.5 - 0.5 * BETA, h1_cell,
                       0.5             , hh_cell)

    ru_edge = rhs_all_u(
        mesh, mats, flow, cnfg, hb_cell, um_edge, ru_edge)

    # compensation for fp round-off
    u2_edge, cu_edge = adv_x_cmp(
        cnfg, uu_edge, k2_step, ru_edge, cu_edge, u2_edge)
    
    if (cnfg.calc_drag and cnfg.anylaw_cd > 0.):
    #-- KE evaluated at t+1/2 above
        cd_edge = computeCd(
            mesh, mats, cnfg, gravity, um_edge,
            c1_edge, c2_edge, 
            z0_edge, n0_edge)

    #-- theta scheme implicit solve
        u2_edge = inv_x_2nd(
            cnfg, u2_edge, .5 * k2_step, cd_edge, uu_edge)
            
        cu_edge = inv_x_1st(
            cnfg, cu_edge, 1. * k2_step, cd_edge, cu_edge)
            
    ttoc = time.time()
    tcpu.momentum_+= (ttoc - ttic)

    hh_cell = cpy_x_vec(cnfg, h2_cell, hh_cell)
    uu_edge = cpy_x_vec(cnfg, u2_edge, uu_edge)

    return  hh_cell, uu_edge, ch_cell, cu_edge


def step_RK32(mesh, mats, flow, cnfg, 
              hh_cell, uu_edge,     # state
              ch_cell, cu_edge):    # compensators

#-- A 3-stage RK scheme, a'la MPAS-A:
#-- L.J. Wicker, W.C. Skamarock (2002): Time-Splitting Methods 
#-- for Elastic Models Using Forward Time Schemes
#-- doi.org/10.1175/1520-0493(2002)130<2088:TSMFEM>2.0.CO;2

#-- but with FB weighting applied within each RK stage:
#-- J.R. Lilly, D. Engwirda, G. Capodaglio, R.L. Higdon and 
#-- M.R. Petersen (2023): CFL Optimized Forward-Backward Runge
#-- Kutta Schemes for the Shallow Water Equations 
#-- doi.org/10.1175/MWR-D-23-0113.1

#-- drag included via 2nd-order IMEX scheme

#-- low-precision support via compensated summation

    start_t = cnfg.timeisnow

    k1_step = 1.0 / 3.0 * cnfg.time_step
    k2_step = 1.0 / 2.0 * cnfg.time_step
    k3_step = 1.0 / 1.0 * cnfg.time_step

    isFB = 1.0 * ("FB" in cnfg.integrate)

    gravity = flow.gravity
    c1_edge = flow.c1_edge
    c2_edge = flow.c2_edge
    z0_edge = flow.z0_edge
    n0_edge = flow.n0_edge

    h1_cell = variables.h1_cell
    u1_edge = variables.u1_edge
    h2_cell = variables.h2_cell
    u2_edge = variables.u2_edge
    h3_cell = variables.h3_cell
    u3_edge = variables.u3_edge
      
    hb_cell = variables.hb_cell

    rh_cell = variables.hh_tend
    ru_edge = variables.uu_tend

#-- 1st RK + FB stage

    cnfg = mark_time(
        cnfg, flow, start_t + 0. / 1. * cnfg.time_step)

    ttic = time.time()

    if cnfg.fb_weight:
        BETA = cnfg.fb_weight[0] * isFB
    else:
        BETA = 0.311875000000000 * isFB

    rh_cell = rhs_all_h(
        mesh, mats, flow, cnfg, hh_cell, uu_edge, rh_cell)

    h1_cell = adv_x_fst(
        cnfg, hh_cell, k1_step, rh_cell, ch_cell, h1_cell)

    ttoc = time.time()
    tcpu.thickness+= (ttoc - ttic)

    ttic = time.time()

    hb_cell = sum_2_way(
        cnfg, hb_cell, 0.0 + 1.0 * BETA, h1_cell,
                       1.0 - 1.0 * BETA, hh_cell)

    ru_edge = rhs_all_u(
        mesh, mats, flow, cnfg, hb_cell, uu_edge, ru_edge)

    u1_edge = adv_x_fst(
        cnfg, uu_edge, k1_step, ru_edge, cu_edge, u1_edge)

    if (cnfg.calc_drag and cnfg.anylaw_cd > 0.):
        cd_edge = computeCd(
            mesh, mats, cnfg, gravity, u1_edge,
            c1_edge, c2_edge, 
            z0_edge, n0_edge)

    #-- euler scheme implicit solve
        u1_edge = inv_x_1st(
            cnfg, u1_edge, 1. * k1_step, cd_edge, uu_edge)

    ttoc = time.time()
    tcpu.momentum_+= (ttoc - ttic)

#-- 2nd RK + FB stage

    cnfg = mark_time(
        cnfg, flow, start_t + 1. / 3. * cnfg.time_step)

    ttic = time.time()

    if cnfg.fb_weight:
        BETA = cnfg.fb_weight[1] * isFB
    else:
        BETA = 0.425000000000000 * isFB

    rh_cell = rhs_all_h(
        mesh, mats, flow, cnfg, h1_cell, u1_edge, rh_cell)

    h2_cell = adv_x_fst(
        cnfg, hh_cell, k2_step, rh_cell, ch_cell, h2_cell)

    ttoc = time.time()
    tcpu.thickness+= (ttoc - ttic)

    ttic = time.time()

    hb_cell = sum_2_way(
        cnfg, hb_cell, 0.0 + 1.0 * BETA, h2_cell,
                       1.0 - 1.0 * BETA, hh_cell)

    # when FB is not in use, the data for h used to advance 
    # u in the second stage needs to be manually set to 
    # the first stage data for h
    hb_cell = sum_2_way(
        cnfg, hb_cell, 0.0 + 1.0 * isFB, hb_cell,
                       1.0 - 1.0 * isFB, h1_cell)
    
    ru_edge = rhs_all_u(
        mesh, mats, flow, cnfg, hb_cell, u1_edge, ru_edge)

    u2_edge = adv_x_fst(
        cnfg, uu_edge, k2_step, ru_edge, cu_edge, u2_edge)

    if (cnfg.calc_drag and cnfg.anylaw_cd > 0.):
        cd_edge = computeCd(
            mesh, mats, cnfg, gravity, u2_edge,
            c1_edge, c2_edge, 
            z0_edge, n0_edge)

    #-- euler scheme implicit solve
        u2_edge = inv_x_1st(
            cnfg, u2_edge, 1. * k2_step, cd_edge, uu_edge)

    ttoc = time.time()
    tcpu.momentum_+= (ttoc - ttic)

#-- 3rd RK + FB stage

    cnfg = mark_time(
        cnfg, flow, start_t + 1. / 2. * cnfg.time_step)

    ttic = time.time()

    if cnfg.fb_weight:
        BETA = cnfg.fb_weight[2] * isFB
    else:
        BETA = 0.362500000000000 * isFB

    rh_cell = rhs_all_h(
        mesh, mats, flow, cnfg, h2_cell, u2_edge, rh_cell)

    # compensation for fp round-off
    h3_cell, ch_cell = adv_x_cmp(
        cnfg, hh_cell, k3_step, rh_cell, ch_cell, h3_cell)

    ttoc = time.time()
    tcpu.thickness+= (ttoc - ttic)

    ttic = time.time()

    hb_cell = sym_3_way(
        cnfg, hb_cell, 0.0 + 1.0 * BETA, h3_cell,
                       1.0 - 2.0 * BETA, h2_cell,
                       0.0 + 1.0 * BETA, hh_cell)
    
    ru_edge = rhs_all_u(
        mesh, mats, flow, cnfg, hb_cell, u2_edge, ru_edge)

    # compensation for fp round-off
    u3_edge, cu_edge = adv_x_cmp(
        cnfg, uu_edge, k3_step, ru_edge, cu_edge, u3_edge)
   
    if (cnfg.calc_drag and cnfg.anylaw_cd > 0.):
    #-- KE evaluated at t+1/2 above
        cd_edge = computeCd(
            mesh, mats, cnfg, gravity, u2_edge,
            c1_edge, c2_edge, 
            z0_edge, n0_edge)

    #-- theta scheme explicit tend.
        u3_edge = inv_x_2nd(
            cnfg, u3_edge, .5 * k3_step, cd_edge, uu_edge)
            
    #-- theta scheme implicit solve
        cu_edge = inv_x_1st(
            cnfg, cu_edge, 1. * k3_step, cd_edge, cu_edge)

    ttoc = time.time()
    tcpu.momentum_+= (ttoc - ttic)

    hh_cell = cpy_x_vec(cnfg, h3_cell, hh_cell)
    uu_edge = cpy_x_vec(cnfg, u3_edge, uu_edge)
    
    return  hh_cell, uu_edge, ch_cell, cu_edge


try:
    # load cython kernels, if compiled
    from _kt import _bnd_x_vec as bnd_x_vec
    from _kt import _set_x_vec as set_x_vec
    from _kt import _cpy_x_vec as cpy_x_vec
    from _kt import _adv_x_fst as adv_x_fst
    from _kt import _adv_x_cmp as adv_x_cmp
    from _kt import _inv_x_1st as inv_x_1st
    from _kt import _inv_x_2nd as inv_x_2nd
    from _kt import _sum_2_way as sum_2_way
    from _kt import _sum_3_way as sum_3_way
    from _kt import _sym_3_way as sym_3_way
    
except ImportError:
    raise RuntimeError("Cython back-end not found")

