
import time
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

""" SWE time integration via various Runge-Kutta methods
"""
#-- Part of the PERISCOPE solver
#-- Darren Engwirda, Jeremy Lilly
#-- d.engwirda@gmail.com
#-- https://github.com/dengwirda/

from _fp import flt32_t, flt64_t
from _fp import reals_t, index_t
from _fp import udata_t, hdata_t, qdata_t
from _fp import utend_t, htend_t, qtend_t

from _dx import calc_drag

from rhs import rhs_tde_d, rhs_all_d
from rhs import rhs_all_u, rhs_slw_u, rhs_fst_u
from rhs import rhs_pgf_u
from rhs import rhs_all_h, rhs_slw_h, rhs_fst_h

@jax.jit
def step_RK33(mesh, mats, flow, cnfg):

#-- A 3-stage 3rd/2nd-order RK scheme:
#-- D. Engwirda (2025): 3-, 4- and 5-stage forward-backward
#-- Runge-Kutta methods for geophysical flows

#-- drag included via a 2nd-order IMEX scheme

    """
    # scheme without forward-backward weights
    b_11 = 0.00 ; b_10 = 1.00 - b_11
    
    b_22 = 0.00 ; b_21 = 1.00
    b_20 = 1.00 - b_22 - b_21
    
    b_33 = 0.00 ; b_32 = 1.00 ; b_31 = 0.00
    b_30 = 1.00 - b_33 - b_32 - b_31
    """

    """
    # 2nd-order CFL=5.000 scheme
    # rational, with robust stability wedge re: background Froude
    # & (linear) 3rd-order cancelling
    b_11 =25./48; b_10 = 1.00 - b_11
    
    b_22 = 4./15; b_21 = 7./15
    b_20 = 1.00 - b_22 - b_21
    
    b_33 = 1./3.; b_32 = 7./48; b_31 = 5./24
    b_30 = 1.00 - b_33 - b_32 - b_31
    """

    # 3rd-order CFL=4.625 scheme
    # rational, with robust stability wedge re: background Froude
    b_11 = 3./4.; b_10 = 1.00 - b_11
    
    b_22 = 4./15; b_21 = 7./15
    b_20 = 1.00 - b_22 - b_21
    
    b_33 = 1./4.; b_32 = 3./16; b_31 = 3./8.
    b_30 = 1.00 - b_33 - b_32 - b_31


    uu_edge = flow.prognostic.uu_edge
    hh_cell = flow.prognostic.hh_cell

    gravity = cnfg.consts.gravity

    rk_base = flow.foundation
    rk_diag = flow.diagnostic

    uk_edge = uu_edge.copy()

    dt_step = cnfg.params.time_step

    k1_step = (1.0 / 3.0) * dt_step
    k2_step = (2.0 / 3.0) * dt_step
    k3_step = (1.0 / 1.0) * dt_step
    
    dz_drag = flow.foundation.dz_drag
    c1_edge = flow.foundation.c1_edge
    c2_edge = flow.foundation.c2_edge
    z0_edge = flow.foundation.z0_edge
    n0_edge = flow.foundation.n0_edge

#-- 1st RK + FB stage

    rk_diag = rhs_all_d(  # eval. diagnostics 
        mesh, mats, cnfg, rk_base, rk_diag, hh_cell, uk_edge)
    
    h0_tend = jnp.zeros(hh_cell.size, dtype=htend_t)
    h0_tend = rhs_all_h(
        mesh, mats, cnfg, rk_base, rk_diag, hh_cell, uk_edge, h0_tend)

    hk_tend = h0_tend.copy()

    h1_cell =(hh_cell - k1_step * hk_tend).astype(hdata_t)

    u0_tend = jnp.zeros(uu_edge.size, dtype=utend_t)
    u0_tend = rhs_slw_u(
        mesh, mats, cnfg, rk_base, rk_diag, hh_cell, uk_edge, u0_tend)
    u0_tend = rhs_fst_u(
        mesh, mats, cnfg, rk_base, rk_diag, hh_cell, uk_edge, u0_tend)

    hb_cell = b_11 * h1_cell + b_10 * hh_cell

    uk_tend = u0_tend.copy()
    uk_tend = rhs_pgf_u(
        mesh, mats, cnfg, rk_base, rk_diag, hb_cell, uk_edge, uk_tend)

    uk_edge =(uu_edge - k1_step * uk_tend).astype(udata_t)


#-- 2nd RK + FB stage

    rk_diag = rhs_all_d(  # eval. diagnostics 
        mesh, mats, cnfg, rk_base, rk_diag, h1_cell, uk_edge)

    hk_tend = jnp.zeros(hh_cell.size, dtype=htend_t)
    hk_tend = rhs_all_h(
        mesh, mats, cnfg, rk_base, rk_diag, h1_cell, uk_edge, hk_tend)

    h2_cell =(hh_cell - k2_step * hk_tend).astype(hdata_t)

    uk_tend = jnp.zeros(uu_edge.size, dtype=utend_t)
    uk_tend = rhs_slw_u(
        mesh, mats, cnfg, rk_base, rk_diag, h1_cell, uk_edge, uk_tend)
    uk_tend = rhs_fst_u(
        mesh, mats, cnfg, rk_base, rk_diag, h1_cell, uk_edge, uk_tend)
    
    hb_cell = b_22 * h2_cell + b_21 * h1_cell \
            + b_20 * hh_cell

    uk_tend = rhs_pgf_u(
        mesh, mats, cnfg, rk_base, rk_diag, hb_cell, uk_edge, uk_tend)

    uk_edge =(uu_edge - k2_step * uk_tend).astype(udata_t)

 
#-- 3rd RK + FB stage

    rk_diag = rhs_all_d(  # eval. diagnostics 
        mesh, mats, cnfg, rk_base, rk_diag, h2_cell, uk_edge)

    hk_tend = jnp.zeros(hh_cell.size, dtype=htend_t)
    hk_tend = rhs_all_h(
        mesh, mats, cnfg, rk_base, rk_diag, h2_cell, uk_edge, hk_tend)

    hk_tend = +1./4. * h0_tend + 3./4. * hk_tend
    
    h3_cell =(hh_cell - k3_step * hk_tend).astype(hdata_t)
    
    uk_tend = jnp.zeros(uu_edge.size, dtype=utend_t)
    uk_tend = rhs_slw_u(
        mesh, mats, cnfg, rk_base, rk_diag, h2_cell, uk_edge, uk_tend)
    uk_tend = rhs_fst_u(
        mesh, mats, cnfg, rk_base, rk_diag, h2_cell, uk_edge, uk_tend)

    hb_cell = b_33 * h3_cell + b_32 * h2_cell \
            + b_31 * h1_cell + b_30 * hh_cell

    uk_tend = +1./4. * u0_tend + 3./4. * uk_tend
    
    uk_tend = rhs_pgf_u(
        mesh, mats, cnfg, rk_base, rk_diag, hb_cell, uk_edge, uk_tend)

    uk_edge =(uu_edge - k3_step * uk_tend).astype(udata_t)
 


    flow = flow.replace(
        prognostic=flow.prognostic.replace(
            uu_edge=uk_edge,
            hh_cell=h3_cell,
        ),
        diagnostic= rk_diag,
    )
    
    return  flow, cnfg


@partial(
    jax.jit, static_argnums=(4,)
)
def step_eqns(mesh, mats, flow, cnfg, step):

    def body(carry, __):
        flow, cnfg = carry
        flow, cnfg = step_RK33(mesh, mats, flow, cnfg)
        return (flow, cnfg), None

    vals, __ = jax.lax.scan(
        body , (flow, cnfg), None, length=step)

    flow, cnfg = vals

    return flow, cnfg 


