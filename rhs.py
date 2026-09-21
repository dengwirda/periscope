
import jax
import jax.numpy as jnp
import numpy as np

""" SWE rhs. evaluations for various Runge-Kutta methods 
"""
#-- Part of the PERISCOPE solver
#-- Darren Engwirda
#-- d.engwirda@gmail.com
#-- https://github.com/dengwirda/

from _fp import flt32_t, flt64_t
from _fp import reals_t, index_t
from _fp import udata_t, hdata_t, qdata_t
from _fp import utend_t, htend_t, qtend_t

from _jv import jxp_prognostic
from _jv import jxp_diagnostic
from _jv import jxp_foundation

from _dx import calc_udry
from _dx import calc_hmap, tend_hadv
from _dx import calc_u_ke, calc_u_pv, calc_perp
from _dx import tend_uadv, tend_upgf

"""
from _dx import calc_obcs, \
                calc_umix, calc_uwav, calc_hmix, \
                tend_umix, tend_hmix, \
                tend_utde, calc_tide, calc_self
"""

def rhs_tde_d(mesh, mats, cnfg, base, diag, hh_cell, uu_edge):
    
#-- evaluate tide tendency diagnostics
    
    """
    zb_cell = base.zb_cell 

    gravity = cnfg.consts.gravity

    xi_tide = diag.xi_tide
    xi_self = diag.xi_self

    # tidal forcing
    xi_tide = calc_tide(mesh, mats, cnfg, gravity, xi_tide)
    
    xi_self = calc_self(mesh, mats, cnfg, hh_cell, zb_cell, 
                                          gravity, xi_self)
    """


def rhs_all_d(mesh, mats, cnfg, base, diag, hh_cell, uu_edge):

#-- evaluate full tendency diagnostics

    zb_cell = base.zb_cell

    ff_cell = base.ff_cell
    ff_edge = base.ff_edge
    ff_dual = base.ff_vert

    gravity = cnfg.consts.gravity

    hr_cell = hh_cell.astype(dtype=reals_t)

    # construct vel^\perp
    vv_edge = calc_perp(mesh, mats, cnfg, uu_edge)
    
    uu_sqr_ = uu_edge ** 2 +  \
              vv_edge ** 2
    uu_mag_ = jnp.sqrt(uu_sqr_)

    # construct thickness
    hh_dual, hh_edge, hh_quad, hh_bias = \
              calc_hmap(mesh, mats, cnfg, gravity, hr_cell, 
                                          uu_edge, vv_edge,
                                          uu_mag_)
    
    """
    # do extrap. for OBCs
    hh_edge, uu_edge = \
              calc_obcs(mesh, mats, cnfg, hh_edge, uu_edge,
                                          gravity, 
                                          hE_prev, uE_prev, 
                                          hE_next, uE_next)
    """

    # apply wet-dry limit
    # here, so shall feedback on nonlinear terms
    uu_edge, vv_edge, nu_thin = \
              calc_udry(mesh, mats, cnfg, hh_edge, 
                                          uu_edge, vv_edge)

    # nonlinear variables: kinetic energy & curl
    ke_cell, ke_bias = calc_u_ke(
        mesh, mats, cnfg, 
        hr_cell, hh_quad, hh_dual, 
        uu_edge, vv_edge, uu_sqr_,
                          +1. / 2. * cnfg.params.time_step)

    uu_tiny = diag.uu_tiny
    pv_tiny = diag.pv_tiny

    rv_dual, pv_dual, rv_wide, pv_wide, \
    rv_cell, pv_cell, \
    pv_edge, pv_bias = calc_u_pv(
        mesh, mats, cnfg, 
        hr_cell, hh_quad, hh_dual,
        ff_dual, ff_edge, ff_cell, 
        uu_edge, vv_edge, uu_mag_,
        uu_tiny, pv_tiny, +1. / 2. * cnfg.params.time_step)

    """
    # waves sub-grid
    nu_wave = calc_uwav(mesh, mats, cnfg, hh_cell, zb_cell,
                                          gravity,
                                          hh_edge,
                                          uu_edge, vv_edge)

    # leith sub-grid
    nu_turb = calc_umix(mesh, mats, cnfg, rv_wide, rv_cell)
    """
    
    return diag.replace(
        vv_edge=vv_edge, 
        hr_cell=hr_cell,
        hh_dual=hh_dual, hh_edge=hh_edge, hh_quad=hh_quad,
        ke_cell=ke_cell,
        rv_dual=rv_dual, pv_dual=pv_dual,
        rv_cell=rv_cell, pv_cell=pv_cell, 
        pv_edge=pv_edge,
        hh_bias=hh_bias, pv_bias=pv_bias
    )


def rhs_slw_h(mesh, mats, cnfg, base, diag, hh_cell, uu_edge, hh_tend):

#-- evaluate slow tendencies dH/dt = RHS(t,U,H)

    return hh_tend


def rhs_fst_h(mesh, mats, cnfg, base, diag, hh_cell, uu_edge, hh_tend):

#-- evaluate fast tendencies dH/dt = RHS(t,U,H)

    zb_cell = base.zb_cell

    gravity = cnfg.consts.gravity
    
    hr_cell = diag.hr_cell
    hh_edge = diag.hh_edge

    # thickness advection
    hh_tend = tend_hadv(mesh, mats, cnfg, hh_edge, hr_cell,
                                          uu_edge,
                                          gravity, 
                                          hh_tend)

    """
    # del^k dissipation
    hh_tend = tend_hmix(mesh, mats, cnfg, hr_cell, zb_cell, 
                                          gravity, 
                                          hh_tend)
    """

    return hh_tend


def rhs_all_h(mesh, mats, cnfg, base, diag, hh_cell, uu_edge, hh_tend):
    
#-- evaluate full tendencies dH/dt = RHS(t,U,H)
    
    hh_tend = rhs_fst_h(
        mesh, mats, cnfg, base, diag, hh_cell, uu_edge, hh_tend)
        
    hh_tend = rhs_slw_h(
        mesh, mats, cnfg, base, diag, hh_cell, uu_edge, hh_tend)
        
    return hh_tend


def rhs_slw_u(mesh, mats, cnfg, base, diag, hh_cell, uu_edge, uu_tend):
    
#-- evaluate slow tendencies dU/dt = RHS(t,U,H)

    zb_cell = base.zb_cell

    ff_cell = base.ff_cell
    ff_edge = base.ff_edge
    ff_dual = base.ff_vert

    gravity = cnfg.consts.gravity

    xi_tide = diag.xi_tide

    hr_cell = diag.hr_cell

    hh_dual = diag.hh_dual
    hh_edge = diag.hh_edge
    hh_quad = diag.hh_quad

    ke_cell = diag.ke_cell
    pv_edge = diag.pv_edge

    # nonlinear advection
    uu_tend = tend_uadv(mesh, mats, cnfg, hh_edge, hh_quad,
                                          uu_edge,
                                          pv_edge, ke_cell,
                                 ff_dual, ff_edge, ff_cell, 
                                          uu_tend)

    """
    # btr-bcl dissipation
    uu_tend = tend_uflt(mesh, mats, cnfg, uu_edge, hh_edge, 
                                          uu_tend)

    # external tend's here re: flux split
    # external geo-pot.
    uu_tend = tend_ugeo(mesh, mats, cnfg, Xi_prev, Xi_next,
                                          hh_cell,
                                          gravity,
                                          uu_tend)

    # tide+SAL geo-pot.
    uu_tend = tend_utde(mesh, mats, cnfg, xi_tide,
                                          hh_cell, zb_cell,
                                          gravity,
                                          uu_tend)

    # external stresses
    uu_tend = tend_utau(mesh, mats, cnfg, Tu_prev, Tu_next,
                                          hh_edge,
                                          uu_tend)
    """
    
    return uu_tend


def rhs_fst_u(mesh, mats, cnfg, base, diag, hh_cell, uu_edge, uu_tend):

#-- evaluate fast tendencies dU/dt = RHS(t,U,H)

    """
    if cnfg.no_u_tend or not cnfg.calc_fast:return uu_tend

    zb_cell = base.zb_cell

    gravity = cnfg.consts.gravity

    hr_cell = diag.hr_cell

    hh_dual = diag.hh_dual
    hh_edge = diag.hh_edge
    hh_quad = diag.hh_quad
    
    nu_turb = diag.nu_turb
    nu_wave = diag.nu_wave
    nu_thin = diag.nu_thin

    # del^k dissipation
    uu_tend = tend_umix(mesh, mats, cnfg, hr_cell, hh_edge, 
                                          hh_quad, hh_dual, 
                                          uu_edge,
                                          nu_turb, nu_wave,
                                          nu_thin,
                                          uu_tend)
    """

    return uu_tend


def rhs_pgf_u(mesh, mats, cnfg, base, diag, hh_cell, uu_edge, uu_tend):

#-- evaluate hPGF tendencies dU/dt = RHS(t,U,H)

    zb_cell = base.zb_cell

    xi_self = diag.xi_self

    gravity = cnfg.consts.gravity

    # pressure gradient
    uu_tend = tend_upgf(mesh, mats, cnfg, hh_cell, zb_cell, 
                                          gravity, 
                                          xi_self,
                                          uu_tend)
    
    return uu_tend


def rhs_all_u(mesh, mats, cnfg, base, diag, hh_cell, uu_edge, uu_tend):

#-- evaluate full tendencies dU/dt = RHS(t,U,H)

    uu_tend = rhs_slw_u(
        mesh, mats, cnfg, base, diag, hh_cell, uu_edge, uu_tend)

    uu_tend = rhs_fst_u(
        mesh, mats, cnfg, base, diag, hh_cell, uu_edge, uu_tend)

    uu_tend = rhs_pgf_u(
        mesh, mats, cnfg, base, diag, hh_cell, uu_edge, uu_tend)

    return uu_tend


