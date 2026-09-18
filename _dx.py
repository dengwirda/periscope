
import time
import math
import jax
import jax.numpy as jnp
import numpy as np

""" SWE spatial discretisation using TRSK-like operators
"""
#-- Part of the PERISCOPE solver
#-- Darren Engwirda
#-- d.engwirda@gmail.com
#-- https://github.com/dengwirda/

from _fp import flt32_t, flt64_t
from _fp import reals_t, index_t

from _jo import op_product, pv_product

def scale_mix(mesh, mats, cnfg):

#-- local gridsize scaling on div^k and del^k operators

    # diam. of equiv. circle
    dx_cell = 2. * np.sqrt(mesh.cell.area / np.pi)

    # cell topo. dissipation
    sf_cell = 1. * cnfg.msh_fixes * \
        np.sqrt(np.abs(mesh.cell.topo - 6))
    sf_cell = np.asarray(sf_cell, dtype=reals_t)

    # smooth near grid-scale
    dx_edge = mats.edge_wing_sums * dx_cell
    dx_edge/= mesh.edge.area
    dx_cell = mats.cell_wing_sums * dx_edge
    dx_cell/= mesh.cell.area
    
    sf_edge = mats.edge_wing_sums * sf_cell
    sf_edge/= mesh.edge.area
    sf_cell = mats.cell_wing_sums * sf_edge
    sf_cell/= mesh.cell.area

    dx_edge = mats.edge_wing_sums * dx_cell
    dx_edge/= mesh.edge.area
    dx_cell = mats.cell_wing_sums * dx_edge
    dx_cell/= mesh.cell.area
    
    sf_edge = mats.edge_wing_sums * sf_cell
    sf_edge/= mesh.edge.area
    sf_cell = mats.cell_wing_sums * sf_edge
    sf_cell/= mesh.cell.area

    if (cnfg.ref_scale > 0.0):
        s2_edge = (dx_edge / cnfg.ref_scale) ** 1
        s4_edge = (dx_edge / cnfg.ref_scale) ** 3
    else:
        s2_edge = np.ones(
            (mesh.edge.size), dtype=reals_t)
        s4_edge = np.ones(
            (mesh.edge.size), dtype=reals_t)

    s2_edge*= (1. + sf_edge)
    s4_edge*= (1. + sf_edge)

    return s2_edge, s4_edge, sf_edge

 
def calc_vars(mesh, mats, flow, cnfg, hh_cell, uu_edge,
                                      qq_cell):

#-- compute diagnostic variables from the current state

    ff_dual = variables.ff_vert
    ff_edge = variables.ff_edge
    ff_cell = variables.ff_cell
    
    Xi_tide = variables.Xi_tide  # lagged values
    Xi_self = variables.Xi_self

    uu_filt = variables.uu_filt

    zb_cell = variables.zb_cell

    gravity = flow.gravity

    vv_edge = calc_perp(mesh, mats, cnfg, uu_edge)

    hh_dual, hh_edge, hh_quad, hh_bias = calc_hmap(
        mesh, mats, cnfg, 
        gravity, hh_cell, uu_edge, vv_edge)

    ke_cell, ke_bias = calc_u_ke(
        mesh, mats, cnfg, 
        hh_cell, hh_quad, hh_dual, uu_edge, vv_edge,
        +1. / 2. * cnfg.time_step)

    rv_dual, pv_dual, r2_dual, p2_dual, \
    rv_cell, pv_cell, \
    pv_edge, pv_bias = calc_u_pv(
        mesh, mats, cnfg, 
        hh_cell, hh_quad, hh_dual, uu_edge, vv_edge,
        ff_dual, ff_edge, ff_cell, 
        +1. / 2. * cnfg.time_step)
        
    nu_turb = variables.nu_turb  # lagged values

    nu_thin = variables.nu_thin

    nu_wave = variables.nu_wave
    os_wave = variables.os_wave

    nu_shoc = variables.nu_shoc
    os_shoc = variables.os_shoc

    return hh_edge, hh_dual, hh_bias, \
           ke_cell, ke_bias, \
           rv_cell, pv_cell, \
           rv_dual, pv_dual, pv_edge, pv_bias, \
           vv_edge, nu_turb, \
           nu_wave, os_wave, nu_shoc, os_shoc, \
           nu_thin, uu_filt, Xi_tide, Xi_self


def invariant(mesh, mats, flow, cnfg, hh_cell, uu_edge,
                                      qq_cell):

#-- compute the discrete energy and enstrophy invariants

    ff_dual = variables.ff_vert
    ff_edge = variables.ff_edge
    ff_cell = variables.ff_cell

    zb_cell = variables.zb_cell

    gravity = flow.gravity

    vv_edge = calc_perp(mesh, mats, cnfg, uu_edge)

    hh_dual, hh_edge, hh_quad, hh_bias = calc_hmap(
        mesh, mats, cnfg, 
        gravity, hh_cell, uu_edge, vv_edge)

    ke_edge = uu_edge ** 2
    ke_edge*= hh_edge * mesh.edge.area
    
    pe_cell = flow.gravity * (
        hh_cell * 0.5 + zb_cell - np.min(zb_cell))

    pe_cell*= hh_cell * mesh.cell.area

    kp_sums = math.fsum(ke_edge) \
            + math.fsum(pe_cell)

    rv_dual, pv_dual, rv_wide, pv_wide, \
    rv_cell, pv_cell, \
    pv_edge, pv_bias = calc_u_pv(
        mesh, mats, cnfg, 
        hh_cell, hh_quad, hh_dual, uu_edge, vv_edge,
        ff_dual, ff_edge, ff_cell, 
        +1. / 2. * cnfg.time_step)

    # include wet-dry ramp in pv budget
    hh_dtol = cnfg.wetdry_h0 + cnfg.hh_tiny
    hh_ramp = hh_dual / hh_dtol / 10. - .01
    hh_ramp = np.maximum(+0.0, 
              np.minimum(+1.0, hh_ramp))

    # pv is curl(u)+f here, so factor hh dependence
    pv_sums = 0.5 * math.fsum(
        mesh.vert.area * (hh_ramp ** 2)
                       * (pv_dual ** 2 / hh_dual))

    return kp_sums, pv_sums


def calc_obcs(mesh, mats, cnfg, 
        hh_edge, uu_edge, 
        gravity, hE_prev, uE_prev, hE_next, uE_next):
        
#-- setup open bnd. conditions
   
    if (hE_prev is None): return hh_edge, uu_edge
    if (uE_prev is None): return hh_edge, uu_edge
   
    ttic = time.time()
        
    hh_edge, uu_edge = _calc_obcs(
        mesh, mats, cnfg, 
        hh_edge, uu_edge, gravity, 
        hE_prev, uE_prev, hE_next, uE_next)
        
    ttoc = time.time()
    tcpu.calc_obcs = tcpu.calc_obcs + (ttoc - ttic)
        
    return hh_edge, uu_edge
    
    
def calc_udry(
        mesh, mats, cnfg, hh_edge, uu_edge, vv_edge):
        
#-- apply wet-dry velocity lim.
   
    nu_thin = variables.nu_thin

    if (cnfg.wetdry_h0 <= 0.): 
        return uu_edge, vv_edge, nu_thin
   
    ttic = time.time()
        
    hh_tiny = cnfg.wetdry_h0 * 10.0
    
    uu_edge, vv_edge, nu_thin = _calc_udry(
        mesh, mats, cnfg, 
            hh_tiny, hh_edge, uu_edge, vv_edge)
        
    ttoc = time.time()
    tcpu.calc_udry = tcpu.calc_udry + (ttoc - ttic)
        
    return uu_edge, vv_edge, nu_thin


def upwinding(mesh, mats, cnfg, 
        ss_wide, ss_dual, ss_cell, uu_edge, vv_edge, 
        ss_edge, up_bias,
        delta_t, sv_tiny, uu_tiny,
        up_kind, up_phi_):

#-- streamline upwind eval.'s

    ttic = time.time()

    ss_edge, up_bias = _upwinding(
        mesh, mats, cnfg, 
        ss_wide, ss_dual, ss_cell, uu_edge, vv_edge, 
        ss_edge, up_bias, 
        delta_t, sv_tiny, uu_tiny, 
        up_kind, up_phi_)

    ttoc = time.time()
    tcpu.upwinding = tcpu.upwinding + (ttoc - ttic)

    return ss_edge, up_bias


def calc_hmap(mesh, mats, cnfg, 
        gravity, hh_cell, uu_edge, vv_edge):

#-- compute discrete thickness

    if (cnfg.option.hh_scheme == "CENTRE"):

        hh_dual = op_product(
            mats.dual.kite_sums, hh_cell) / mesh.vert.area
        hh_edge = op_product(
            mats.edge.wing_sums, hh_cell) / mesh.edge.area

        hh_quad =(4.0 * hh_edge + op_product(
            mats.edge.dual_sums, hh_dual) ) / 6.0

        hh_bias = jnp.zeros(hh_edge.size, dtype=reals_t)

    else:  # option.hh_scheme == "UPWIND"

        hh_dual = op_product(
            mats.dual.kite_sums, hh_cell) / mesh.vert.area
        hh_edge = op_product(
            mats.edge.wing_sums, hh_cell) / mesh.edge.area

        hh_quad =(4.0 * hh_edge + op_product(
            mats.edge.dual_sums, hh_dual) ) / 6.0

    #-- compute the upwind thickness blend
        uu_wave = jnp.sqrt(uu_edge ** 2 + vv_edge ** 2)

        cel1 = mesh.edge.cell[:, 0] - 1
        cel2 = mesh.edge.cell[:, 1] - 1

        cel1 = jnp.where(cel1 >= 0, cel1, cel2)
        cel2 = jnp.where(cel2 >= 0, cel2, cel1)

        h1_cell = hh_cell[cel1].astype(reals_t)
        h2_cell = hh_cell[cel2].astype(reals_t)

        c1_wave = uu_wave + jnp.sqrt(gravity * h1_cell)
        c2_wave = uu_wave + jnp.sqrt(gravity * h2_cell)

    #-- upwind if the wavespeed ratio >> 1
        hh_bias = jnp.where(c2_wave>c1_wave, 
            jnp.maximum(jnp.sqrt(c2_wave/c1_wave - 1.), 
                                 h2_cell/h1_cell - 1.),
            jnp.maximum(jnp.sqrt(c1_wave/c2_wave - 1.), 
                                 h1_cell/h2_cell - 1.)
        )
        hh_bias = jnp.minimum(+1.0, hh_bias)

        hh_bias = hh_bias ** 3

        hh_edge = jnp.where(uu_edge >= 0.0, 
            hh_bias * h1_cell + (1.-hh_bias) * hh_edge,
            hh_bias * h2_cell + (1.-hh_bias) * hh_edge
        )
        
    return hh_dual, hh_edge, hh_quad, hh_bias
    

def calc_u_ke(mesh, mats, cnfg, 
        hh_cell, hh_edge, hh_dual, uu_edge, vv_edge,
        delta_t):

#-- reconstruct kinetic energy

    method_ke = cnfg.consts.ke_method
    weight_ke = cnfg.consts.ke_weight

#-- calc. kinetic energy on edges: 1./2 * |u|^2
    k1_edge = 1.0 * uu_edge ** 2
    k2_edge = 0.5 *(uu_edge ** 2 + vv_edge ** 2)

    ke_edge =(1.0 - method_ke) * k1_edge + \
             (0.0 + method_ke) * k2_edge

#-- remap kinetic energy M_(c,e) * 1./2 * |u|^2
    ke_edge = ke_edge / hh_edge ** 2
    ke_cell = op_product(
        mats.cell.wing_sums, ke_edge) / mesh.cell.area

    ke_cell = ke_cell * hh_cell ** 2

    ke_bias = jnp.zeros(hh_edge.size, dtype=reals_t)

    return ke_cell, ke_bias


def _build_pv(mesh, mats, cnfg, 
        hh_cell, hh_quad, hh_dual, ff_dual, ff_edge, ff_cell,
        uu_edge, vv_edge, 
        delta_t):
           
#-- compute discrete vorticity
              
    rv_dual = op_product(mats.dual.curl_sums, uu_edge)
   #rv_dual = rv_dual *  mesh.vert.slip

    rv_edge = op_product(mats.edge.dual_sums, rv_dual)

    rv_dual = rv_dual /  mesh.vert.area
    rv_edge = rv_edge /  mesh.quad.area

    pv_dual = rv_dual + ff_dual
    pv_edge = rv_edge + ff_edge

    rv_wide = op_product(mats.dual.edge_sums, rv_edge)
    rv_wide = rv_wide /  3.0 #!!

    rv_cell = op_product(mats.cell.kite_sums, rv_dual)
    rv_cell = rv_cell /  mesh.cell.area

    pv_wide = rv_wide + ff_dual
    pv_cell = rv_cell + ff_cell

    return rv_dual, pv_dual, rv_wide, pv_wide, \
           jnp.sqrt( jnp.sum(pv_wide ** 2 ) ), \
           rv_cell, pv_cell, \
           rv_edge, pv_edge
              
              
def calc_u_pv(mesh, mats, cnfg, 
        hh_cell, hh_quad, hh_dual, ff_dual, ff_edge, ff_cell,
        uu_edge, vv_edge,
        delta_t):
  
#-- compute discrete vorticity
  
    rv_dual, pv_dual, rv_wide, pv_wide, pv_rms_, \
    rv_cell, pv_cell, \
    rv_edge, pv_edge =  _build_pv(
        mesh, mats, cnfg, 
        hh_cell, hh_quad, hh_dual, 
        ff_dual, ff_edge, ff_cell, 
        uu_edge, vv_edge, delta_t)
    
    """
    uu_tiny = cnfg.uu_tiny * 1
    pv_tiny = cnfg.pv_tiny * 1
    pv_tiny = max (pv_tiny, 
        2.0 * np.finfo(reals_t).eps * pv_rms_)

    pv_edge, pv_bias =  upwinding(
        mesh, mats, cnfg, 
        pv_wide, pv_dual, pv_cell, uu_edge, vv_edge, 
        pv_edge, up_edge,
        delta_t, pv_tiny, uu_tiny, 
        cnfg.option.pv_scheme, 
        cnfg.consts.pv_upwind)
    """

    pv_bias = jnp.zeros(pv_edge.size, dtype=reals_t)

    return rv_dual, pv_dual, rv_wide, pv_wide, \
           rv_cell, pv_cell, \
           pv_edge, pv_bias

              
def calc_perp(mesh, mats, cnfg, uu_edge):

#-- get tangential velocity

    vv_edge =-op_product(mats.edge.lsqr_perp, uu_edge)

    return vv_edge
              
              
def tend_hadv(mesh, mats, cnfg, hh_edge, hh_cell, 
                                uu_edge,
                                gravity, 
                                hh_tend):

#-- div. for thickness flux

    c0 = 0. #!! cnfg.sound_spd
    gamma = (c0>0.) * gravity / (c0+1.)**2

    uh_flux = uu_edge * hh_edge \
            * (1.0 + 0.5 * gamma * hh_edge)

    hh_tend = hh_tend + ( op_product(
        mats.cell.flux_sums, uh_flux) / mesh.cell.area
            / (1.0 + 1.0 * gamma * hh_cell)
    )

    return hh_tend
    

def tend_uadv(mesh, mats, cnfg, 
        hh_edge, hh_quad, uu_edge, pv_edge, ke_cell,
        ff_dual, ff_edge, ff_cell,
        uu_tend):

#-- energy-neutral UV. flux

    pv_weight = cnfg.consts.pv_weight

    pv_sub_ =(pv_edge - ff_edge * pv_weight) / hh_quad
    pv_add_ =(pv_edge + ff_edge * pv_weight) / hh_quad
    uh_flux = uu_edge * hh_edge

    pv_flux = pv_product(mats.edge.flux_perp, uh_flux, 
                                              pv_sub_, 
                                              pv_add_)

    ke_grad = op_product(mats.edge.grad_norm, ke_cell)

    uu_tend = uu_tend \
        + mesh.edge.mask * (ke_grad - 0.500 * pv_flux)

    return uu_tend
    
    
def tend_upgf(mesh, mats, cnfg, hh_cell, zb_cell,
                                gravity,
                                xi_self,  
                                uu_tend):

#-- get z pressure gradient

    sal_const = 0.0; sal_scale = 1.0

    cel1 = mesh.edge.cell[:, 0] - 1
    cel2 = mesh.edge.cell[:, 1] - 1

    cel1 = jnp.where(cel1 >= 0, cel1, cel2)
    cel2 = jnp.where(cel2 >= 0, cel2, cel1)

    h1_cell = hh_cell[cel1].astype(reals_t)
    h2_cell = hh_cell[cel2].astype(reals_t)

    hh_min_ = 2.0 * ( h1_cell * h2_cell / 
                    ( h1_cell + h2_cell ) )

    zt_cell = zb_cell + hh_cell

#-- surface pressure gradient g * G * (h + z_b)
    zt_grad = op_product(mats.edge.grad_norm, zt_cell)

#-- attract.+loading gradient g * G * filt(z_t)
    xi_grad = op_product(mats.edge.grad_norm, xi_self)

#-- scalar, depth-weighted SAL: alpha * G * xi
    zt_grad = zt_grad - xi_grad * (sal_const * 
        jnp.minimum(1.0, jnp.sqrt(hh_min_/sal_scale)))
    
    uu_tend = uu_tend + \
        gravity * mesh.edge.mask * zt_grad

    return uu_tend

    
def calc_umix(mesh, mats, cnfg, rv_dual, rv_cell):

#-- compute leith viscosities

    nu_turb = variables.nu_turb

    if (cnfg.leith_chi == 0): return nu_turb

    ttic = time.time()

    nu_turb = _calc_umix(
        mesh, mats, cnfg, rv_dual, rv_cell)
    
    ttoc = time.time()
    tcpu.calc_umix = tcpu.calc_umix + (ttoc - ttic)

    return nu_turb


def calc_uwav(mesh, mats, cnfg, hh_cell, zb_cell,
                                gravity,
                                hh_edge, 
                                uu_edge, vv_edge):

#-- compute waves dissipation

    nu_wave = variables.nu_wave

    if (cnfg.waves_chi == 0): return nu_wave

    ttic = time.time()

    hh_tiny = cnfg.wetdry_h0 * 10.0

    nu_wave = _calc_uwav(
        mesh, mats, cnfg, 
            hh_cell, zb_cell, gravity, 
            hh_tiny, hh_edge, uu_edge, vv_edge)
    
    ttoc = time.time()
    tcpu.calc_uwav = tcpu.calc_uwav + (ttoc - ttic)

    return nu_wave


def calc_hmix(mesh, mats, cnfg, hh_cell, zb_cell,
                                gravity,
                                hh_edge, 
                                uu_edge, vv_edge):

#-- compute shock dissipation

    nu_shoc = variables.nu_shoc

    if (cnfg.shock_chi == 0): return nu_shoc

    ttic = time.time()

    hh_tiny = cnfg.wetdry_h0 * 10.0

    nu_shoc = _calc_hmix(
        mesh, mats, cnfg, 
            hh_cell, zb_cell, gravity, 
            hh_tiny, hh_edge, uu_edge, vv_edge)
    
    ttoc = time.time()
    tcpu.calc_hmix = tcpu.calc_hmix + (ttoc - ttic)

    return nu_shoc


def tend_umix(mesh, mats, cnfg, hh_cell, hh_edge, 
                                hh_quad, hh_dual, 
                                uu_edge,
                                nu_turb, nu_wave,
                                nu_thin,
                                uu_tend):

#-- viscous del^k operators

    if (cnfg.uu_visc_k == 0): return uu_tend

    ttic = time.time()
            
    hh_tiny = cnfg.wetdry_h0 * 100.

    uu_tend = _tend_umix(
        mesh, mats, cnfg, 
            hh_cell, hh_edge, hh_quad, hh_dual, 
            uu_edge, 
            nu_turb, nu_wave, nu_thin, 
            hh_tiny, uu_tend)

    ttoc = time.time()
    tcpu.tend_umix = tcpu.tend_umix + (ttoc - ttic)

    return uu_tend
    
    
def tend_hmix(mesh, mats, cnfg, hh_cell, zb_cell, 
                                gravity,
                                nu_shoc, 
                                hh_tend):

#-- diffusive del^k operators

    if (cnfg.hh_diff_k == 0): return hh_tend

    ttic = time.time()

    hh_tiny = cnfg.wetdry_h0 * 100.

    hh_tend = _tend_hmix(
        mesh, mats, cnfg, 
            hh_cell, zb_cell, 
            gravity, nu_shoc, hh_tiny, hh_tend)
    
    ttoc = time.time()
    tcpu.tend_hmix = tcpu.tend_hmix + (ttoc - ttic)

    return hh_tend


def calc_tide(mesh, mats, cnfg, gravity, Xi_tide):

#-- calc. ext. tidal forcing 

    if (cnfg.tidal_frc ==""): return Xi_tide

    ttic = time.time()

    tnow = cnfg.timeisnow

    Xi_tide = _calc_tide(
        mesh, mats, cnfg, tnow, gravity, Xi_tide)
        
    ttoc = time.time()
    tcpu.calc_tide = tcpu.calc_tide + (ttoc - ttic)

    return Xi_tide


def calc_self(mesh, mats, cnfg, hh_cell, zb_cell, 
                                gravity, 
                                Xi_self):

#-- calc. self loads geo-pot.

    if (cnfg.sal_solve ==""): return Xi_self

    ttic = time.time()

    tnow = cnfg.timeisnow

    Xi_self = _calc_self(
        mesh, mats, cnfg, 
        tnow, hh_cell, zb_cell, gravity, Xi_self)
        
    ttoc = time.time()
    tcpu.calc_self = tcpu.calc_self + (ttoc - ttic)

    return Xi_self


def tend_ugeo(mesh, mats, cnfg, Xi_prev, Xi_next,
                                hh_cell, 
                                gravity,
                                uu_tend):

#-- get grad of ext. geo-pot.

    if (Xi_prev is None or
            cnfg.no_geopot): return uu_tend

    ttic = time.time()

    hh_tiny = 1. * cnfg.hh_tiny

    uu_tend = _tend_ugeo(
        mesh, mats, cnfg, 
            gravity, hh_tiny,
            Xi_prev, Xi_next, hh_cell, uu_tend)
        
    ttoc = time.time()
    tcpu.tend_ugeo = tcpu.tend_ugeo + (ttoc - ttic)

    return uu_tend


def tend_utde(mesh, mats, cnfg, Xi_tide,
                                hh_cell, zb_cell,
                                gravity, 
                                uu_tend):

#-- get grad of tide geo-pot.

    if (cnfg.tidal_frc== ""):return uu_tend

    ttic = time.time()

    hh_tiny = cnfg.hh_tiny * 1.

    uu_tend = _tend_utde(
        mesh, mats, cnfg, 
            gravity, hh_tiny,
            Xi_tide, hh_cell, zb_cell, uu_tend)
        
    ttoc = time.time()
    tcpu.calc_tide = tcpu.calc_tide + (ttoc - ttic)

    return uu_tend
    
    
def tend_utau(mesh, mats, cnfg, Tu_prev, Tu_next,
                                hh_edge, 
                                uu_tend):

#-- forcing from ext. stress

    if (Tu_prev is None or
            cnfg.no_stress): return uu_tend

    ttic = time.time()

    hh_tiny = cnfg.wetdry_h0 * 10.0

    uu_tend = _tend_utau(
        mesh, mats, cnfg, hh_tiny,
            Tu_prev, Tu_next, hh_edge, uu_tend)
    
    ttoc = time.time()
    tcpu.tend_utau = tcpu.tend_utau + (ttoc - ttic)

    return uu_tend


def tend_uflt(mesh, mats, cnfg, uu_edge, hh_edge, 
                                uu_tend):

#-- filtered btr-bcl drag c_d

    if (cnfg.fltlaw_cd<=0. or
        cnfg.fltlaw_t0<=0.): return uu_tend

    ttic = time.time()

    uu_filt = variables.uu_filt

    uu_filt, uu_tend = _tend_uflt(
        mesh, mats, cnfg, 
            uu_edge, hh_edge, uu_filt, uu_tend)

    ttoc = time.time()
    tcpu.calc_drag = tcpu.calc_drag + (ttoc - ttic)

    return uu_tend


def calc_drag(mesh, mats, cnfg, gravity, dz_drag,
                                c1_edge, c2_edge,
                                z0_edge, 
                                n0_edge):

#-- composite bottom drag c_d

    ttic = time.time()
    
    ke_cell = variables.ke_cell  # from prev. eval.
    ke_edge = variables.ke_edge

    hh_edge = variables.hh_edge
    hh_quad = variables.hh_quad

    hh_tiny = cnfg.hh_tiny * 1.
    ke_tiny = cnfg.ke_tiny * 1.

    cd_edge = _calc_drag(
        mesh, mats, cnfg, hh_tiny, ke_tiny, 
            gravity, hh_edge, hh_quad,
            ke_cell, ke_edge, dz_drag,
            c1_edge, c2_edge,
            z0_edge, n0_edge)
            
    ttoc = time.time()
    tcpu.calc_drag = tcpu.calc_drag + (ttoc - ttic)

    return cd_edge


