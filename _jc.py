
import numpy as np
import jax
import jax.numpy as jnp
from flax import struct
from scipy.sparse import csr_matrix

from _fp import flt32_t, flt64_t
from _fp import reals_t, index_t
from _fp import udata_t, hdata_t, qdata_t

@struct.dataclass
class jxp_slv_params:
    time_step: flt64_t = 0.0
    next_step: flt64_t = 0.0
    timeisnow: flt64_t = 0.0

@struct.dataclass
class jxp_slv_consts:
    timestart: flt64_t = 0.0
    forc_ramp: flt64_t = 0.0

    gravity:   flt64_t = 9.80616

    cfl_limit: flt32_t = 0.0
    dt_margin: flt32_t = 0.0
    dt_cycles: flt32_t = 0.0

    pv_upwind: flt32_t = 0.0
    pv_weight: flt32_t = 0.0
    ke_upwind: flt32_t = 0.0
    ke_weight: flt32_t = 0.0
    ke_method: flt32_t = 0.0

    ref_scale: flt32_t = 0.0
    msh_fixes: flt32_t = 0.0
    uu_visc_2: flt32_t = 0.0
    uu_visc_4: flt32_t = 0.0
    hh_diff_2: flt32_t = 0.0
    hh_diff_4: flt32_t = 0.0

    leith_chi: flt32_t = 0.0
    leith_max: flt32_t = 0.0
    waves_chi: flt32_t = 0.0
    waves_max: flt32_t = 0.0

    linlaw_cd: flt32_t = 0.0
    sqrlaw_cd: flt32_t = 0.0
    loglaw_z0: flt32_t = 0.0
    loglaw_lo: flt32_t = 0.0
    loglaw_hi: flt32_t = 0.0
    manlaw_n0: flt32_t = 0.0
    manlaw_lo: flt32_t = 0.0
    manlaw_hi: flt32_t = 0.0

    wetdry_h0: flt32_t = 0.0

    sound_spd: flt32_t = 0.0

    sal_nfilt: flt32_t = 0.0
    sal_const: flt32_t = 0.0
    sal_scale: flt32_t = 0.0

@struct.dataclass
class jxp_usr_option:
    integrate: str = struct.field(pytree_node=False)
    
    hh_scheme: str = struct.field(pytree_node=False)
    pv_scheme: str = struct.field(pytree_node=False)
    ke_scheme: str = struct.field(pytree_node=False)

    wetdry_on: bool= struct.field(pytree_node=False)

    tidal_frc: str = struct.field(pytree_node=False)
    sal_solve: str = struct.field(pytree_node=False)

    no_advrot: bool= struct.field(pytree_node=False)
    no_geopot: bool= struct.field(pytree_node=False)
    no_stress: bool= struct.field(pytree_node=False)

@struct.dataclass
class jxp_parameters:
    params:     jxp_slv_params
    consts:     jxp_slv_consts
    option:     jxp_usr_option


def usr_to_jax(mesh, mats, flow, cnfg):

    cell_size = mesh.cell.size
    edge_size = mesh.edge.size
    dual_size = mesh.vert.size

    cnfg.jx = jxp_parameters(
        params = jxp_slv_params(
            time_step=cnfg.time_step,
            next_step=cnfg.time_step,
            timeisnow=cnfg.timeisnow,
        ),
        consts = jxp_slv_consts(
            timestart=cnfg.timestart,
            forc_ramp=cnfg.forc_ramp,

            cfl_limit=cnfg.cfl_limit,
            dt_margin=cnfg.dt_margin,
            dt_cycles=cnfg.dt_cycles,

            pv_upwind=cnfg.pv_upwind,
            pv_weight=cnfg.pv_weight,
            ke_upwind=cnfg.ke_upwind,
            ke_weight=cnfg.ke_weight,
            ke_method=cnfg.ke_method,

            ref_scale=cnfg.ref_scale,
            msh_fixes=cnfg.msh_fixes,
            uu_visc_2=cnfg.uu_visc_2,
            uu_visc_4=cnfg.uu_visc_4,
            hh_diff_2=cnfg.hh_diff_2,
            hh_diff_4=cnfg.hh_diff_4,

            leith_chi=cnfg.leith_chi,
            leith_max=cnfg.leith_max,
            waves_chi=cnfg.waves_chi,
            waves_max=cnfg.waves_max,

            linlaw_cd=cnfg.linlaw_cd,
            sqrlaw_cd=cnfg.linlaw_cd,
            loglaw_z0=cnfg.loglaw_z0,
            loglaw_lo=cnfg.loglaw_lo,
            loglaw_hi=cnfg.loglaw_hi,
            manlaw_n0=cnfg.manlaw_n0,
            manlaw_lo=cnfg.manlaw_lo,
            manlaw_hi=cnfg.manlaw_hi,

            wetdry_h0=cnfg.wetdry_h0,

            sound_spd=cnfg.sound_spd,
   
            sal_nfilt=cnfg.sal_nfilt,         
            sal_const=cnfg.sal_const,
            sal_scale=cnfg.sal_scale,
        ),
        option = jxp_usr_option(
            integrate=cnfg.integrate,

            hh_scheme=cnfg.hh_scheme,
            pv_scheme=cnfg.pv_scheme,
            ke_scheme=cnfg.ke_scheme,

            wetdry_on=cnfg.wetdry_h0 > 0.0,

            tidal_frc=cnfg.tidal_frc,
            sal_solve=cnfg.sal_solve,

            no_advrot=cnfg.no_advrot,
            no_geopot=cnfg.no_geopot,
            no_stress=cnfg.no_stress,
        ),
    )

    return cnfg


