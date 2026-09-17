
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
    time_step: float
    timestart: float
    timeisnow: float

    cfl_limit: float
    dt_margin: float
    dt_cycles: float

    pv_upwind: float
    pv_weight: float
    ke_upwind: float
    ke_weight: float
    ke_method: float

    leith_chi: float
    leith_max: float
    waves_chi: float
    waves_max: float

    linlaw_cd: float
    sqrlaw_cd: float
    loglaw_z0: float
    loglaw_lo: float
    loglaw_hi: float
    manlaw_n0: float
    manlaw_lo: float
    manlaw_hi: float

    wetdry_h0: float

@struct.dataclass
class jxp_usr_option:
    integrate: str = struct.field(pytree_node=False)
    hh_scheme: str = struct.field(pytree_node=False)
    pv_scheme: str = struct.field(pytree_node=False)
    ke_scheme: str = struct.field(pytree_node=False)

   #tidal_frc: str = struct.field(pytree_node=False)
   #sal_solve: str = struct.field(pytree_node=False)

@struct.dataclass
class jxp_parameters:
    params:     jxp_slv_params
    option:     jxp_usr_option


def usr_to_jax(mesh, mats, flow, cnfg):

    cell_size = mesh.cell.size
    edge_size = mesh.edge.size
    dual_size = mesh.vert.size

    cnfg.jx = jxp_parameters(
        params = jxp_slv_params(
            time_step=cnfg.time_step,
            timestart=cnfg.timestart,
            timeisnow=cnfg.timeisnow,

            cfl_limit=cnfg.cfl_limit,
            dt_margin=cnfg.dt_margin,
            dt_cycles=cnfg.dt_cycles,

            pv_upwind=cnfg.pv_upwind,
            pv_weight=cnfg.pv_weight,
            ke_upwind=cnfg.ke_upwind,
            ke_weight=cnfg.ke_weight,
            ke_method=cnfg.ke_method,

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
        ),
        option = jxp_usr_option(
            integrate=cnfg.integrate,
            hh_scheme=cnfg.hh_scheme,
            pv_scheme=cnfg.pv_scheme,
            ke_scheme=cnfg.ke_scheme,
        ),
    )

    return cnfg


