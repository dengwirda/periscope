
import numpy as np
import jax
import jax.numpy as jnp
from flax import struct
from scipy.sparse import csr_matrix

from _fp import flt32_t, flt64_t
from _fp import reals_t, index_t
from _fp import udata_t, hdata_t, qdata_t

@struct.dataclass
class jxp_prognostic:
    hh_cell:    jnp.ndarray
    uu_edge:    jnp.ndarray

@struct.dataclass
class jxp_diagnostic:
    vv_edge:    jnp.ndarray

    hh_dual:    jnp.ndarray
    hh_edge:    jnp.ndarray
    hh_quad:    jnp.ndarray            

    rv_dual:    jnp.ndarray
    pv_dual:    jnp.ndarray
    rv_wide:    jnp.ndarray
    pv_wide:    jnp.ndarray
    rv_edge:    jnp.ndarray
    pv_edge:    jnp.ndarray
    rv_cell:    jnp.ndarray
    pv_cell:    jnp.ndarray

    ke_edge:    jnp.ndarray
    ke_dual:    jnp.ndarray
    ke_cell:    jnp.ndarray

    hh_bias:    jnp.ndarray
    pv_bias:    jnp.ndarray
    ke_bias:    jnp.ndarray

    nu_turb:    jnp.ndarray
    nu_wave:    jnp.ndarray
    nu_shoc:    jnp.ndarray
    nu_thin:    jnp.ndarray

@struct.dataclass
class jxp_foundation:
    ff_cell:    jnp.ndarray
    ff_edge:    jnp.ndarray 
    ff_vert:    jnp.ndarray

    zb_cell:    jnp.ndarray

    dz_drag:    jnp.ndarray
    c1_edge:    jnp.ndarray
    c2_edge:    jnp.ndarray
    z0_edge:    jnp.ndarray
    n0_edge:    jnp.ndarray

    msh_fix:    jnp.ndarray
    msh_nu2:    jnp.ndarray
    msh_nu4:    jnp.ndarray
    visc_u2:    jnp.ndarray
    visc_u4:    jnp.ndarray
    diff_h2:    jnp.ndarray
    diff_h4:    jnp.ndarray

@struct.dataclass
class jxp_statistics:
    hh_min_:    jnp.ndarray
    uu_min_:    jnp.ndarray
    qq_min_:    jnp.ndarray
    hh_max_:    jnp.ndarray
    uu_max_:    jnp.ndarray
    qq_max_:    jnp.ndarray

    zt_rms_:    jnp.ndarray
    ke_ave_:    jnp.ndarray
    ke_rms_:    jnp.ndarray
    ke_max_:    jnp.ndarray
    dk_ave_:    jnp.ndarray
    dk_rms_:    jnp.ndarray
    dk_max_:    jnp.ndarray

@struct.dataclass
class jxp_flow_state:
    prognostic: jxp_prognostic
    diagnostic: jxp_diagnostic
    foundation: jxp_foundation
   #metric_var: jxp_statistics
    
def var_to_jax(mesh, mats, flow, cnfg):

    cell_size = mesh.cell.size
    edge_size = mesh.edge.size
    dual_size = mesh.vert.size

    flow.jx = jxp_flow_state(
        diagnostic= jxp_diagnostic(
            vv_edge=jnp.zeros(edge_size, dtype=reals_t),

            hh_dual=jnp.zeros(dual_size, dtype=reals_t),
            hh_edge=jnp.zeros(edge_size, dtype=reals_t),
            hh_quad=jnp.zeros(edge_size, dtype=reals_t),

            rv_dual=jnp.zeros(dual_size, dtype=reals_t),
            pv_dual=jnp.zeros(dual_size, dtype=reals_t),
            rv_wide=jnp.zeros(dual_size, dtype=reals_t),
            pv_wide=jnp.zeros(dual_size, dtype=reals_t),
            rv_edge=jnp.zeros(edge_size, dtype=reals_t),               
            pv_edge=jnp.zeros(edge_size, dtype=reals_t),
            rv_cell=jnp.zeros(cell_size, dtype=reals_t),
            pv_cell=jnp.zeros(cell_size, dtype=reals_t),

            ke_dual=jnp.zeros(dual_size, dtype=reals_t),
            ke_edge=jnp.zeros(edge_size, dtype=reals_t),            
            ke_cell=jnp.zeros(cell_size, dtype=reals_t),

            hh_bias=jnp.zeros(edge_size, dtype=reals_t),
            pv_bias=jnp.zeros(edge_size, dtype=reals_t),
            ke_bias=jnp.zeros(edge_size, dtype=reals_t),

            nu_turb=jnp.zeros(edge_size, dtype=reals_t),
            nu_wave=jnp.zeros(edge_size, dtype=reals_t),
            nu_shoc=jnp.zeros(edge_size, dtype=reals_t),
            nu_thin=jnp.zeros(edge_size, dtype=reals_t),
        ),
        prognostic= jxp_prognostic(
            hh_cell=jnp.asarray(flow.hh_cell),
            uu_edge=jnp.asarray(flow.uu_edge),
        ),
        foundation= jxp_foundation(
            ff_cell=jnp.asarray(flow.ff_cell),
            ff_edge=jnp.asarray(flow.ff_edge),
            ff_vert=jnp.asarray(flow.ff_vert),

            zb_cell=jnp.asarray(flow.zb_cell),

            dz_drag=jnp.asarray(flow.dz_drag),
            c1_edge=jnp.asarray(flow.c1_edge),
            c2_edge=jnp.asarray(flow.c2_edge),
            z0_edge=jnp.asarray(flow.z0_edge),
            n0_edge=jnp.asarray(flow.n0_edge),

            msh_fix=jnp.asarray(flow.msh_fix),
            msh_nu2=jnp.asarray(flow.msh_nu2),
            msh_nu4=jnp.asarray(flow.msh_nu4),
            visc_u2=jnp.asarray(flow.visc_u2),
            visc_u4=jnp.asarray(flow.visc_u4),
            diff_h2=jnp.asarray(flow.diff_h2),
            diff_h4=jnp.asarray(flow.diff_h4),
        ),
    )

    return flow


