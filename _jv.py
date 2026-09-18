
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
    hh_cell:    jnp.ndarray | None = None
    uu_edge:    jnp.ndarray | None = None

@struct.dataclass
class jxp_diagnostic:
    vv_edge:    jnp.ndarray | None = None

    hr_cell:    jnp.ndarray | None = None

    hh_dual:    jnp.ndarray | None = None
    hh_edge:    jnp.ndarray | None = None
    hh_quad:    jnp.ndarray | None = None           

    pv_edge:    jnp.ndarray | None = None
    rv_dual:    jnp.ndarray | None = None
    pv_dual:    jnp.ndarray | None = None
    rv_cell:    jnp.ndarray | None = None
    pv_cell:    jnp.ndarray | None = None

    ke_cell:    jnp.ndarray | None = None

    hh_bias:    jnp.ndarray | None = None
    pv_bias:    jnp.ndarray | None = None
    ke_bias:    jnp.ndarray | None = None

    nu_turb:    jnp.ndarray | None = None
    nu_wave:    jnp.ndarray | None = None
    nu_shoc:    jnp.ndarray | None = None
    nu_thin:    jnp.ndarray | None = None

    xi_self:    jnp.ndarray | None = None
    xi_tide:    jnp.ndarray | None = None

@struct.dataclass
class jxp_foundation:
    ff_cell:    jnp.ndarray | None = None
    ff_edge:    jnp.ndarray | None = None
    ff_vert:    jnp.ndarray | None = None

    zb_cell:    jnp.ndarray | None = None

    dz_drag:    jnp.ndarray | None = None
    c1_edge:    jnp.ndarray | None = None
    c2_edge:    jnp.ndarray | None = None
    z0_edge:    jnp.ndarray | None = None
    n0_edge:    jnp.ndarray | None = None

    msh_fix:    jnp.ndarray | None = None
    msh_nu2:    jnp.ndarray | None = None
    msh_nu4:    jnp.ndarray | None = None
    visc_u2:    jnp.ndarray | None = None
    visc_u4:    jnp.ndarray | None = None
    diff_h2:    jnp.ndarray | None = None
    diff_h4:    jnp.ndarray | None = None

@struct.dataclass
class jxp_statistics:
    hh_min_:    jnp.ndarray | None = None
    uu_min_:    jnp.ndarray | None = None
    qq_min_:    jnp.ndarray | None = None
    hh_max_:    jnp.ndarray | None = None
    uu_max_:    jnp.ndarray | None = None
    qq_max_:    jnp.ndarray | None = None

    zt_rms_:    jnp.ndarray | None = None
    ke_ave_:    jnp.ndarray | None = None
    ke_rms_:    jnp.ndarray | None = None
    ke_max_:    jnp.ndarray | None = None
    dk_ave_:    jnp.ndarray | None = None
    dk_rms_:    jnp.ndarray | None = None
    dk_max_:    jnp.ndarray | None = None

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

            hr_cell=jnp.zeros(cell_size, dtype=reals_t),

            hh_dual=jnp.zeros(dual_size, dtype=reals_t),
            hh_edge=jnp.zeros(edge_size, dtype=reals_t),
            hh_quad=jnp.zeros(edge_size, dtype=reals_t),

            pv_edge=jnp.zeros(edge_size, dtype=reals_t),
            rv_dual=jnp.zeros(dual_size, dtype=reals_t),
            pv_dual=jnp.zeros(dual_size, dtype=reals_t),
            rv_cell=jnp.zeros(cell_size, dtype=reals_t),
            pv_cell=jnp.zeros(cell_size, dtype=reals_t),

            ke_cell=jnp.zeros(cell_size, dtype=reals_t),

            hh_bias=jnp.zeros(edge_size, dtype=reals_t),
            pv_bias=jnp.zeros(edge_size, dtype=reals_t),
            ke_bias=jnp.zeros(edge_size, dtype=reals_t),

            nu_turb=jnp.zeros(edge_size, dtype=reals_t),
            nu_wave=jnp.zeros(edge_size, dtype=reals_t),
            nu_shoc=jnp.zeros(edge_size, dtype=reals_t),
            nu_thin=jnp.zeros(edge_size, dtype=reals_t),

            xi_self=jnp.zeros(cell_size, dtype=reals_t),
            xi_tide=jnp.zeros(cell_size, dtype=reals_t),
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


