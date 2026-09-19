
import numpy as np
import jax
import jax.numpy as jnp
from flax import struct
from scipy.sparse import csr_matrix

from _fp import flt32_t, flt64_t
from _fp import reals_t, index_t
from _fp import udata_t, hdata_t, qdata_t

@struct.dataclass
class jxp_sparse_mat:
    indx:       jnp.ndarray
    vals:       jnp.ndarray

@struct.dataclass
class jxp_cell_coeff:
    flux_sums:  jxp_sparse_mat
    kite_sums:  jxp_sparse_mat
    wing_sums:  jxp_sparse_mat
    edge_sums:  jxp_sparse_mat
    dual_sums:  jxp_sparse_mat

@struct.dataclass
class jxp_edge_coeff:
    tail_sums:  jxp_sparse_mat
    wing_sums:  jxp_sparse_mat
    dual_sums:  jxp_sparse_mat
    cell_sums:  jxp_sparse_mat
    grad_norm:  jxp_sparse_mat
    grad_perp:  jxp_sparse_mat
    flux_perp:  jxp_sparse_mat
    lsqr_perp:  jxp_sparse_mat

@struct.dataclass
class jxp_quad_coeff:
    curl_sums:  jxp_sparse_mat

@struct.dataclass
class jxp_dual_coeff:
    flux_sums:  jxp_sparse_mat
    kite_sums:  jxp_sparse_mat
    tail_sums:  jxp_sparse_mat
    cell_sums:  jxp_sparse_mat
    edge_sums:  jxp_sparse_mat
    curl_sums:  jxp_sparse_mat

@struct.dataclass
class jxp_mats_tuple:
    cell:       jxp_cell_coeff
    edge:       jxp_edge_coeff
    quad:       jxp_quad_coeff
    dual:       jxp_dual_coeff

def ops_to_jax(mesh, mats, flow, cnfg):

    cell_size = mesh.cell.size
    edge_size = mesh.edge.size
    dual_size = mesh.vert.size

    mats.jx = jxp_mats_tuple(
        cell= jxp_cell_coeff(
            flux_sums=csr_to_jax(mats.cell_flux_sums),
            kite_sums=csr_to_jax(mats.cell_kite_sums),
            wing_sums=csr_to_jax(mats.cell_wing_sums),
            edge_sums=csr_to_jax(mats.cell_edge_sums),
            dual_sums=csr_to_jax(mats.cell_vert_sums),
        ),
        edge= jxp_edge_coeff(
            tail_sums=csr_to_jax(mats.edge_tail_sums),
            wing_sums=csr_to_jax(mats.edge_wing_sums),
            dual_sums=csr_to_jax(mats.edge_vert_sums),
            cell_sums=csr_to_jax(mats.edge_cell_sums),
            grad_norm=csr_to_jax(mats.edge_grad_norm),
            grad_perp=csr_to_jax(mats.edge_grad_perp),
            flux_perp=csr_to_jax(mats.edge_flux_perp),
            lsqr_perp=csr_to_jax(mats.edge_lsqr_perp),
        ),
        quad= jxp_quad_coeff(
            curl_sums=csr_to_jax(mats.quad_curl_sums),
        ),
        dual= jxp_dual_coeff(
            flux_sums=csr_to_jax(mats.dual_flux_sums),
            kite_sums=csr_to_jax(mats.dual_kite_sums),
            tail_sums=csr_to_jax(mats.dual_tail_sums),
            cell_sums=csr_to_jax(mats.dual_cell_sums),
            edge_sums=csr_to_jax(mats.dual_edge_sums),
            curl_sums=csr_to_jax(mats.dual_curl_sums),
        ),
    )

    return mats


def csr_to_jax(csr):
#-- convert a scipy sparse operator to a dense padded gather
#-- form suited to JAX/GPU: row r's sparse entries become
#-- idx[r, :], wgt[r, :], padded with (0, 0.0).

    nrow = csr.shape[0]
    cnnz = np.diff(csr.indptr)
    kmax = int(cnnz.max())

    ridx = np.repeat(np.arange(nrow), cnnz)
    rpos = np.arange(csr.indices.size) - csr.indptr[ridx]

    indx = np.zeros((nrow, kmax), dtype=index_t)
    vals = np.zeros((nrow, kmax), dtype=reals_t)

    indx[ridx, rpos] = csr.indices
    vals[ridx, rpos] = csr.data

    return jxp_sparse_mat(jnp.asarray(indx), 
                          jnp.asarray(vals))


def idx_gather(mat, idx):
#-- JAX-based indexing operation, sans bounds checks
    return mat.at[idx].get(
        mode="promise_in_bounds", wrap_negative_indices=False,
    )


def op_product(mat, vec):
#-- JAX-based stencil reduction for linear operators
#-- sum M_ij * vec_j
    """
    return jnp.sum(vec[mat.indx] * mat.vals, axis=1)
    """

    tmp = vec.at[mat.indx].get(
        mode="promise_in_bounds", wrap_negative_indices=False,
    )
    return jnp.sum(tmp * mat.vals, axis=1)


def pv_product(mat, flx, sub, add):
#-- JAX-based stencil reduction for pv-adv operators
#-- sum W_ij * flx_j * (sub_i + add_j)
    """
    return jnp.sum(mat.vals * 
        flx[mat.indx] * (sub[:, None]+add[mat.indx]),
        axis=1,
    )
    """

    _tf = flx.at[mat.indx].get(
        mode="promise_in_bounds", wrap_negative_indices=False,
    )
    _ta = add.at[mat.indx].get(
        mode="promise_in_bounds", wrap_negative_indices=False,
    )
    return jnp.sum(mat.vals * 
        _tf * (sub[:,None] + _ta), axis=1,
    )


