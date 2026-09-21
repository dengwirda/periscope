
import numpy as np
import jax
import jax.numpy as jnp
from flax import struct
from scipy.sparse import csr_matrix

from _fp import flt32_t, flt64_t
from _fp import reals_t, index_t
from _fp import udata_t, hdata_t, qdata_t

@struct.dataclass
class jxp_cell_tuple:
    irev:       jnp.ndarray | None = None
    ifwd:       jnp.ndarray | None = None
    xpos:       jnp.ndarray | None = None
    ypos:       jnp.ndarray | None = None
    zpos:       jnp.ndarray | None = None
    xlon:       jnp.ndarray | None = None
    ylat:       jnp.ndarray | None = None
    vert:       jnp.ndarray | None = None
    edge:       jnp.ndarray | None = None
    cell:       jnp.ndarray | None = None
    topo:       jnp.ndarray | None = None
    mask:       jnp.ndarray | None = None
    gate:       jnp.ndarray | None = None
    open:       jnp.ndarray | None = None
    wall:       jnp.ndarray | None = None
    area:       jnp.ndarray | None = None

@struct.dataclass
class jxp_edge_tuple:
    irev:       jnp.ndarray | None = None
    ifwd:       jnp.ndarray | None = None
    xpos:       jnp.ndarray | None = None
    ypos:       jnp.ndarray | None = None
    zpos:       jnp.ndarray | None = None
    xprp:       jnp.ndarray | None = None
    yprp:       jnp.ndarray | None = None
    zprp:       jnp.ndarray | None = None
    xnrm:       jnp.ndarray | None = None
    ynrm:       jnp.ndarray | None = None
    znrm:       jnp.ndarray | None = None
    xlon:       jnp.ndarray | None = None
    ylat:       jnp.ndarray | None = None
    vlen:       jnp.ndarray | None = None
    dlen:       jnp.ndarray | None = None
    clen:       jnp.ndarray | None = None
    slen:       jnp.ndarray | None = None
    spac:       jnp.ndarray | None = None
    beta:       jnp.ndarray | None = None
    cos_:       jnp.ndarray | None = None
    sin_:       jnp.ndarray | None = None
    tail:       jnp.ndarray | None = None
    wing:       jnp.ndarray | None = None
    lhs_:       jnp.ndarray | None = None
    rhs_:       jnp.ndarray | None = None
    vert:       jnp.ndarray | None = None
    cell:       jnp.ndarray | None = None
    edge:       jnp.ndarray | None = None
    topo:       jnp.ndarray | None = None
    mask:       jnp.ndarray | None = None
    gate:       jnp.ndarray | None = None
    slip:       jnp.ndarray | None = None
    perp:       jnp.ndarray | None = None
    open:       jnp.ndarray | None = None
    wall:       jnp.ndarray | None = None
    area:       jnp.ndarray | None = None

@struct.dataclass
class jxp_quad_tuple:
    area:       jnp.ndarray | None = None

@struct.dataclass
class jxp_vert_tuple:
    irev:       jnp.ndarray | None = None
    ifwd:       jnp.ndarray | None = None
    xpos:       jnp.ndarray | None = None
    ypos:       jnp.ndarray | None = None
    zpos:       jnp.ndarray | None = None
    xlon:       jnp.ndarray | None = None
    ylat:       jnp.ndarray | None = None
    kite:       jnp.ndarray | None = None
    edge:       jnp.ndarray | None = None
    cell:       jnp.ndarray | None = None
    mask:       jnp.ndarray | None = None
    gate:       jnp.ndarray | None = None
    slip:       jnp.ndarray | None = None
    open:       jnp.ndarray | None = None
    wall:       jnp.ndarray | None = None
    area:       jnp.ndarray | None = None

@struct.dataclass
class jxp_mesh_tuple:
    cell:       jxp_cell_tuple
    edge:       jxp_edge_tuple
    quad:       jxp_quad_tuple
    vert:       jxp_vert_tuple

def msh_to_jax(mesh, mats, flow, cnfg):

    cell_size = mesh.cell.size
    edge_size = mesh.edge.size
    dual_size = mesh.vert.size

    # adj. cells used in local upwind stencils
    edge_lhs_ = mesh.edge.cell[:, 0] - 1
    edge_rhs_ = mesh.edge.cell[:, 1] - 1
    edge_lhs_[edge_lhs_<0] = edge_rhs_[edge_lhs_<0]
    edge_rhs_[edge_rhs_<0] = edge_lhs_[edge_rhs_<0]

    mesh.jx = jxp_mesh_tuple(
        cell= jxp_cell_tuple(
            irev=jnp.asarray(mesh.cell.irev),
            ifwd=jnp.asarray(mesh.cell.ifwd),
            xpos=jnp.asarray(mesh.cell.xpos),
            ypos=jnp.asarray(mesh.cell.ypos),
            zpos=jnp.asarray(mesh.cell.zpos),
            xlon=jnp.asarray(mesh.cell.xlon),
            ylat=jnp.asarray(mesh.cell.ylat),
            vert=jnp.asarray(mesh.cell.vert),
            edge=jnp.asarray(mesh.cell.edge),
            cell=jnp.asarray(mesh.cell.cell),
            topo=jnp.asarray(mesh.cell.topo),
            mask=jnp.asarray(mesh.cell.mask),
            gate=jnp.asarray(mesh.cell.gate),
            open=jnp.asarray(mesh.cell.open),
            wall=jnp.asarray(mesh.cell.wall),
            area=jnp.asarray(mesh.cell.area),
        ),
        edge= jxp_edge_tuple(
            irev=jnp.asarray(mesh.edge.irev),
            ifwd=jnp.asarray(mesh.edge.ifwd),
            xpos=jnp.asarray(mesh.edge.xpos),
            ypos=jnp.asarray(mesh.edge.ypos),
            zpos=jnp.asarray(mesh.edge.zpos),
            xprp=jnp.asarray(mesh.edge.xprp),
            yprp=jnp.asarray(mesh.edge.yprp),
            zprp=jnp.asarray(mesh.edge.zprp),
            xnrm=jnp.asarray(mesh.edge.xnrm),
            ynrm=jnp.asarray(mesh.edge.ynrm),
            znrm=jnp.asarray(mesh.edge.znrm),
            xlon=jnp.asarray(mesh.edge.xlon),
            ylat=jnp.asarray(mesh.edge.ylat),
            vlen=jnp.asarray(mesh.edge.vlen),
            dlen=jnp.asarray(mesh.edge.dlen),
            clen=jnp.asarray(mesh.edge.clen),
            slen=jnp.asarray(mesh.edge.slen),
            spac=jnp.asarray(mesh.edge.spac),
            beta=jnp.asarray(mesh.edge.beta),
            cos_=jnp.asarray(mesh.edge.cos_),
            sin_=jnp.asarray(mesh.edge.sin_),
            tail=jnp.asarray(mesh.edge.tail),
            wing=jnp.asarray(mesh.edge.wing),
            lhs_=jnp.asarray(edge_lhs_),
            rhs_=jnp.asarray(edge_rhs_),
            vert=jnp.asarray(mesh.edge.vert),
            cell=jnp.asarray(mesh.edge.cell),
            edge=jnp.asarray(mesh.edge.edge),
            topo=jnp.asarray(mesh.edge.topo),
            mask=jnp.asarray(mesh.edge.mask),
            gate=jnp.asarray(mesh.edge.gate),
            slip=jnp.asarray(mesh.edge.slip),
            perp=jnp.asarray(mesh.edge.perp),
            open=jnp.asarray(mesh.edge.open),
            wall=jnp.asarray(mesh.edge.wall),
            area=jnp.asarray(mesh.edge.area),
        ),
        quad= jxp_quad_tuple(
            area=jnp.asarray(mesh.quad.area),
        ),
        vert= jxp_vert_tuple(
            irev=jnp.asarray(mesh.vert.irev),
            ifwd=jnp.asarray(mesh.vert.ifwd),
            xpos=jnp.asarray(mesh.vert.xpos),
            ypos=jnp.asarray(mesh.vert.ypos),
            zpos=jnp.asarray(mesh.vert.zpos),
            xlon=jnp.asarray(mesh.vert.xlon),
            ylat=jnp.asarray(mesh.vert.ylat),
            kite=jnp.asarray(mesh.vert.kite),
            edge=jnp.asarray(mesh.vert.edge),
            cell=jnp.asarray(mesh.vert.cell),
            mask=jnp.asarray(mesh.vert.mask),
            gate=jnp.asarray(mesh.vert.gate),
            slip=jnp.asarray(mesh.vert.slip), 
            open=jnp.asarray(mesh.vert.open),
            wall=jnp.asarray(mesh.vert.wall),
            area=jnp.asarray(mesh.vert.area),
        ),
    )

    return mesh


