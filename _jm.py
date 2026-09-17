
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
    xpos:       jnp.ndarray
    ypos:       jnp.ndarray
    zpos:       jnp.ndarray
    xlon:       jnp.ndarray
    ylat:       jnp.ndarray
    vert:       jnp.ndarray
    edge:       jnp.ndarray
    cell:       jnp.ndarray
    topo:       jnp.ndarray
    mask:       jnp.ndarray
    area:       jnp.ndarray

@struct.dataclass
class jxp_edge_tuple:
    xpos:       jnp.ndarray
    ypos:       jnp.ndarray
    zpos:       jnp.ndarray
    xprp:       jnp.ndarray
    yprp:       jnp.ndarray
    zprp:       jnp.ndarray
    xnrm:       jnp.ndarray
    ynrm:       jnp.ndarray
    znrm:       jnp.ndarray
    xlon:       jnp.ndarray
    ylat:       jnp.ndarray
    vlen:       jnp.ndarray
    dlen:       jnp.ndarray
    clen:       jnp.ndarray
    slen:       jnp.ndarray
    spac:       jnp.ndarray
    beta:       jnp.ndarray
    cos_:       jnp.ndarray
    sin_:       jnp.ndarray
    tail:       jnp.ndarray
    wing:       jnp.ndarray
    vert:       jnp.ndarray
    cell:       jnp.ndarray
    edge:       jnp.ndarray
    topo:       jnp.ndarray
    mask:       jnp.ndarray
    area:       jnp.ndarray

@struct.dataclass
class jxp_quad_tuple:
    area:       jnp.ndarray

@struct.dataclass
class jxp_vert_tuple:
    xpos:       jnp.ndarray
    ypos:       jnp.ndarray
    zpos:       jnp.ndarray
    xlon:       jnp.ndarray
    ylat:       jnp.ndarray
    kite:       jnp.ndarray
    edge:       jnp.ndarray
    cell:       jnp.ndarray
    mask:       jnp.ndarray
    area:       jnp.ndarray

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

    mesh.jx = jxp_mesh_tuple(
        cell= jxp_cell_tuple(
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
            area=jnp.asarray(mesh.cell.area),
        ),
        edge= jxp_edge_tuple(
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
            vert=jnp.asarray(mesh.edge.vert),
            cell=jnp.asarray(mesh.edge.cell),
            edge=jnp.asarray(mesh.edge.edge),
            topo=jnp.asarray(mesh.edge.topo),
            mask=jnp.asarray(mesh.edge.mask),
            area=jnp.asarray(mesh.edge.area),
        ),
        quad= jxp_quad_tuple(
            area=jnp.asarray(mesh.quad.area),
        ),
        vert= jxp_vert_tuple(
            xpos=jnp.asarray(mesh.vert.xpos),
            ypos=jnp.asarray(mesh.vert.ypos),
            zpos=jnp.asarray(mesh.vert.zpos),
            xlon=jnp.asarray(mesh.vert.xlon),
            ylat=jnp.asarray(mesh.vert.ylat),
            kite=jnp.asarray(mesh.vert.kite),
            edge=jnp.asarray(mesh.vert.edge),
            cell=jnp.asarray(mesh.vert.cell),
            mask=jnp.asarray(mesh.vert.mask),
            area=jnp.asarray(mesh.vert.area),
        ),
    )

    return mesh


