
import numpy as np
import jax
import jax.numpy as jnp
from flax import struct
from scipy.sparse import csr_matrix

from _fp import flt32_t, flt64_t
from _fp import reals_t, index_t
from _fp import udata_t, hdata_t, qdata_t

from _jo import ops_to_jax
from _jm import msh_to_jax
from _jv import var_to_jax
from _jc import usr_to_jax

def all_to_jax(mesh, mats, flow, cnfg):

    mesh = msh_to_jax(mesh, mats, flow, cnfg)
    mats = ops_to_jax(mesh, mats, flow, cnfg)
    flow = var_to_jax(mesh, mats, flow, cnfg)

    cnfg = usr_to_jax(mesh, mats, flow, cnfg)

    return mesh, mats, flow, cnfg


