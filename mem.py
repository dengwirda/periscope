
import numpy as np

""" Util. to setup memory pools for global & scratch arrays.
"""
#-- Part of the PERISCOPE solver
#-- Darren Engwirda
#-- d.engwirda@gmail.com
#-- https://github.com/dengwirda/

from _fp import flt32_t, flt64_t
from _fp import reals_t, index_t
from _fp import udata_t, hdata_t, qdata_t
from _fp import utend_t, htend_t, qtend_t

class base: pass
variables = base()

VERT_SIZE = 4   # number of vec's in pools
EDGE_SIZE = 4
CELL_SIZE = 4
vert_pool = []
edge_pool = []
cell_pool = []

def init_pool(mesh):
    for i in range(VERT_SIZE):
        vert_pool.append(
            np.empty(mesh.vert.size, dtype=reals_t))
            
    for i in range(EDGE_SIZE):
        edge_pool.append(
            np.empty(mesh.edge.size, dtype=reals_t))
            
    for i in range(CELL_SIZE):
        cell_pool.append(
            np.empty(mesh.cell.size, dtype=reals_t))
    
    variables.is_zero = \
            np.zeros(mesh._max_size, dtype=reals_t)

    variables.hh_cell = \
            np.empty(mesh.cell.size, dtype=hdata_t)
    variables.uu_edge = \
            np.empty(mesh.edge.size, dtype=udata_t)
    variables.qq_cell = \
            np.empty(mesh.cell.size, dtype=qdata_t)

    variables.hh_min_ = \
            np.empty(mesh.cell.size, dtype=reals_t)
    variables.uu_min_ = \
            np.empty(mesh.edge.size, dtype=reals_t)
    variables.qq_min_ = \
            np.empty(mesh.cell.size, dtype=reals_t)
    variables.hh_max_ = \
            np.empty(mesh.cell.size, dtype=reals_t)
    variables.uu_max_ = \
            np.empty(mesh.edge.size, dtype=reals_t)
    variables.qq_max_ = \
            np.empty(mesh.cell.size, dtype=reals_t)
    variables.zt_rms_ = \
            np.zeros(mesh.cell.size, dtype=reals_t)    
    variables.ke_ave_ = \
            np.zeros(mesh.cell.size, dtype=reals_t)
    variables.ke_rms_ = \
            np.zeros(mesh.cell.size, dtype=reals_t)
    variables.ke_max_ = \
            np.zeros(mesh.cell.size, dtype=reals_t)
    variables.dk_ave_ = \
            np.zeros(mesh.edge.size, dtype=reals_t)
    variables.dk_rms_ = \
            np.zeros(mesh.edge.size, dtype=reals_t)
    variables.dk_max_ = \
            np.zeros(mesh.edge.size, dtype=reals_t)

    variables.hh_tend = \
            np.zeros(mesh.cell.size, dtype=htend_t)
    variables.uu_tend = \
            np.zeros(mesh.edge.size, dtype=utend_t)
    variables.qh_tend = \
            np.zeros(mesh.cell.size, dtype=qtend_t)

    variables.h0_tend = \
            np.zeros(mesh.cell.size, dtype=htend_t)
    variables.u0_tend = \
            np.zeros(mesh.edge.size, dtype=utend_t)
    variables.s0_tend = \
            np.zeros(mesh.edge.size, dtype=utend_t)

    variables.hb_cell = \
            np.empty(mesh.cell.size, dtype=hdata_t)

    variables.h1_cell = \
            np.empty(mesh.cell.size, dtype=hdata_t)
    variables.h2_cell = \
            np.empty(mesh.cell.size, dtype=hdata_t)
    variables.h3_cell = \
            np.empty(mesh.cell.size, dtype=hdata_t)
    
    variables.uk_edge = \
            np.empty(mesh.edge.size, dtype=udata_t)
    
    variables.hh_dual = \
            np.empty(mesh.vert.size, dtype=reals_t)
            
    variables.rv_dual = \
            np.empty(mesh.vert.size, dtype=reals_t)
    variables.pv_dual = \
            np.empty(mesh.vert.size, dtype=reals_t)

    variables.rv_wide = \
            np.empty(mesh.vert.size, dtype=reals_t)
    variables.pv_wide = \
            np.empty(mesh.vert.size, dtype=reals_t)
            
    variables.vv_edge = \
            np.empty(mesh.edge.size, dtype=reals_t)
            
    variables.hh_edge = \
            np.empty(mesh.edge.size, dtype=reals_t)  
    variables.hh_quad = \
            np.empty(mesh.edge.size, dtype=reals_t)
    
    variables.rv_edge = \
            np.empty(mesh.edge.size, dtype=reals_t)               
    variables.pv_edge = \
            np.empty(mesh.edge.size, dtype=reals_t)

    variables.hh_bias = \
            np.zeros(mesh.edge.size, dtype=reals_t)
    variables.pv_bias = \
            np.zeros(mesh.edge.size, dtype=reals_t)
    variables.ke_bias = \
            np.zeros(mesh.edge.size, dtype=reals_t)
    
    variables.nu_turb = \
            np.zeros(mesh.edge.size, dtype=reals_t)
    variables.nu_wave = \
            np.zeros(mesh.edge.size, dtype=reals_t)
    variables.nu_shoc = \
            np.zeros(mesh.edge.size, dtype=reals_t)
    variables.nu_thin = \
            np.zeros(mesh.edge.size, dtype=reals_t)

    variables.os_wave = \
            np.empty(mesh.edge.size, dtype=reals_t)
    variables.os_shoc = \
            np.empty(mesh.cell.size, dtype=reals_t)
        
    variables.cd_edge = \
            np.empty(mesh.edge.size, dtype=reals_t)
    variables.cd_save = \
            np.empty(mesh.edge.size, dtype=reals_t)

    variables.ke_diss = \
            np.empty(mesh.edge.size, dtype=reals_t)
            
    variables.rv_cell = \
            np.empty(mesh.cell.size, dtype=reals_t)
    variables.pv_cell = \
            np.empty(mesh.cell.size, dtype=reals_t)

    variables.ke_edge = \
            np.zeros(mesh.edge.size, dtype=reals_t)
    variables.ke_dual = \
            np.zeros(mesh.vert.size, dtype=reals_t)
    variables.ke_cell = \
            np.zeros(mesh.cell.size, dtype=reals_t)

    variables.uu_filt = \
            np.zeros(mesh.edge.size, dtype=reals_t)

    variables.Xi_tide = \
            np.zeros(mesh.cell.size, dtype=reals_t)
    variables.Xi_self = \
            np.zeros(mesh.cell.size, dtype=reals_t)

def get_vec_v():
    if (len(vert_pool) > 0):
        return vert_pool.pop()
    else:
        raise RuntimeError("Pool exhausted: verts")
        
def put_vec_v(vec): vert_pool.append(vec)

def get_vec_e():
    if (len(edge_pool) > 0):
        return edge_pool.pop()
    else:
        raise RuntimeError("Pool exhausted: edges")
        
def put_vec_e(vec): edge_pool.append(vec)

def get_vec_c():
    if (len(cell_pool) > 0):
        return cell_pool.pop()
    else:
        raise RuntimeError("Pool exhausted: cells")
        
def put_vec_c(vec): cell_pool.append(vec)

