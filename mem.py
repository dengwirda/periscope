
import numpy as np

""" Util. to setup memory pools for global & scratch arrays.
"""
#-- Part of the PERISCOPE solver
#-- Darren Engwirda
#-- d.engwirda@gmail.com
#-- https://github.com/dengwirda/

from _fp import flt32_t, flt64_t
from _fp import reals_t, index_t

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
    
    variables.hh_cell = \
            np.empty(mesh.cell.size, dtype=flt64_t)
    variables.uu_edge = \
            np.empty(mesh.edge.size, dtype=flt64_t)

    variables.hh_min_ = \
            np.empty(mesh.cell.size, dtype=reals_t)
    variables.uu_min_ = \
            np.empty(mesh.edge.size, dtype=reals_t)
    variables.hh_max_ = \
            np.empty(mesh.cell.size, dtype=reals_t)
    variables.uu_max_ = \
            np.empty(mesh.edge.size, dtype=reals_t)

    variables.ch_cell = \
            np.zeros(mesh.cell.size, dtype=flt64_t)
    variables.cu_edge = \
            np.zeros(mesh.edge.size, dtype=flt64_t)

    variables.hh_tend = \
            np.zeros(mesh.cell.size, dtype=flt64_t)
    variables.uu_tend = \
            np.zeros(mesh.edge.size, dtype=flt64_t)

    variables.hb_cell = \
            np.empty(mesh.cell.size, dtype=flt64_t)

    variables.h1_cell = \
            np.empty(mesh.cell.size, dtype=flt64_t)
    variables.u1_edge = \
            np.empty(mesh.edge.size, dtype=flt64_t)
    variables.h2_cell = \
            np.empty(mesh.cell.size, dtype=flt64_t)
    variables.u2_edge = \
            np.empty(mesh.edge.size, dtype=flt64_t)
    variables.h3_cell = \
            np.empty(mesh.cell.size, dtype=flt64_t)
    variables.u3_edge = \
            np.empty(mesh.edge.size, dtype=flt64_t)

    variables.hh_dual = \
            np.empty(mesh.vert.size, dtype=reals_t)
            
    variables.rv_dual = \
            np.empty(mesh.vert.size, dtype=reals_t)
    variables.pv_dual = \
            np.empty(mesh.vert.size, dtype=reals_t)

    variables.r2_dual = \
            np.empty(mesh.vert.size, dtype=reals_t)
    variables.p2_dual = \
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

    variables.nu_shoc = \
            np.zeros(mesh.edge.size, dtype=reals_t)
    variables.hs_shoc = \
            np.empty(mesh.cell.size, dtype=reals_t)
        
    variables.cd_edge = \
            np.empty(mesh.edge.size, dtype=reals_t)
            
    variables.rv_cell = \
            np.empty(mesh.cell.size, dtype=reals_t)
    variables.pv_cell = \
            np.empty(mesh.cell.size, dtype=reals_t)

    variables.ke_edge = \
            np.empty(mesh.edge.size, dtype=reals_t)
    variables.ke_dual = \
            np.empty(mesh.vert.size, dtype=reals_t)
    variables.ke_cell = \
            np.empty(mesh.cell.size, dtype=reals_t)

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

