
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
EDGE_SIZE = 8
CELL_SIZE = 8
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
    
    variables.hh_tend = \
            np.empty(mesh.cell.size, dtype=reals_t)
    variables.uu_tend = \
            np.empty(mesh.edge.size, dtype=reals_t)

    variables.hh_tend[:] = 0.0
    variables.uu_tend[:] = 0.0

    variables.hh_dual = \
            np.empty(mesh.vert.size, dtype=reals_t)
            
    variables.rv_dual = \
            np.empty(mesh.vert.size, dtype=reals_t)
    variables.pv_dual = \
            np.empty(mesh.vert.size, dtype=reals_t)
    variables.ke_dual = \
            np.empty(mesh.vert.size, dtype=reals_t)    

    variables.r2_dual = \
            np.empty(mesh.vert.size, dtype=reals_t)
    variables.p2_dual = \
            np.empty(mesh.vert.size, dtype=reals_t)
            
    variables.vv_edge = \
            np.empty(mesh.edge.size, dtype=reals_t)
            
    variables.hh_edge = \
            np.empty(mesh.edge.size, dtype=reals_t)  
    variables.h2_edge = \
            np.empty(mesh.edge.size, dtype=reals_t)
    
    variables.rv_edge = \
            np.empty(mesh.edge.size, dtype=reals_t)               
    variables.pv_edge = \
            np.empty(mesh.edge.size, dtype=reals_t)

    variables.hh_bias = \
            np.empty(mesh.edge.size, dtype=reals_t)
    variables.pv_bias = \
            np.empty(mesh.edge.size, dtype=reals_t)
    variables.ke_bias = \
            np.empty(mesh.edge.size, dtype=reals_t)
    
    variables.nu_edge = \
            np.empty(mesh.edge.size, dtype=reals_t)
        
    variables.nu_edge[:] = 0.0
        
    variables.cd_edge = \
            np.empty(mesh.edge.size, dtype=reals_t)
            
    variables.rv_cell = \
            np.empty(mesh.cell.size, dtype=reals_t)
    variables.pv_cell = \
            np.empty(mesh.cell.size, dtype=reals_t)
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

