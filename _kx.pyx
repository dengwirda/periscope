
#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=True
#cython: cpow=True

""" SWE spatial discretisation using TRSK-like operators
"""
#-- Darren Engwirda

import numpy as np
cimport numpy as np
cimport cython

from cython.parallel import prange, parallel
from libc.stdint cimport int32_t

from _fp import flt32_t, flt64_t
from _fp import reals_t, index_t

from lib cimport sqrtf as sqrt_r
from lib cimport logf as log_r
from lib cimport fabsf as fabs_r

from mem import variables
from mem import get_vec_v, get_vec_e, get_vec_c, \
                put_vec_v, put_vec_e, put_vec_c

ctypedef float   FLT32_t
ctypedef double  FLT64_t
ctypedef int32_t INDEX_t
ctypedef float   REALS_t  # or double

def _computeBC(mesh, trsk, cnfg,
    np.ndarray[REALS_t, ndim=1] hh_edge,
    np.ndarray[REALS_t, ndim=1] uu_edge,
    const REALS_t gg_cell,
    np.ndarray[REALS_t, ndim=1] hE_edge,
    np.ndarray[REALS_t, ndim=1] uE_edge
            ):
            
#-- update u, h variables to set open BCs
            
    cdef INDEX_t edge, eidx
    cdef REALS_t RADIATE, UU_PRED
    
    cdef REALS_t ZERO = 0.0
    cdef REALS_t TWO_ = 2.0
    
    cdef INDEX_t cnfg_numthread = cnfg.numthread
    cdef INDEX_t cnfg_chunksize = cnfg.chunksize
    
    cdef REALS_t cnfg_time_step = cnfg.time_step
        
    cdef INDEX_t NBCS = mesh.edge.open.size
    
    cdef REALS_t *HH_EDGE = &hh_edge[0]
    cdef REALS_t *UU_EDGE = &uu_edge[0]
    cdef REALS_t *HE_EDGE = &hE_edge[0]
    cdef REALS_t *UE_EDGE = &uE_edge[0]
    
    cdef INDEX_t[::1] mesh_edge_open = mesh.edge.open
    
    cdef INDEX_t *MESH_EDGE_OPEN = &mesh_edge_open[0]
    
    cdef REALS_t[::1] mesh_edge_slen = mesh.edge.slen
    
    cdef REALS_t *MESH_EDGE_SLEN = &mesh_edge_slen[0]
    
    if (NBCS > 0):

        with nogil, parallel(num_threads=cnfg_numthread):
          
            for edge in prange(0, NBCS, schedule="static", 
                    chunksize=cnfg_chunksize):
                    
                eidx = MESH_EDGE_OPEN[edge]
                    
                #-- Flather-type bnd. condition
                RADIATE = sqrt_r(
                    gg_cell / HH_EDGE[eidx])
                    
                #-- CFL-like limiter
                RADIATE = min(RADIATE, TWO_ * 
                    MESH_EDGE_SLEN[eidx] / cnfg_time_step)
                
                UU_PRED = UE_EDGE[eidx] + RADIATE * (
                    HH_EDGE[eidx] - HE_EDGE[eidx]
                    )
                
                if (UU_PRED <=  ZERO):
                
                #-- inflow: prescribe mass flux
                    UU_EDGE[eidx] = UU_PRED
                    HH_EDGE[eidx] = HE_EDGE[eidx]
                    
                else:
                
                #-- outflow: radiate deviations
                    UU_EDGE[eidx] = UU_PRED
                   #HH_EDGE[eidx] = upwind, from scheme
                          
    return hh_edge, uu_edge
            

def _upwinding(mesh, trsk, cnfg, 
    np.ndarray[REALS_t, ndim=1] sw_dual,
    np.ndarray[REALS_t, ndim=1] ss_dual, 
    np.ndarray[REALS_t, ndim=1] ss_cell,
    np.ndarray[REALS_t, ndim=1] uu_edge, 
    np.ndarray[REALS_t, ndim=1] vv_edge,
    np.ndarray[REALS_t, ndim=1] ss_edge,
    np.ndarray[REALS_t, ndim=1] up_bias, 
    const REALS_t delta_t,
    const REALS_t ss_tiny, const REALS_t uu_tiny,
    up_kind, 
    const REALS_t up_min_, const REALS_t up_max_
              ):
    
#-- streamline upwinding for a variable S
    
    cdef INDEX_t vert, edge, iptr, xidx
    cdef REALS_t xval
    cdef REALS_t dN_EDGE, dP_EDGE
    cdef REALS_t DS_EDGE, UM_EDGE, SS_WIND
    
    cdef REALS_t ZERO = 0.0
    cdef REALS_t HALF = 0.5
    cdef REALS_t ONE_ = 1.0
    
    cdef INDEX_t cnfg_numthread = cnfg.numthread
    cdef INDEX_t cnfg_chunksize = cnfg.chunksize
        
    cdef INDEX_t NVRT = mesh.vert.size
    cdef INDEX_t NEDG = mesh.edge.size
    cdef INDEX_t NCEL = mesh.cell.size
    
    cdef REALS_t *SW_DUAL = &sw_dual[0]
    cdef REALS_t *SS_DUAL = &ss_dual[0]
    cdef REALS_t *SS_CELL = &ss_cell[0]
    cdef REALS_t *UU_EDGE = &uu_edge[0]
    cdef REALS_t *VV_EDGE = &vv_edge[0]
    cdef REALS_t *SS_EDGE = &ss_edge[0]
    cdef REALS_t *UP_BIAS = &up_bias[0]
    
    cdef INDEX_t[::1] grad_norm_xptr = \
        trsk.edge_grad_norm.indptr
    cdef INDEX_t[::1] grad_norm_xidx = \
        trsk.edge_grad_norm.indices
    cdef REALS_t[::1] grad_norm_xval = \
        trsk.edge_grad_norm.data
    
    cdef INDEX_t *GRAD_NORM_XPTR = &grad_norm_xptr[0]
    cdef INDEX_t *GRAD_NORM_XIDX = &grad_norm_xidx[0]
    cdef REALS_t *GRAD_NORM_XVAL = &grad_norm_xval[0]
    
    cdef INDEX_t[::1] grad_perp_xptr = \
        trsk.edge_grad_perp.indptr
    cdef INDEX_t[::1] grad_perp_xidx = \
        trsk.edge_grad_perp.indices
    cdef REALS_t[::1] grad_perp_xval = \
        trsk.edge_grad_perp.data
    
    cdef INDEX_t *GRAD_PERP_XPTR = &grad_perp_xptr[0]
    cdef INDEX_t *GRAD_PERP_XIDX = &grad_perp_xidx[0]
    cdef REALS_t *GRAD_PERP_XVAL = &grad_perp_xval[0]
       
    cdef INDEX_t[::1] vert_tail_xptr = \
        trsk.dual_tail_sums.indptr
    cdef INDEX_t[::1] vert_tail_xidx = \
        trsk.dual_tail_sums.indices
    cdef REALS_t[::1] vert_tail_xval = \
        trsk.dual_tail_sums.data
    
    cdef INDEX_t *VERT_EDGE_XPTR = &vert_tail_xptr[0]
    cdef INDEX_t *VERT_EDGE_XIDX = &vert_tail_xidx[0]
    cdef REALS_t *VERT_EDGE_XVAL = &vert_tail_xval[0]
    
    cdef INDEX_t[::1] edge_vert_xptr = \
        trsk.edge_vert_sums.indptr
    cdef INDEX_t[::1] edge_vert_xidx = \
        trsk.edge_vert_sums.indices
    
    cdef INDEX_t *EDGE_VERT_XPTR = &edge_vert_xptr[0]
    cdef INDEX_t *EDGE_VERT_XIDX = &edge_vert_xidx[0]
    
    cdef REALS_t[::1] mesh_edge_clen = mesh.edge.clen
    cdef REALS_t[::1] mesh_edge_vlen = mesh.edge.vlen
    cdef REALS_t[::1] mesh_edge_slen = mesh.edge.slen
    
    cdef REALS_t *MESH_EDGE_CLEN = &mesh_edge_clen[0]
    cdef REALS_t *MESH_EDGE_VLEN = &mesh_edge_vlen[0]
    cdef REALS_t *MESH_EDGE_SLEN = &mesh_edge_slen[0]
    
    cdef REALS_t[::1] mesh_dual_area = mesh.vert.area
    
    cdef REALS_t *MESH_DUAL_AREA = &mesh_dual_area[0]
        
    cdef np.ndarray[REALS_t] ds_vert = get_vec_v()
        
    cdef REALS_t *DS_VERT = &ds_vert[0]
        
    if   (up_kind == "APVM"):
              
        #-- APVM: anticipated upstream method; lagrangian
        #-- formulation. Upwind departure points, appears
        #-- to be inconsistent in time...
  
        with nogil, parallel(num_threads=cnfg_numthread):
      
            for edge in prange(0, NEDG, schedule="static", 
                    chunksize=cnfg_chunksize):        
            #-- dN_edge = edge_grad_norm * ss_cell
                dN_EDGE = ZERO
                for iptr in range(GRAD_NORM_XPTR[edge +0], 
                                  GRAD_NORM_XPTR[edge +1]):
                        
                    xval = GRAD_NORM_XVAL[iptr]
                    xidx = GRAD_NORM_XIDX[iptr]
                        
                    dN_EDGE = \
                        dN_EDGE + (xval * SS_CELL[xidx])
                        
            #-- dP_edge = edge_grad_perp * ss_dual
                dP_EDGE = ZERO
                for iptr in range(GRAD_PERP_XPTR[edge +0],
                                  GRAD_PERP_XPTR[edge +1]):
                        
                    xval = GRAD_PERP_XVAL[iptr]
                    xidx = GRAD_PERP_XIDX[iptr]
                        
                    dP_EDGE = \
                        dP_EDGE + (xval * SS_DUAL[xidx])
                    
            #-- lagrangian APVM, scale w. flow
                SS_WIND = UU_EDGE[edge] * dN_EDGE \
                        + VV_EDGE[edge] * dP_EDGE
                    
                UP_BIAS[edge] = ZERO          
                
                SS_EDGE[edge]-= delta_t * SS_WIND
            
    elif (up_kind == "AUST-CONST"):

        #-- AUST: anticipated upstream method; APVM meets
        #-- LUST? Upwinds in multi-dimensional sense, vs.
        #-- LUST, which upwinds via tangential dir. only.

        #-- const. upwinding version

        with nogil, parallel(num_threads=cnfg_numthread):

            for edge in prange(0, NEDG, schedule="static", 
                    chunksize=cnfg_chunksize):      
            #-- dN_edge = edge_grad_norm * ss_cell
                dN_EDGE = ZERO
                for iptr in range(GRAD_NORM_XPTR[edge +0], 
                                  GRAD_NORM_XPTR[edge +1]):
                        
                    xval = GRAD_NORM_XVAL[iptr]
                    xidx = GRAD_NORM_XIDX[iptr]
                        
                    dN_EDGE = \
                        dN_EDGE + (xval * SS_CELL[xidx])
                        
            #-- dP_edge = edge_grad_perp * ss_dual
                dP_EDGE = ZERO
                for iptr in range(GRAD_PERP_XPTR[edge +0],
                                  GRAD_PERP_XPTR[edge +1]):
                        
                    xval = GRAD_PERP_XVAL[iptr]
                    xidx = GRAD_PERP_XIDX[iptr]
                        
                    dP_EDGE = \
                        dP_EDGE + (xval * SS_DUAL[xidx])
                    
                UP_BIAS[edge] = up_min_
                    
            #-- upwind APVM, scale w. grid spacing
                UM_EDGE = uu_tiny + sqrt_r (
                    UU_EDGE[edge] * UU_EDGE[edge] +
                    VV_EDGE[edge] * VV_EDGE[edge]
                    )

                SS_WIND = ONE_ / UM_EDGE * (
                    UU_EDGE[edge] * dN_EDGE +
                    VV_EDGE[edge] * dP_EDGE
                    )
                  
                SS_EDGE[edge]-= UP_BIAS[edge] \
                    * SS_WIND * MESH_EDGE_SLEN[edge]

    elif (up_kind == "AUST-ADAPT"):
        
        #-- AUST: anticipated upstream method; APVM meets
        #-- LUST? Upwinds in multi-dimensional sense, vs.
        #-- LUST, which upwinds via tangential dir. only.

        #-- adapt. upwinding version

        with nogil, parallel(num_threads=cnfg_numthread):

            for vert in prange(0, NVRT, schedule="static", 
                    chunksize=cnfg_chunksize):
            #-- ds_vert = |large - small| stencils
                DS_VERT[vert] = ( 
                    SS_DUAL[vert] - SW_DUAL[vert])
                    
                DS_VERT[vert]*= DS_VERT[vert]
                
            for edge in prange(0, NEDG, schedule="static", 
                    chunksize=cnfg_chunksize):    
            #-- up_bias = edge_dual_maps * ds_vert
                UP_BIAS[edge] = ZERO
                for iptr in range(EDGE_VERT_XPTR[edge +0], 
                                  EDGE_VERT_XPTR[edge +1]):
                        
                    xidx = EDGE_VERT_XIDX[iptr]
                        
                    UP_BIAS[edge]+= (HALF * DS_VERT[xidx])
                
                # rms difference at verticies
                UP_BIAS[edge] = sqrt_r(UP_BIAS[edge])
                        
            #-- dN_edge = edge_grad_norm * ss_cell
                dN_EDGE = ZERO
                for iptr in range(GRAD_NORM_XPTR[edge +0], 
                                  GRAD_NORM_XPTR[edge +1]):
                        
                    xval = GRAD_NORM_XVAL[iptr]
                    xidx = GRAD_NORM_XIDX[iptr]
                        
                    dN_EDGE = \
                        dN_EDGE + (xval * SS_CELL[xidx])
                        
            #-- dP_edge = edge_grad_perp * ss_dual
                dP_EDGE = ZERO
                for iptr in range(GRAD_PERP_XPTR[edge +0],
                                  GRAD_PERP_XPTR[edge +1]):
                        
                    xval = GRAD_PERP_XVAL[iptr]
                    xidx = GRAD_PERP_XIDX[iptr]
                        
                    dP_EDGE = \
                        dP_EDGE + (xval * SS_DUAL[xidx])
                    
            #-- a measure of 'difference' on edges
                DS_EDGE = ss_tiny + (HALF *(
                   fabs_r(
                dN_EDGE * MESH_EDGE_CLEN[edge])
                 + fabs_r(
                dP_EDGE * MESH_EDGE_VLEN[edge])
                         ) )
                
                UP_BIAS[edge]/= DS_EDGE

            #-- upwind APVM, scale w. grid spacing
                UM_EDGE = uu_tiny + sqrt_r (
                    UU_EDGE[edge] * UU_EDGE[edge] +
                    VV_EDGE[edge] * VV_EDGE[edge]
                    )

                SS_WIND = ONE_ / UM_EDGE * (
                    UU_EDGE[edge] * dN_EDGE +
                    VV_EDGE[edge] * dP_EDGE
                    )
                    
                UP_BIAS[edge] = up_min_ + min (
                    up_max_ - up_min_, UP_BIAS[edge]
                    )
                  
                SS_EDGE[edge]-= UP_BIAS[edge] \
                    * SS_WIND * MESH_EDGE_SLEN[edge]
               
    put_vec_v  (ds_vert)
              
    return ss_edge, up_bias
    

def _computeHH(mesh, trsk, cnfg,
    np.ndarray[REALS_t, ndim=1] hh_cell,
    np.ndarray[REALS_t, ndim=1] uu_edge
              ):
    
#-- compute edge & dual centred thickness
    
    cdef INDEX_t vert, edge, iptr, xidx
    cdef INDEX_t cel1, cel2
    cdef REALS_t xval
    cdef REALS_t HH_WIND
    
    cdef REALS_t ZERO = 0.0
    cdef REALS_t HALF = 0.5
    cdef REALS_t ONE_ = 1.0
    cdef REALS_t FOUR = 4.0
    cdef REALS_t SIX_ = 6.0
    
    cdef INDEX_t cnfg_numthread = cnfg.numthread
    cdef INDEX_t cnfg_chunksize = cnfg.chunksize
    
    cdef REALS_t up_max_ = cnfg.hh_max_up
    cdef REALS_t up_min_ = cnfg.hh_min_up
    
    cdef INDEX_t NVRT = mesh.vert.size
    cdef INDEX_t NEDG = mesh.edge.size
    cdef INDEX_t NCEL = mesh.cell.size
    
    cdef REALS_t *HH_CELL = &hh_cell[0]
    cdef REALS_t *UU_EDGE = &uu_edge[0]
    
    cdef INDEX_t[::1] edge_wing_xptr = \
        trsk.edge_wing_sums.indptr
    cdef INDEX_t[::1] edge_wing_xidx = \
        trsk.edge_wing_sums.indices
    cdef REALS_t[::1] edge_wing_xval = \
        trsk.edge_wing_sums.data
    
    cdef INDEX_t *EDGE_WING_XPTR = &edge_wing_xptr[0]
    cdef INDEX_t *EDGE_WING_XIDX = &edge_wing_xidx[0]
    cdef REALS_t *EDGE_WING_XVAL = &edge_wing_xval[0]
       
    cdef INDEX_t[::1] dual_kite_xptr = \
        trsk.dual_kite_sums.indptr
    cdef INDEX_t[::1] dual_kite_xidx = \
        trsk.dual_kite_sums.indices
    cdef REALS_t[::1] dual_kite_xval = \
        trsk.dual_kite_sums.data
    
    cdef INDEX_t *DUAL_KITE_XPTR = &dual_kite_xptr[0]
    cdef INDEX_t *DUAL_KITE_XIDX = &dual_kite_xidx[0]
    cdef REALS_t *DUAL_KITE_XVAL = &dual_kite_xval[0]
    
    cdef INDEX_t[::1] edge_vert_xptr = \
        trsk.edge_vert_sums.indptr
    cdef INDEX_t[::1] edge_vert_xidx = \
        trsk.edge_vert_sums.indices
    
    cdef INDEX_t *EDGE_VERT_XPTR = &edge_vert_xptr[0]
    cdef INDEX_t *EDGE_VERT_XIDX = &edge_vert_xidx[0]
    
    cdef REALS_t[::1] mesh_dual_area = mesh.vert.area
    cdef REALS_t[::1] mesh_edge_area = mesh.edge.area
    
    cdef REALS_t *MESH_DUAL_AREA = &mesh_dual_area[0]
    cdef REALS_t *MESH_EDGE_AREA = &mesh_edge_area[0]

    cdef INDEX_t[:, ::1] mesh_edge_cell = mesh.edge.cell
    
    cdef np.ndarray[REALS_t] hh_dual = variables.hh_dual
    cdef np.ndarray[REALS_t] hh_edge = variables.hh_edge
    cdef np.ndarray[REALS_t] h2_edge = variables.h2_edge
    cdef np.ndarray[REALS_t] up_bias = variables.hh_bias
    
    cdef REALS_t *HH_DUAL = &hh_dual[0]
    cdef REALS_t *HH_EDGE = &hh_edge[0]
    cdef REALS_t *H2_EDGE = &h2_edge[0]
    cdef REALS_t *UP_BIAS = &up_bias[0]
   
    if (cnfg.hh_scheme == "CENTRE"):
    
        with nogil, parallel(num_threads=cnfg_numthread):
        
            for vert in prange(0, NVRT, schedule="static", 
                    chunksize=cnfg_chunksize):
            #-- compute dual-centred thickness
                HH_DUAL[vert] = ZERO
                for iptr in range(DUAL_KITE_XPTR[vert +0], 
                                  DUAL_KITE_XPTR[vert +1]):
                        
                    xval = DUAL_KITE_XVAL[iptr]
                    xidx = DUAL_KITE_XIDX[iptr]
                        
                    HH_DUAL[vert]+= (xval * HH_CELL[xidx])
                    
                HH_DUAL[vert]/= MESH_DUAL_AREA[vert]
        
            for edge in prange(0, NEDG, schedule="static", 
                    chunksize=cnfg_chunksize):
            #-- compute edge-centred thickness
                HH_EDGE[edge] = ZERO
                UP_BIAS[edge] = ZERO
                for iptr in range(EDGE_WING_XPTR[edge +0], 
                                  EDGE_WING_XPTR[edge +1]):
                        
                    xval = EDGE_WING_XVAL[iptr]
                    xidx = EDGE_WING_XIDX[iptr]
                        
                    HH_EDGE[edge]+= (xval * HH_CELL[xidx])
                    
                HH_EDGE[edge]/= MESH_EDGE_AREA[edge]
                
            #-- compute for PV; simpson's rule
                H2_EDGE[edge] = (HH_EDGE[edge] * FOUR)
                for iptr in range(EDGE_VERT_XPTR[edge +0], 
                                  EDGE_VERT_XPTR[edge +1]):
                        
                    xidx = EDGE_VERT_XIDX[iptr]
                        
                    H2_EDGE[edge]+= (ONE_ * HH_DUAL[xidx])
                    
                H2_EDGE[edge]/= SIX_
            
    else:  # hh_scheme == "UPWIND"
    
        with nogil, parallel(num_threads=cnfg_numthread):
        
            for vert in prange(0, NVRT, schedule="static", 
                    chunksize=cnfg_chunksize):
            #-- compute dual-centred thickness               
                HH_DUAL[vert] = ZERO
                for iptr in range(DUAL_KITE_XPTR[vert +0], 
                                  DUAL_KITE_XPTR[vert +1]):
                        
                    xval = DUAL_KITE_XVAL[iptr]
                    xidx = DUAL_KITE_XIDX[iptr]
                        
                    HH_DUAL[vert]+= (xval * HH_CELL[xidx])
                    
                HH_DUAL[vert]/= MESH_DUAL_AREA[vert]
                
                HH_DUAL[vert] = max (ZERO , HH_DUAL[vert])
        
            for edge in prange(0, NEDG, schedule="static", 
                    chunksize=cnfg_chunksize):
                        
                cel1 = mesh_edge_cell[edge, 0] - 1
                cel2 = mesh_edge_cell[edge, 1] - 1
                
                if (cel1 < 0): cel1 = cel2
                if (cel2 < 0): cel2 = cel1
                        
            #-- compute edge-centred thickness
                HH_EDGE[edge] = ZERO
                for iptr in range(EDGE_WING_XPTR[edge +0], 
                                  EDGE_WING_XPTR[edge +1]):
                        
                    xval = EDGE_WING_XVAL[iptr]
                    xidx = EDGE_WING_XIDX[iptr]
                        
                    HH_EDGE[edge]+= (xval * HH_CELL[xidx])
                    
                HH_EDGE[edge]/= MESH_EDGE_AREA[edge]
                
                HH_EDGE[edge] = max (ZERO , HH_EDGE[edge])
                
            #-- compute upwind thickness blend
                if (UU_EDGE[edge] >= ZERO):
                    HH_WIND = HH_CELL[cel1]
                else:
                    HH_WIND = HH_CELL[cel2]
                       
                UP_BIAS[edge] = fabs_r(
                    HH_CELL[cel2] - HH_CELL[cel1]) \
                       / min(HH_CELL[cel1], HH_CELL[cel2])
                
                UP_BIAS[edge] = \
                    min(up_max_, UP_BIAS[edge])
                UP_BIAS[edge] = \
                    max(up_min_, UP_BIAS[edge])
                
                UP_BIAS[edge]*=  UP_BIAS[edge]
                
                HH_EDGE[edge] = \
                    UP_BIAS[edge] * HH_WIND + \
                    (ONE_ - UP_BIAS[edge])* HH_EDGE[edge]
                 
            #-- compute for PV; simpson's rule       
                H2_EDGE[edge] = (HH_EDGE[edge] * FOUR)
                for iptr in range(EDGE_VERT_XPTR[edge +0], 
                                  EDGE_VERT_XPTR[edge +1]):
                        
                    xidx = EDGE_VERT_XIDX[iptr]
                        
                    H2_EDGE[edge]+= (ONE_ * HH_DUAL[xidx])
                    
                H2_EDGE[edge]/= SIX_

    return hh_dual, hh_edge, h2_edge, up_bias

    
def _computeKE(mesh, trsk, cnfg, 
    np.ndarray[REALS_t, ndim=1] uu_edge,
    np.ndarray[REALS_t, ndim=1] vv_edge
              ):
    
#-- compute the kinetic energy 1/2 |u|^2
    
    cdef INDEX_t cell, iptr, xidx
    cdef REALS_t xval
    
    cdef REALS_t ZERO = 0.0
    cdef REALS_t HALF = 0.5
    
    cdef INDEX_t cnfg_numthread = cnfg.numthread
    cdef INDEX_t cnfg_chunksize = cnfg.chunksize
    
    cdef INDEX_t NVRT = mesh.vert.size
    cdef INDEX_t NEDG = mesh.edge.size
    cdef INDEX_t NCEL = mesh.cell.size
    
    cdef REALS_t *UU_EDGE = &uu_edge[0]
    cdef REALS_t *VV_EDGE = &vv_edge[0]
    
    cdef INDEX_t[::1] cell_wing_xptr = \
        trsk.cell_wing_sums.indptr
    cdef INDEX_t[::1] cell_wing_xidx = \
        trsk.cell_wing_sums.indices
    cdef REALS_t[::1] cell_wing_xval = \
        trsk.cell_wing_sums.data
    
    cdef INDEX_t *CELL_WING_XPTR = &cell_wing_xptr[0]
    cdef INDEX_t *CELL_WING_XIDX = &cell_wing_xidx[0]
    cdef REALS_t *CELL_WING_XVAL = &cell_wing_xval[0]
    
    cdef REALS_t[::1] mesh_cell_area = mesh.cell.area
    
    cdef REALS_t *MESH_CELL_AREA = &mesh_cell_area[0]
    
    cdef np.ndarray[REALS_t] ke_cell = variables.ke_cell
        
    cdef REALS_t *KE_CELL = &ke_cell[0]
    
    with nogil, parallel(num_threads=cnfg_numthread):
        
        for cell in prange(0, NCEL, schedule="static", 
                chunksize=cnfg_chunksize):
        #-- remap kinetic energy M_(c,e) * 1/2 * |u|^2
            KE_CELL[cell] = ZERO
            for iptr in range(CELL_WING_XPTR[cell +0],
                              CELL_WING_XPTR[cell +1]):
                    
                xval = CELL_WING_XVAL[iptr]
                xidx = CELL_WING_XIDX[iptr]
                
                KE_CELL[cell]+= \
                    HALF * xval * ( 
                        UU_EDGE[xidx] * UU_EDGE[xidx] 
                      + VV_EDGE[xidx] * VV_EDGE[xidx])
        
            KE_CELL[cell]/= MESH_CELL_AREA[cell]
    
    return ke_cell
    
    
def _computePV(mesh, trsk, cnfg, 
    np.ndarray[REALS_t, ndim=1] hh_cell, 
    np.ndarray[REALS_t, ndim=1] hh_edge, 
    np.ndarray[REALS_t, ndim=1] hh_dual,
    np.ndarray[REALS_t, ndim=1] uu_edge, 
    np.ndarray[REALS_t, ndim=1] vv_edge,
    np.ndarray[FLT32_t, ndim=1] ff_dual,
    np.ndarray[FLT32_t, ndim=1] ff_edge,
    np.ndarray[FLT32_t, ndim=1] ff_cell
              ):
    
#-- compute potential and relative curl u
    
    cdef INDEX_t vert, edge, cell, iptr, xidx
    cdef REALS_t xval
    
    cdef REALS_t ZERO = 0.0
    cdef REALS_t ONE_ = 1.0
    
    cdef INDEX_t cnfg_numthread = cnfg.numthread
    cdef INDEX_t cnfg_chunksize = cnfg.chunksize
    cdef REALS_t cnfg_do_advect = (not cnfg.no_advect)
    
    cdef INDEX_t NVRT = mesh.vert.size
    cdef INDEX_t NEDG = mesh.edge.size
    cdef INDEX_t NCEL = mesh.cell.size
    
    cdef REALS_t *HH_CELL = &hh_cell[0]
    cdef REALS_t *HH_EDGE = &hh_edge[0]
    cdef REALS_t *HH_DUAL = &hh_dual[0]
    cdef REALS_t *UU_EDGE = &uu_edge[0]
    cdef REALS_t *VV_EDGE = &vv_edge[0]
    cdef FLT32_t *FF_DUAL = &ff_dual[0]
    cdef FLT32_t *FF_EDGE = &ff_edge[0]
    cdef FLT32_t *FF_CELL = &ff_cell[0]
    
    cdef INDEX_t[::1] dual_curl_xptr = \
        trsk.dual_curl_sums.indptr
    cdef INDEX_t[::1] dual_curl_xidx = \
        trsk.dual_curl_sums.indices
    cdef REALS_t[::1] dual_curl_xval = \
        trsk.dual_curl_sums.data
    
    cdef INDEX_t *DUAL_CURL_XPTR = &dual_curl_xptr[0]
    cdef INDEX_t *DUAL_CURL_XIDX = &dual_curl_xidx[0]
    cdef REALS_t *DUAL_CURL_XVAL = &dual_curl_xval[0]
    
    cdef INDEX_t[::1] edge_vert_xptr = \
        trsk.edge_vert_sums.indptr
    cdef INDEX_t[::1] edge_vert_xidx = \
        trsk.edge_vert_sums.indices
    
    cdef INDEX_t *EDGE_VERT_XPTR = &edge_vert_xptr[0]
    cdef INDEX_t *EDGE_VERT_XIDX = &edge_vert_xidx[0]
       
    cdef INDEX_t[::1] cell_kite_xptr = \
        trsk.cell_kite_sums.indptr
    cdef INDEX_t[::1] cell_kite_xidx = \
        trsk.cell_kite_sums.indices
    cdef REALS_t[::1] cell_kite_xval = \
        trsk.cell_kite_sums.data
    
    cdef INDEX_t *CELL_KITE_XPTR = &cell_kite_xptr[0]
    cdef INDEX_t *CELL_KITE_XIDX = &cell_kite_xidx[0]
    cdef REALS_t *CELL_KITE_XVAL = &cell_kite_xval[0]
       
    cdef INDEX_t[::1] vert_tail_xptr = \
        trsk.dual_tail_sums.indptr
    cdef INDEX_t[::1] vert_tail_xidx = \
        trsk.dual_tail_sums.indices
    cdef REALS_t[::1] vert_tail_xval = \
        trsk.dual_tail_sums.data
    
    cdef INDEX_t *VERT_TAIL_XPTR = &vert_tail_xptr[0]
    cdef INDEX_t *VERT_TAIL_XIDX = &vert_tail_xidx[0]
    cdef REALS_t *VERT_TAIL_XVAL = &vert_tail_xval[0]
       
    cdef REALS_t[::1] mesh_cell_area = mesh.cell.area
    cdef REALS_t[::1] mesh_dual_area = mesh.vert.area
    cdef REALS_t[::1] mesh_quad_area = mesh.quad.area
    
    cdef REALS_t *MESH_CELL_AREA = &mesh_cell_area[0]
    cdef REALS_t *MESH_DUAL_AREA = &mesh_dual_area[0]
    cdef REALS_t *MESH_QUAD_AREA = &mesh_quad_area[0]
    
    cdef REALS_t[::1] mesh_vert_slip = mesh.vert.slip
    
    cdef REALS_t *MESH_VERT_SLIP = &mesh_vert_slip[0]
    
    cdef np.ndarray[REALS_t] rv_dual = variables.rv_dual
    cdef np.ndarray[REALS_t] pv_dual = variables.pv_dual
    
    cdef REALS_t *RV_DUAL = &rv_dual[0]
    cdef REALS_t *PV_DUAL = &pv_dual[0]
    
    cdef np.ndarray[REALS_t] rv_cell = variables.rv_cell
    cdef np.ndarray[REALS_t] pv_cell = variables.pv_cell
        
    cdef REALS_t *RV_CELL = &rv_cell[0]
    cdef REALS_t *PV_CELL = &pv_cell[0]
    
    cdef np.ndarray[REALS_t] r2_dual = variables.r2_dual
    cdef np.ndarray[REALS_t] p2_dual = variables.p2_dual
    
    cdef REALS_t *R2_DUAL = &r2_dual[0]
    cdef REALS_t *P2_DUAL = &p2_dual[0]
    
    cdef np.ndarray[REALS_t] rv_edge = variables.rv_edge
    cdef np.ndarray[REALS_t] pv_edge = variables.pv_edge
        
    cdef REALS_t *RV_EDGE = &rv_edge[0]
    cdef REALS_t *PV_EDGE = &pv_edge[0]
   
    with nogil, parallel(num_threads=cnfg_numthread):
    
        for vert in prange(0, NVRT, schedule="static",
                chunksize=cnfg_chunksize):
        #-- compute dual-centred vorticity
            RV_DUAL[vert] = ZERO
            for iptr in range(DUAL_CURL_XPTR[vert +0], 
                              DUAL_CURL_XPTR[vert +1]):
                    
                xval = DUAL_CURL_XVAL[iptr]
                xidx = DUAL_CURL_XIDX[iptr]
                    
                RV_DUAL[vert]+= (xval * UU_EDGE[xidx])
            
            # circulation, not curl(u) yet
            RV_DUAL[vert]*= (
                ONE_ - MESH_VERT_SLIP[vert]
                        )
            RV_DUAL[vert]*=(cnfg_do_advect)
            
        for edge in prange(0, NEDG, schedule="static", 
                chunksize=cnfg_chunksize):
        #-- compute edge-centred vorticity
            RV_EDGE[edge] = ZERO
            for iptr in range(EDGE_VERT_XPTR[edge +0], 
                              EDGE_VERT_XPTR[edge +1]):
                    
                xidx = EDGE_VERT_XIDX[iptr]
                    
                RV_EDGE[edge]+= (ONE_ * RV_DUAL[xidx])
        
            RV_EDGE[edge]/= MESH_QUAD_AREA[edge]
            
            PV_EDGE[edge] = \
                (ONE_ / HH_EDGE[edge]) * \
                    (RV_EDGE[edge] + FF_EDGE[edge])
            
        for vert in prange(0, NVRT, schedule="static", 
                chunksize=cnfg_chunksize):
        #-- average rhombi to dual -- a'la Gassmann
            R2_DUAL[vert] = ZERO
            for iptr in range(VERT_TAIL_XPTR[vert +0], 
                              VERT_TAIL_XPTR[vert +1]):
                
                xval = VERT_TAIL_XVAL[iptr]    
                xidx = VERT_TAIL_XIDX[iptr]
                    
                R2_DUAL[vert]+= (xval * RV_EDGE[xidx])
             
            R2_DUAL[vert]/= MESH_DUAL_AREA[vert]
            
            P2_DUAL[vert] = \
                (ONE_ / HH_DUAL[vert]) * \
                    (R2_DUAL[vert] + FF_DUAL[vert])
            
            # circulation to curl(u) final
            RV_DUAL[vert]/= MESH_DUAL_AREA[vert]
        
            PV_DUAL[vert] = \
                (ONE_ / HH_DUAL[vert]) * \
                    (RV_DUAL[vert] + FF_DUAL[vert])
                        
        for cell in prange(0, NCEL, schedule="static", 
                chunksize=cnfg_chunksize):
        #-- compute cell-centred vorticity
            RV_CELL[cell] = ZERO
            for iptr in range(CELL_KITE_XPTR[cell +0], 
                              CELL_KITE_XPTR[cell +1]):
                    
                xval = CELL_KITE_XVAL[iptr]
                xidx = CELL_KITE_XIDX[iptr]
                    
                RV_CELL[cell]+= (xval * RV_DUAL[xidx])
        
            RV_CELL[cell]/= MESH_CELL_AREA[cell]
        
            PV_CELL[cell] = \
                (ONE_ / HH_CELL[cell]) * \
                    (RV_CELL[cell] + FF_CELL[cell])
                  
    return rv_dual, pv_dual, \
           r2_dual, p2_dual, \
           rv_cell, pv_cell, \
           rv_edge, pv_edge
    
    
def _advect_UH(mesh, trsk, cnfg, 
    np.ndarray[REALS_t, ndim=1] hh_edge,
    np.ndarray[REALS_t, ndim=1] uu_edge,
    np.ndarray[REALS_t, ndim=1] hh_tend
              ):
    
#-- compute thickness advection: div uh
    
    cdef INDEX_t cell, iptr, xidx
    cdef REALS_t xval
  
    cdef REALS_t UH_TEND
    
    cdef REALS_t ZERO = 0.0
    
    cdef INDEX_t cnfg_numthread = cnfg.numthread
    cdef INDEX_t cnfg_chunksize = cnfg.chunksize
    
    cdef INDEX_t NVRT = mesh.vert.size
    cdef INDEX_t NEDG = mesh.edge.size
    cdef INDEX_t NCEL = mesh.cell.size
    
    cdef REALS_t *HH_EDGE = &hh_edge[0]
    cdef REALS_t *UU_EDGE = &uu_edge[0]
    cdef REALS_t *HH_TEND = &hh_tend[0]
    
    cdef INDEX_t[::1] cell_flux_xptr = \
        trsk.cell_flux_sums.indptr
    cdef INDEX_t[::1] cell_flux_xidx = \
        trsk.cell_flux_sums.indices
    cdef REALS_t[::1] cell_flux_xval = \
        trsk.cell_flux_sums.data
    
    cdef INDEX_t *CELL_FLUX_XPTR = &cell_flux_xptr[0]
    cdef INDEX_t *CELL_FLUX_XIDX = &cell_flux_xidx[0]
    cdef REALS_t *CELL_FLUX_XVAL = &cell_flux_xval[0]
    
    cdef REALS_t[::1] mesh_cell_area = mesh.cell.area
    
    cdef REALS_t *MESH_CELL_AREA = &mesh_cell_area[0]
    
    with nogil, parallel(num_threads=cnfg_numthread):
        
        for cell in prange(0, NCEL, schedule="static", 
                chunksize=cnfg_chunksize):
        #-- divergence of edge thickness fluxes D * uh
            UH_TEND = ZERO
            for iptr in range(CELL_FLUX_XPTR[cell +0], 
                              CELL_FLUX_XPTR[cell +1]):
                    
                xval = CELL_FLUX_XVAL[iptr]
                xidx = CELL_FLUX_XIDX[iptr]
                
                UH_TEND = UH_TEND + (
                                xval * UU_EDGE[xidx] * 
                                       HH_EDGE[xidx]
                                    )
        
            HH_TEND[cell]+= \
                UH_TEND / MESH_CELL_AREA [cell]
    
    return hh_tend
    
    
def _advect_UV(mesh, trsk, cnfg, 
    np.ndarray[REALS_t, ndim=1] hh_edge,
    np.ndarray[REALS_t, ndim=1] uu_edge,
    np.ndarray[REALS_t, ndim=1] pv_edge,
    np.ndarray[REALS_t, ndim=1] ke_cell,
    np.ndarray[REALS_t, ndim=1] uu_tend
              ):
    
#-- mometum advection: qhu^\perp + grad K
    
    cdef INDEX_t edge, iptr, xidx
    cdef REALS_t xval
    
    cdef REALS_t KE_GRAD, UV_FLUX
    
    cdef REALS_t HALF = 0.5
    cdef REALS_t ZERO = 0.0
    
    cdef INDEX_t cnfg_numthread = cnfg.numthread
    cdef INDEX_t cnfg_chunksize = cnfg.chunksize
    cdef REALS_t cnfg_do_advect = (not cnfg.no_advect)
    
    cdef INDEX_t NVRT = mesh.vert.size
    cdef INDEX_t NEDG = mesh.edge.size
    cdef INDEX_t NCEL = mesh.cell.size
    
    cdef REALS_t *HH_EDGE = &hh_edge[0]
    cdef REALS_t *UU_EDGE = &uu_edge[0]
    cdef REALS_t *PV_EDGE = &pv_edge[0]
    cdef REALS_t *KE_CELL = &ke_cell[0]
    cdef REALS_t *UU_TEND = &uu_tend[0]

    cdef INDEX_t[::1] edge_flux_xptr = \
        trsk.edge_flux_perp.indptr
    cdef INDEX_t[::1] edge_flux_xidx = \
        trsk.edge_flux_perp.indices
    cdef REALS_t[::1] edge_flux_xval = \
        trsk.edge_flux_perp.data
    
    cdef INDEX_t *EDGE_FLUX_XPTR = &edge_flux_xptr[0]
    cdef INDEX_t *EDGE_FLUX_XIDX = &edge_flux_xidx[0]
    cdef REALS_t *EDGE_FLUX_XVAL = &edge_flux_xval[0]
       
    cdef INDEX_t[::1] grad_norm_xptr = \
        trsk.edge_grad_norm.indptr
    cdef INDEX_t[::1] grad_norm_xidx = \
        trsk.edge_grad_norm.indices
    cdef REALS_t[::1] grad_norm_xval = \
        trsk.edge_grad_norm.data
    
    cdef INDEX_t *GRAD_NORM_XPTR = &grad_norm_xptr[0]
    cdef INDEX_t *GRAD_NORM_XIDX = &grad_norm_xidx[0]
    cdef REALS_t *GRAD_NORM_XVAL = &grad_norm_xval[0]
    
    with nogil, parallel(num_threads=cnfg_numthread):

        for edge in prange(0, NEDG, schedule="static", 
                chunksize=cnfg_chunksize):
        #-- gradient of kinetic energy G * 1/2 * |u|^2
            KE_GRAD = ZERO
            for iptr in range(GRAD_NORM_XPTR[edge +0], 
                              GRAD_NORM_XPTR[edge +1]):
                    
                xval = GRAD_NORM_XVAL[iptr]
                xidx = GRAD_NORM_XIDX[iptr]
                
                KE_GRAD = KE_GRAD + xval * KE_CELL[xidx]
                
            KE_GRAD = KE_GRAD * cnfg_do_advect

        #-- energy neutral flux 1/2 * (W*qhu + q*W*hu)
            UV_FLUX = ZERO
            for iptr in range(EDGE_FLUX_XPTR[edge +0], 
                              EDGE_FLUX_XPTR[edge +1]):
                    
                xval = EDGE_FLUX_XVAL[iptr]
                xidx = EDGE_FLUX_XIDX[iptr]
                    
                UV_FLUX = UV_FLUX - xval * \
                    UU_EDGE[xidx] * \
                    HH_EDGE[xidx] * \
                        (PV_EDGE[edge] + PV_EDGE[xidx]
                        )
                
            UU_TEND[edge]+= KE_GRAD + (HALF * UV_FLUX)
        
    return uu_tend
    

def _computeGZ(mesh, trsk, cnfg, 
    np.ndarray[REALS_t, ndim=1] hh_cell,
    np.ndarray[FLT32_t, ndim=1] zb_cell,
        const REALS_t gg_cell,
    np.ndarray[REALS_t, ndim=1] uu_tend
              ):
    
#-- pressure gradient: grad g * (h + z_b)
    
    cdef INDEX_t edge, iptr, xidx
    cdef REALS_t xval
    
    cdef REALS_t GZ_EDGE
    
    cdef REALS_t ZERO = 0.0
    
    cdef INDEX_t cnfg_numthread = cnfg.numthread
    cdef INDEX_t cnfg_chunksize = cnfg.chunksize
    
    cdef INDEX_t NVRT = mesh.vert.size
    cdef INDEX_t NEDG = mesh.edge.size
    cdef INDEX_t NCEL = mesh.cell.size
    
    cdef REALS_t *HH_CELL = &hh_cell[0]
    cdef FLT32_t *ZB_CELL = &zb_cell[0]
    cdef REALS_t *UU_TEND = &uu_tend[0]
    
    cdef INDEX_t[::1] grad_norm_xptr = \
        trsk.edge_grad_norm.indptr
    cdef INDEX_t[::1] grad_norm_xidx = \
        trsk.edge_grad_norm.indices
    cdef REALS_t[::1] grad_norm_xval = \
        trsk.edge_grad_norm.data
    
    cdef INDEX_t *GRAD_NORM_XPTR = &grad_norm_xptr[0]
    cdef INDEX_t *GRAD_NORM_XIDX = &grad_norm_xidx[0]
    cdef REALS_t *GRAD_NORM_XVAL = &grad_norm_xval[0]

    with nogil, parallel(num_threads=cnfg_numthread):

        for edge in prange(0, NEDG, schedule="static", 
                chunksize=cnfg_chunksize):
        #-- surface pressure gradient g * G * (h + zb)
            GZ_EDGE = ZERO
            for iptr in range(GRAD_NORM_XPTR[edge +0], 
                              GRAD_NORM_XPTR[edge +1]):
                    
                xval = GRAD_NORM_XVAL[iptr]
                xidx = GRAD_NORM_XIDX[iptr]
                    
                GZ_EDGE = GZ_EDGE + \
                    xval* gg_cell * (
                        HH_CELL[xidx] + ZB_CELL[xidx])
                        
            UU_TEND[edge] = UU_TEND[edge] + GZ_EDGE
    
    return uu_tend

    
def _computeVV(mesh, trsk, cnfg, 
    np.ndarray[REALS_t, ndim=1] uu_edge
              ):
    
#-- tangent reconstruction: F^\perp <== F
    
    cdef INDEX_t edge, iptr, xidx
    cdef REALS_t xval
    
    cdef REALS_t ZERO = 0.0
    
    cdef INDEX_t cnfg_numthread = cnfg.numthread
    cdef INDEX_t cnfg_chunksize = cnfg.chunksize
    
    cdef INDEX_t NVRT = mesh.vert.size
    cdef INDEX_t NEDG = mesh.edge.size
    cdef INDEX_t NCEL = mesh.cell.size
    
    cdef REALS_t *UU_EDGE = &uu_edge[0]

    cdef INDEX_t[::1] edge_lsqr_xptr = \
        trsk.edge_lsqr_perp.indptr
    cdef INDEX_t[::1] edge_lsqr_xidx = \
        trsk.edge_lsqr_perp.indices
    cdef REALS_t[::1] edge_lsqr_xval = \
        trsk.edge_lsqr_perp.data
    
    cdef INDEX_t *EDGE_LSQR_XPTR = &edge_lsqr_xptr[0]
    cdef INDEX_t *EDGE_LSQR_XIDX = &edge_lsqr_xidx[0]
    cdef REALS_t *EDGE_LSQR_XVAL = &edge_lsqr_xval[0]

    cdef np.ndarray[REALS_t] vv_edge = variables.vv_edge
        
    cdef REALS_t *VV_EDGE = &vv_edge[0]
    
    with nogil, parallel(num_threads=cnfg_numthread):
        
        for edge in prange(0, NEDG, schedule="static", 
                chunksize=cnfg_chunksize):
        #-- LSQR .^perp reconstruction: v = W_lsqr * u
            VV_EDGE[edge] = ZERO
            for iptr in range(EDGE_LSQR_XPTR[edge +0], 
                              EDGE_LSQR_XPTR[edge +1]):
                    
                xval = EDGE_LSQR_XVAL[iptr]
                xidx = EDGE_LSQR_XIDX[iptr]
                    
                VV_EDGE[edge]+= (xval * UU_EDGE[xidx])
    
    return vv_edge


def _computeNu(mesh, trsk, cnfg, 
    np.ndarray[REALS_t, ndim=1] rv_dual,
    np.ndarray[REALS_t, ndim=1] rv_cell
              ):
          
#-- leith sub-grid model for turb. nu_2^u
    
    cdef INDEX_t edge, cell, iptr, xidx
    cdef REALS_t xval
    cdef REALS_t dN_EDGE, dP_EDGE, NU_SCAL
    
    cdef REALS_t ZERO = 0.0
    cdef REALS_t TWO_ = 2.0
    
    cdef INDEX_t cnfg_numthread = cnfg.numthread
    cdef INDEX_t cnfg_chunksize = cnfg.chunksize
    cdef REALS_t cnfg_leith_chi = cnfg.leith_chi
    
    cdef INDEX_t NVRT = mesh.vert.size
    cdef INDEX_t NEDG = mesh.edge.size
    cdef INDEX_t NCEL = mesh.cell.size
    
    cdef REALS_t *RV_DUAL = &rv_dual[0]
    cdef REALS_t *RV_CELL = &rv_cell[0]
    
    cdef REALS_t[::1] nu_max_ = cnfg.leith_max
    cdef REALS_t *NU_MAX_ = &nu_max_[0]
    
    cdef INDEX_t[::1] grad_norm_xptr = \
        trsk.edge_grad_norm.indptr
    cdef INDEX_t[::1] grad_norm_xidx = \
        trsk.edge_grad_norm.indices
    cdef REALS_t[::1] grad_norm_xval = \
        trsk.edge_grad_norm.data
    
    cdef INDEX_t *GRAD_NORM_XPTR = &grad_norm_xptr[0]
    cdef INDEX_t *GRAD_NORM_XIDX = &grad_norm_xidx[0]
    cdef REALS_t *GRAD_NORM_XVAL = &grad_norm_xval[0]
    
    cdef INDEX_t[::1] grad_perp_xptr = \
        trsk.edge_grad_perp.indptr
    cdef INDEX_t[::1] grad_perp_xidx = \
        trsk.edge_grad_perp.indices
    cdef REALS_t[::1] grad_perp_xval = \
        trsk.edge_grad_perp.data
    
    cdef INDEX_t *GRAD_PERP_XPTR = &grad_perp_xptr[0]
    cdef INDEX_t *GRAD_PERP_XIDX = &grad_perp_xidx[0]
    cdef REALS_t *GRAD_PERP_XVAL = &grad_perp_xval[0]
    
    cdef REALS_t[::1] mesh_edge_slen = mesh.edge.slen
    
    cdef REALS_t *MESH_EDGE_SLEN = &mesh_edge_slen[0]
    
    cdef np.ndarray[REALS_t] nu_edge = variables.nu_edge
    
    cdef REALS_t *NU_TURB = &nu_edge[0]

    with nogil, parallel(num_threads=cnfg_numthread):
  
        for edge in prange(0, NEDG, schedule="static", 
                chunksize=cnfg_chunksize):        
        #-- dN_edge = edge_grad_norm * rv_cell
            dN_EDGE = ZERO
            for iptr in range(GRAD_NORM_XPTR[edge +0], 
                              GRAD_NORM_XPTR[edge +1]):
                    
                xval = GRAD_NORM_XVAL[iptr]
                xidx = GRAD_NORM_XIDX[iptr]
                    
                dN_EDGE = \
                    dN_EDGE + (xval * RV_CELL[xidx])
                    
        #-- dP_edge = edge_grad_perp * rv_dual
            dP_EDGE = ZERO
            for iptr in range(GRAD_PERP_XPTR[edge +0],
                              GRAD_PERP_XPTR[edge +1]):
                    
                xval = GRAD_PERP_XVAL[iptr]
                xidx = GRAD_PERP_XIDX[iptr]
                    
                dP_EDGE = \
                    dP_EDGE + (xval * RV_DUAL[xidx])
        
            # nu_2 = (chi * len)^3 * |grad (curl u)|
            NU_TURB[edge] = sqrt_r(
                            dN_EDGE * dN_EDGE + 
                            dP_EDGE * dP_EDGE )
            
            NU_SCAL = (cnfg_leith_chi * MESH_EDGE_SLEN[edge])
            NU_SCAL*= TWO_  # slen is only half
                    
            NU_TURB[edge]*= NU_SCAL * NU_SCAL * NU_SCAL
            
            NU_TURB[edge] = min(NU_TURB[edge], NU_MAX_[edge])
        
    return nu_edge


def _computeDU(mesh, trsk, cnfg, 
    np.ndarray[REALS_t, ndim=1] uu_edge,
    np.ndarray[REALS_t, ndim=1] uu_tend
              ):
    
#-- viscosity dissipation: nu_k^u * div^k
    
    cdef INDEX_t edge, cell, iptr, xidx
    cdef REALS_t xval
    
    cdef REALS_t ZERO = 0.0
    
    cdef INDEX_t cnfg_numthread = cnfg.numthread
    cdef INDEX_t cnfg_chunksize = cnfg.chunksize
    
    cdef INDEX_t NVRT = mesh.vert.size
    cdef INDEX_t NEDG = mesh.edge.size
    cdef INDEX_t NCEL = mesh.cell.size
    
    cdef REALS_t *UU_EDGE = &uu_edge[0]
    cdef REALS_t *UU_TEND = &uu_tend[0]
    
    cdef REALS_t[::1] d2_visc = cnfg.du_visc_2
    cdef REALS_t *D2_VISC = &d2_visc[0]
    
    cdef REALS_t[::1] d4_visc = cnfg.du_visc_4
    cdef REALS_t *D4_VISC = &d4_visc[0]
    
    cdef INDEX_t[::1] cell_flux_xptr = \
        trsk.cell_flux_sums.indptr
    cdef INDEX_t[::1] cell_flux_xidx = \
        trsk.cell_flux_sums.indices
    cdef REALS_t[::1] cell_flux_xval = \
        trsk.cell_flux_sums.data
    
    cdef INDEX_t *CELL_FLUX_XPTR = &cell_flux_xptr[0]
    cdef INDEX_t *CELL_FLUX_XIDX = &cell_flux_xidx[0]
    cdef REALS_t *CELL_FLUX_XVAL = &cell_flux_xval[0]
       
    cdef INDEX_t[::1] grad_norm_xptr = \
        trsk.edge_grad_norm.indptr
    cdef INDEX_t[::1] grad_norm_xidx = \
        trsk.edge_grad_norm.indices
    cdef REALS_t[::1] grad_norm_xval = \
        trsk.edge_grad_norm.data
    
    cdef INDEX_t *GRAD_NORM_XPTR = &grad_norm_xptr[0]
    cdef INDEX_t *GRAD_NORM_XIDX = &grad_norm_xidx[0]
    cdef REALS_t *GRAD_NORM_XVAL = &grad_norm_xval[0]
    
    cdef REALS_t[::1] mesh_cell_area = mesh.cell.area
    
    cdef REALS_t *MESH_CELL_AREA = &mesh_cell_area[0]
    
    cdef np.ndarray[REALS_t] d2_edge = get_vec_e()
    cdef np.ndarray[REALS_t] d4_edge = get_vec_e()
        
    cdef REALS_t *D2_EDGE = &d2_edge[0]
    cdef REALS_t *D4_EDGE = &d4_edge[0]
    
    cdef np.ndarray[REALS_t] du_cell = get_vec_c()
        
    cdef REALS_t *DU_CELL = &du_cell[0]
        
    with nogil, parallel(num_threads=cnfg_numthread):
            
        for cell in prange(0, NCEL, schedule="static", 
                chunksize=cnfg_chunksize):
        #-- compute div(u.n)
            DU_CELL[cell] = ZERO
            for iptr in range(CELL_FLUX_XPTR[cell +0], 
                              CELL_FLUX_XPTR[cell +1]):
                    
                xval = CELL_FLUX_XVAL[iptr]
                xidx = CELL_FLUX_XIDX[iptr]
                    
                DU_CELL[cell]+= (xval * UU_EDGE[xidx])
              
            DU_CELL[cell]/= MESH_CELL_AREA[cell]
            
        for edge in prange(0, NEDG, schedule="static", 
                chunksize=cnfg_chunksize):
        #-- D^2 = vk * grad(div(u.n))
            D2_EDGE[edge] = ZERO
            for iptr in range(GRAD_NORM_XPTR[edge +0], 
                              GRAD_NORM_XPTR[edge +1]):
                    
                xval = GRAD_NORM_XVAL[iptr]
                xidx = GRAD_NORM_XIDX[iptr]
                    
                D2_EDGE[edge]+= (xval * DU_CELL[xidx])
                
            D4_EDGE[edge] = (
                   D2_EDGE[edge] * D4_VISC[edge]
                   )
                
        for cell in prange(0, NCEL, schedule="static", 
                chunksize=cnfg_chunksize):
        #-- compute div(D^2)
            DU_CELL[cell] = ZERO
            for iptr in range(CELL_FLUX_XPTR[cell +0], 
                              CELL_FLUX_XPTR[cell +1]):
                    
                xval = CELL_FLUX_XVAL[iptr]
                xidx = CELL_FLUX_XIDX[iptr]
                    
                DU_CELL[cell]+= (xval * D4_EDGE[xidx])
         
            DU_CELL[cell]/= MESH_CELL_AREA[cell]

        for edge in prange(0, NEDG, schedule="static", 
                chunksize=cnfg_chunksize):
        #-- D^4 = vk * grad(div(D^2))
            D4_EDGE[edge] = ZERO
            for iptr in range(GRAD_NORM_XPTR[edge +0], 
                              GRAD_NORM_XPTR[edge +1]):
                    
                xval = GRAD_NORM_XVAL[iptr]
                xidx = GRAD_NORM_XIDX[iptr]
                    
                D4_EDGE[edge]+= (xval * DU_CELL[xidx])
                
            UU_TEND[edge]-= (
                D2_VISC[edge] * D2_EDGE[edge]
              - D4_VISC[edge] * D4_EDGE[edge]
                )
            
    put_vec_e  (d4_edge)       
    put_vec_e  (d2_edge)
    put_vec_c  (du_cell)
                     
    return uu_tend
    

def _computeVU(mesh, trsk, cnfg,
    np.ndarray[REALS_t, ndim=1] uu_edge,
    np.ndarray[REALS_t, ndim=1] nu_edge,
    np.ndarray[REALS_t, ndim=1] uu_tend
              ):
    
#-- viscosity dissipation: nu_k^u * del^k
    
    cdef INDEX_t vert, edge, cell, iptr, xidx
    cdef REALS_t xval
    
    cdef REALS_t ZERO = 0.0
    cdef REALS_t ONE_ = 1.0
    
    cdef INDEX_t cnfg_numthread = cnfg.numthread
    cdef INDEX_t cnfg_chunksize = cnfg.chunksize
    
    cdef INDEX_t NVRT = mesh.vert.size
    cdef INDEX_t NEDG = mesh.edge.size
    cdef INDEX_t NCEL = mesh.cell.size
    
    cdef REALS_t *UU_EDGE = &uu_edge[0]
    cdef REALS_t *NU_TURB = &nu_edge[0]
    cdef REALS_t *UU_TEND = &uu_tend[0]
    
    cdef REALS_t[::1] v2_visc = cnfg.uu_visc_2
    cdef REALS_t *V2_VISC = &v2_visc[0]
    
    cdef REALS_t[::1] v4_visc = cnfg.uu_visc_4
    cdef REALS_t *V4_VISC = &v4_visc[0]
    
    cdef INDEX_t[::1] cell_flux_xptr = \
        trsk.cell_flux_sums.indptr
    cdef INDEX_t[::1] cell_flux_xidx = \
        trsk.cell_flux_sums.indices
    cdef REALS_t[::1] cell_flux_xval = \
        trsk.cell_flux_sums.data
    
    cdef INDEX_t *CELL_FLUX_XPTR = &cell_flux_xptr[0]
    cdef INDEX_t *CELL_FLUX_XIDX = &cell_flux_xidx[0]
    cdef REALS_t *CELL_FLUX_XVAL = &cell_flux_xval[0]
       
    cdef INDEX_t[::1] dual_curl_xptr = \
        trsk.dual_curl_sums.indptr
    cdef INDEX_t[::1] dual_curl_xidx = \
        trsk.dual_curl_sums.indices
    cdef REALS_t[::1] dual_curl_xval = \
        trsk.dual_curl_sums.data
    
    cdef INDEX_t *DUAL_CURL_XPTR = &dual_curl_xptr[0]
    cdef INDEX_t *DUAL_CURL_XIDX = &dual_curl_xidx[0]
    cdef REALS_t *DUAL_CURL_XVAL = &dual_curl_xval[0]
       
    cdef INDEX_t[::1] grad_norm_xptr = \
        trsk.edge_grad_norm.indptr
    cdef INDEX_t[::1] grad_norm_xidx = \
        trsk.edge_grad_norm.indices
    cdef REALS_t[::1] grad_norm_xval = \
        trsk.edge_grad_norm.data
    
    cdef INDEX_t *GRAD_NORM_XPTR = &grad_norm_xptr[0]
    cdef INDEX_t *GRAD_NORM_XIDX = &grad_norm_xidx[0]
    cdef REALS_t *GRAD_NORM_XVAL = &grad_norm_xval[0]
    
    cdef INDEX_t[::1] grad_perp_xptr = \
        trsk.edge_grad_perp.indptr
    cdef INDEX_t[::1] grad_perp_xidx = \
        trsk.edge_grad_perp.indices
    cdef REALS_t[::1] grad_perp_xval = \
        trsk.edge_grad_perp.data
    
    cdef INDEX_t *GRAD_PERP_XPTR = &grad_perp_xptr[0]
    cdef INDEX_t *GRAD_PERP_XIDX = &grad_perp_xidx[0]
    cdef REALS_t *GRAD_PERP_XVAL = &grad_perp_xval[0]
    
    cdef REALS_t[::1] mesh_cell_area = mesh.cell.area
    cdef REALS_t[::1] mesh_dual_area = mesh.vert.area
    
    cdef REALS_t *MESH_CELL_AREA = &mesh_cell_area[0]
    cdef REALS_t *MESH_DUAL_AREA = &mesh_dual_area[0]
    
    cdef REALS_t[::1] mesh_edge_mask = mesh.edge.fmsk
    
    cdef REALS_t *MESH_EDGE_MASK = &mesh_edge_mask[0]
    
    cdef REALS_t[::1] mesh_vert_slip = mesh.vert.slip
    
    cdef REALS_t *MESH_VERT_SLIP = &mesh_vert_slip[0]
    
    cdef np.ndarray[REALS_t] v2_edge = get_vec_e()
    cdef np.ndarray[REALS_t] v4_edge = get_vec_e()
        
    cdef REALS_t *V2_EDGE = &v2_edge[0]
    cdef REALS_t *V4_EDGE = &v4_edge[0]
    
    cdef np.ndarray[REALS_t] rv_dual = get_vec_v()
    cdef np.ndarray[REALS_t] du_cell = get_vec_c()
        
    cdef REALS_t *RV_DUAL = &rv_dual[0]
    cdef REALS_t *DU_CELL = &du_cell[0]
    
    with nogil, parallel(num_threads=cnfg_numthread):
        
        for cell in prange(0, NCEL, schedule="static", 
                chunksize=cnfg_chunksize):
        #-- compute div(u.n)
            DU_CELL[cell] = ZERO
            for iptr in range(CELL_FLUX_XPTR[cell +0], 
                              CELL_FLUX_XPTR[cell +1]):
                    
                xval = CELL_FLUX_XVAL[iptr]
                xidx = CELL_FLUX_XIDX[iptr]
                    
                DU_CELL[cell]+= (xval * UU_EDGE[xidx])
            
            DU_CELL[cell]/= MESH_CELL_AREA[cell]
            
        for vert in prange(0, NVRT, schedule="static", 
                chunksize=cnfg_chunksize):
        #-- compute rot(u.n)
            RV_DUAL[vert] = ZERO
            for iptr in range(DUAL_CURL_XPTR[vert +0], 
                              DUAL_CURL_XPTR[vert +1]):
                    
                xval = DUAL_CURL_XVAL[iptr]
                xidx = DUAL_CURL_XIDX[iptr]
                    
                RV_DUAL[vert]+= (xval * UU_EDGE[xidx])
         
            RV_DUAL[vert]*= (
                     ONE_ - MESH_VERT_SLIP[vert]
                        )     
            RV_DUAL[vert]/= MESH_DUAL_AREA[vert]
            
        for edge in prange(0, NEDG, schedule="static", 
                chunksize=cnfg_chunksize):
        #-- V^2 = vk * grad(div(u.n)) - 
        #--       vk * grad(rot(u.n))
            V2_EDGE[edge] = ZERO
            for iptr in range(GRAD_NORM_XPTR[edge +0], 
                              GRAD_NORM_XPTR[edge +1]):
                    
                xval = GRAD_NORM_XVAL[iptr]
                xidx = GRAD_NORM_XIDX[iptr]
                    
                V2_EDGE[edge]+= (xval * DU_CELL[xidx])
                
            for iptr in range(GRAD_PERP_XPTR[edge +0], 
                              GRAD_PERP_XPTR[edge +1]):
                    
                xval = GRAD_PERP_XVAL[iptr]
                xidx = GRAD_PERP_XIDX[iptr]
                    
                V2_EDGE[edge]-= (xval * RV_DUAL[xidx])
                
            V2_EDGE[edge]*= MESH_EDGE_MASK[edge]
            
            V4_EDGE[edge] = (
                   V2_EDGE[edge] * V4_VISC[edge]
                   )
                
        for cell in prange(0, NCEL, schedule="static", 
                chunksize=cnfg_chunksize):
        #-- compute div(V^2)
            DU_CELL[cell] = ZERO
            for iptr in range(CELL_FLUX_XPTR[cell +0], 
                              CELL_FLUX_XPTR[cell +1]):
                    
                xval = CELL_FLUX_XVAL[iptr]
                xidx = CELL_FLUX_XIDX[iptr]
                    
                DU_CELL[cell]+= (xval * V4_EDGE[xidx])
         
            DU_CELL[cell]/= MESH_CELL_AREA[cell]

        for vert in prange(0, NVRT, schedule="static", 
                chunksize=cnfg_chunksize):
        #-- compute rot(V^2)
            RV_DUAL[vert] = ZERO
            for iptr in range(DUAL_CURL_XPTR[vert +0], 
                              DUAL_CURL_XPTR[vert +1]):
                    
                xval = DUAL_CURL_XVAL[iptr]
                xidx = DUAL_CURL_XIDX[iptr]
                    
                RV_DUAL[vert]+= (xval * V4_EDGE[xidx])
         
            RV_DUAL[vert]/= MESH_DUAL_AREA[vert]
            
        for edge in prange(0, NEDG, schedule="static", 
                chunksize=cnfg_chunksize):
        #-- V^4 = vk * grad(div(V^2)) - 
        #--       vk * grad(rot(V^2))
            V4_EDGE[edge] = ZERO
            for iptr in range(GRAD_NORM_XPTR[edge +0], 
                              GRAD_NORM_XPTR[edge +1]):
                    
                xval = GRAD_NORM_XVAL[iptr]
                xidx = GRAD_NORM_XIDX[iptr]
                    
                V4_EDGE[edge]+= (xval * DU_CELL[xidx])
                
            for iptr in range(GRAD_PERP_XPTR[edge +0], 
                              GRAD_PERP_XPTR[edge +1]):
                    
                xval = GRAD_PERP_XVAL[iptr]
                xidx = GRAD_PERP_XIDX[iptr]
                    
                V4_EDGE[edge]-= (xval * RV_DUAL[xidx])
                
            UU_TEND[edge]-= (
                V2_VISC[edge] * V2_EDGE[edge]
              + NU_TURB[edge] * V2_EDGE[edge]
              - V4_VISC[edge] * V4_EDGE[edge]
                )
                     
    put_vec_e  (v4_edge)       
    put_vec_e  (v2_edge)
    put_vec_c  (du_cell)
    put_vec_v  (rv_dual)
                     
    return uu_tend
    
    
def _computeVH(mesh, trsk, cnfg, 
    np.ndarray[REALS_t, ndim=1] hh_cell,
    np.ndarray[FLT32_t, ndim=1] zb_cell,
        const REALS_t gg_cell,
    np.ndarray[REALS_t, ndim=1] hh_tend
              ):
    
#-- thickness dissipation: nu_k^h * del^k
    
    cdef INDEX_t edge, cell, iptr, xidx
    cdef INDEX_t cel1, cel2
    cdef REALS_t xval, last
    
    cdef REALS_t ZERO = 0.0
    cdef REALS_t ONE_ = 1.0
    cdef REALS_t HALF = 0.5
    
    cdef INDEX_t cnfg_numthread = cnfg.numthread
    cdef INDEX_t cnfg_chunksize = cnfg.chunksize
    
    cdef INDEX_t NVRT = mesh.vert.size
    cdef INDEX_t NEDG = mesh.edge.size
    cdef INDEX_t NCEL = mesh.cell.size
    
    cdef REALS_t *HH_CELL = &hh_cell[0]
    cdef FLT32_t *ZB_CELL = &zb_cell[0]
    cdef REALS_t *HH_TEND = &hh_tend[0]
    
    cdef REALS_t[::1] v2_diff = cnfg.hh_diff_2
    cdef REALS_t *V2_DIFF = &v2_diff[0]
    
    cdef REALS_t[::1] v4_diff = cnfg.hh_diff_4
    cdef REALS_t *V4_DIFF = &v4_diff[0]
    
    cdef INDEX_t[::1] cell_flux_xptr = \
        trsk.cell_flux_sums.indptr
    cdef INDEX_t[::1] cell_flux_xidx = \
        trsk.cell_flux_sums.indices
    cdef REALS_t[::1] cell_flux_xval = \
        trsk.cell_flux_sums.data
    
    cdef INDEX_t *CELL_FLUX_XPTR = &cell_flux_xptr[0]
    cdef INDEX_t *CELL_FLUX_XIDX = &cell_flux_xidx[0]
    cdef REALS_t *CELL_FLUX_XVAL = &cell_flux_xval[0]
    
    cdef INDEX_t[::1] grad_norm_xptr = \
        trsk.edge_grad_norm.indptr
    cdef INDEX_t[::1] grad_norm_xidx = \
        trsk.edge_grad_norm.indices
    cdef REALS_t[::1] grad_norm_xval = \
        trsk.edge_grad_norm.data
    
    cdef INDEX_t *GRAD_NORM_XPTR = &grad_norm_xptr[0]
    cdef INDEX_t *GRAD_NORM_XIDX = &grad_norm_xidx[0]
    cdef REALS_t *GRAD_NORM_XVAL = &grad_norm_xval[0]
       
    cdef REALS_t[::1] mesh_cell_area = mesh.cell.area
    cdef REALS_t[::1] mesh_edge_mask = mesh.edge.fmsk
    
    cdef REALS_t *MESH_CELL_AREA = &mesh_cell_area[0]
    cdef REALS_t *MESH_EDGE_MASK = &mesh_edge_mask[0]
    
    cdef np.ndarray[REALS_t] v2_cell = get_vec_c()
    cdef np.ndarray[REALS_t] v4_cell = get_vec_c()
        
    cdef REALS_t *V2_CELL = &v2_cell[0]
    cdef REALS_t *V4_CELL = &v4_cell[0]
    
    cdef np.ndarray[REALS_t] hz_edge = get_vec_e()
    cdef np.ndarray[REALS_t] ok_edge = get_vec_e()
    cdef np.ndarray[REALS_t] ok_cell = get_vec_c()
        
    cdef REALS_t *HZ_EDGE = &hz_edge[0]
    cdef REALS_t *OK_EDGE = &ok_edge[0]
    cdef REALS_t *OK_CELL = &ok_cell[0]
    
    cdef np.ndarray[INDEX_t, ndim=2] \
        mesh_edge_cell = mesh.edge.cell
    
    with nogil, parallel(num_threads=cnfg_numthread):
        
        for edge in prange(0, NEDG, schedule="static", 
                chunksize=cnfg_chunksize):
                
            cel1 = mesh_edge_cell[edge, 0] - 1
            cel2 = mesh_edge_cell[edge, 1] - 1
            
            if (cel1 < 0): cel1 = cel2
            if (cel2 < 0): cel2 = cel1
                
        #-- pressure gradient tend.: g * G * (hh + zb)
            HZ_EDGE[edge] = ZERO
            for iptr in range(GRAD_NORM_XPTR[edge +0], 
                              GRAD_NORM_XPTR[edge +1]):
                    
                xval = GRAD_NORM_XVAL[iptr]
                xidx = GRAD_NORM_XIDX[iptr]
                    
                HZ_EDGE[edge]+= \
                    xval * gg_cell * (
                        HH_CELL[xidx] + ZB_CELL[xidx])
         
        #-- flux-limiter: don't diffuse to topography!
            OK_EDGE[edge] = MESH_EDGE_MASK[edge]
            OK_EDGE[edge]*=(
                min(ZB_CELL[cel1]+ HH_CELL[cel1],
                    ZB_CELL[cel2]+ HH_CELL[cel2])
              > max(ZB_CELL[cel1], ZB_CELL[cel2])
                   )
                        
            HZ_EDGE[edge]*= OK_EDGE[edge]
            
        for cell in prange(0, NCEL, schedule="static", 
                chunksize=cnfg_chunksize):
        #-- compute div(H.n)
            V2_CELL[cell] = ZERO
            OK_CELL[cell] = ZERO
            for iptr in range(CELL_FLUX_XPTR[cell +0], 
                              CELL_FLUX_XPTR[cell +1]):
                    
                xval = CELL_FLUX_XVAL[iptr]
                xidx = CELL_FLUX_XIDX[iptr]
                    
                V2_CELL[cell]+= (xval * HZ_EDGE[xidx])
                
                # -ve if any edge limits
                OK_CELL[cell]-= (ONE_ - OK_EDGE[xidx])
         
            V2_CELL[cell]/= MESH_CELL_AREA[cell]
            
        for edge in prange(0, NEDG, schedule="static", 
                chunksize=cnfg_chunksize):
        #-- pressure gradient tend.: g * G * (hh + zb)
            last = HZ_EDGE[edge]
            HZ_EDGE[edge] = ZERO
            for iptr in range(GRAD_NORM_XPTR[edge +0], 
                              GRAD_NORM_XPTR[edge +1]):
                    
                xval = GRAD_NORM_XVAL[iptr]
                xidx = GRAD_NORM_XIDX[iptr]
                    
                HZ_EDGE[edge]+= \
                    xval * gg_cell * (
                        V2_CELL[xidx] * V4_DIFF[xidx])
                
                # if incident edges okay        
                OK_EDGE[edge]*=(OK_CELL[xidx] >= ZERO)
                        
            HZ_EDGE[edge]*= OK_EDGE[edge]
            
        #-- flux limiter: don't allow up-gradient flux
        #-- see M. Xue (2000): MWR.
            HZ_EDGE[edge]*= (
                HZ_EDGE[edge]* last < ZERO)
            
        for cell in prange(0, NCEL, schedule="static", 
                chunksize=cnfg_chunksize):
        #-- compute div(H^2)
            V4_CELL[cell] = ZERO
            for iptr in range(CELL_FLUX_XPTR[cell +0], 
                              CELL_FLUX_XPTR[cell +1]):
                    
                xval = CELL_FLUX_XVAL[iptr]
                xidx = CELL_FLUX_XIDX[iptr]
                    
                V4_CELL[cell]+= (xval * HZ_EDGE[xidx])
         
            V4_CELL[cell]/= MESH_CELL_AREA[cell]
            
            HH_TEND[cell]-= (
                V2_DIFF[cell] * V2_CELL[cell]
              - V4_DIFF[cell] * V4_CELL[cell]
                )
    
    put_vec_c  (ok_cell)
    put_vec_e  (ok_edge)
    put_vec_c  (v4_cell)       
    put_vec_c  (v2_cell)
    put_vec_e  (hz_edge)
    
    return hh_tend
    
    
def _computeTU(mesh, trsk, cnfg, 
    np.ndarray[FLT32_t, ndim=1] Tu_edge,
    np.ndarray[REALS_t, ndim=1] hh_edge,
    np.ndarray[REALS_t, ndim=1] uu_tend
              ):
    
#-- forcing due to external stresses: tau / h
    
    cdef INDEX_t edge, cell, iptr, xidx
    cdef REALS_t xval
    
    cdef INDEX_t cnfg_numthread = cnfg.numthread
    cdef INDEX_t cnfg_chunksize = cnfg.chunksize
    
    cdef INDEX_t NVRT = mesh.vert.size
    cdef INDEX_t NEDG = mesh.edge.size
    cdef INDEX_t NCEL = mesh.cell.size
    
    cdef FLT32_t *TU_EDGE = &Tu_edge[0]
    cdef REALS_t *HH_EDGE = &hh_edge[0]
    cdef REALS_t *UU_TEND = &uu_tend[0]
    
    with nogil, parallel(num_threads=cnfg_numthread):
    
        for edge in prange(0, NEDG, schedule="static", 
                chunksize=cnfg_chunksize):
                
            UU_TEND[edge]-= (
                    TU_EDGE[edge] / HH_EDGE[edge]
                    )
        
    return uu_tend
    
    
def _computeCd(mesh, trsk, cnfg, 
        const REALS_t hh_tiny,
    np.ndarray[REALS_t, ndim=1] hh_cell,
    np.ndarray[REALS_t, ndim=1] uu_edge,
    np.ndarray[REALS_t, ndim=1] vv_edge
              ):
    
#-- cd = cd_lin + (cd_sqr + cd_log) * |u| / h
    
    cdef REALS_t VONK = 0.4  # von karman
    
    cdef REALS_t ZERO = 0.0
    cdef REALS_t HALF = 0.5
    cdef REALS_t TWO_ = 2.0
    cdef REALS_t ONE_ = 1.0
    
    cdef INDEX_t edge, iptr, xidx
    cdef INDEX_t cel1, cel2
    cdef REALS_t xval
    
    cdef INDEX_t cnfg_numthread = cnfg.numthread
    cdef INDEX_t cnfg_chunksize = cnfg.chunksize
    
    cdef REALS_t hh_edge, ke_edge
    
    cdef REALS_t loglaw_z0 = cnfg.loglaw_z0
    cdef REALS_t loglaw_hi = cnfg.loglaw_hi
    cdef REALS_t loglaw_lo = cnfg.loglaw_lo
    
    cdef REALS_t sqrlaw_cd = cnfg.sqrlaw_cd
    cdef REALS_t linlaw_cd = cnfg.linlaw_cd
    
    cdef INDEX_t NVRT = mesh.vert.size
    cdef INDEX_t NEDG = mesh.edge.size
    cdef INDEX_t NCEL = mesh.cell.size
    
    cdef REALS_t *HH_CELL = &hh_cell[0]
    cdef REALS_t *UU_EDGE = &uu_edge[0]
    cdef REALS_t *VV_EDGE = &vv_edge[0]
    
    cdef np.ndarray[REALS_t] cd_edge = variables.cd_edge
    
    cdef REALS_t *CD_EDGE = &cd_edge[0]
    
    cdef INDEX_t[:, ::1] mesh_edge_cell = mesh.edge.cell
    
    with nogil, parallel(num_threads=cnfg_numthread):
    
        for edge in prange(0, NEDG, schedule="static", 
                chunksize=cnfg_chunksize):

            cel1 = mesh_edge_cell[edge, 0] - 1
            cel2 = mesh_edge_cell[edge, 1] - 1
            
            if (cel1 < 0): cel1 = cel2
            if (cel2 < 0): cel2 = cel1

            hh_edge = TWO_ * HH_CELL[cel1] * \
                             HH_CELL[cel2] / \
                    (HH_CELL[cel1] + HH_CELL[cel2])
                   
            ke_edge = HALF * (
                UU_EDGE[edge] * UU_EDGE[edge] 
              + VV_EDGE[edge] * VV_EDGE[edge]
                )
                   
            hh_edge = max(hh_tiny, hh_edge)
            
            CD_EDGE[edge] = ZERO
       
            if (loglaw_z0 > ZERO):            
                # NB. log(1+z/z0) "fix" to loglaw
                CD_EDGE[edge] = (VONK / log_r (
                    ONE_ + HALF * hh_edge / loglaw_z0)
                    )
                    
                CD_EDGE[edge]*= CD_EDGE[edge]
                    
                CD_EDGE[edge] = \
                    min(CD_EDGE[edge], loglaw_hi)
                    
                CD_EDGE[edge] = \
                    max(CD_EDGE[edge], loglaw_lo)
            
            CD_EDGE[edge]+= sqrlaw_cd
            
            CD_EDGE[edge]*= sqrt_r(TWO_ * ke_edge) / hh_edge
            
            CD_EDGE[edge]+= linlaw_cd
    
    return cd_edge

