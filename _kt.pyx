
#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=True
#cython: cpow=True

""" Utilities for thread-parallel RK-FB time integration
"""
#-- Part of the PERISCOPE solver
#-- Darren Engwirda
#-- d.engwirda@gmail.com
#-- https://github.com/dengwirda/

import numpy as np
cimport numpy as np
cimport cython

from cython.parallel import prange, parallel
from libc.stdint cimport int32_t

from _fp import flt32_t, flt64_t
from _fp import reals_t, index_t

ctypedef float   FLT32_t
ctypedef double  FLT64_t
ctypedef int32_t INDEX_t
ctypedef float   REALS_t  # or double

def _bnd_x_vec(cnfg,
    np.ndarray[REALS_t, ndim=1] xx_data,
    np.ndarray[REALS_t, ndim=1] xx_min_,
    np.ndarray[REALS_t, ndim=1] xx_max_
              ):

#-- update minmax bnds given data in xx

    cdef INDEX_t ipos
    cdef INDEX_t NDAT = xx_data.size

    cdef INDEX_t numthread = cnfg.numthread
    cdef INDEX_t chunksize = cnfg.chunksize

    cdef REALS_t *XX_DATA = &xx_data[+0]
    cdef REALS_t *XX_MIN_ = &xx_min_[+0]
    cdef REALS_t *XX_MAX_ = &xx_max_[+0]

    with nogil, parallel(num_threads=numthread):

        for ipos in prange(0, NDAT, schedule="static",
                chunksize=chunksize):
                    
            XX_MIN_[ipos] = min(
                XX_MIN_[ipos], XX_DATA[ipos])
                
            XX_MAX_[ipos] = max(
                XX_MAX_[ipos], XX_DATA[ipos])

    return xx_min_, xx_max_


def _set_x_vec(cnfg,
        const REALS_t xx_fill,
    np.ndarray[REALS_t, ndim=1] xx_data,
              ):

#-- xx = fv, straight-up vector fill - that's it!

    cdef INDEX_t ipos
    cdef INDEX_t NDAT = xx_data.size

    cdef INDEX_t numthread = cnfg.numthread
    cdef INDEX_t chunksize = cnfg.chunksize

    cdef REALS_t *XX_DATA = &xx_data[+0]

    with nogil, parallel(num_threads=numthread):

        for ipos in prange(0, NDAT, schedule="static",
                chunksize=chunksize):

            XX_DATA[ipos] = xx_fill

    return xx_data


def _cpy_x_vec(cnfg, 
    np.ndarray[REALS_t, ndim=1] xx_data,
    np.ndarray[REALS_t, ndim=1] yy_data
              ):

#-- yy = xx, straight-up vector copy - that's it!
    
    cdef INDEX_t ipos
    cdef INDEX_t NDAT = xx_data.size
    
    cdef INDEX_t numthread = cnfg.numthread
    cdef INDEX_t chunksize = cnfg.chunksize
    
    cdef REALS_t *XX_DATA = &xx_data[+0]
    cdef REALS_t *YY_DATA = &yy_data[+0]
    
    with nogil, parallel(num_threads=numthread):
    
        for ipos in prange(0, NDAT, schedule="static", 
                chunksize=chunksize):       
        
            YY_DATA[ipos] = XX_DATA[ipos]
            
    return yy_data
    

def _adv_x_cmp(cnfg, 
    np.ndarray[REALS_t, ndim=1] xx_data,
    const REALS_t rh_coef, 
    np.ndarray[FLT64_t, ndim=1] rh_data, 
    np.ndarray[REALS_t, ndim=1] cx_data,
    np.ndarray[REALS_t, ndim=1] yy_data
              ):

#-- yy = xx + beta * rh, with fp error compensation
    
    cdef INDEX_t ipos
    cdef INDEX_t NDAT = xx_data.size
    
    cdef REALS_t RH_TEMP
    
    cdef FLT64_t ZERO = 0.0

    cdef INDEX_t numthread = cnfg.numthread
    cdef INDEX_t chunksize = cnfg.chunksize
    
    cdef REALS_t *XX_DATA = &xx_data[+0]
    cdef FLT64_t *RH_DATA = &rh_data[+0]
    cdef REALS_t *CX_DATA = &cx_data[+0]
    cdef REALS_t *YY_DATA = &yy_data[+0]
    
    with nogil, parallel(num_threads=numthread):
    
        for ipos in prange(0, NDAT, schedule="static",
                chunksize=chunksize):        
        #-- compensated state update; new compensator
        
            RH_TEMP       =-RH_DATA[ipos] * rh_coef - \
                            CX_DATA[ipos]
        
            RH_DATA[ipos] = ZERO

            YY_DATA[ipos] = XX_DATA[ipos] + RH_TEMP
            
            CX_DATA[ipos] =(YY_DATA[ipos] - 
                            XX_DATA[ipos])- RH_TEMP
            
    return yy_data, cx_data
    
    
def _adv_x_fst(cnfg, 
    np.ndarray[REALS_t, ndim=1] xx_data,
    const REALS_t rh_coef, 
    np.ndarray[FLT64_t, ndim=1] rh_data, 
    np.ndarray[REALS_t, ndim=1] cx_data,
    np.ndarray[REALS_t, ndim=1] yy_data
              ):
    
#-- yy = xx + beta * rh, with fp error compensation
    
    cdef INDEX_t ipos
    cdef INDEX_t NDAT = xx_data.size
    
    cdef REALS_t RH_TEMP
    
    cdef FLT64_t ZERO = 0.0

    cdef INDEX_t numthread = cnfg.numthread
    cdef INDEX_t chunksize = cnfg.chunksize
    
    cdef REALS_t *XX_DATA = &xx_data[+0]
    cdef FLT64_t *RH_DATA = &rh_data[+0]
    cdef REALS_t *CX_DATA = &cx_data[+0]
    cdef REALS_t *YY_DATA = &yy_data[+0]
    
    with nogil, parallel(num_threads=numthread):
    
        for ipos in prange(0, NDAT, schedule="static", 
                chunksize=chunksize):
        #-- compensated state update; old compensator
        
            RH_TEMP       =-RH_DATA[ipos] * rh_coef - \
                            CX_DATA[ipos]
        
            RH_DATA[ipos] = ZERO

            YY_DATA[ipos] = XX_DATA[ipos] + RH_TEMP
              
    return yy_data
    

def _inv_x_1st(cnfg, 
    np.ndarray[REALS_t, ndim=1] xx_data,
    const REALS_t rh_coef, 
    np.ndarray[REALS_t, ndim=1] cd_data,
    np.ndarray[REALS_t, ndim=1] rh_data
              ):
              
    cdef INDEX_t ipos
    cdef INDEX_t NDAT = xx_data.size
    
    cdef REALS_t ONE_ = 1.0
    
    cdef INDEX_t numthread = cnfg.numthread
    cdef INDEX_t chunksize = cnfg.chunksize
    
    cdef REALS_t *XX_DATA = &xx_data[+0]
    cdef REALS_t *CD_DATA = &cd_data[+0]
    cdef REALS_t *RH_DATA = &rh_data[+0]
    
    with nogil, parallel(num_threads=numthread):
    
        for ipos in prange(0, NDAT, schedule="static", 
                chunksize=chunksize):
        #-- 1st-order inversion for implicit drag
                     
            XX_DATA[ipos]/= CD_DATA[ipos] * rh_coef + \
                            ONE_
                        
    return xx_data

    
def _inv_x_2nd(cnfg, 
    np.ndarray[REALS_t, ndim=1] xx_data,
    const REALS_t rh_coef, 
    np.ndarray[REALS_t, ndim=1] cd_data,
    np.ndarray[REALS_t, ndim=1] rh_data
              ):
              
    cdef INDEX_t ipos
    cdef INDEX_t NDAT = xx_data.size
    
    cdef REALS_t ONE_ = 1.0
    
    cdef INDEX_t numthread = cnfg.numthread
    cdef INDEX_t chunksize = cnfg.chunksize
    
    cdef REALS_t *XX_DATA = &xx_data[+0]
    cdef REALS_t *CD_DATA = &cd_data[+0]
    cdef REALS_t *RH_DATA = &rh_data[+0]
    
    with nogil, parallel(num_threads=numthread):
    
        for ipos in prange(0, NDAT, schedule="static", 
                chunksize=chunksize):
        #-- 2nd-order inversion for implicit drag
                    
            XX_DATA[ipos]-= CD_DATA[ipos] * rh_coef * \
                            RH_DATA[ipos]
                            
            XX_DATA[ipos]/= CD_DATA[ipos] * rh_coef + \
                            ONE_
                        
    return xx_data
    
    
def _sum_2_way(cnfg, 
    np.ndarray[REALS_t, ndim=1] yy_data,
    const REALS_t x1_coef, 
    np.ndarray[REALS_t, ndim=1] x1_data,
    const REALS_t x2_coef, 
    np.ndarray[REALS_t, ndim=1] x2_data
              ):
    
#-- yy = sum( bi * xi ), 2-array version
    
    cdef INDEX_t ipos
    
    cdef INDEX_t NDAT = x1_data.size
    
    cdef INDEX_t numthread = cnfg.numthread
    cdef INDEX_t chunksize = cnfg.chunksize
    
    cdef REALS_t *X1_DATA = &x1_data[+0]
    cdef REALS_t *X2_DATA = &x2_data[+0]
    cdef REALS_t *YY_DATA = &yy_data[+0]
    
    with nogil, parallel(num_threads=numthread):
    
        for ipos in prange(0, NDAT, schedule="static", 
                chunksize=chunksize):        
        #-- 2-way forward-backward averaging
        
            YY_DATA[ipos] = X1_DATA[ipos] * x1_coef + \
                            X2_DATA[ipos] * x2_coef
              
    return yy_data
    

def _sum_3_way(cnfg, 
    np.ndarray[REALS_t, ndim=1] yy_data,
    const REALS_t x1_coef, 
    np.ndarray[REALS_t, ndim=1] x1_data,
    const REALS_t x2_coef, 
    np.ndarray[REALS_t, ndim=1] x2_data,
    const REALS_t x3_coef, 
    np.ndarray[REALS_t, ndim=1] x3_data
              ):
    
#-- yy = sum( bi * xi ), 3-array version
    
    cdef INDEX_t ipos
    
    cdef INDEX_t NDAT = x1_data.size
    
    cdef INDEX_t numthread = cnfg.numthread
    cdef INDEX_t chunksize = cnfg.chunksize
    
    cdef REALS_t *X1_DATA = &x1_data[+0]
    cdef REALS_t *X2_DATA = &x2_data[+0]
    cdef REALS_t *X3_DATA = &x3_data[+0]
    cdef REALS_t *YY_DATA = &yy_data[+0]
    
    with nogil, parallel(num_threads=numthread):
    
        for ipos in prange(0, NDAT, schedule="static", 
                chunksize=chunksize):        
        #-- 3-way forward-backward averaging
            
            YY_DATA[ipos] = X1_DATA[ipos] * x1_coef + \
                            X2_DATA[ipos] * x2_coef + \
                            X3_DATA[ipos] * x3_coef
                                     
    return yy_data

    
def _sym_3_way(cnfg, 
    np.ndarray[REALS_t, ndim=1] yy_data,
    const REALS_t x1_coef, 
    np.ndarray[REALS_t, ndim=1] x1_data,
    const REALS_t x2_coef, 
    np.ndarray[REALS_t, ndim=1] x2_data,
    const REALS_t x3_coef, 
    np.ndarray[REALS_t, ndim=1] x3_data
              ):
    
#-- yy = b1 * (x1 + x3) + b2 * x2, symmetric sums
    
    cdef INDEX_t ipos
    
    cdef INDEX_t NDAT = x1_data.size
    
    cdef INDEX_t numthread = cnfg.numthread
    cdef INDEX_t chunksize = cnfg.chunksize
    
    cdef REALS_t *X1_DATA = &x1_data[+0]
    cdef REALS_t *X2_DATA = &x2_data[+0]
    cdef REALS_t *X3_DATA = &x3_data[+0]
    cdef REALS_t *YY_DATA = &yy_data[+0]
    
    with nogil, parallel(num_threads=numthread):
    
        for ipos in prange(0, NDAT, schedule="static",
                chunksize=chunksize):        
        #-- 3-way forward-backward averaging
            
            YY_DATA[ipos] = X2_DATA[ipos] * x2_coef + \
                           (X1_DATA[ipos] + 
                            X3_DATA[ipos])* x1_coef
                                     
    return yy_data

