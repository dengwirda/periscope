
#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: initializedcheck=False
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
from cython.parallel import threadid

from _fp import flt32_t, flt64_t
from _kp cimport FLT32_t, FLT64_t, LOCAL_t
from _kp cimport FLTXX_t, FLTXT_t

from _fp import reals_t, index_t, bytes_t
from _kp cimport REALS_t, INDEX_t, BYTES_t

from _fp import udata_t, hdata_t, qdata_t
from _kp cimport UDATA_t, HDATA_t, QDATA_t

from _fp import utend_t, htend_t, qtend_t
from _kp cimport UTEND_t, HTEND_t, QTEND_t

from lib cimport sqrtf as sqrt_r
from lib cimport cbrtf as cbrt_r
from lib cimport fabsf as fabs_r
from lib cimport isnan as isnan_r

def _cfl_adapt(mesh, mats, cnfg,
        const REALS_t gravity,
        const REALS_t hh_thin,
        const REALS_t rk_scal,
        const REALS_t rk_adv_,
        const REALS_t cfl_num,
        const REALS_t dt_prev,
        const REALS_t dt_jump,
    np.ndarray[HDATA_t, ndim=1] hh_prev,
    np.ndarray[HDATA_t, ndim=1] hh_cell,
    np.ndarray[UDATA_t, ndim=1] uu_prev,
    np.ndarray[UDATA_t, ndim=1] uu_edge
              ):

#-- project new adaptive time-step based on CFL estimate

    cdef INDEX_t edge, cel1, cel2

    cdef REALS_t SPACING
    cdef REALS_t UU_PROJ, UU_LAST, UU_NORM
    cdef LOCAL_t GH_WAVE, CC_WAVE, CC_LAST
    cdef REALS_t H1_PROJ, H2_PROJ, H1_LAST, H2_LAST
    
    cdef REALS_t ZERO = 0.0
    cdef REALS_t HALF = 0.5
    cdef REALS_t ONE_ = 1.0
    cdef REALS_t TWO_ = 2.0

    cdef REALS_t PROJ = 5.0 / 1.0
    cdef REALS_t REDO =-7.0 / 8.0

    cdef REALS_t FAIL = 1.0 + 1.E-04

    cdef INDEX_t numthread = cnfg.numthread
    cdef INDEX_t chunksize = cnfg.chunksize
    cdef INDEX_t chunkcell = cnfg.chunkcell
    cdef INDEX_t chunkedge = cnfg.chunkedge
    cdef INDEX_t chunkvert = cnfg.chunkvert

    cdef INDEX_t NVRT = mesh.vert.size
    cdef INDEX_t NEDG = mesh.edge.size
    cdef INDEX_t NCEL = mesh.cell.size
    
    cdef FLT64_t dt_inf_ = np.inf
    cdef FLT64_t dt_max_ = \
        ONE_ / max(cnfg.ff_max_, 1.E-16)  # coriolis
    cdef FLT64_t dt_fail, dt_proj
    cdef FLT64_t*at_fail
    cdef FLT64_t*at_proj

    cdef FLT64_t[::1] dt_fail_all = \
        np.full(numthread, dt_inf_, dtype=flt64_t)
    cdef FLT64_t[::1] dt_proj_all = \
        np.full(numthread, dt_inf_, dtype=flt64_t)

    cdef HDATA_t *HH_PREV = &hh_prev[0]
    cdef HDATA_t *HH_CELL = &hh_cell[0]
    cdef UDATA_t *UU_PREV = &uu_prev[0]
    cdef UDATA_t *UU_EDGE = &uu_edge[0]
 
    cdef FLT64_t dt_scal = cfl_num * rk_scal
    cdef FLT64_t dt_okay =    FAIL * \
                 cfl_num * rk_scal / dt_prev

    # (omega/k)^2 = c^2 = (f/k)^2 + g*h
    # let k = 2*pi/2*rros = pi/(g*h)^1/2/f
    # c^2 = g*h/pi^2 + g*h = (1 + 1/pi^2)*g*h
    cdef REALS_t fr_wav_    
    cdef REALS_t ff_wav_ = \
        gravity * ( ONE_ + (ONE_ / np.pi) ** 2 )
    
    cdef REALS_t[::1] mesh_edge_spac = mesh.edge.spac
    
    cdef REALS_t *MESH_EDGE_SPAC = &mesh_edge_spac[0]

    cdef INDEX_t[:, ::1] mesh_edge_cell = mesh.edge.cell

    with nogil, parallel(num_threads=numthread):

        # thread-local reduction tricks:
        # https://github.com/cython/cython/issues/3585
        dt_fail = dt_inf_; at_fail =&dt_fail
        dt_proj = dt_max_; at_proj =&dt_proj

        for edge in prange(0, NEDG, schedule="static", 
                chunksize=chunkedge):

            cel1 = mesh_edge_cell[edge, 0] - 1
            cel2 = mesh_edge_cell[edge, 1] - 1
            
            if (cel1 < 0): cel1 = cel2
            if (cel2 < 0): cel2 = cel1

            H1_LAST = HH_CELL[cel1]
            H1_PROJ = HH_CELL[cel1] + \
                PROJ * (HH_CELL[cel1] - HH_PREV[cel1])

            H2_LAST = HH_CELL[cel2]
            H2_PROJ = HH_CELL[cel2] + \
                PROJ * (HH_CELL[cel2] - HH_PREV[cel2])

            H1_LAST = max(H1_LAST, HH_PREV[cel1])
            H1_LAST = max(H1_LAST, hh_thin)

            H1_PROJ = max(H1_PROJ, H1_LAST)
           #H1_PROJ = max(H1_PROJ, hh_thin)  # done above

            H2_LAST = max(H2_LAST, HH_PREV[cel2])
            H2_LAST = max(H2_LAST, hh_thin)

            H2_PROJ = max(H2_PROJ, H2_LAST)
           #H2_PROJ = max(H2_PROJ, hh_thin)  # done above

            UU_LAST = UU_EDGE[edge]
            UU_PROJ = UU_EDGE[edge] + \
                PROJ * (UU_EDGE[edge] - UU_PREV[edge])

            UU_LAST = max(+fabs_r (UU_LAST), 
                          +fabs_r (UU_PREV[edge]))
            UU_PROJ = max(+fabs_r (UU_LAST), 
                          +fabs_r (UU_PROJ))

            # approx poincare wave-speed
            GH_WAVE = sqrt_r(
                ff_wav_ * max(H1_LAST , H2_LAST))

            # 1.0 + sqrt(froude) measure
            fr_wav_ = sqrt_r(
                ONE_+ sqrt_r( UU_LAST / GH_WAVE))

            # |u| + amplify * wave-speed
            CC_LAST = UU_LAST * rk_adv_ + fr_wav_ * GH_WAVE
    
            # as above, for forward projected state
            GH_WAVE = sqrt_r(
                ff_wav_ * max(H1_PROJ , H2_PROJ))

            fr_wav_ = sqrt_r(
                ONE_+ sqrt_r( UU_PROJ / GH_WAVE))

            CC_WAVE = UU_PROJ * rk_adv_ + fr_wav_ * GH_WAVE
            
            SPACING = MESH_EDGE_SPAC[edge]

            if (CC_LAST > dt_okay * SPACING):
                at_fail[0] = min(
                    at_fail[0], SPACING / CC_LAST)
            else:
                at_proj[0] = min(
                    at_proj[0], SPACING / CC_WAVE)

            if (isnan_r(HH_CELL[cel1]) or 
                isnan_r(HH_CELL[cel2]) ):
                at_fail[0] = min( at_fail[0], dt_prev* HALF)

        dt_fail_all[threadid()] = at_fail[0]
        dt_proj_all[threadid()] = at_proj[0]
            
    dt_redo = dt_scal * np.min(dt_fail_all)
    dt_next = dt_scal * np.min(dt_proj_all)

    if (dt_redo == dt_inf_):
        if (dt_prev > ZERO and dt_next > ZERO):
            if (dt_next > dt_prev * dt_jump):
                dt_next = dt_prev * dt_jump
    else: 
        dt_next = dt_redo * REDO

    return dt_next


def _bnd_x_vec(cnfg,
    np.ndarray[FLTXX_t, ndim=1] xx_data,
    np.ndarray[REALS_t, ndim=1] xx_min_,
    np.ndarray[REALS_t, ndim=1] xx_max_
              ):

#-- update minmax bnds given data in xx

    cdef INDEX_t ipos
    cdef INDEX_t NDAT = xx_data.size

    cdef INDEX_t numthread = cnfg.numthread
    cdef INDEX_t numchunks = cnfg.numchunks

    cdef INDEX_t chunksize = max(1, 
       ((NDAT // numthread // numchunks) // 8) * 8
        )

    cdef FLTXX_t *XX_DATA = &xx_data[+0]
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


def _nrm_x_vec(cnfg,
    np.ndarray[FLTXX_t, ndim=1] xx_data,
    np.ndarray[REALS_t, ndim=1] xx_ave_,
    np.ndarray[REALS_t, ndim=1] xx_rms_,
    np.ndarray[REALS_t, ndim=1] xx_max_
              ):

#-- update minmax norm given data in xx

    cdef INDEX_t ipos
    cdef INDEX_t NDAT = xx_data.size

    cdef INDEX_t numthread = cnfg.numthread
    cdef INDEX_t numchunks = cnfg.numchunks

    cdef INDEX_t chunksize = max(1,
       ((NDAT // numthread // numchunks) // 8) * 8
        )

    cdef FLTXX_t *XX_DATA = &xx_data[+0]
    cdef REALS_t *XX_AVE_ = &xx_ave_[+0]
    cdef REALS_t *XX_RMS_ = &xx_rms_[+0]
    cdef REALS_t *XX_MAX_ = &xx_max_[+0]

    with nogil, parallel(num_threads=numthread):

        for ipos in prange(0, NDAT, schedule="static",
                chunksize=chunksize):
            
            XX_AVE_[ipos]+= XX_DATA[ipos]
        
            XX_RMS_[ipos]+= XX_DATA[ipos] * \
                            XX_DATA[ipos]
                
            XX_MAX_[ipos] = max(
                XX_MAX_[ipos], XX_DATA[ipos])

    return xx_ave_, xx_rms_, xx_max_


def _nrm_z_vec(cnfg,
    np.ndarray[FLT32_t, ndim=1] zb_data,
    np.ndarray[FLTXX_t, ndim=1] hh_data,
    np.ndarray[REALS_t, ndim=1] zz_rms_
              ):

#-- update minmax norm given data in zz

    cdef INDEX_t ipos
    cdef INDEX_t NDAT = hh_data.size

    cdef FLT64_t ZZ_VAL_

    cdef INDEX_t numthread = cnfg.numthread
    cdef INDEX_t numchunks = cnfg.numchunks

    cdef INDEX_t chunksize = max(1,
       ((NDAT // numthread // numchunks) // 8) * 8
        )

    cdef FLT32_t *ZB_DATA = &zb_data[+0]
    cdef FLTXX_t *HH_DATA = &hh_data[+0]
    cdef REALS_t *ZZ_RMS_ = &zz_rms_[+0]

    with nogil, parallel(num_threads=numthread):

        for ipos in prange(0, NDAT, schedule="static",
                chunksize=chunksize):
                    
            ZZ_VAL_ = \
                ZB_DATA[ipos] + HH_DATA[ipos]

            ZZ_RMS_[ipos]+= ZZ_VAL_ * ZZ_VAL_

    return zz_rms_


def _set_x_vec(cnfg,
    np.ndarray[FLTXX_t, ndim=1] xx_data,
        const FLTXX_t xx_fill
              ):

#-- xx = fv, straight-up vector fill - that's it!

    cdef INDEX_t ipos
    cdef INDEX_t NDAT = xx_data.size

    cdef INDEX_t numthread = cnfg.numthread
    cdef INDEX_t numchunks = cnfg.numchunks

    cdef INDEX_t chunksize = max(1,
       ((NDAT // numthread // numchunks) // 8) * 8
        )

    cdef FLTXX_t *XX_DATA = &xx_data[+0]

    with nogil, parallel(num_threads=numthread):

        for ipos in prange(0, NDAT, schedule="static",
                chunksize=chunksize):

            XX_DATA[ipos] = xx_fill

    return xx_data


def _cpy_x_vec(cnfg, 
    np.ndarray[FLTXX_t, ndim=1] xx_data,
    np.ndarray[FLTXX_t, ndim=1] yy_data
              ):

#-- yy = xx, straight-up vector copy - that's it!
    
    cdef INDEX_t ipos
    cdef INDEX_t NDAT = xx_data.size
    
    cdef INDEX_t numthread = cnfg.numthread
    cdef INDEX_t numchunks = cnfg.numchunks

    cdef INDEX_t chunksize = max(1,
       ((NDAT // numthread // numchunks) // 8) * 8
        )
    
    cdef FLTXX_t *XX_DATA = &xx_data[+0]
    cdef FLTXX_t *YY_DATA = &yy_data[+0]
    
    with nogil, parallel(num_threads=numthread):
    
        for ipos in prange(0, NDAT, schedule="static", 
                chunksize=chunksize):       
        
            YY_DATA[ipos] = XX_DATA[ipos]
            
    return yy_data
    

def _inc_x_rhs(cnfg, 
    np.ndarray[FLTXX_t, ndim=1] xx_data,
        const REALS_t rh_coef, 
    np.ndarray[FLTXT_t, ndim=1] rh_data,
    np.ndarray[FLTXX_t, ndim=1] yy_data
              ):
    
    cdef INDEX_t ipos
    cdef INDEX_t NDAT = xx_data.size
    
    cdef REALS_t RH_TEMP
    
    cdef REALS_t ZERO = 0.0

    cdef INDEX_t numthread = cnfg.numthread
    cdef INDEX_t numchunks = cnfg.numchunks

    cdef INDEX_t chunksize = max(1,
       ((NDAT // numthread // numchunks) // 8) * 8
        )
    
    cdef FLTXX_t *XX_DATA = &xx_data[+0]
    cdef FLTXT_t *RH_DATA = &rh_data[+0]
    cdef FLTXX_t *YY_DATA = &yy_data[+0]
    
    with nogil, parallel(num_threads=numthread):
    
        for ipos in prange(0, NDAT, schedule="static",
                chunksize=chunksize):
        #-- yy = xx + beta * rh
        
            RH_TEMP       =-RH_DATA[ipos] * rh_coef
        
            RH_DATA[ipos] = ZERO

            YY_DATA[ipos] = XX_DATA[ipos] + RH_TEMP
            
    return yy_data
    

def _inv_x_1st(cnfg, 
    np.ndarray[FLTXX_t, ndim=1] xx_data,
        const REALS_t rh_coef, 
    np.ndarray[REALS_t, ndim=1] cd_data,
    np.ndarray[FLTXX_t, ndim=1] rh_data
              ):

    cdef INDEX_t ipos
    cdef INDEX_t NDAT = xx_data.size
    
    cdef REALS_t ONE_ = 1.0
    
    cdef INDEX_t numthread = cnfg.numthread
    cdef INDEX_t numchunks = cnfg.numchunks

    cdef INDEX_t chunksize = max(1,
       ((NDAT // numthread // numchunks) // 8) * 8
        )
    
    cdef FLTXX_t *XX_DATA = &xx_data[+0]
    cdef REALS_t *CD_DATA = &cd_data[+0]
    cdef FLTXX_t *RH_DATA = &rh_data[+0]
    
    with nogil, parallel(num_threads=numthread):
    
        for ipos in prange(0, NDAT, schedule="static", 
                chunksize=chunksize):
        #-- 1-stage inversion for implicit drag
                     
            XX_DATA[ipos]/= CD_DATA[ipos] * rh_coef + \
                            ONE_
                        
    return xx_data

    
def _inv_x_2nd(cnfg, 
    np.ndarray[FLTXX_t, ndim=1] xx_data,
        const REALS_t rh_coef, 
    np.ndarray[REALS_t, ndim=1] cd_data,
    np.ndarray[FLTXX_t, ndim=1] rh_data
              ):
              
    cdef INDEX_t ipos
    cdef INDEX_t NDAT = xx_data.size
    
    cdef REALS_t ONE_ = 1.0
    
    cdef INDEX_t numthread = cnfg.numthread
    cdef INDEX_t numchunks = cnfg.numchunks

    cdef INDEX_t chunksize = max(1,
       ((NDAT // numthread // numchunks) // 8) * 8
        )
    
    # oscillatory if not off-centred
    cdef REALS_t  ex_coef = (1. - 1./4.) * rh_coef
    cdef REALS_t  im_coef = (1. + 1./4.) * rh_coef

    cdef FLTXX_t *XX_DATA = &xx_data[+0]
    cdef REALS_t *CD_DATA = &cd_data[+0]
    cdef FLTXX_t *RH_DATA = &rh_data[+0]
    
    with nogil, parallel(num_threads=numthread):
    
        for ipos in prange(0, NDAT, schedule="static", 
                chunksize=chunksize):
        #-- 2-stage inversion for implicit drag
                    
            XX_DATA[ipos]-= CD_DATA[ipos] * ex_coef * \
                            RH_DATA[ipos]
                            
            XX_DATA[ipos]/= CD_DATA[ipos] * im_coef + \
                            ONE_
                        
    return xx_data

    
def _sum_2_way(cnfg, 
    np.ndarray[FLTXX_t, ndim=1] yy_data,
        const FLTXX_t x1_coef, 
    np.ndarray[FLTXX_t, ndim=1] x1_data,
        const FLTXX_t x2_coef, 
    np.ndarray[FLTXX_t, ndim=1] x2_data
              ):
    
#-- yy = sum( bi * xi ), 2-array version
    
    cdef INDEX_t ipos
    
    cdef INDEX_t NDAT = x1_data.size
    
    cdef INDEX_t numthread = cnfg.numthread
    cdef INDEX_t numchunks = cnfg.numchunks

    cdef INDEX_t chunksize = max(1,
       ((NDAT // numthread // numchunks) // 8) * 8
        )
    
    cdef FLTXX_t *X1_DATA = &x1_data[+0]
    cdef FLTXX_t *X2_DATA = &x2_data[+0]
    cdef FLTXX_t *YY_DATA = &yy_data[+0]
    
    with nogil, parallel(num_threads=numthread):
    
        for ipos in prange(0, NDAT, schedule="static", 
                chunksize=chunksize):        
        #-- 2-way forward-backward averaging
        
            YY_DATA[ipos] = X1_DATA[ipos] * x1_coef + \
                            X2_DATA[ipos] * x2_coef
              
    return yy_data
    

def _sum_3_way(cnfg, 
    np.ndarray[FLTXX_t, ndim=1] yy_data,
        const FLTXX_t x1_coef, 
    np.ndarray[FLTXX_t, ndim=1] x1_data,
        const FLTXX_t x2_coef, 
    np.ndarray[FLTXX_t, ndim=1] x2_data,
        const FLTXX_t x3_coef, 
    np.ndarray[FLTXX_t, ndim=1] x3_data
              ):
    
#-- yy = sum( bi * xi ), 3-array version
    
    cdef INDEX_t ipos
    
    cdef INDEX_t NDAT = x1_data.size
    
    cdef INDEX_t numthread = cnfg.numthread
    cdef INDEX_t numchunks = cnfg.numchunks

    cdef INDEX_t chunksize = max(1,
       ((NDAT // numthread // numchunks) // 8) * 8
        )
    
    cdef FLTXX_t *X1_DATA = &x1_data[+0]
    cdef FLTXX_t *X2_DATA = &x2_data[+0]
    cdef FLTXX_t *X3_DATA = &x3_data[+0]
    cdef FLTXX_t *YY_DATA = &yy_data[+0]
    
    with nogil, parallel(num_threads=numthread):
    
        for ipos in prange(0, NDAT, schedule="static", 
                chunksize=chunksize):        
        #-- 3-way forward-backward averaging
            
            YY_DATA[ipos] = X1_DATA[ipos] * x1_coef + \
                            X2_DATA[ipos] * x2_coef + \
                            X3_DATA[ipos] * x3_coef
                                     
    return yy_data

    
def _sym_3_way(cnfg, 
    np.ndarray[FLTXX_t, ndim=1] yy_data,
        const FLTXX_t x1_coef, 
    np.ndarray[FLTXX_t, ndim=1] x1_data,
        const FLTXX_t x2_coef, 
    np.ndarray[FLTXX_t, ndim=1] x2_data,
        const FLTXX_t x3_coef, 
    np.ndarray[FLTXX_t, ndim=1] x3_data
              ):
    
#-- yy = b1 * (x1 + x3) + b2 * x2, symmetric sums
    
    cdef INDEX_t ipos
    
    cdef INDEX_t NDAT = x1_data.size
    
    cdef INDEX_t numthread = cnfg.numthread
    cdef INDEX_t numchunks = cnfg.numchunks

    cdef INDEX_t chunksize = max(1,
       ((NDAT // numthread // numchunks) // 8) * 8
        )
    
    cdef FLTXX_t *X1_DATA = &x1_data[+0]
    cdef FLTXX_t *X2_DATA = &x2_data[+0]
    cdef FLTXX_t *X3_DATA = &x3_data[+0]
    cdef FLTXX_t *YY_DATA = &yy_data[+0]
    
    with nogil, parallel(num_threads=numthread):
    
        for ipos in prange(0, NDAT, schedule="static",
                chunksize=chunksize):        
        #-- 3-way forward-backward averaging
            
            YY_DATA[ipos] = X2_DATA[ipos] * x2_coef + \
                           (X1_DATA[ipos] + 
                            X3_DATA[ipos])* x1_coef
                                     
    return yy_data

