
#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: initializedcheck=False
#cython: cdivision=True
#cython: cpow=True

""" Solvers for self attraction and loading (SAL) inputs
"""
#-- Part of the PERISCOPE solver
#-- Darren Engwirda
#-- d.engwirda@gmail.com
#-- https://github.com/dengwirda/

import os
import numpy as np
cimport numpy as np
cimport cython

from cython.parallel import prange, parallel

from _fp import flt32_t, flt64_t
from _kp cimport FLT32_t, FLT64_t

from _fp import reals_t, index_t
from _kp cimport REALS_t, INDEX_t, LOCAL_t

from _fp import udata_t, hdata_t, qdata_t
from _kp cimport UDATA_t, HDATA_t, QDATA_t

from lib cimport sqrt as sqrt_r
from lib cimport sin as sin_r
from lib cimport asin as asin_r
from lib cimport cos as cos_r
from lib cimport acos as acos_r

from mem import variables
from mem import get_vec_v, get_vec_e, get_vec_c, \
                put_vec_v, put_vec_e, put_vec_c

xa_love = np.loadtxt(  # load love numbers
    os.path.join(os.path.abspath(
    os.path.dirname( __file__ )),"lln.dat"), comments="#")

xa_init = False
xa_scal = np.ones(xa_love.shape[0], dtype=reals_t)

xa_Arcn = np.empty(0); xa_Brcn = np.empty(0)
xa_sin_ = np.empty(0); xa_cos_ = np.empty(0)
xa_Rexp = np.empty(0); xa_Iexp = np.empty(0)

pp_poly = np.empty(0); qq_poly = np.empty(0)
rr_poly = np.empty(0)

h1 = xa_love[1, 0]; l1 = xa_love[1, 1]; k1 = xa_love[1, 2]
xa_love[1, 0] = +2./3. * h1 - 2./3. * l1
xa_love[1, 1] = -1./3. * h1 + 1./3. * l1
xa_love[1, 2] = -1./3. * h1 - 2./3. * l1 - 1.

def _setup_sal(mesh, mats, cnfg, 
        const INDEX_t nord):

    global xa_Arcn, xa_Brcn, xa_sin_, xa_cos_
    global xa_Rexp, xa_Iexp
    global xa_init

    cdef INDEX_t indx, mdeg, ndeg

    cdef REALS_t RHOW = 1035.0  # water
    cdef REALS_t RHOE = 5517.0  # solid

    cdef REALS_t ZERO = 0.0
    cdef REALS_t ONE_ = 1.0
    cdef REALS_t HALF = 0.5
    cdef FLT64_t PI = np.pi

    cdef LOCAL_t SCAL = 3.0 * RHOW / RHOE

    cdef INDEX_t NCEL = mesh.cell.size

    if (xa_init): return

    onum = nord + 1
    xa_Arcn = np.zeros((onum, onum), dtype=reals_t)
    xa_Brcn = np.zeros((onum, onum), dtype=reals_t)

    pp_poly = np.zeros((NCEL), dtype=reals_t)
    qq_poly = np.zeros((NCEL), dtype=reals_t)
    rr_poly = np.zeros((NCEL), dtype=reals_t)

    xa_sin_ = np.zeros((NCEL), dtype=reals_t)
    xa_cos_ = np.zeros((NCEL), dtype=reals_t)

    xa_Rexp = np.zeros((NCEL, onum), dtype=reals_t)
    xa_Iexp = np.zeros((NCEL, onum), dtype=reals_t)

    for mdeg in range(+0, nord + 1):
        for cell in range(+0, NCEL):

        #-- exponential coeff. for transformation

            xlon = mesh.cell.xlon [cell]

            xa_Rexp[cell, mdeg] = cos_r(mdeg * xlon)
            xa_Iexp[cell, mdeg] = sin_r(mdeg * xlon)

    for cell in range(+0, NCEL):
        if (True):

            ylat = mesh.cell.ylat [cell]

            xa_sin_[cell] = sin_r (HALF * PI - ylat)
            xa_cos_[cell] = cos_r (HALF * PI - ylat)

    for mdeg in range(+0, nord + 1):
        for ndeg in range(mdeg, nord + 1):

        #-- SH coeff: (1 + k' - h') / (2 * n - 1)

            indx = (nord + 1) * mdeg - \
                mdeg * (mdeg + 1) / 2 + ndeg + 1

            xa_scal[indx] =  SCAL * (ONE_ 
              + xa_love[ndeg, 2] 
              - xa_love[ndeg, 0]) / (2* ndeg + 1)

    for mdeg in range(+0, nord + 1):
        for ndeg in range(mdeg, nord + 1):
            
        #-- a recurrence for Legendre polynomials 

            if (mdeg == ndeg): continue

            xa_Arcn[ndeg, mdeg] = (sqrt_r(
                (2*ndeg - 1) * (2*ndeg + 1)) / 
                    ((ndeg - mdeg) * (ndeg + mdeg)))

            xa_Brcn[ndeg, mdeg] = (sqrt_r(
                (2*ndeg + 1) * 
                (ndeg + mdeg - 1)*(ndeg - mdeg - 1)) / 
            ((ndeg - mdeg) * (ndeg + mdeg) * (2*ndeg - 3)))

    xa_init = True


def _assoc_pmn(mesh, mats, cnfg,
        const INDEX_t ndeg, 
        const INDEX_t mdeg):

#-- compute assoc. Legendre polynomials: P, Q, R

    cdef INDEX_t cell, ideg

    cdef REALS_t ZERO = 0.0
    cdef REALS_t ONE_ = 1.0
    cdef REALS_t TWO_ = 2.0
    cdef REALS_t FOUR = 4.0
    cdef REALS_t PI = np.pi

    cdef INDEX_t numthread = cnfg.numthread
    cdef INDEX_t chunksize = cnfg.chunksize
    cdef INDEX_t chunkcell = cnfg.chunkcell
    cdef INDEX_t chunkedge = cnfg.chunkedge
    cdef INDEX_t chunkvert = cnfg.chunkvert

    cdef INDEX_t NCEL = mesh.cell.size

    cdef np.ndarray[REALS_t,ndim=1] _p_sin_ = xa_sin_
    cdef np.ndarray[REALS_t,ndim=1] _p_cos_ = xa_cos_

    cdef REALS_t *XA_SIN_ = &_p_sin_[0]
    cdef REALS_t *XA_COS_ = &_p_cos_[0]

    cdef np.ndarray[REALS_t,ndim=1] _p_ppol = pp_poly
    cdef np.ndarray[REALS_t,ndim=1] _p_qpol = qq_poly
    cdef np.ndarray[REALS_t,ndim=1] _p_rpol = rr_poly

    cdef REALS_t *PP_POLY = &_p_ppol[0]
    cdef REALS_t *QQ_POLY = &_p_qpol[0]
    cdef REALS_t *RR_POLY = &_p_rpol[0]

    cdef REALS_t[:, ::1] XA_ARCN = xa_Arcn
    cdef REALS_t[:, ::1] XA_BRCN = xa_Brcn

    with nogil, parallel(num_threads=numthread):

        if (ndeg == mdeg + 0):

            for cell in prange(0, NCEL, schedule="static",
                    chunksize=chunkcell):

                PP_POLY[cell] = sqrt_r (ONE_ / 
                   (FOUR * PI)) * XA_SIN_[cell] ** mdeg

                for ideg in range(1, mdeg + 1):

                    PP_POLY[cell]*= sqrt_r (
                   (TWO_ * ideg + ONE_) / (TWO_ * ideg))

        if (ndeg == mdeg + 1):

            for cell in prange(0, NCEL, schedule="static",
                    chunksize=chunkcell):

                QQ_POLY[cell] = \
                    XA_ARCN[ndeg, mdeg] * XA_COS_[cell] * \
                                          PP_POLY[cell]

        if (ndeg >= mdeg + 2):

            for cell in prange(0, NCEL, schedule="static",
                    chunksize=chunkcell):

                RR_POLY[cell] = \
                    XA_ARCN[ndeg, mdeg] * XA_COS_[cell] * \
                                          QQ_POLY[cell] - \
                    XA_BRCN[ndeg, mdeg] * PP_POLY[cell]

    return


def _calc_self(mesh, mats, cnfg, 
        const FLT64_t time,
    np.ndarray[HDATA_t, ndim=1] hh_cell,
    np.ndarray[FLT32_t, ndim=1] zb_cell,
        const REALS_t gravity, 
    np.ndarray[REALS_t, ndim=1] Xi_cell
              ):
    
#-- compute the self attraction & loading

    cdef INDEX_t vert, cell, filt, iptr, xidx
    cdef REALS_t xval

    cdef REALS_t ZERO = 0.0

    cdef INDEX_t numthread = cnfg.numthread
    cdef INDEX_t chunksize = cnfg.chunksize
    cdef INDEX_t chunkcell = cnfg.chunkcell
    cdef INDEX_t chunkedge = cnfg.chunkedge
    cdef INDEX_t chunkvert = cnfg.chunkvert

    cdef INDEX_t numfilter = cnfg.sal_nfilt
 
    cdef INDEX_t NCEL = mesh.cell.size
    cdef INDEX_t NVRT = mesh.vert.size

    cdef LOCAL_t  XI_SUM_
    cdef HDATA_t *HH_CELL = &hh_cell[0]
    cdef FLT32_t *ZB_CELL = &zb_cell[0]
    cdef REALS_t *XI_CELL = &Xi_cell[0]

    cdef np.ndarray[REALS_t] xi_vert = get_vec_v()

    cdef REALS_t *XI_VERT = &xi_vert[0]

    cdef INDEX_t *DUAL_KITE_XPTR = ptr_index_t(
             mats.dual_kite_sums.indptr)
    cdef INDEX_t *DUAL_KITE_XIDX = ptr_index_t(
             mats.dual_kite_sums.indices)
    cdef REALS_t *DUAL_KITE_XVAL = ptr_reals_t(
             mats.dual_kite_sums.data)

    cdef INDEX_t *CELL_KITE_XPTR = ptr_index_t(
             mats.cell_kite_sums.indptr)
    cdef INDEX_t *CELL_KITE_XIDX = ptr_index_t(
             mats.cell_kite_sums.indices)
    cdef REALS_t *CELL_KITE_XVAL = ptr_reals_t(
             mats.cell_kite_sums.data)

    cdef REALS_t *MESH_DUAL_AREA = ptr_reals_t(
                  mesh.vert.area)
    cdef REALS_t *MESH_CELL_AREA = ptr_reals_t(
                  mesh.cell.area)

    with nogil, parallel(num_threads=numthread):

        for cell in prange(0, NCEL, schedule="static",
                    chunksize=chunksize):

            XI_CELL[cell] = ZB_CELL[cell] + HH_CELL[cell]

        for filt in range(numfilter):
            #-- SAL is a smoothing operator, so filter at
            #-- grid-scale (over kernel of scheme)

            for vert in prange(0, NVRT, schedule="static",
                    chunksize=chunkvert):
            #-- remap Xi to duals to smooth at grid-scale
                XI_SUM_ = ZERO
                for iptr in range(DUAL_KITE_XPTR[vert +0], 
                                  DUAL_KITE_XPTR[vert +1]):
                    
                    xval = DUAL_KITE_XVAL[iptr]
                    xidx = DUAL_KITE_XIDX[iptr]

                    XI_SUM_ = \
                        XI_SUM_ + xval * XI_CELL[xidx]

                XI_VERT[vert] = \
                        XI_SUM_ / MESH_DUAL_AREA[vert]

            for cell in prange(0, NCEL, schedule="static",
                    chunksize=chunkcell):
            #-- remap Xi to cells to smooth at grid-scale
                XI_SUM_ = ZERO
                for iptr in range(CELL_KITE_XPTR[cell +0], 
                                  CELL_KITE_XPTR[cell +1]):
                    
                    xval = CELL_KITE_XVAL[iptr]
                    xidx = CELL_KITE_XIDX[iptr]

                    XI_SUM_ = \
                        XI_SUM_ + xval * XI_VERT[xidx]

                XI_CELL[cell] = \
                        XI_SUM_ / MESH_CELL_AREA[cell]

    put_vec_v   (xi_vert)

    return Xi_cell


cdef INDEX_t* ptr_index_t(INDEX_t[::1] buffer):
    cdef INDEX_t *BUFFER = &buffer[+0]
    return BUFFER


cdef REALS_t* ptr_reals_t(REALS_t[::1] buffer):
    cdef REALS_t *BUFFER = &buffer[+0]
    return BUFFER


