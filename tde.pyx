
#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: initializedcheck=False
#cython: cdivision=True
#cython: cpow=True

""" K constituent tidal potential due to sun-moon system
"""
#-- Part of the PERISCOPE solver
#-- Darren Engwirda
#-- d.engwirda@gmail.com
#-- https://github.com/dengwirda/

import numpy as np
cimport numpy as np
cimport cython

import datetime as dt

from cython.parallel import prange, parallel

from _fp import flt32_t, flt64_t
from _kp cimport FLT32_t, FLT64_t
from _fp import reals_t, index_t
from _kp cimport REALS_t, INDEX_t, LOCAL_t

from lib cimport sin as sin_r
from lib cimport asin as asin_r
from lib cimport cos as cos_r
from lib cimport acos as acos_r
from lib cimport tan as tan_r
from lib cimport atan as atan_r
from lib cimport sqrt as sqrt_r

tc_mark = flt64_t (0)  # time at last eval
tc_ncon = index_t (0)  # num. constituents
tc_nmax = index_t (8)  # max. constituents
tc_ampl = np.zeros(tc_nmax, dtype=flt64_t)
tc_freq = np.zeros(tc_nmax, dtype=flt64_t)
tc_node = np.zeros(tc_nmax, dtype=flt64_t)
tc_phas = np.zeros(tc_nmax, dtype=flt64_t)
tc_astr = np.zeros(tc_nmax, dtype=flt64_t)
tc_love = np.zeros(tc_nmax, dtype=flt64_t)
tc_type = np.zeros(tc_nmax, dtype=index_t)
tc_time = np.zeros(tc_nmax, dtype=flt64_t)
tc_scal = np.zeros(tc_nmax, dtype=flt64_t)

def _fix_angle(FLT64_t angle):
#-- map angle onto [0.0, 360.0] domain
    cdef INDEX_t cycle = int(angle / 360.0)
    angle -= cycle * 360.0
    if (angle < 0.0): angle += 360.0
    return angle


def _orbit_tde(
        const INDEX_t year,
        const INDEX_t dayj,
        const INDEX_t hour):

    cdef INDEX_t leap = int((year - 1901) / 4.0)
    cdef INDEX_t yr19 = year - 1900
    cdef INDEX_t day_ = dayj + leap - 1

    cdef FLT64_t I, s, p, h, N, xi, pc, nu, nup, nup2
    cdef FLT64_t PI = np.pi

    cdef FLT64_t Nr, pr, Ir, nur, xir, p1, nupr, nup2r

    # N: lon of the moon's node (N, Table 1, Schureman)
    N = 259.1560564 - 19.328185764 * yr19 - \
            0.0529539336 * day_ - 0.0022064139 * hour
    N = _fix_angle(N)
    Nr = N * PI / 180.0

    # p: lunar perigee (small p, Table 1)
    p = 334.3837214 + 40.66246584 * yr19 + \
            0.1114040160 * day_ + 0.0046418340 * hour
    p = _fix_angle(p)
    pr = p * PI / 180.0

    Ir = acos_r(0.91369490 - 0.0356926 * cos_r(Nr))
    I = _fix_angle(Ir * 180.0 / PI)

    nur = asin_r(0.0897056 * sin_r(Nr) / sin_r(Ir))
    nu = nur * 180.0 / PI

    xir = Nr - \
        2.0 * atan_r(0.64412 * tan_r(Nr / 2.0)) - nur
    xi = xir * 180.0 / PI

    pc = _fix_angle(p - xi)

    # h: mean lon of the sun (small h, Table 1)
    h = 280.1895014 - 0.238724988 * yr19 + \
            0.9856473288 * day_ + 0.0410686387 * hour
    h = _fix_angle(h)

    # p1: solar perigee (small p1, Table 1)
    p1 = 281.2208569 + 0.01717836 * yr19 + \
            0.0000470640 * day_ + 0.0000019610 * hour
    p1 = _fix_angle(p1)

    # s: mean lon of the moon (small s, Table 1)
    s = 277.02562060 + 129.38482032 * yr19 + \
            13.176396768 * day_ + 0.5490165320 * hour
    s = _fix_angle(s)

    nupr = atan_r(sin_r(nur) 
        / (cos_r(nur) + 0.3347660 / sin_r(2.0 * Ir)))
    nup = nupr * 180.0 / PI

    nup2r = 0.5 * atan_r(sin_r(2.0 * nur) / 
        (cos_r(2.0 * nur) + 0.0726184 / sin_r(Ir) ** 2))
    nup2 = nup2r * 180.0 / PI

    return I, s, p, h, N, xi, pc, nu, nup, nup2


def _setup_tde(mesh, mats, cnfg, 
        const FLT64_t time):

    global tc_ncon, tc_mark

    cdef FLT64_t r_I, r_s, r_p, r_h, r_N, r_xi, r_pc, \
                 r_nu, r_nup, r_nup2
    cdef FLT64_t n_I, n_s, n_p, n_h, n_N, n_xi, n_pc, \
                 n_nu, n_nup, n_nup2
    cdef FLT64_t T, PI = np.pi
    cdef INDEX_t r_year, r_day_, r_hour
    cdef INDEX_t n_year, n_day_, n_hour

    cdef INDEX_t numthread = cnfg.numthread
    cdef INDEX_t chunksize = cnfg.chunksize

    cdef INDEX_t NCON = 0

    cdef np.ndarray[FLT64_t,ndim=1] _p_ampl = tc_ampl
    cdef np.ndarray[FLT64_t,ndim=1] _p_freq = tc_freq
    cdef np.ndarray[FLT64_t,ndim=1] _p_node = tc_node
    cdef np.ndarray[FLT64_t,ndim=1] _p_phas = tc_phas
    cdef np.ndarray[FLT64_t,ndim=1] _p_astr = tc_astr
    cdef np.ndarray[FLT64_t,ndim=1] _p_love = tc_love
    
    cdef FLT64_t *TC_AMPL = &_p_ampl[0]
    cdef FLT64_t *TC_FREQ = &_p_freq[0]
    cdef FLT64_t *TC_NODE = &_p_node[0]
    cdef FLT64_t *TC_PHAS = &_p_phas[0]
    cdef FLT64_t *TC_ASTR = &_p_astr[0]
    cdef FLT64_t *TC_LOVE = &_p_love[0]
    
    cdef np.ndarray[INDEX_t,ndim=1] _p_type = tc_type

    cdef INDEX_t *TC_TYPE = &_p_type[0]
    
    if (time >= tc_mark):
        tc_ncon = 0  # re-eval tidal coeff
       #tc_mark = time + 3600.0  # eval coeff in 1hr
        tc_mark = time + 1.E+32  # just initial eval

    if (tc_ncon > 0): return tc_ncon

    # eval phase at reference time
    # eval amplitude at current time

    now_ = dt.datetime.fromisoformat(cnfg.datestart)

    r_year = now_.year  # ref
    r_day_ = now_.timetuple().tm_yday
    r_hour = now_.hour

    r_I, r_s, r_p, r_h, r_N, r_xi, r_pc, r_nu, \
    r_nup, r_nup2 = _orbit_tde(r_year, r_day_, r_hour)

    now_+= dt.timedelta(seconds=time)

    n_year = now_.year  # now
    n_day_ = now_.timetuple().tm_yday
    n_hour = now_.hour

    n_I, n_s, n_p, n_h, n_N, n_xi, n_pc, n_nu, \
    n_nup, n_nup2 = _orbit_tde(n_year, n_day_, n_hour)

    T = _fix_angle(180.0 + r_hour * (360.0 / 24.0))

    if ("M2" in cnfg.tidal_frc):
        TC_AMPL[NCON] = 0.242334  # MPAS-O
       #TC_AMPL[NCON] = 0.244100  # TPXO et al?
        TC_FREQ[NCON] = 1.4051890E-04
        TC_LOVE[NCON] = 0.693
        TC_ASTR[NCON] = 0.0
        TC_ASTR[NCON]*= PI / 180.0
        TC_NODE[NCON] = \
            cos_r(0.5 * n_I * PI / 180.0) ** 4 / 0.91544
        TC_PHAS[NCON] = _fix_angle(
            2.0 * (T - r_s + r_h) + 2.0 * (r_xi - r_nu))
        TC_PHAS[NCON]*= PI / 180.0
        TC_TYPE[NCON] = 2 - 0
        NCON += 1

    if ("K1" in cnfg.tidal_frc):
        TC_AMPL[NCON] = 0.141565
        TC_FREQ[NCON] = 0.7292117E-04
        TC_LOVE[NCON] = 0.736
        TC_ASTR[NCON] = 0.0
        TC_ASTR[NCON]*= PI / 180.0
        TC_NODE[NCON] = sqrt_r(
            0.8965 * sin_r(2.0 * n_I * PI / 180.0) ** 2 +
            0.6001 * sin_r(2.0 * n_I * PI / 180.0) *
                cos_r(n_nu * PI / 180.0) + 0.1006)
        TC_PHAS[NCON] = _fix_angle(T + r_h - 90.0 - r_nup)
        TC_PHAS[NCON]*= PI / 180.0
        TC_TYPE[NCON] = 1 - 0
        NCON += 1

    if ("S2" in cnfg.tidal_frc):
        TC_AMPL[NCON] = 0.112743
        TC_FREQ[NCON] = 1.4544410E-04
        TC_LOVE[NCON] = 0.693
        TC_ASTR[NCON] = 0.0 
        TC_ASTR[NCON]*= PI / 180.0
        TC_NODE[NCON] = 1.0
        TC_PHAS[NCON] = _fix_angle(2.0 * T)
        TC_PHAS[NCON]*= PI / 180.0
        TC_TYPE[NCON] = 2 - 0
        NCON += 1

    if ("O1" in cnfg.tidal_frc):
        TC_AMPL[NCON] = 0.100661
        TC_FREQ[NCON] = 0.6759774E-04
        TC_LOVE[NCON] = 0.695
        TC_ASTR[NCON] = 0.0
        TC_ASTR[NCON]*= PI / 180.0
        TC_NODE[NCON] = sin_r(n_I * PI / 180.0) * \
            cos_r(0.5 * n_I * PI / 180.0) ** 2 / 0.37988
        TC_PHAS[NCON] = _fix_angle(T -
            2.0 * r_s + r_h + 90.0 + 2.0 * r_xi - r_nu)
        TC_PHAS[NCON]*= PI / 180.0
        TC_TYPE[NCON] = 1 - 0
        NCON += 1

    if ("P1" in cnfg.tidal_frc):
        TC_AMPL[NCON] = 0.046848
        TC_FREQ[NCON] = 0.7252295E-04
        TC_LOVE[NCON] = 0.706
        TC_ASTR[NCON] = 0.0 
        TC_ASTR[NCON]*= PI / 180.0
        TC_NODE[NCON] = 1.0 
        TC_PHAS[NCON] = _fix_angle(T - r_h + 90.0)
        TC_PHAS[NCON]*= PI / 180.0
        TC_TYPE[NCON] = 1 - 0
        NCON += 1
    
    if ("N2" in cnfg.tidal_frc):
        TC_AMPL[NCON] = 0.046397
        TC_FREQ[NCON] = 1.3787970E-04
        TC_LOVE[NCON] = 0.693
        TC_ASTR[NCON] = 0.0
        TC_ASTR[NCON]*= PI / 180.0
        TC_NODE[NCON] = \
            cos_r(0.5 * n_I * PI / 180.0) ** 4 / 0.91544
        TC_PHAS[NCON] = _fix_angle(2.0 * (T + r_h)
            - 3.0 * r_s + r_p + 2.0 * (r_xi - r_nu))
        TC_PHAS[NCON]*= PI / 180.0
        TC_TYPE[NCON] = 2 - 0
        NCON += 1

    if ("K2" in cnfg.tidal_frc):
        TC_AMPL[NCON] = 0.030684
        TC_FREQ[NCON] = 1.4584230E-04
        TC_LOVE[NCON] = 0.693 
        TC_ASTR[NCON] = 0.0
        TC_ASTR[NCON]*= PI / 180.0
        TC_NODE[NCON] = 1.0 / 1000.0 + sqrt_r(
            19.0444 * sin_r(n_I * PI / 180.0) ** 4 +
            2.77020 * sin_r(n_I * PI / 180.0) ** 2 *
            cos_r(2 * n_nu * PI / 180.0) + 0.0981)
        TC_PHAS[NCON] = \
            _fix_angle(2.0 * (T + r_h) - 2.0 * r_nup2) 
        TC_PHAS[NCON]*= PI / 180.0
        TC_TYPE[NCON] = 2 - 0
        NCON += 1

    if ("Q1" in cnfg.tidal_frc):
        TC_AMPL[NCON] = 0.019273
        TC_FREQ[NCON] = 0.6495854E-04
        TC_LOVE[NCON] = 0.695
        TC_ASTR[NCON] = 0.0
        TC_ASTR[NCON]*= PI / 180.0
        TC_NODE[NCON] = sin_r(n_I * PI / 180.0) * \
            cos_r(0.5 * n_I * PI / 180.0) ** 2 / 0.37988
        TC_PHAS[NCON] = \
            _fix_angle(T - 3.0 * r_s + r_h 
                + r_p + 90.0 + 2.00 * r_xi - r_nu)
        TC_PHAS[NCON]*= PI / 180.0
        TC_TYPE[NCON] = 1 - 0
        NCON += 1

    tc_ncon = NCON
    return tc_ncon


def _calc_tide(mesh, mats, cnfg, 
        const FLT64_t time,
        const REALS_t gravity,
    np.ndarray[REALS_t, ndim=1] Xi_cell
              ):
    
#-- compute the tidal potential at time=t
    
    cdef INDEX_t cell, icon, jcon
    cdef FLT64_t xlon, ylat

    cdef INDEX_t tc_pass
    cdef FLT64_t tc_invf, tc_offt, tc_fun1, tc_fun2

    cdef REALS_t ZERO = 0.0
    cdef FLT64_t ONE_ = 1.0
    cdef FLT64_t TWO_ = 2.0
    cdef FLT64_t PI = np.pi
    
    cdef INDEX_t numthread = cnfg.numthread
    cdef INDEX_t chunksize = cnfg.chunksize
    
    cdef INDEX_t NCEL = mesh.cell.size

    cdef REALS_t *XI_CELL = &Xi_cell[0]

    cdef np.ndarray[FLT64_t,ndim=1] _p_ampl = tc_ampl
    cdef np.ndarray[FLT64_t,ndim=1] _p_freq = tc_freq
    cdef np.ndarray[FLT64_t,ndim=1] _p_node = tc_node
    cdef np.ndarray[FLT64_t,ndim=1] _p_phas = tc_phas
    cdef np.ndarray[FLT64_t,ndim=1] _p_astr = tc_astr
    cdef np.ndarray[FLT64_t,ndim=1] _p_love = tc_love
    
    cdef FLT64_t *TC_AMPL = &_p_ampl[0]
    cdef FLT64_t *TC_FREQ = &_p_freq[0]
    cdef FLT64_t *TC_NODE = &_p_node[0]
    cdef FLT64_t *TC_PHAS = &_p_phas[0]
    cdef FLT64_t *TC_ASTR = &_p_astr[0]
    cdef FLT64_t *TC_LOVE = &_p_love[0]

    cdef np.ndarray[INDEX_t,ndim=1] _p_type = tc_type    

    cdef INDEX_t *TC_TYPE = &_p_type[0]
    
    cdef np.ndarray[FLT64_t,ndim=1] _p_time = tc_time
    cdef np.ndarray[FLT64_t,ndim=1] _p_scal = tc_scal

    cdef FLT64_t *TC_TIME = &_p_time[0]
    cdef FLT64_t *TC_SCAL = &_p_scal[0]

    cdef FLT64_t[::1] mesh_cell_xlon = mesh.cell.xlon
    cdef FLT64_t[::1] mesh_cell_ylat = mesh.cell.ylat
    
    cdef FLT64_t *MESH_CELL_XLON = &mesh_cell_xlon[0]
    cdef FLT64_t *MESH_CELL_YLAT = &mesh_cell_ylat[0]

    cdef INDEX_t NCON = \
        _setup_tde (mesh, mats, cnfg, time)

    with nogil, parallel(num_threads=numthread):

        for icon in range(NCON):
            tc_invf = TWO_ * PI / TC_FREQ[icon]
            
            tc_pass = int(time / tc_invf)
            tc_offt = time - tc_pass * tc_invf

            TC_SCAL[icon] = (
                TC_AMPL[icon] * 
                TC_NODE[icon] * TC_LOVE[icon])

            TC_TIME[icon] = (
                TC_FREQ[icon] * tc_offt + 
                TC_PHAS[icon] + TC_ASTR[icon])

        for cell in prange(0, NCEL, schedule="static", 
                chunksize=chunksize):

            xlon =  MESH_CELL_XLON[cell]
            ylat =  MESH_CELL_YLAT[cell]

        #-- apply diurnal + semi-diurnal constituents

            tc_fun1 = sin_r(ylat * TWO_)
            tc_fun2 = cos_r(ylat)
            tc_fun2 = tc_fun2 * tc_fun2

            XI_CELL[cell] = ZERO
            for jcon in range(NCON):
                if (TC_TYPE[jcon] == 1):
                    XI_CELL[cell]+= (
                    TC_SCAL[jcon] * tc_fun1 * 
                      cos_r(
                    TC_TIME[jcon] + xlon * ONE_ )
                    )

                if (TC_TYPE[jcon] == 2):
                    XI_CELL[cell]+= (
                    TC_SCAL[jcon] * tc_fun2 * 
                      cos_r(
                    TC_TIME[jcon] + xlon * TWO_ ) 
                    )
            
            XI_CELL[cell]*= gravity 

    return  Xi_cell


