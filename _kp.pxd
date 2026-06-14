
from libc.stdint cimport int8_t, int32_t

""" typedef's for discrete SWE spatio-temporal operators
"""
#-- Part of the PERISCOPE solver
#-- Darren Engwirda
#-- d.engwirda@gmail.com
#-- https://github.com/dengwirda/

ctypedef float   FLT32_t
ctypedef double  FLT64_t

ctypedef int32_t INDEX_t
ctypedef float   REALS_t  # or double
ctypedef double  LOCAL_t  # registers

ctypedef int8_t  BYTES_t

ctypedef fused   FLTXX_t: # underlying 
    FLT32_t
    FLT64_t

ctypedef fused   FLTXT_t: # tendencies
    FLT32_t
    FLT64_t

ctypedef float   UDATA_t  # underlying
ctypedef double  HDATA_t
ctypedef float   QDATA_t

ctypedef double  UTEND_t  # tendencies
ctypedef double  HTEND_t
ctypedef double  QTEND_t

