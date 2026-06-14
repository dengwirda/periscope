
import os
import numpy as np
import argparse

""" SWE: solve the nonlinear SWE on generalised MPAS meshes.
"""
#-- Part of the PERISCOPE solver
#-- Darren Engwirda
#-- d.engwirda@gmail.com
#-- https://github.com/dengwirda/

from stb import strtobool
from slv import swe

from _fp import flt32_t, flt64_t
from _fp import reals_t, index_t

def sec(time):
#-- parse WdXhYmZs time interval in to elapsed seconds
    import re
    tosec = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    try:
        part = re.findall(
            r"(\d+(?:\.\d+)?)([smhd])",time)
        secs = +0.0
        for val, unit in part:
            secs += float(val) * tosec[unit]
        return secs
    except:
        raise ValueError(
            "Couldn't read interval:", time)


if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--mesh-file", dest="mesh_file", type=str,
        required=True, 
        help="Path to user INITIAL conditions file.")
        
    parser.add_argument(
        "--forc-file", dest="forc_file", type=str,
        default="",
        required=False, 
        help="Path to user FORCING tendencies file.")

    parser.add_argument(
        "--soln-file", dest="soln_file", type=str,
        default="",
        required=False, 
        help="Path to user output {OUT_+MESH-FILE}.")

#-- general compute config.

    parser.add_argument(
        "--verbosity", dest="verbosity", type=int,
        default=0,
        required=False, 
        help="How much output to print; larger = more.")

    parser.add_argument(
        "--equations", dest="equations", type=str,
        default="shallow-water",
        required=False,
        help="Eqn. select = shallow-water, shallow-exner.")
        
    parser.add_argument(
        "--numthread", dest="numthread", type=int,
        default=1,
        required=False, help="Number of parallel threads.")

    parser.add_argument(
        "--numdomain", dest="numdomain", type=int,
        default=1,
        required=False, help="Number of parallel domains.")
        
    parser.add_argument(
        "--numchunks", dest="numchunks", type=int,
        default=2,
        required=False, help="Stride of parallel threads.")

    parser.add_argument(
        "--save-vars", dest="save_vars", type=str,
        default="uu_edge, hh_cell, qh_cell, " + \
                "ke_cell, du_cell, rv_dual",
        required=False,
        help="Selected ouput variables to save to file.")

    parser.add_argument(
        "--stat-vars", dest="stat_vars", type=str,
        default="hh_cell, ke_cell",
        required=False,
        help="Selected stats variables to save to file.")

#-- temporal scheme config.

    parser.add_argument(
        "--forc-ramp", dest="forc_ramp", type=float,
        default=0.0,        
        required=False, 
        help="Length of ramp applied to forcing.")

    parser.add_argument(
        "--num-steps", dest="iteration", type=int,
        default=np.inf,
        required=False, help="Number of time steps.")

    parser.add_argument(
        "--time-span", dest="time_span", type=str,
        default=None,
        required=False, help="Length of integration.")

    parser.add_argument(
        "--time-step", dest="time_step", type=float,
        default=-1.,
        required=False, help="Length of time step.")
        
    parser.add_argument(
        "--timestart", dest="timestart", type=float,
        default=0.0,
        required=False, help="Simulation start time.")

    parser.add_argument(
        "--datestart", dest="datestart", type=str,
        default="1991-01-01 00:00:00",
        required=False, help="Simulation start date.")

    parser.add_argument(
        "--cfl-limit", dest="cfl_limit", type=float,
        default=0.9875,
        required=False, help="Adapt step CFL threshold.")

    parser.add_argument(
        "--dt-margin", dest="dt_margin", type=float,
        default=1.6667,
        required=False, help="Adapt step max. increase.")

    parser.add_argument(
        "--dt-cycles", dest="dt_cycles", type=int,
        default=10,
        required=False, help="Adapt step num. try-redo.")

    parser.add_argument(
        "--integrate", dest="integrate", type=str,
        default="RK33-FB",
        required=False, 
        help="Time integration scheme = RK33-FB, RK43-FB.")
    
    parser.add_argument(
        "--save-freq", dest="save_freq", type=int,
        required=False, 
        default=np.inf, 
        help="Save output to file at each FREQ-th step.")

    parser.add_argument(
        "--save-time", dest="save_time", type=str,
        required=False, 
        default=None, 
        help="Save output to file at each Wd Xh Ym Zs time.")

    parser.add_argument(
        "--stat-freq", dest="stat_freq", type=int,
        required=False, 
        default=np.inf, 
        help="Evaluate statistics at each FREQ-th step.")

    parser.add_argument(
        "--stat-time", dest="stat_time", type=str,
        required=False, 
        default=None, 
        help="Evaluate statistics at each Wd Xh Ym Zs time.")

#-- spatial scheme options

    parser.add_argument(
        "--hh-scheme", dest="hh_scheme", type=str,
        default="UPWIND",
        required=False, 
        help="HH.-flux formulation = UPWIND, CENTRE.")

    parser.add_argument(
        "--pv-upwind", dest="pv_upwind", type=float,
        default=1.0000,
        required=False, help="Upwind PV.-flux bias.")

    parser.add_argument(
        "--pv-scheme", dest="pv_scheme", type=str,
        default="AUST-adapt",
        required=False, 
        help="PV.-flux formulation = AUST-adapt, AUST-const, " +
             "APVM, CENTRE.")

    parser.add_argument(
        "--pv-weight", dest="pv_weight", type=float,
        default=0.1000,
        required=False, help="Set linear/nonlinear PV split.")
    
    parser.add_argument(
        "--ke-upwind", dest="ke_upwind", type=float,
        default=0.1000,
        required=False, help="Upwind KE.-edge bias.")

    parser.add_argument(
        "--ke-scheme", dest="ke_scheme", type=str,
        default="CENTRE",
        required=False, 
        help="KE.-grad formulation = AUST-adapt, AUST-const, " +
             "APVM, CENTRE.")
    
    parser.add_argument(
        "--ke-weight", dest="ke_weight", type=float,
        default=1.0000,
        required=False, help="Set KE.-cell vs KE.-dual.")

    parser.add_argument(
        "--ke-method", dest="ke_method", type=float,
        default=1.0000,
        required=False, help="Set KE.-LSQR vs KE.-TRSK.")

#-- dissipation strategies

    parser.add_argument(
        "--ref-scale", dest="ref_scale", type=float,
        default=30.E+3,
        required=False, help="Ref-len. for visc. scale.")

    parser.add_argument(
        "--msh-fixes", dest="msh_fixes", type=float,
        default=1.E+00,
        required=False, help="Mesh quality visc. scale.")
        
    parser.add_argument(
        "--hh-diff-2", dest="hh_diff_2", type=float,
        default=0.E+00,
        required=False, help="DEL^2(H) diffusion coeff.")

    parser.add_argument(
        "--hh-diff-4", dest="hh_diff_4", type=float,
        default=0.E+00,
        required=False, help="DEL^4(H) diffusion coeff.")

    parser.add_argument(
        "--uu-visc-2", dest="uu_visc_2", type=float,
        default=0.E+00,
        required=False, help="DEL^2(U) viscosity coeff.")

    parser.add_argument(
        "--uu-visc-4", dest="uu_visc_4", type=float,
        default=0.E+00,
        required=False, help="DEL^4(U) viscosity coeff.")

    parser.add_argument(
        "--leith-chi", dest="leith_chi", type=float,
        default=0.E+00,
        required=False, help="Leith model scalar coeff.")
        
    parser.add_argument(
        "--leith-max", dest="leith_max", type=float,
        default=np.inf,
        required=False, help="Leith model max. damping.")

    parser.add_argument(
        "--waves-chi", dest="waves_chi", type=float,
        default=2.50E-02,
        required=False, help="Waves model scalar coeff.")

    parser.add_argument(
        "--waves-max", dest="waves_max", type=float,
        default=np.inf,
        required=False, help="Waves model max. damping.")

    parser.add_argument(
        "--shock-chi", dest="shock_chi", type=float,
        default=0.E+00,
        required=False, help="Shock model scalar coeff.")

    parser.add_argument(
        "--shock-max", dest="shock_max", type=float,
        default=np.inf,
        required=False, help="Shock model max. damping.")

    parser.add_argument(
        "--shock-cut", dest="shock_cut", type=float,
        default=1.E+00,
        required=False, help="Shock model cutoff value.")

#-- drag parameterisations

    parser.add_argument(
        "--linlaw-cd", dest="linlaw_cd", type=float,
        default=0.E+00,
        required=False, help="Linear-law Cd drag coeff.")
        
    parser.add_argument(
        "--sqrlaw-cd", dest="sqrlaw_cd", type=float,
        default=0.E+00,
        required=False, help="Square-law Cd drag coeff.")

    parser.add_argument(
        "--loglaw-z0", dest="loglaw_z0", type=float,
        default=0.E+00,
        required=False, help="Log-law roughness length.")

    parser.add_argument(
        "--loglaw-lo", dest="loglaw_lo", type=float,
        default=0.E+00,
        required=False, help="Log-law minimum Cd coeff.")

    parser.add_argument(
        "--loglaw-hi", dest="loglaw_hi", type=float,
        default=0.E+00,
        required=False, help="Log-law maximum Cd coeff.")
        
    parser.add_argument(
        "--manlaw-n0", dest="manlaw_n0", type=float,
        default=0.E+00,
        required=False, help="Manning roughness length.")

    parser.add_argument(
        "--manlaw-lo", dest="manlaw_lo", type=float,
        default=0.E+00,
        required=False, help="Manning minimum Cd coeff.")

    parser.add_argument(
        "--manlaw-hi", dest="manlaw_hi", type=float,
        default=0.E+00,
        required=False, help="Manning maximum Cd coeff.")

    parser.add_argument(
        "--fltlaw-cd", dest="fltlaw_cd", type=float,
        default=0.E+00,
        required=False, help="Filter-law Cd drag coeff.")

    parser.add_argument(
        "--fltlaw-h0", dest="fltlaw_h0", type=float,
        default=1.E+00,
        required=False, help="Filter-law folding depth.")

    parser.add_argument(
        "--fltlaw-t0", dest="fltlaw_t0", type=float,
        default=1.E+00,
        required=False, help="Filter-law folding scale.")
    
    parser.add_argument(
        "--wetdry-h0", dest="wetdry_h0", type=float,
        default=0.E+00,
        required=False, help="Wet-dry limit. thickness.")

#-- physics scheme options

    parser.add_argument(
        "--sound-spd", dest="sound_spd", type=float,
        default=-1.0,
        required=False, 
        help="Fluid speed-of-sound; <= 0 to disable.")

    parser.add_argument(
        "--tidal-frc", dest="tidal_frc", type=str,
        default="",
        required=False,
        help="List of ext. tidal constituents to apply.")

    parser.add_argument(
        "--SAL-const", dest="sal_const", type=float,
        default=0.E+00,
        required=False,
        help="Self attraction and loading scalar coeff.")

    parser.add_argument(
        "--SAL-nfilt", dest="sal_nfilt", type=int,
        default=2,
        required=False,
        help="Self attraction and loading num. filters.")

    parser.add_argument(
        "--SAL-scale", dest="sal_scale", type=float,
        default=1.E+00,
        required=False,
        help="Self attraction and loading scalar depth.")

    parser.add_argument(
        "--SAL-solve", dest="sal_solve", type=str,
        default="",
        required=False,
        help="Self attraction and loading solver: " + 
                    "SCALAR, INLINE.")

#-- misc. flags and config.

    parser.add_argument(
        "--PGF-limit", dest="pgf_limit", 
        type=lambda x: bool(strtobool(str(x.strip()))),
        required=False, 
        default=False, help="Enable PGF slope limit.")

    parser.add_argument(
        "--no-u-tend", dest="no_u_tend", 
        type=lambda x: bool(strtobool(str(x.strip()))),
        required=False, 
        default=False, help="Disable uu-tend. terms.")
        
    parser.add_argument(
        "--no-h-tend", dest="no_h_tend", 
        type=lambda x: bool(strtobool(str(x.strip()))),
        required=False, 
        default=False, help="Disable hh-tend. terms.")

    parser.add_argument(
        "--no-advect", dest="no_advect", 
        type=lambda x: bool(strtobool(str(x.strip()))),
        required=False, 
        default=False, help="Disable mom.-advection.")

    parser.add_argument(
        "--no-rotate", dest="no_rotate", 
        type=lambda x: bool(strtobool(str(x.strip()))),
        required=False, 
        default=False, help="Disable coriolis terms.")

    parser.add_argument(
        "--no-geopot", dest="no_geopot", 
        type=lambda x: bool(strtobool(str(x.strip()))),
        required=False, 
        default=False, help="Disable geopot forcing.")
    
    parser.add_argument(
        "--no-stress", dest="no_stress", 
        type=lambda x: bool(strtobool(str(x.strip()))),
        required=False, 
        default=False, help="Disable stress forcing.")

    parser.add_argument(
        "--FB-weight", dest="fb_weight", type=float,
        required=False,
        nargs="*",
        help="Forward-backward weights for integrators.")

    cnfg = parser.parse_args()

    # sanitise time stepping params
    if (cnfg.time_step <= 0 and cnfg.cfl_limit <= 0):
        raise ValueError(
            "TIME-STEP or CFL-LIMIT must be set.")

    if (cnfg.time_step >= 0.0): cnfg.cfl_limit = -1.

    if (cnfg.iteration == np.inf):
        cnfg.iteration = np.iinfo(index_t).max
   
    if (cnfg.stat_freq == np.inf):
        cnfg.stat_freq = np.iinfo(index_t).max

    if (cnfg.stat_time is not None):
        cnfg.stat_time = flt64_t(sec(cnfg.stat_time))
        cnfg.stat_freq = cnfg.stat_time
    else:
        cnfg.stat_time = -1.0
        cnfg.stat_freq = index_t(
            min(cnfg.iteration, cnfg.stat_freq))

    if (cnfg.save_freq == np.inf):
        cnfg.save_freq = np.iinfo(index_t).max

    if (cnfg.save_time is not None):
        cnfg.save_time = flt64_t(sec(cnfg.save_time))
        cnfg.save_freq = cnfg.save_time
    else:
        cnfg.save_time = -1.0
        cnfg.save_freq = index_t(
            min(cnfg.iteration, cnfg.save_freq))

    if (cnfg.time_span is not None):
        cnfg.time_stop = flt64_t(sec(cnfg.time_span))
    else:
        cnfg.time_stop = -1.0

    if (cnfg.time_step > 0. and cnfg.time_stop > 0.):
        num_iters = cnfg.time_stop / \
                    cnfg.time_step
        if index_t(num_iters) != num_iters:
            raise ValueError(
            "TIME-STOP not a multiple of TIME-STEP")
        cnfg.iteration = index_t(num_iters)

    if (cnfg.time_step > 0. and cnfg.stat_time > 0.):
        num_iters = cnfg.stat_time / \
                    cnfg.time_step
        if index_t(num_iters) != num_iters:
            raise ValueError(
            "STAT-TIME not a multiple of TIME-STEP")
        cnfg.stat_freq = index_t(num_iters)

    if (cnfg.time_step > 0. and cnfg.save_time > 0.):
        num_iters = cnfg.save_time / \
                    cnfg.time_step
        if index_t(num_iters) != num_iters:
            raise ValueError(
            "SAVE-TIME not a multiple of TIME-STEP")
        cnfg.save_freq = index_t(num_iters)

    # santise discretisation params
    cnfg.integrate = cnfg.integrate.upper()
    cnfg.equations = cnfg.equations.upper()
    cnfg.hh_scheme = cnfg.hh_scheme.upper()
    cnfg.ke_scheme = cnfg.ke_scheme.upper()
    cnfg.pv_scheme = cnfg.pv_scheme.upper()
    
    cnfg.save_vars = cnfg.save_vars.lower()
    cnfg.stat_vars = cnfg.stat_vars.lower()
    
    cnfg.tidal_frc = cnfg.tidal_frc.upper()
    cnfg.sal_solve = cnfg.sal_solve.upper()

    # mesh, forcing, & solution i/o
    path, file = os.path.split(cnfg.mesh_file)
    if (cnfg.soln_file == ""): 
        cnfg.soln_file = \
            os.path.join(path,  "out_" + file)

    swe(cnfg)  # call to the solver

