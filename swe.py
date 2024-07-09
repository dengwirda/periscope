
import os
import time
import copy
import numpy as np
import netCDF4 as nc
import argparse

""" SWE: solve the nonlinear SWE on generalised MPAS meshes.
"""
#-- Part of the PERISCOPE solver
#-- Darren Engwirda
#-- d.engwirda@gmail.com
#-- https://github.com/dengwirda/

from stb import strtobool

from _fp import flt32_t, flt64_t
from _fp import reals_t, index_t

from log import tcpu

from msh import load_mesh, sort_mesh, \
                load_flow, sort_flow, \
                load_forc, sort_forc
from ops import operators
from mem import init_pool, variables

from io_ import init_file, save_step

from _dt import step_eqns, step_bnds, mark_time
from _dx import invariant, scalingVk

def swe(cnfg):

    print(
    "#"+"=========================================="*2+"\n" + 
    "#              o                     \n" +
    "#   ,_   _  _  `  .   _, __  ,_   _  \n" +
    "# _/|_)_(/_/ (_(_/_)_(__(_)_/|_)_(/_ \n" +
    "#  /|                       /|       \n" +
    "# (/                       (/        \n" +
    "#"+"=========================================="*2+"\n"
         )

    # sanitise user config. params    
    cnfg.save_freq = \
        min(cnfg.iteration, cnfg.save_freq)
    cnfg.stat_freq = \
        min(cnfg.iteration, cnfg.stat_freq)
    
    cnfg.calc_slow = True
    cnfg.calc_fast = True
    cnfg.calc_drag = True
    cnfg.timeisnow = cnfg.timestart
    
    cnfg.integrate = cnfg.integrate.upper()
    cnfg.equations = cnfg.equations.upper()
    cnfg.hh_scheme = cnfg.hh_scheme.upper()
    cnfg.ke_upwind = cnfg.ke_upwind.upper()
    cnfg.ke_scheme = cnfg.ke_scheme.upper()
    cnfg.pv_upwind = cnfg.pv_upwind.upper()
    cnfg.pv_scheme = cnfg.pv_scheme.upper()
    
    cnfg.save_vars = cnfg.save_vars.lower()
    
    if ("CENTRE" in cnfg.ke_scheme): 
        cnfg.ke_upwind = "NONE"
    if ("CENTRE" in cnfg.pv_scheme): 
        cnfg.pv_upwind = "NONE"

    # mesh, forcing & solution i/o 
    name = cnfg.mesh_file
    forc = cnfg.forc_file
    save = cnfg.soln_file
    path, file = os.path.split(name)
    if (save == ""): 
        save = os.path.join(path, "out_"+file)

    print("Loading input assets...")
    
    ttic = time.time()

    # load mesh + init. conditions
    mesh = load_mesh(name)
    flow = load_flow(name, mesh, lean=True)
    flow = load_forc(forc, flow, lean=True)

    ttoc = time.time()
    print("*READ done (sec):", round(ttoc - ttic, 2))
    
    print("")
    print("Creating output file...")

    ttic = time.time()

    init_file(name, cnfg, save, mesh, flow)

    ttoc = time.time()
    print("*SAVE done (sec):", round(ttoc - ttic, 2))

    print("")
    print("Reordering mesh data...")

    ttic = time.time()

    mesh = sort_mesh(mesh, True)
    flow = sort_flow(flow, mesh, lean=True)
    flow = sort_forc(flow, mesh, lean=True)

    h0_cell = flow.hh_cell
    u0_edge = flow.uu_edge

    h0_cell = np.maximum(cnfg.wetdry_h0, h0_cell)

    ttoc = time.time()
    print("*SORT done (sec):", round(ttoc - ttic, 2))

    print("")
    print("Forming coefficients...")

    ttic = time.time()

    mesh.cell.mask[flow.is_mask]= True
    mesh.edge.mask[flow.uu_mask]= True
    mesh.vert.mask[flow.rv_mask]= True
    
    mesh.edge.open = np.asarray(
        np.argwhere(
    flow.is_open).ravel(), dtype=index_t)
    
    # set sparse spatial operators
    mats = operators(mesh)

    # remap fe,fc is more accurate?
    flow.ff_edge = mats.edge_tail_sums*flow.ff_vert
    flow.ff_edge/= mesh.edge.area
    
    flow.ff_cell = mats.cell_kite_sums*flow.ff_vert
    flow.ff_cell/= mesh.cell.area

    flow.ff_vert = np.asarray(
           flow.ff_vert, dtype=flt32_t)
    flow.ff_edge = np.asarray(
           flow.ff_edge, dtype=flt32_t)
    flow.ff_cell = np.asarray(
           flow.ff_cell, dtype=flt32_t)

    flow.ff_cell*= (not cnfg.no_rotate)
    flow.ff_edge*= (not cnfg.no_rotate)
    flow.ff_vert*= (not cnfg.no_rotate)
    
    ttoc = time.time()
    print("*FORM done (sec):", round(ttoc - ttic, 2))
   
    print("")
    print("Integrating the flow...")

    ttic = time.time(); next = 0; freq = 0;
    
    kp_sums = np.zeros(1 + cnfg.iteration 
        // max(+1, cnfg.stat_freq), dtype=reals_t)
    en_sums = np.zeros(1 + cnfg.iteration 
        // max(+1, cnfg.stat_freq), dtype=reals_t)
    
    init_pool(mesh)  # alloc. internal arrays

    hh_cell = variables.hh_cell
    hh_cell[:] = h0_cell
    hh_min_ = variables.hh_min_
    hh_min_[:] = h0_cell    
    hh_max_ = variables.hh_max_
    hh_max_[:] = h0_cell
    
    uu_edge = variables.uu_edge
    uu_edge[:] = u0_edge
    uu_min_ = variables.uu_min_
    uu_min_[:] = u0_edge    
    uu_max_ = variables.uu_max_
    uu_max_[:] = u0_edge    

    uu_edge[mesh.edge.mask] = 0.  # ensure BC
    uu_edge[mesh.edge.open] = \
            u0_edge [mesh.edge.open]
    
    mesh.edge.slip = mesh.edge.mask * \
                     flow.bc_slip
    mesh.edge.slip  [mesh.edge.open] = reals_t(1.)
    
    mesh.vert.slip = mats.dual_edge_sums * \
                     mesh.edge.slip
    mesh.vert.slip/= np.maximum(+1, 
                     mats.dual_edge_sums * \
                     mesh.edge.mask)
    
    mesh.cell.fmsk = reals_t(1.0 - mesh.cell.mask)
    mesh.edge.fmsk = reals_t(1.0 - mesh.edge.mask)
    mesh.vert.fmsk = reals_t(1.0 - mesh.vert.mask)
    
    cnfg.anylaw_cd = \
        max([cnfg.linlaw_cd, cnfg.sqrlaw_cd, 
             cnfg.loglaw_z0, cnfg.manlaw_n0] 
           )

    cnfg.du_visc_k = \
        max (cnfg.du_visc_2, cnfg.du_visc_4)
    cnfg.uu_visc_k = \
        max (cnfg.uu_visc_2, cnfg.uu_visc_4)
    cnfg.uu_visc_k = \
        max (cnfg.uu_visc_k, cnfg.leith_chi)
    cnfg.hh_diff_k = \
        max (cnfg.hh_diff_2, cnfg.hh_diff_4)
    
    s2_edge, s4_edge,\
    s2_cell, s4_cell = scalingVk(mesh, mats, cnfg)
    
    cnfg.leith_max = np.asarray(
        (cnfg.leith_max * s2_edge), dtype=reals_t)
    
    cnfg.du_visc_2 = np.asarray(
        (cnfg.du_visc_2 * s2_edge), dtype=reals_t)
    cnfg.du_visc_4 = np.asarray(
        (cnfg.du_visc_4 * s4_edge), dtype=reals_t)
    cnfg.uu_visc_2 = np.asarray(
        (cnfg.uu_visc_2 * s2_edge), dtype=reals_t)
    cnfg.uu_visc_4 = np.asarray(
        (cnfg.uu_visc_4 * s4_edge), dtype=reals_t)
    cnfg.hh_diff_2 = np.asarray(
        (cnfg.hh_diff_2 * s2_cell), dtype=reals_t)
    cnfg.hh_diff_4 = np.asarray(
        (cnfg.hh_diff_4 * s4_cell), dtype=reals_t)
   
    cnfg.du_visc_4 = np.sqrt(cnfg.du_visc_4)
    cnfg.uu_visc_4 = np.sqrt(cnfg.uu_visc_4)
    cnfg.hh_diff_4 = np.sqrt(cnfg.hh_diff_4)
    
    ch_cell = variables.ch_cell
    cu_edge = variables.cu_edge

    flow.prev = flow.next  # if forc. time-invariant...

    for step in range(0, cnfg.iteration + 1):

        tnow = cnfg.timeisnow

        if (step > 0):
        #-- 0-th step is just to write ICs to output...
            if (flow.xx_time is not None):
                # find needed forcing step to interp.
                need = np.searchsorted(
                    flow.xx_time, 
                        cnfg.timeisnow + cnfg.time_step)
                        
                need = min(need, flow.xx_time.size - 1)
                    
                if (need > flow.step): 
                # a piecewise linear interp. for now...
                    flow.prev = copy.deepcopy(flow.next)
                    flow = load_forc(forc, flow, step=need)
                    flow = sort_forc(flow, mesh)
                    
            hh_cell, uu_edge, \
            ch_cell, cu_edge = step_eqns(
                mesh, mats, flow, cnfg, hh_cell, uu_edge,
                                        ch_cell, cu_edge
            )
                      
            hh_min_, hh_max_, \
            uu_min_, uu_max_ = step_bnds(
                mesh, mats, flow, cnfg, hh_cell, uu_edge,
                                        hh_min_, hh_max_,
                                        uu_min_, uu_max_
            )
                
       #if (np.min(hh_min_) <= 0.0): 
       #    print("-ve layer thickness:", np.min(hh_min_) )
                 
        cnfg = mark_time(cnfg, flow, tnow + cnfg.time_step)
            
        if (cnfg.stat_freq > 0 and 
                step % cnfg.stat_freq == 0):
        #-- eval. statistics at every stat steps
            kp_sums[next], \
            en_sums[next] = invariant(
                mesh, mats, flow, cnfg, hh_cell, uu_edge
            )

            print ( 
                 "*STEP, d(K+P), d(Q^2):",
                f"{step:>12}",
                f"{rdf(kp_sums[next], kp_sums[+0]):>24}",
                f"{rdf(en_sums[next], en_sums[+0]):>24}"
            )

            next = next + 1

        if (cnfg.save_freq > 0 and 
                step % cnfg.save_freq == 0):
        #-- & save all state at every save steps
            save_step(save, mesh, mats,
                      flow, cnfg, freq, hh_cell, uu_edge
            )

            freq = freq + 1

    ttoc = time.time()

    print("")
    print("Run complete; runtime:")
    print("*wall-time (sec):", round(ttoc - ttic, 2))
    print("*file-i/o. (sec):", round(tcpu.filewrite, 2))
    print("*thickness (sec):", round(tcpu.thickness, 2))
    print("*momentum_ (sec):", round(tcpu.momentum_, 2))
    print("*computeBC (sec):", round(tcpu.computeBC, 2))
    print("*limiterWD (sec):", round(tcpu.limiterWD, 2))
    print("*upwinding (sec):", round(tcpu.upwinding, 2)) 
    print("*compute_H (sec):", round(tcpu.compute_H, 2))
    print("*advect_UH (sec):", round(tcpu.advect_UH, 2))
    print("*computeKE (sec):", round(tcpu.computeKE, 2))    
    print("*computePV (sec):", round(tcpu.computePV, 2))
    print("*advect_UV (sec):", round(tcpu.advect_UV, 2))
    print("*computeGZ (sec):", round(tcpu.computeGZ, 2))
    print("*computeVV (sec):", round(tcpu.computeVV, 2))
    print("*computeNu (sec):", round(tcpu.computeNu, 2))
    print("*computeDU (sec):", round(tcpu.computeDU, 2))
    print("*computeVU (sec):", round(tcpu.computeVU, 2))
    print("*computeVH (sec):", round(tcpu.computeVH, 2))
    print("*computeXI (sec):", round(tcpu.computeXI, 2))
    print("*computeTU (sec):", round(tcpu.computeTU, 2))
    print("*computeCd (sec):", round(tcpu.computeCd, 2))

    data = nc.Dataset(save, "a", format="NETCDF4")

    data.variables["kp_sums"][:] = kp_sums
    data.variables["en_sums"][:] = en_sums
    
    data.variables["hh_min_"][:] = \
               hh_min_[mesh.cell.irev - 1]
    data.variables["hh_max_"][:] = \
               hh_max_[mesh.cell.irev - 1]
               
    data.variables["uu_min_"][:] = \
               uu_min_[mesh.edge.irev - 1]
    data.variables["uu_max_"][:] = \
               uu_max_[mesh.edge.irev - 1]
    
    # xt variables are tmp scratch
    
    xt_dual = mats.dual_tail_sums * cnfg.du_visc_2
    xt_dual/= mesh.vert.area
    
    data.variables["d2_visc"][:] = \
               xt_dual[mesh.vert.irev - 1]
    
    xt_dual = mats.dual_tail_sums * cnfg.du_visc_4
    xt_dual/= mesh.vert.area
    
    data.variables["d4_visc"][:] = \
               xt_dual[mesh.vert.irev - 1] ** 2
    
    xt_dual = mats.dual_tail_sums * cnfg.uu_visc_2
    xt_dual/= mesh.vert.area
    
    data.variables["u2_visc"][:] = \
               xt_dual[mesh.vert.irev - 1]
    
    xt_dual = mats.dual_tail_sums * cnfg.uu_visc_4
    xt_dual/= mesh.vert.area
    
    data.variables["u4_visc"][:] = \
               xt_dual[mesh.vert.irev - 1] ** 2
    
    data.variables["h2_diff"][:] = \
        cnfg.hh_diff_2[mesh.cell.irev - 1]
    data.variables["h4_diff"][:] = \
        cnfg.hh_diff_4[mesh.cell.irev - 1] ** 2
    
    data.close()


def rdf(xval, yval):
#-- return relative change -- floor'd to zero near eps
    eps_ = np.finfo(reals_t).eps
    rdel = (xval - yval) / (yval + eps_)
    rdel*= (abs(rdel) >= 1 * eps_)
    return  rdel


if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

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

    parser.add_argument(
        "--time-step", dest="time_step", type=float,
        required=True, help="Length of time steps.")
        
    parser.add_argument(
        "--timestart", dest="timestart", type=float,
        default=0.0,
        required=False, 
        help="Time at simulation start {TIME = 0}.")

    parser.add_argument(
        "--num-steps", dest="iteration", type=int,
        required=True, help="Number of time steps.")

    parser.add_argument(
        "--integrate", dest="integrate", type=str,
        default="RK32-FB",
        required=False, 
        help="Time integration scheme = {RK32-FB}, " +
                                        "RK22-FB, ")

    parser.add_argument(
        "--sub-steps", dest="sub_steps", type=int,
        default=0,
        required=False, help="Number of fast steps; " + 
                            "for slow-fast integrators.")
                            
    parser.add_argument(
        "--save-freq", dest="save_freq", type=int,
        required=False, 
        default=np.iinfo(int).max, 
        help="Save output to file at each FREQ-th step.")

    parser.add_argument(
        "--stat-freq", dest="stat_freq", type=int,
        required=False, 
        default=np.iinfo(int).max, 
        help="Evaluate statistics at each FREQ-th step.")

    parser.add_argument(
        "--save-vars", dest="save_vars", type=str,
        default=
        "uu_edge,hh_cell,ke_cell,du_cell,rv_dual",
        required=False,
        help="Selected ouput variables to save to file.")

    parser.add_argument(
        "--equations", dest="equations", type=str,
        default="shallow-water",
        required=False,
        help="Eqn. selection = {shallow-water}.")
        
    parser.add_argument(
        "--numthread", dest="numthread", type=int,
        default=1,
        required=False,
        help="Number of parallel threads = {1}.")
        
    parser.add_argument(
        "--chunksize", dest="chunksize", type=int,
        default=4096,
        required=False,
        help="Stride of parallel decomp. = {4096}.")
        
    parser.add_argument(
        "--hh-min-up", dest="hh_min_up", type=float,
        default=0.0,
        required=False,
        help="Upwind HH.-edge bias {BIAS = +0./ 1.}.")

    parser.add_argument(
        "--hh-max-up", dest="hh_max_up", type=float,
        default=1.0,
        required=False,
        help="Upwind HH.-edge bias {BIAS = +1./ 1.}.")
        
    parser.add_argument(
        "--hh-scheme", dest="hh_scheme", type=str,
        default="CENTRE",
        required=False, 
        help="HH.-flux formulation = {UPWIND}, CENTRE.")

    parser.add_argument(
        "--pv-upwind", dest="pv_upwind", type=str,
        default="AUST-adapt",
        required=False, 
        help="Upstream formulation for PV = APVM, " + 
             "{AUST-adapt}, AUST-const.")

    parser.add_argument(
        "--pv-min-up", dest="pv_min_up", type=float,
        default=1./80.,
        required=False,
        help="Upwind PV.-flux bias {BIAS = +1./80.}.")

    parser.add_argument(
        "--pv-max-up", dest="pv_max_up", type=float,
        default=7./ 8.,
        required=False,
        help="Upwind PV.-flux bias {BIAS = +7./ 8.}.")
    
    parser.add_argument(
        "--pv-scheme", dest="pv_scheme", type=str,
        default="UPWIND",
        required=False, 
        help="PV.-flux formulation = {UPWIND}, CENTRE.")

    parser.add_argument(
        "--ke-upwind", dest="ke_upwind", type=str,
        default="AUST-const",
        required=False, 
        help="Upstream formulation for KE = APVM, " + 
             "{AUST-const}, AUST-adapt.")

    parser.add_argument(
        "--ke-min-up", dest="ke_min_up", type=float,
        default=1./80.,
        required=False,
        help="Upwind KE.-edge bias {BIAS = +1./80.}.")

    parser.add_argument(
        "--ke-max-up", dest="ke_max_up", type=float,
        default=1./ 8.,
        required=False,
        help="Upwind KE.-edge bias {BIAS = +1./ 8.}.")

    parser.add_argument(
        "--ke-scheme", dest="ke_scheme", type=str,
        default="CENTRE",
        required=False, 
        help="KE.-grad formulation = {CENTRE}, SKINNY.")

    parser.add_argument(
        "--ref-scale", dest="ref_scale", type=float,
        default=30.E+3,
        required=False,
        help="Ref-len. for visc. scales {DX = 30.E+03}.")

    parser.add_argument(
        "--msh-fixes", dest="msh_fixes", type=float,
        default=1.E+00,
        required=False,
        help="Mesh quality visc. scales {MF = +1.E+00}.")
        
    parser.add_argument(
        "--hh-diff-2", dest="hh_diff_2", type=float,
        default=0.E+00,
        required=False,
        help="DEL^2(H) damping coeff. {DIFF = +0.E+00}.")

    parser.add_argument(
        "--hh-diff-4", dest="hh_diff_4", type=float,
        default=0.E+00,
        required=False,
        help="DEL^4(H) damping coeff. {DIFF = +0.E+00}.")

    parser.add_argument(
        "--du-visc-2", dest="du_visc_2", type=float,
        default=0.E+00,
        required=False,
        help="DIV^2(U) damping coeff. {VISC = +0.E+00}.")

    parser.add_argument(
        "--du-visc-4", dest="du_visc_4", type=float,
        default=0.E+00,
        required=False,
        help="DIV^4(U) damping coeff. {VISC = +0.E+00}.")

    parser.add_argument(
        "--uu-visc-2", dest="uu_visc_2", type=float,
        default=0.E+00,
        required=False,
        help="DEL^2(U) damping coeff. {VISC = +0.E+00}.")

    parser.add_argument(
        "--uu-visc-4", dest="uu_visc_4", type=float,
        default=0.E+00,
        required=False,
        help="DEL^4(U) damping coeff. {VISC = +0.E+00}.")

    parser.add_argument(
        "--leith-chi", dest="leith_chi", type=float,
        default=0.E+00,
        required=False,
        help="Leith model coefficient {SCAL = +0.E+00}.")
        
    parser.add_argument(
        "--leith-max", dest="leith_max", type=float,
        default=0.E+00,
        required=False,
        help="Leith model max damping {VISC = +0.E+00}.")

    parser.add_argument(
        "--linlaw-cd", dest="linlaw_cd", type=float,
        default=0.E+00,
        required=False,
        help="Linear-law Cd coefficient {Cd = +0.E+00}.")
        
    parser.add_argument(
        "--sqrlaw-cd", dest="sqrlaw_cd", type=float,
        default=0.E+00,
        required=False,
        help="Square-law Cd coefficient {Cd = +0.E+00}.")

    parser.add_argument(
        "--loglaw-z0", dest="loglaw_z0", type=float,
        default=0.E+00,
        required=False,
        help="Log-law roughness lengths {Z0 = +0.E+00}.")

    parser.add_argument(
        "--loglaw-lo", dest="loglaw_lo", type=float,
        default=0.E+00,
        required=False,
        help="Log-law minimum Cd coeff. {Cd > +0.E+00}.")

    parser.add_argument(
        "--loglaw-hi", dest="loglaw_hi", type=float,
        default=0.E+00,
        required=False,
        help="Log-law maximum Cd coeff. {Cd < +0.E+00}.")
        
    parser.add_argument(
        "--manlaw-n0", dest="manlaw_n0", type=float,
        default=0.E+00,
        required=False,
        help="Manning roughness lengths {N0 = +0.E+00}.")

    parser.add_argument(
        "--manlaw-lo", dest="manlaw_lo", type=float,
        default=0.E+00,
        required=False,
        help="Manning minimum Cd coeff. {Cd > +0.E+00}.")

    parser.add_argument(
        "--manlaw-hi", dest="manlaw_hi", type=float,
        default=0.E+00,
        required=False,
        help="Manning maximum Cd coeff. {Cd < +0.E+00}.")
    
    parser.add_argument(
        "--wetdry-h0", dest="wetdry_h0", type=float,
        default=0.E+00,
        required=False,
        help="Thickness of wet-dry limiter {1.E-04}.")

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
        "--FB-weight", dest="fb_weight", type=float,
        required=False,
        nargs="*",
        help="Forward-backward weights for integrators.")

    swe(parser.parse_args())
    
