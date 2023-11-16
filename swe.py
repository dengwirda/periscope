
import os
import time
import numpy as np
import netCDF4 as nc
import argparse

""" SWE: solve the nonlinear SWE on generalised MPAS meshes.
"""
#-- Authors: Darren Engwirda

from stb import strtobool

from _fp import flt32_t, flt64_t
from _fp import reals_t, index_t

from log import tcpu

from msh import load_mesh, load_flow, \
                sort_mesh, sort_flow
from ops import trsk_mats
from mem import init_pool

from io_ import init_file, save_step

from _dx import HH_TINY, invariant, scalingVk
from _dt import step_eqns

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
    
    cnfg.save_freq = min(
        cnfg.iteration, cnfg.save_freq)
    cnfg.stat_freq = min(
        cnfg.save_freq, cnfg.stat_freq)
    
    cnfg.calc_slow = True
    cnfg.calc_fast = True
    cnfg.calc_drag = True
    
    cnfg.integrate = cnfg.integrate.upper()
    cnfg.equations = cnfg.equations.upper()
    cnfg.hh_scheme = cnfg.hh_scheme.upper()
    cnfg.ke_upwind = cnfg.ke_upwind.upper()
    cnfg.ke_scheme = cnfg.ke_scheme.upper()
    cnfg.pv_upwind = cnfg.pv_upwind.upper()
    cnfg.pv_scheme = cnfg.pv_scheme.upper()
    
    if ("CENTRE" in cnfg.ke_scheme): 
        cnfg.ke_upwind = "NONE"
    if ("CENTRE" in cnfg.pv_scheme): 
        cnfg.pv_upwind = "NONE"
    
    cnfg.anylaw_cd = max([
        cnfg.linlaw_cd, cnfg.sqrlaw_cd, cnfg.loglaw_z0
        ] )
    
    name = cnfg.mesh_file
    path, file = os.path.split(name)
    save = os.path.join(path, "out_" + file)

    print("Loading input assets...")
    
    ttic = time.time()

    # load mesh + init. conditions
    mesh = load_mesh(name)
    flow = load_flow(name, mesh, lean=True)

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

    u0_edge = flow.uu_edge[-1, :, 0]
    uu_edge = u0_edge
    
    h0_cell = flow.hh_cell[-1, :, 0]
    hh_cell = h0_cell
    
    hh_cell = np.maximum(HH_TINY, hh_cell)
    
    ttoc = time.time()
    print("*SORT done (sec):", round(ttoc - ttic, 2))

    print("")
    print("Forming coefficients...")

    ttic = time.time()

    # set sparse spatial operators
    trsk = trsk_mats(mesh)

    # remap fe,fc is more accurate?
    flow.ff_edge = trsk.edge_tail_sums*flow.ff_vert
    flow.ff_edge = \
        (flow.ff_edge / mesh.edge.area)

    flow.ff_cell = trsk.cell_kite_sums*flow.ff_vert
    flow.ff_cell = \
        (flow.ff_cell / mesh.cell.area)

    flow.ff_cell*= (not cnfg.no_rotate)
    flow.ff_edge*= (not cnfg.no_rotate)
    flow.ff_vert*= (not cnfg.no_rotate)
    
    flow.zb_cell = np.asarray(
        flow.zb_cell, dtype=flt32_t)

    flow.ff_cell = np.asarray(
        flow.ff_cell, dtype=flt32_t)
    flow.ff_edge = np.asarray(
        flow.ff_edge, dtype=flt32_t)
    flow.ff_vert = np.asarray(
        flow.ff_vert, dtype=flt32_t)
    
    # always round IC's to flt32_t
    hh_cell = np.ascontiguousarray(
             hh_cell, dtype=flt32_t)
    uu_edge = np.ascontiguousarray(
             uu_edge, dtype=flt32_t)
    
    hh_cell = np.ascontiguousarray(
             hh_cell, dtype=reals_t)
    uu_edge = np.ascontiguousarray(
             uu_edge, dtype=reals_t)
    
    kp_sums = np.zeros((cnfg.iteration 
            // cnfg.stat_freq + 1), dtype=reals_t)
    en_sums = np.zeros((cnfg.iteration 
            // cnfg.stat_freq + 1), dtype=reals_t)

    ttoc = time.time()
    print("*FORM done (sec):", round(ttoc - ttic, 2))
   
    print("")
    print("Integrating the flow...")

    ttic = time.time(); next = 0; freq = 0
    
    init_pool(mesh)  # alloc. internal arrays
    
    mesh.cell.mask = flow.is_mask
    mesh.cell.fmsk = reals_t(1.0 - mesh.cell.mask)
    mesh.edge.mask = flow.uu_mask
    mesh.edge.mask[np.logical_or.reduce((
        mesh.edge.cell[:, 0] < 1,
        mesh.edge.cell[:, 1] < 1))] = True
    mesh.edge.fmsk = reals_t(1.0 - mesh.edge.mask)
    mesh.vert.mask = flow.rv_mask
    mesh.vert.mask[np.logical_or.reduce((
        mesh.vert.cell[:, 0] < 1,
        mesh.vert.cell[:, 1] < 1,
        mesh.vert.cell[:, 2] < 1))] = True
    mesh.vert.fmsk = reals_t(1.0 - mesh.vert.mask)
    
    uu_edge[mesh.edge.mask] = 0.  # ensure BC
    
    cnfg.du_damp_k = \
        max (cnfg.du_damp_2, cnfg.du_damp_4)
    cnfg.vu_damp_k = \
        max (cnfg.vu_damp_2, cnfg.vu_damp_4)
    cnfg.vh_damp_k = \
        max (cnfg.vh_damp_2, cnfg.vh_damp_4)
    
    s2_edge, s4_edge,\
    s2_cell, s4_cell = scalingVk(mesh, trsk, cnfg)
    
    cnfg.du_damp_2 = np.asarray(
        (cnfg.du_damp_2 * s2_edge), dtype=reals_t)
    cnfg.du_damp_4 = np.asarray(
        (cnfg.du_damp_4 * s4_edge), dtype=reals_t)
    cnfg.vu_damp_2 = np.asarray(
        (cnfg.vu_damp_2 * s2_edge), dtype=reals_t)
    cnfg.vu_damp_4 = np.asarray(
        (cnfg.vu_damp_4 * s4_edge), dtype=reals_t)
    cnfg.vh_damp_2 = np.asarray(
        (cnfg.vh_damp_2 * s2_cell), dtype=reals_t)
    cnfg.vh_damp_4 = np.asarray(
        (cnfg.vh_damp_4 * s4_cell), dtype=reals_t)
    
    cnfg.du_damp_4 = np.sqrt(cnfg.du_damp_4)
    cnfg.vu_damp_4 = np.sqrt(cnfg.vu_damp_4)
    cnfg.vh_damp_4 = np.sqrt(cnfg.vh_damp_4)
    
    cu_edge = uu_edge * 0.; ch_cell = hh_cell * 0.

    for step in range(0, cnfg.iteration + 1):

        if (step > 0):
        #-- 0-th step is just to write ICs to output...
             hh_cell, uu_edge, \
             ch_cell, cu_edge = step_eqns(
                 mesh, trsk, flow, cnfg, hh_cell, uu_edge,
                                         ch_cell, cu_edge)

        if (step % cnfg.stat_freq == 0):
        #-- eval. statistics on stat steps
            kp_sums[next], \
            en_sums[next] = invariant(
                mesh, trsk, flow, cnfg, hh_cell, uu_edge)

            print("*STEP, d(K+P), d(Q^2):",
                 f"{step:>12}",
                 f"{rdf(kp_sums[next], kp_sums[+0]):>24}",
                 f"{rdf(en_sums[next], en_sums[+0]):>24}"
            )

            next = next + 1

        if (step % cnfg.save_freq == 0):
        #-- & save all state on save steps
            save_step(save, mesh, trsk,
                      flow, cnfg, freq, hh_cell, uu_edge)

            freq = freq + 1

    ttoc = time.time()

    print("")
    print("Timing data:")
    print("*wall-time (sec):", round(ttoc - ttic, 2))
    print("*file-i/o. (sec):", round(tcpu.filewrite, 2))
    print("*thickness (sec):", round(tcpu.thickness, 2))
    print("*momentum_ (sec):", round(tcpu.momentum_, 2))
    print("*upwinding (sec):", round(tcpu.upwinding, 2)) 
    print("*compute_H (sec):", round(tcpu.compute_H, 2))
    print("*advect_UH (sec):", round(tcpu.advect_UH, 2))
    print("*computeKE (sec):", round(tcpu.computeKE, 2))    
    print("*computePV (sec):", round(tcpu.computePV, 2))
    print("*advect_UV (sec):", round(tcpu.advect_UV, 2))
    print("*computeGZ (sec):", round(tcpu.computeGZ, 2))
    print("*computeVV (sec):", round(tcpu.computeVV, 2))
    print("*computeDU (sec):", round(tcpu.computeDU, 2))
    print("*computeVU (sec):", round(tcpu.computeVU, 2))
    print("*computeVH (sec):", round(tcpu.computeVH, 2))
    print("*computeCd (sec):", round(tcpu.computeCd, 2))

    data = nc.Dataset(save, "a", format="NETCDF4")

    data.variables["kp_sums"][:] = kp_sums
    data.variables["en_sums"][:] = en_sums
    
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
        "--srcs-file", dest="srcs_file", type=str,
        required=False, 
        help="Path to user FORCING tendencies file.")

    parser.add_argument(
        "--time-step", dest="time_step", type=float,
        required=True, help="Length of time steps.")

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
        "--wall-slip", dest="wall_slip", type=float,
        default=0.0,
        required=False,
        help="No-slip/free-slip factor = {+0}, +1.")
        
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
        default=1./40.,
        required=False,
        help="Upwind PV.-flux bias {BIAS = +1./40.}.")

    parser.add_argument(
        "--pv-max-up", dest="pv_max_up", type=float,
        default=3./ 4.,
        required=False,
        help="Upwind PV.-flux bias {BIAS = +3./ 4.}.")
    
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
        help="KE.-grad formulation = {CENTRE}, " +
                                     "CENTRE+SKINNY, " +
                                     "UPWIND, " + 
                                     "UPWIND+SKINNY.")

    parser.add_argument(
        "--hh-expect", dest="hh_expect", type=float,
        default=1.E+03,
        required=False,
        help="Ref. HH. magnitude {|HH| = +1.E+03}.")
        
    parser.add_argument(
        "--uu-expect", dest="uu_expect", type=float,
        default=1.E+00,
        required=False,
        help="Ref. UU. magnitude {|UU| = +1.E+00}.")

    parser.add_argument(
        "--dx-damp-r", dest="dx_damp_r", type=float,
        default=30.E+3,
        required=False,
        help="Ref-len. for viscosity scaling {DX = 30E+3}.")
        
    parser.add_argument(
        "--vh-damp-2", dest="vh_damp_2", type=float,
        default=0.E+00,
        required=False,
        help="DEL^2(H) damping coeff. {DAMP = +0.E+00}.")

    parser.add_argument(
        "--vh-damp-4", dest="vh_damp_4", type=float,
        default=0.E+00,
        required=False,
        help="DEL^4(H) damping coeff. {DAMP = +0.E+00}.")

    parser.add_argument(
        "--du-damp-2", dest="du_damp_2", type=float,
        default=0.E+00,
        required=False,
        help="DIV^2(U) damping coeff. {DAMP = +0.E+00}.")

    parser.add_argument(
        "--du-damp-4", dest="du_damp_4", type=float,
        default=0.E+00,
        required=False,
        help="DIV^4(U) damping coeff. {DAMP = +0.E+00}.")

    parser.add_argument(
        "--vu-damp-2", dest="vu_damp_2", type=float,
        default=0.E+00,
        required=False,
        help="DEL^2(U) damping coeff. {DAMP = +0.E+00}.")

    parser.add_argument(
        "--vu-damp-4", dest="vu_damp_4", type=float,
        default=0.E+00,
        required=False,
        help="DEL^4(U) damping coeff. {DAMP = +0.E+00}.")
        
    parser.add_argument(
        "--vu-du-mul", dest="vu_du_mul", type=float,
        default=1.E+00,
        required=False,
        help="DEL^k(U) div. amplifier {MUL. = +1.E+00}.")

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
    
    
