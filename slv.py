
import time
import copy
import numpy as np

""" SLV: solve the nonlinear SWE on generalised MPAS meshes.
"""
#-- Part of the PERISCOPE solver
#-- Darren Engwirda
#-- d.engwirda@gmail.com
#-- https://github.com/dengwirda/

import jax
jax.config.update("jax_enable_x64", True)

from _fp import flt32_t, flt64_t
from _fp import reals_t, index_t
from _fp import udata_t, hdata_t, qdata_t

from _jx import all_to_jax

from log import tcpu

from msh import load_mesh, sort_mesh, \
                load_flow, sort_flow, \
                load_forc, sort_forc, \
                init_wall, init_obcs
from ops import operators

from io_ import init_file, save_step, save_last

from _dt import step_eqns
from _dx import invariant, scale_mix

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

    cnfg.calc_tide = True
    cnfg.calc_slow = True
    cnfg.calc_fast = True
    cnfg.calc_drag = True
    
    cnfg.timeisnow = cnfg.timestart
    cnfg.stat_step = +0.0
    cnfg.stat_prev = +0
    cnfg.save_step = +0.0
    cnfg.save_prev = +0

    cnfg.completed = False

    # mesh, forcing & solution i/o 
    name = cnfg.mesh_file
    forc = cnfg.forc_file
    save = cnfg.soln_file
    
    print("Loading input assets...")
    
    ttic = time.time()

    # load mesh + init. conditions
    mesh = load_mesh(name)
    flow = load_flow(name, mesh, lean=True)
    flow = load_forc(forc, flow, lean=True)

    # offset, if ICs are a restart
    cnfg.timestart+= flow.elapsed
    cnfg.timeisnow+= flow.elapsed
    
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
    
    flow.hh_cell = \
        np.maximum(cnfg.wetdry_h0 / 2., flow.hh_cell)

    ttoc = time.time()
    print("*SORT done (sec):", round(ttoc - ttic, 2))

    print("")
    print("Forming coefficients...")

    ttic = time.time()

    # set basic wall masks + lists
    mesh = init_wall(mesh, flow)

    # set sparse spatial operators
    mats = operators(mesh)

    # set domain boundary stencils
    mesh = init_obcs(mesh, flow, mats)

    ttoc = time.time()
    print("*FORM done (sec):", round(ttoc - ttic, 2))
   
    print("")
    print("Integrating the flow...")

    kp_sum_ = []; en_sum_ = [];
    
    uu_edge = flow.uu_edge
    uu_edge[mesh.edge.mask] = 0.  # ensure BC
    uu_edge[mesh.edge.open] =flow.uu_edge[mesh.edge.open]
    flow.uu_edge = uu_edge
    
    # start forward integrations
    flow, cnfg = pre (mesh, mats, flow, cnfg)

    # do host-to-device transfer
    mesh, mats, flow, cnfg = all_to_jax(
                      mesh, mats, flow, cnfg)

    """
    # uncomment to inspect compiled XLA
    from _jo import op_product, pv_product

    # linear kernel
    f = jax.jit(op_product)
    compiled = f.lower(mats.jx.cell.flux_sums, uu_edge).compile()
    print(compiled.as_text())

    # pvflux kernel
    p1_edge = uu_edge + 1.
    p2_edge = uu_edge - 1.

    g = jax.jit(pv_product)
    compiled = g.lower(mats.jx.edge.flux_perp, uu_edge, 
                                      p1_edge, p2_edge).compile()
    print(compiled.as_text())

    # full timestep
    h = jax.jit(step_eqns, static_argnums=(4,))
    compiled = h.lower(mesh.jx, mats.jx, 
                       flow.jx, cnfg.jx, cnfg.iteration).compile()

    with open("step_eqns_xla_up.txt", "w") as f:
        f.write(compiled.as_text())

    raise Exception()
    """


    save_step(save,
        mesh.jx, mats.jx, flow.jx, cnfg.jx, step=0)


    ttic = time.time(); next = +0; freq = +0
    """
    flow.prev = flow.next  # if forc. time-invariant...

    cnfg = init_step (mesh, mats, flow, cnfg, 
                      hh_cell, uu_edge, 
                      qq_cell)

    cnfg.time_stop+= cnfg.timeisnow * (cnfg.time_stop>0.)
    cnfg.stat_next = cnfg.timeisnow
    cnfg.save_next = cnfg.timeisnow
    """

    """
    # device-to-host copy seems quite slow!
    save_step(save, 
        mesh.jx, mats.jx, flow.jx, cnfg.jx, step=0)
    """


    """
    # uncomment to produce profile trace
    opts = jax.profiler.ProfileOptions()
    opts.gpu_enable_cupti_activity_graph_trace = True
    opts.gpu_dump_graph_node_mapping = True

    with jax.profiler.trace("/tmp/jpx_trace",
                            create_perfetto_trace=True,
                            profiler_options=opts):
    """
    # main time-stepping loop
    flow.jx, cnfg.jx = step_eqns(
        mesh.jx, mats.jx, flow.jx, cnfg.jx, cnfg.iteration)
        
    jax.block_until_ready(flow.jx)

    ttoc = time.time()


    save_step(save, 
        mesh.jx, mats.jx, flow.jx, cnfg.jx, step=1)

    kp_sums, pv_sums = invariant(
        mesh.jx, mats.jx, flow.jx, cnfg.jx)

    print(np.float64(kp_sums), np.float64(pv_sums))



    #WIP hacky device-to-host
    hh_cell = np.asarray(flow.jx.prognostic.hh_cell, dtype=hdata_t)
    uu_edge = np.asarray(flow.jx.prognostic.uu_edge, dtype=udata_t)

    print(np.min(hh_cell), np.max(hh_cell))
    print(np.min(uu_edge), np.max(uu_edge))


    
    """
    save_last(save, mesh, mats, flow, cnfg, step, 
              kp_sum_, en_sum_,
              hh_min_, hh_max_,
              uu_min_, uu_max_,
              qq_min_, qq_max_,
                       zt_rms_,
              ke_ave_, ke_rms_, ke_max_,
              dk_ave_, dk_rms_, dk_max_)
    """

    print("")
    print("Run complete; runtime:")
    print("*wall-time (sec):", round(ttoc - ttic, 2))
    print("*file-i/o. (sec):", round(tcpu.filewrite, 2))
    """
    print("*evaluate_ (sec):", round(tcpu.evaluate_, 2))    
    print("*thickness (sec):", round(tcpu.thickness, 2))
    print("*momentum_ (sec):", round(tcpu.momentum_, 2))
    print("*finalise_ (sec):", round(tcpu.finalise_, 2))
    print("*calc-obcs (sec):", round(tcpu.calc_obcs, 2))
    print("*calc-udry (sec):", round(tcpu.calc_udry, 2))
    print("*upwinding (sec):", round(tcpu.upwinding, 2))
    print("*calc-hmap (sec):", round(tcpu.calc_hmap, 2))
    print("*tend-hadv (sec):", round(tcpu.tend_hadv, 2))
    print("*calc-perp (sec):", round(tcpu.calc_perp, 2))
    print("*calc-u-ke (sec):", round(tcpu.calc_u_ke, 2))
    print("*calc-u-pv (sec):", round(tcpu.calc_u_pv, 2))
    print("*tend-uadv (sec):", round(tcpu.tend_uadv, 2))
    print("*tend-upgf (sec):", round(tcpu.tend_upgf, 2))
    print("*calc-umix (sec):", round(tcpu.calc_umix, 2))
    print("*calc-uwav (sec):", round(tcpu.calc_uwav, 2))
    print("*tend-umix (sec):", round(tcpu.tend_umix, 2))
    print("*calc-hmix (sec):", round(tcpu.calc_hmix, 2))    
    print("*tend-hmix (sec):", round(tcpu.tend_hmix, 2))
    print("*calc-tide (sec):", round(tcpu.calc_tide, 2))
    print("*calc-sal_ (sec):", round(tcpu.calc_self, 2))
    print("*tend-ugeo (sec):", round(tcpu.tend_ugeo, 2))
    print("*tend-utau (sec):", round(tcpu.tend_utau, 2))
    print("*calc-drag (sec):", round(tcpu.calc_drag, 2))
    """


def out(done, freq, mark, step, time):
#-- return TRUE if an output step has been reached
    if (freq > 0 and (isinstance(freq, index_t) 
        and step % freq == 0)): return True

    if (freq > 0 and (isinstance(freq, flt64_t) 
        and time - mark >= 0)): return True

    return  done


def rdf(xval, yval):
#-- return relative change -- floor'd to zero near eps
    eps_ = np.finfo(reals_t).eps
    rdel = (xval - yval) / (yval + eps_)
    return  rdel * (abs (rdel) >= +1 * eps_)


def pre(mesh, mats, flow, cnfg):
#-- do various init. ops for flow + config. at pre-run

    # remap coriolis onto msh DoFs
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

    cnfg.ff_max_ = np.max(np.abs(flow.ff_edge))

    flow.h0_rms_ = \
        np.sqrt(np.mean(flow.hh_cell ** 2))
    flow.u0_rms_ = \
        np.sqrt(np.mean(flow.uu_edge ** 2))
    flow.p0_rms_ = \
        np.sqrt(np.mean(flow.ff_cell ** 2))

    flow.c0_rms_ = flow.u0_rms_ + \
        np.sqrt (flow.gravity * flow.h0_rms_)

    flow.hh_tiny = 100. * \
        np.finfo(flt64_t).eps * flow.h0_rms_
    flow.uu_tiny = 100. * \
        np.finfo(flt64_t).eps * flow.c0_rms_
    flow.pv_tiny = 100. * \
        np.finfo(flt64_t).eps * flow.p0_rms_
    flow.pv_tiny+= flow.uu_tiny

    # const. scaling on drag param.
    cnfg.anylaw_cd = \
        max([cnfg.linlaw_cd, cnfg.sqrlaw_cd, 
             cnfg.loglaw_z0, cnfg.manlaw_n0
           ] )
    
    flow.c1_edge = flow.c1_edge * cnfg.linlaw_cd
    flow.c2_edge = flow.c2_edge * cnfg.sqrlaw_cd
    flow.z0_edge = flow.z0_edge * cnfg.loglaw_z0
    flow.n0_edge = flow.n0_edge * cnfg.manlaw_n0

    # subgrid drag thickness scale
    flow.dz_drag = np.asarray (
        mats.edge_wing_sums * (
        np.maximum(0., flow.zb_drag-flow.zb_cell
        ) ), dtype=flt32_t)
    flow.dz_drag /= mesh.edge.area

    # mesh scaling for dissipation
    cnfg.uu_visc_k = \
        max (cnfg.uu_visc_2, cnfg.uu_visc_4)
    cnfg.uu_visc_k = \
        max (cnfg.uu_visc_k, cnfg.leith_chi)
    cnfg.uu_visc_k = \
        max (cnfg.uu_visc_k, cnfg.waves_chi)
    cnfg.uu_visc_k = \
        max (cnfg.uu_visc_k, cnfg.wetdry_h0)

    cnfg.hh_diff_k = \
        max (cnfg.hh_diff_2, cnfg.hh_diff_4)
    cnfg.hh_diff_k = \
        max (cnfg.hh_diff_k, cnfg.shock_chi)

    s2_edge, s4_edge, msh_fix = \
        scale_mix(mesh, mats, cnfg)

    flow.msh_fix = msh_fix
    flow.msh_nu2 = s2_edge
    flow.msh_nu4 = s4_edge

    flow.visc_u2 = np.asarray(
        (cnfg.uu_visc_2*s2_edge), dtype=reals_t)
    flow.visc_u4 = np.asarray(
        (cnfg.uu_visc_4*s4_edge), dtype=reals_t)

    flow.diff_h2 = np.asarray(
        (cnfg.hh_diff_2*s2_edge), dtype=reals_t)
    flow.diff_h4 = np.asarray(
        (cnfg.hh_diff_4*s4_edge), dtype=reals_t)
   
    flow.diff_h4 = np.sqrt(flow.diff_h4)

    return flow, cnfg

