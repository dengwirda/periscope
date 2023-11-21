
import time
import numpy as np
import netCDF4 as nc

""" NETCDF-4 output for SWE-solver; MPAS-style variables  
"""
#-- Darren Engwirda

from _fp import flt32_t, flt64_t
from _fp import reals_t, index_t

from log import tcpu

from _dx import diag_vars

class base: pass
save = base()
save.uu_edge = False  # True to write to file
save.vv_edge = False
save.hh_cell = False
save.hh_edge = False
save.hh_dual = False
save.du_cell = False
save.uh_cell = False
save.ke_bias = False
save.ke_cell = False
save.pv_bias = False
save.pv_dual = False
save.rv_dual = False
save.pv_cell = False
save.rv_cell = False

def save_step(save, mesh, trsk, flow, cnfg, step, hh_cell, uu_edge):

    hh_edge, hh_dual, ke_cell, ke_bias, \
    rv_cell, pv_cell, rv_dual, pv_dual, \
    pv_edge, pv_bias, vv_edge = diag_vars (
        mesh, trsk, flow, cnfg, hh_cell, uu_edge
        )

    ttic = time.time()

    data = nc.Dataset(save, "a", format="NETCDF4")

    # xt variables are tmp scratch

    if (save.uu_edge):
        data.variables["uu_edge"][step, :, :] = \
            np.reshape(uu_edge[
                mesh.edge.irev - 1], (1, mesh.edge.size, 1))
                
    if (save.vv_edge):
        data.variables["vv_edge"][step, :, :] = \
            np.reshape(vv_edge[
                mesh.edge.irev - 1], (1, mesh.edge.size, 1))
                
    if (save.hh_cell):         
        data.variables["hh_cell"][step, :, :] = \
            np.reshape(hh_cell[
                mesh.cell.irev - 1], (1, mesh.cell.size, 1))
                
    if (save.hh_edge):         
        data.variables["hh_edge"][step, :, :] = \
            np.reshape(hh_edge[
                mesh.edge.irev - 1], (1, mesh.edge.size, 1))
                
    if (save.hh_dual):         
        data.variables["hh_dual"][step, :, :] = \
            np.reshape(hh_dual[
                mesh.vert.irev - 1], (1, mesh.vert.size, 1))

    if (save.du_cell):
        xt_cell = trsk.cell_flux_sums * uu_edge
        xt_cell/= mesh.cell.area

        data.variables["du_cell"][step, :, :] = \
            np.reshape(xt_cell[
                mesh.cell.irev - 1], (1, mesh.cell.size, 1))

    if (save.uh_cell):
        xt_edge = uu_edge * hh_edge
        xt_cell = trsk.cell_flux_sums * xt_edge
        xt_cell/= mesh.cell.area

        data.variables["uh_cell"][step, :, :] = \
            np.reshape(xt_cell[
                mesh.cell.irev - 1], (1, mesh.cell.size, 1))

    if (save.ke_cell):
        data.variables["ke_cell"][step, :, :] = \
            np.reshape(ke_cell[
                mesh.cell.irev - 1], (1, mesh.cell.size, 1)) 

    if (save.pv_bias)
        xt_dual = trsk.dual_tail_sums * pv_bias
        xt_dual/= mesh.vert.area

        data.variables["pv_bias"][step, :, :] = \
            np.reshape(xt_dual[
                mesh.vert.irev - 1], (1, mesh.vert.size, 1))
    
    if (save.pv_dual):        
        data.variables["pv_dual"][step, :, :] = \
            np.reshape(pv_dual[
                mesh.vert.irev - 1], (1, mesh.vert.size, 1))
    
    if (save.rv_dual):
        data.variables["rv_dual"][step, :, :] = \
            np.reshape(rv_dual[
                mesh.vert.irev - 1], (1, mesh.vert.size, 1))
                
    if (save.pv_cell):        
        data.variables["pv_cell"][step, :, :] = \
            np.reshape(pv_cell[
                mesh.cell.irev - 1], (1, mesh.cell.size, 1))
    
    if (save.rv_cell):
        data.variables["rv_cell"][step, :, :] = \
            np.reshape(rv_cell[
                mesh.cell.irev - 1], (1, mesh.cell.size, 1))

    data.close()
    
    ttoc = time.time()
    tcpu.filewrite = tcpu.filewrite + (ttoc - ttic)
    
    
def init_file(name, cnfg, save, mesh, flow):

    ttic = time.time()
    
    data = nc.Dataset(save, "w", format="NETCDF4")
    data.on_a_sphere = "YES"
    data.sphere_radius = mesh.rsph
    data.config_gravity = flow.grav
    data.is_periodic = "NO"
    data.source = "PERISCOPE"
    
    for attr in vars(cnfg):  # add user-opts to output
        vals = getattr(cnfg, attr)
        if (vals is None): continue
        if (type(vals) == list): continue
        if (type(vals) == bool): vals = int(vals)
        setattr (data, attr, vals)

    data.createDimension("Time", None)
    data.createDimension("Step", None)
    data.createDimension("TWO", 2)
    data.createDimension("nCells", mesh.cell.size)
    data.createDimension("nEdges", mesh.edge.size)
    data.createDimension("nVertices", mesh.vert.size)
    data.createDimension("nVertLevels", 1)
    data.createDimension("maxEdges", np.max(mesh.cell.topo) * 1)
    data.createDimension("maxEdges2", np.max(mesh.cell.topo) * 2)
    data.createDimension("vertexDegree", 3)

    data.createVariable("lonCell", "f8", ("nCells"))
    data["lonCell"][:] = mesh.cell.xlon
    data.createVariable("latCell", "f8", ("nCells"))
    data["latCell"][:] = mesh.cell.ylat
    data.createVariable("xCell", "f8", ("nCells"))
    data["xCell"][:] = mesh.cell.xpos
    data.createVariable("yCell", "f8", ("nCells"))
    data["yCell"][:] = mesh.cell.ypos
    data.createVariable("zCell", "f8", ("nCells"))
    data["zCell"][:] = mesh.cell.zpos
    data.createVariable("areaCell", "f8", ("nCells"))
    data["areaCell"][:] = mesh.cell.area
    data.createVariable(
        "verticesOnCell", "i4", ("nCells", "maxEdges"))
    data["verticesOnCell"][:, :] = mesh.cell.vert
    data.createVariable(
        "edgesOnCell", "i4", ("nCells", "maxEdges"))
    data["edgesOnCell"][:, :] = mesh.cell.edge
    data.createVariable(
        "cellsOnCell", "i4", ("nCells", "maxEdges"))
    data["cellsOnCell"][:, :] = mesh.cell.cell
    data.createVariable("nEdgesOnCell", "i4", ("nCells"))
    data["nEdgesOnCell"][:] = mesh.cell.topo

    data.createVariable("lonEdge", "f8", ("nEdges"))
    data["lonEdge"][:] = mesh.edge.xlon
    data.createVariable("latEdge", "f8", ("nEdges"))
    data["latEdge"][:] = mesh.edge.ylat
    data.createVariable("xEdge", "f8", ("nEdges"))
    data["xEdge"][:] = mesh.edge.xpos
    data.createVariable("yEdge", "f8", ("nEdges"))
    data["yEdge"][:] = mesh.edge.ypos
    data.createVariable("zEdge", "f8", ("nEdges"))
    data["zEdge"][:] = mesh.edge.zpos
    data.createVariable("dvEdge", "f8", ("nEdges"))
    data["dvEdge"][:] = mesh.edge.vlen
    data.createVariable("dcEdge", "f8", ("nEdges"))
    data["dcEdge"][:] = mesh.edge.clen
    data.createVariable("angleEdge", "f8", ("nEdges"))
    data["angleEdge"][:] = mesh.edge.beta
    data.createVariable(
        "verticesOnEdge", "i4", ("nEdges", "TWO"))
    data["verticesOnEdge"][:, :] = mesh.edge.vert
    data.createVariable(
        "weightsOnEdge", "f8", ("nEdges", "maxEdges2"))
    data["weightsOnEdge"][:, :] = mesh.edge.wmul
    data.createVariable(
        "cellsOnEdge", "i4", ("nEdges", "TWO"))
    data["cellsOnEdge"][:, :] = mesh.edge.cell
    data.createVariable(
        "edgesOnEdge", "i4", ("nEdges", "maxEdges2"))
    data["edgesOnEdge"][:, :] = mesh.edge.edge
    data.createVariable("nEdgesOnEdge", "i4", ("nEdges"))
    data["nEdgesOnEdge"][:] = mesh.edge.topo

    data.createVariable("lonVertex", "f8", ("nVertices"))
    data["lonVertex"][:] = mesh.vert.xlon
    data.createVariable("latVertex", "f8", ("nVertices"))
    data["latVertex"][:] = mesh.vert.ylat
    data.createVariable("xVertex", "f8", ("nVertices"))
    data["xVertex"][:] = mesh.vert.xpos
    data.createVariable("yVertex", "f8", ("nVertices"))
    data["yVertex"][:] = mesh.vert.ypos
    data.createVariable("zVertex", "f8", ("nVertices"))
    data["zVertex"][:] = mesh.vert.zpos
    data.createVariable("areaTriangle", "f8", ("nVertices"))
    data["areaTriangle"][:] = mesh.vert.area
    data.createVariable(
        "kiteAreasOnVertex", "f8", ("nVertices", "vertexDegree"))
    data["kiteAreasOnVertex"][:, :] = mesh.vert.kite
    data.createVariable(
        "edgesOnVertex", "i4", ("nVertices", "vertexDegree"))
    data["edgesOnVertex"][:, :] = mesh.vert.edge
    data.createVariable(
        "cellsOnVertex", "i4", ("nVertices", "vertexDegree"))
    data["cellsOnVertex"][:, :] = mesh.vert.cell
   
    data.createVariable("is_mask", "i1", ("nCells"))
    data["is_mask"].long_name = "TRUE for cells masked out of flow"
    data["is_mask"][:] = flow.is_mask
   
    data.createVariable("zb_cell", "f4", ("nCells"))
    data["zb_cell"].long_name = "Elevation of bottom surface"
    data["zb_cell"][:] = flow.zb_cell

    data.createVariable("ff_cell", "f4", ("nCells"))
    data["ff_cell"].long_name = "Coriolis parameter on cells"
    data["ff_cell"][:] = flow.ff_cell
    data.createVariable("ff_edge", "f4", ("nEdges"))
    data["ff_edge"].long_name = "Coriolis parameter on edges"
    data["ff_edge"][:] = flow.ff_edge
    data.createVariable("ff_vert", "f4", ("nVertices"))
    data["ff_vert"].long_name = "Coriolis parameter on duals"
    data["ff_vert"][:] = flow.ff_vert

    data.createVariable("h2_diff", "f4", ("nCells"))
    data["h2_diff"].long_name = "DEL^2(H) diffusion coefficient"
    data.createVariable("h4_diff", "f4", ("nCells"))
    data["h4_diff"].long_name = "DEL^4(H) diffusion coefficient"
    
    data.createVariable("d2_visc", "f4", ("nVertices"))
    data["d2_visc"].long_name = \
        "DIV^2(U) viscosity coefficient, remapped to duals"
    data.createVariable("d4_visc", "f4", ("nVertices"))
    data["d4_visc"].long_name = \
        "DIV^4(U) viscosity coefficient, remapped to duals"
    
    data.createVariable("u2_visc", "f4", ("nVertices"))
    data["u2_visc"].long_name = \
        "DEL^2(U) viscosity coefficient, remapped to duals"
    data.createVariable("u4_visc", "f4", ("nVertices"))
    data["u4_visc"].long_name = \
        "DEL^4(U) viscosity coefficient, remapped to duals"
    
    data.createVariable(
        "u0_edge", "f4", ("nEdges", "nVertLevels"))
    data["u0_edge"].long_name = "Normal velocity initial conditions" 
    data["u0_edge"][:] = flow.uu_edge[-1, :, :]
    data.createVariable(
        "h0_cell", "f4", ("nCells", "nVertLevels"))    
    data["h0_cell"].long_name = "Layer thickness initial conditions"
    data["h0_cell"][:] = flow.hh_cell[-1, :, :]

    data.createVariable("kp_sums", "f4", ("Step"))
    data["kp_sums"].long_name = \
        "Energetics invariant: total KE+PE over time"
    data.createVariable("en_sums", "f4", ("Step"))
    data["en_sums"].long_name = \
        "Rotational invariant: total PV**2 over time"

    if ("uu_edge" in cnfg.save_vars):
        data.createVariable(
            "uu_edge", "f4", ("Time", "nEdges", "nVertLevels"))
        data["uu_edge"].long_name = "Normal velocity on edges" 
        save.uu_edge = True
    
    if ("hh_cell" in cnfg.save_vars):   
        data.createVariable(
            "hh_cell", "f4", ("Time", "nCells", "nVertLevels"))    
        data["hh_cell"].long_name = "Layer thickness on cells"
        save.hh_cell = True
        
    if ("hh_edge" in cnfg.save_vars):   
        data.createVariable(
            "hh_edge", "f4", ("Time", "nEdges", "nVertLevels"))    
        data["hh_edge"].long_name = "Layer thickness on edges"
        save.hh_edge = True
        
    if ("hh_dual" in cnfg.save_vars):   
        data.createVariable(
            "hh_dual", "f4", ("Time", "nVertices", "nVertLevels"))    
        data["hh_dual"].long_name = "Layer thickness on vertices"
        save.hh_dual = True

    if ("du_cell" in cnfg.save_vars):
        data.createVariable(
            "du_cell", "f4", ("Time", "nCells", "nVertLevels"))
        data["du_cell"].long_name = \
            "Divergence of velocity on cells"
        save.du_cell = True
    
    if ("uh_cell" in cnfg.save_vars):
        data.createVariable(
            "uh_cell", "f4", ("Time", "nCells", "nVertLevels"))
        data["uh_cell"].long_name = \
            "Divergence of thickness flux on cells"
        save.uh_cell = True

    if ("ke_bias" in cnfg.save_vars):
        data.createVariable(
            "ke_bias", "f4", ("Time", "nVertices", "nVertLevels"))
        data["ke_bias"].long_name = \
            "Upwind-bias for KE, remapped to duals"
        save.ke_bias = True
    
    if ("pv_bias" in cnfg.save_vars):
        data.createVariable(
            "pv_bias", "f4", ("Time", "nVertices", "nVertLevels"))
        data["pv_bias"].long_name = \
            "Upwind-bias for PV, remapped to duals"
        save.pv_bias = True

    if ("ke_cell" in cnfg.save_vars):
        data.createVariable(
            "ke_cell", "f4", ("Time", "nCells", "nVertLevels"))
        data["ke_cell"].long_name = "Kinetic energy on cells"
        save.ke_cell = True
        
    if ("pv_dual" in cnfg.save_vars):
        data.createVariable(
            "pv_dual", "f4", ("Time", "nVertices", "nVertLevels"))
        data["pv_dual"].long_name = "Potential vorticity on duals"
        save.pv_dual = True       
    
    if ("rv_dual" in cnfg.save_vars):
        data.createVariable(
            "rv_dual", "f4", ("Time", "nVertices", "nVertLevels"))
        data["rv_dual"].long_name = "Relative vorticity on duals"
        save.rv_dual = True
        
    if ("pv_cell" in cnfg.save_vars):
        data.createVariable(
            "pv_cell", "f4", ("Time", "nCells", "nVertLevels"))
        data["pv_cell"].long_name = "Potential vorticity on cells"
        save.pv_cell = True       
    
    if ("rv_cell" in cnfg.save_vars):
        data.createVariable(
            "rv_cell", "f4", ("Time", "nCells", "nVertLevels"))
        data["rv_cell"].long_name = "Relative vorticity on cells"
        save.rv_cell = True

    data.close()
    
    ttoc = time.time()
    tcpu.filewrite = tcpu.filewrite + (ttoc - ttic)
    
