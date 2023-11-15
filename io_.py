
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

def save_step(save, mesh, trsk, flow, cnfg, step, hh_cell, uu_edge):

    hh_edge, hh_dual, ke_cell, ke_bias, \
    rv_cell, pv_cell, rv_dual, pv_dual, \
    pv_edge, pv_bias = diag_vars (
        mesh, trsk, flow, cnfg, hh_cell, uu_edge)

    ttic = time.time()

    data = nc.Dataset(
        save, "a", format="NETCDF4")

    # xt variables are tmp scratch

    data.variables["uu_edge"][step, :, :] = \
        np.reshape(uu_edge[
            mesh.edge.irev - 1], (1, mesh.edge.size, 1))
    data.variables["hh_cell"][step, :, :] = \
        np.reshape(hh_cell[
            mesh.cell.irev - 1], (1, mesh.cell.size, 1))

    xt_cell = trsk.cell_flux_sums * uu_edge
    xt_cell/= mesh.cell.area

    data.variables["du_cell"][step, :, :] = \
        np.reshape(xt_cell[
            mesh.cell.irev - 1], (1, mesh.cell.size, 1))

    xt_edge = uu_edge * hh_edge
    xt_cell = trsk.cell_flux_sums * xt_edge
    xt_cell/= mesh.cell.area

    data.variables["uh_cell"][step, :, :] = \
        np.reshape(xt_cell[
            mesh.cell.irev - 1], (1, mesh.cell.size, 1))

    """
    xt_dual = trsk.dual_tail_sums * ke_bias
    xt_dual/= mesh.vert.area

    data.variables["ke_bias"][step, :, :] = \
        np.reshape(xt_dual[
            mesh.vert.irev - 1], (1, mesh.vert.size, 1))
    """

    data.variables["ke_cell"][step, :, :] = \
        np.reshape(ke_cell[
            mesh.cell.irev - 1], (1, mesh.cell.size, 1)) 

    xt_dual = trsk.dual_tail_sums * pv_bias
    xt_dual/= mesh.vert.area

    data.variables["pv_bias"][step, :, :] = \
        np.reshape(xt_dual[
            mesh.vert.irev - 1], (1, mesh.vert.size, 1))
            
    data.variables["pv_dual"][step, :, :] = \
        np.reshape(pv_dual[
            mesh.vert.irev - 1], (1, mesh.vert.size, 1))
    data.variables["rv_dual"][step, :, :] = \
        np.reshape(rv_dual[
            mesh.vert.irev - 1], (1, mesh.vert.size, 1))

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

    data.createVariable(
        "uu_edge", "f4", ("Time", "nEdges", "nVertLevels"))
    data["uu_edge"].long_name = "Normal velocity on edges"    
    data.createVariable(
        "hh_cell", "f4", ("Time", "nCells", "nVertLevels"))    
    data["hh_cell"].long_name = "Layer thickness on cells"

    data.createVariable(
        "du_cell", "f4", ("Time", "nCells", "nVertLevels"))
    data["du_cell"].long_name = \
        "Divergence of velocity on cells"
        
    data.createVariable(
        "uh_cell", "f4", ("Time", "nCells", "nVertLevels"))
    data["uh_cell"].long_name = \
        "Divergence of thickness flux on cells"

    """
    data.createVariable(
        "ke_bias", "f4", ("Time", "nVertices", "nVertLevels"))
    data["ke_bias"].long_name = \
        "Upwind-bias for KE, averaged to duals"
    """
    data.createVariable(
        "pv_bias", "f4", ("Time", "nVertices", "nVertLevels"))
    data["pv_bias"].long_name = \
        "Upwind-bias for PV, averaged to duals"

    data.createVariable(
        "ke_cell", "f4", ("Time", "nCells", "nVertLevels"))
    data["ke_cell"].long_name = "Kinetic energy on cells"
    data.createVariable(
        "pv_dual", "f4", ("Time", "nVertices", "nVertLevels"))
    data["pv_dual"].long_name = "Potential vorticity on duals"
    data.createVariable(
        "rv_dual", "f4", ("Time", "nVertices", "nVertLevels"))
    data["rv_dual"].long_name = "Relative vorticity on duals"

    data.close()
    
    ttoc = time.time()
    tcpu.filewrite = tcpu.filewrite + (ttoc - ttic)
    
