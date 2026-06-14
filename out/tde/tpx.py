
import time
import sys
import os
import numpy as np
import netCDF4 as nc

import argparse

from datetime import datetime
from pytides.tide import Tide
from pytides.constituent import primary_5
from progressbar import ProgressBar, Percentage, Bar, ETA

sys.path.insert(
    1, os.path.join(sys.path[0], "..", ".."))

from stb import strtobool

from msh import load_mesh, cell_quad
from map import flatten
from map import interp2d
from ops import operators

from _fp import flt32_t, flt64_t
from _fp import reals_t, index_t

# remap TPXO10 data onto a solution and compare

HERE = os.path.abspath(os.path.dirname(__file__))

HMIN = 2.5  # min. depth for analysis

def discrepancy(Ao, Am, Po, Pm):
    return np.sqrt(np.maximum(
        0., .5 * (Ao ** 2 + Am ** 2 - 
        2. * Ao * Am * np.cos((Po - Pm) * np.pi/180.0)
        ) ) )

def calc_harmonics(args, mesh, tpxo, data, out_):

    print("Computing harmonics...")

    zt_tide = data.tidal_frc  # constituents used
    zt_tide = zt_tide.split(",")

    zt_data = np.asarray(
        data["hh_cell"][:], dtype=np.float32)
    zb_cell = np.asarray(
        data["zb_cell"][:], dtype=np.float32)
    zt_data = zt_data[:, :, 0] + zb_cell

    zt_step = int(data.dimensions["Time"].size)
    zt_time = np.linspace(0., zt_step - 1, zt_step)
    zt_time*= data.time_step * data.save_freq / 3600.
    zt_time+= data.timestart / 3600.

    t0_time = datetime.fromisoformat(data.datestart)
    times = Tide._times(t0_time, zt_time)

    keep = zt_time > args.skip_days * 24.

    zt_data = zt_data[keep, :]; times = times[keep]

    widgets = ["Harmonic analysis: ", Percentage(), " ", 
               Bar(), " ", ETA()]
    waitbar = ProgressBar(
        widgets=widgets, maxval=mesh.cell.size).start()

    for cell in range(mesh.cell.size):
        if (cell % 2500 == 0): waitbar.update(cell)

        tide_decomp = Tide.decompose(
            heights=zt_data[:, cell], 
                t=times, constituents=primary_5)

        for icon in range(len(zt_tide)):
            name = zt_tide[icon] + "_amp_solver"
            out_[name][cell] = \
                tide_decomp.model["amplitude"][icon + 1]

            name = zt_tide[icon] + "_phs_solver"
            out_[name][cell] = \
                tide_decomp.model["phase"][icon + 1]

    waitbar.finish()


def calc_remapping(args, mesh, tpxo, data, out_):

    print("Remapping TPXO data...")

    zt_tide = data.tidal_frc  # constituents used
    zt_tide = zt_tide.split(",")

    ncon = int(tpxo.dimensions["nc"].size)
    cons = tpxo["con"]
    xlon = np.asarray(tpxo["lon_z"][:], dtype=np.float64)
    ylat = np.asarray(tpxo["lat_z"][:], dtype=np.float64)

    xlon = xlon[:, 0] * np.pi / 180.
    ylat = ylat[0, :] * np.pi / 180.

    for icon in range(ncon):
        con_amp = np.asarray(
            tpxo["ha"][ icon], dtype=np.float32)
        con_phs = np.asarray(
            tpxo["hp"][ icon], dtype=np.float32)

        con_amp = con_amp.T
        con_phs = con_phs.T

        # interp. TPXO onto mesh via (quasi-)remap
        ha_cell = interp2d(
            ylat, xlon, con_amp, mesh.cell.ylat, 
                                 mesh.cell.xlon)
        ha_vert = interp2d(
            ylat, xlon, con_amp, mesh.vert.ylat, 
                                 mesh.vert.xlon)
        
        HA_cell = cell_quad(mesh, ha_cell, ha_vert)

        hp_cell = interp2d(
            ylat, xlon, con_phs, mesh.cell.ylat, 
                                 mesh.cell.xlon)
        hp_vert = interp2d(
            ylat, xlon, con_phs, mesh.vert.ylat, 
                                 mesh.vert.xlon)

        HP_cell = cell_quad(mesh, hp_cell, hp_vert)

        out_["M2_amp_tpxo10"][:] = HA_cell
        out_["M2_phs_tpxo10"][:] = HP_cell
        break


if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--soln-file", dest="soln_file", type=str,
        required=True, help="Path to user soln file.")

    parser.add_argument(
        "--tpxo-file", dest="tpxo_file", type=str,
        required=True, help="Path to TPXO data file.")

    parser.add_argument(
        "--skip-days", dest="skip_days", type=int,
        required=True, help="Excluded from analysis.")

    parser.add_argument(
        "--full-calc", dest="full_calc",
        type=lambda x: bool(strtobool(str(x.strip()))),
        default=False,
        required=False, help="TRUE for full compute.")

    args = parser.parse_args()

    print("Loading the TPXO file...")

    tpxo = nc.Dataset(args.tpxo_file, "r")

    print("Loading the mesh file...")
    
    mesh = load_mesh(args.soln_file)

    out_ = nc.Dataset(args.soln_file, "a")
    
    print("Creating the analysis...")

    try:
        flat = flt64_t(mesh.sphere_flatten)
    except:
        flat = flt64_t(0.0)

    """
    obs_ ["lat"] = 180. / np.pi * (
        flatten(obs_["lat"] * np.pi / 180., flat)
    )
    """

    data = nc.Dataset(args.soln_file, "r")
    zt_tide = data.tidal_frc  # constituents used
    print("-"+zt_tide)

    zt_tide = zt_tide.split(",")
    for icon in range(len(zt_tide)):
        name = zt_tide[icon] + "_amp_tpxo10"
        if (name not in out_.variables):
            out_.createVariable(name, "f4", ("nCells"))

        name = zt_tide[icon] + "_phs_tpxo10"
        if (name not in out_.variables):
            out_.createVariable(name, "f4", ("nCells"))

        name = zt_tide[icon] + "_amp_solver"
        if (name not in out_.variables):
            out_.createVariable(name, "f4", ("nCells"))

        name = zt_tide[icon] + "_phs_solver"
        if (name not in out_.variables):
            out_.createVariable(name, "f4", ("nCells"))

    if (args.full_calc):
        calc_harmonics(args, mesh, tpxo, data, out_)

    if (args.full_calc):
        calc_remapping(args, mesh, tpxo, data, out_)

    err = discrepancy(
        np.asarray(out_["M2_amp_solver"][:]), 
        np.asarray(out_["M2_amp_tpxo10"][:]),
        np.asarray(out_["M2_phs_solver"][:]), 
        np.asarray(out_["M2_phs_tpxo10"][:]))

    mask = np.logical_and.reduce((  # shallow
        np.asarray(out_["hh_max_"][:,0])>HMIN ,
        np.asarray(out_["h0_cell"][:,0])<1000.,
        ) )

    print(np.sum(mesh.cell.area[mask] * err[mask]) 
        / np.sum(mesh.cell.area[mask]), "[shallow]")

    mask = np.logical_and.reduce((  # deep
        np.asarray(out_["hh_max_"][:,0])>HMIN ,
        np.asarray(out_["h0_cell"][:,0])>1000.,
        ) )

    print(np.sum(mesh.cell.area[mask] * err[mask]) 
        / np.sum(mesh.cell.area[mask]), "[deep]")

    mask = np.logical_and.reduce((  # deep, low-lat
        np.asarray(out_["hh_max_"][:,0])>HMIN ,
        np.asarray(out_["h0_cell"][:,0])>1000.,
        np.abs(mesh.cell.ylat)<=66.*np.pi/180.,
        ) )

    print(np.sum(mesh.cell.area[mask] * err[mask]) 
        / np.sum(mesh.cell.area[mask]), "[deep+low-lat]")

    out_.close()
