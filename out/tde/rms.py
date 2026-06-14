
import time
import sys
import os
import copy
import numpy as np
import netCDF4 as nc

import argparse
import jigsawpy

from datetime import datetime
from pytides.tide import Tide
from pytides.constituent import primary_5

sys.path.insert(
    1, os.path.join(sys.path[0], "..", ".."))

from stb import strtobool

from msh import load_mesh, circ_dist
from map import flatten
from map import find_cell

from _fp import flt32_t, flt64_t
from _fp import reals_t, index_t

# Analysis for tide gauge data

if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--soln-file", dest="soln_file", type=str,
        required=True, help="Path to user soln file.")

    parser.add_argument(
        "--tide-file", dest="tide_file", type=str,
        required=True, help="Path to tide data file.")

    parser.add_argument(
        "--skip-days", dest="skip_days", type=int,
        required=True, help="Excluded from analysis.")

    args = parser.parse_args()

    obs_ = np.loadtxt(
    args.tide_file, skiprows=1, delimiter=",", 
    quotechar='"',
    dtype={"names":(
                "station", "lon", "lat", "source", 
                "m2_amp", "m2_phs",  
                "s2_amp", "s2_phs",
                "n2_amp", "n2_phs",
                "k2_amp", "k2_phs",
                "k1_amp", "k1_phs",
                "o1_amp", "o1_phs",
                "p1_amp", "p1_phs",
                "q1_amp", "q1_phs", "kind"),
         "formats":(
                "U64", np.float64, np.float64, "U16",
                np.float32, np.float32,
                np.float32, np.float32,
                np.float32, np.float32,
                np.float32, np.float32,
                np.float32, np.float32,
                np.float32, np.float32,
                np.float32, np.float32,
                np.float32, np.float32, "U8")
          } )

    print("Loading the mesh file...")
    
    mesh = load_mesh(args.soln_file)
    
    print("Creating the analysis...")

    try:
        flat = flt64_t(mesh.sphere_flatten)
    except:
        flat = flt64_t(0.0)

    obs_ ["lat"] = 180. / np.pi * (
        flatten(obs_["lat"] * np.pi / 180., flat)
    )

    wrap = obs_["lon"] < 0.
    obs_ [wrap]["lon"] += 360.

    near = find_cell(mesh, obs_["lon"] * np.pi / 180.,
                           obs_["lat"] * np.pi / 180.)

    data = nc.Dataset(args.soln_file, "r")
    zt_tide = data.tidal_frc  # constituents used
    print(zt_tide)

    zt_data = np.asarray(
        data["hh_cell"][:], dtype=np.float32)
    zb_cell = np.asarray(
        data["zb_cell"][:], dtype=np.float32)
    zt_data = zt_data[:, near, 0] + zb_cell[near]

    zt_step = int(data.dimensions["Time"].size) - 1
    zt_hour = zt_step * data.time_step * data.save_freq / 3600.
    zt_time = np.linspace(
        0., zt_hour * 3600., int(np.ceil(zt_hour)) * 4) / 3600.
    zt_time+= data.timestart / 3600.

    t0_time = datetime.fromisoformat(data.datestart)
    times = Tide._times(t0_time, zt_time)

    keep = zt_time > args.skip_days * 24.

    rms_ = copy.deepcopy(obs_)

    rms_[:]["m2_amp"] = 0.; rms_[:]["s2_amp"] = 0.
    rms_[:]["n2_amp"] = 0.; rms_[:]["k1_amp"] = 0.
    rms_[:]["o1_amp"] = 0.;

    used = np.zeros(obs_.size, dtype=bool)

    nobs = 0; pts_ = []; val_ = []; ind_ = []
    for s in range(obs_.size):
        amp = (obs_[s]["m2_amp"], obs_[s]["s2_amp"],
               obs_[s]["n2_amp"], obs_[s]["k1_amp"],
               obs_[s]["o1_amp"])
        phs = (obs_[s]["m2_phs"], obs_[s]["s2_phs"],
               obs_[s]["n2_phs"], obs_[s]["k1_phs"],
               obs_[s]["o1_phs"])
        tide_model_ = Tide(
            constituents=primary_5, amplitudes=amp, phases=phs)

        zssh = tide_model_.at(times)

        rms_[s]["m2_amp"] = \
            np.sqrt(np.sum(zssh** 2) / zssh.size)

        if (np.isnan(rms_[s]["m2_amp"])): continue

        cell = near[s]

        if (zb_cell[cell] > -250.):
           #print("too shallow:", zb_cell [cell])
            continue

        pos1 = np.zeros((1, 2), dtype=np.float64)
        pos1[0, 0] = mesh.cell.xlon[cell]
        pos1[0, 1] = mesh.cell.ylat[cell]
        pos2 = np.zeros((1, 2), dtype=np.float64)
        pos2[0, 0] = obs_[s]["lon"] * np.pi / 180.
        pos2[0, 1] = obs_[s]["lat"] * np.pi / 180.
        dist = circ_dist(mesh.rsph, pos1, pos2)

        if (dist > np.sqrt(mesh.cell.area[cell])):
           #print("outside msh:", dist)
            continue

        ppos = np.zeros((4, 2), dtype=np.float32)
        ppos[0, 0] = obs_[s]["lon"] - 1.
        ppos[0, 1] = obs_[s]["lat"] - 1.
        ppos[1, 0] = obs_[s]["lon"] + 1.
        ppos[1, 1] = obs_[s]["lat"] - 1.
        ppos[2, 0] = obs_[s]["lon"] + 1.
        ppos[2, 1] = obs_[s]["lat"] + 1.
        ppos[3, 0] = obs_[s]["lon"] - 1.
        ppos[3, 1] = obs_[s]["lat"] + 1.
        if obs_[s]["lon"] < -1.: ppos[:, 0]+= 360.
        pts_.append(ppos)

        vals = np.zeros((4, 1), dtype=np.float32)
        vals[0, 0] = rms_[s]["m2_amp"]
        vals[1, 0] = rms_[s]["m2_amp"]
        vals[2, 0] = rms_[s]["m2_amp"]
        vals[3, 0] = rms_[s]["m2_amp"]
        val_.append(vals)

        inds = np.zeros((1, 4), dtype=  np.int32)
        inds[0, 0] = 4 * nobs + 0
        inds[0, 1] = 4 * nobs + 1
        inds[0, 2] = 4 * nobs + 2
        inds[0, 3] = 4 * nobs + 3
        ind_.append(inds)

        used[s] = True; nobs = nobs + 1

    pts_ = np.concatenate(pts_)
    val_ = np.concatenate(val_)
    ind_ = np.concatenate(ind_)

   #pts_[pts_[:, 0] < -3., 0]+= 360.

    mesh = jigsawpy.jigsaw_msh_t()
    mesh.vert2 = np.zeros(pts_.shape[0], dtype=mesh.VERT2_t)
    mesh.vert2["coord"][:] = pts_
    mesh.value = np.zeros(pts_.shape[0], dtype=mesh.REALS_t)
    mesh.value[:] = np.squeeze(val_) * np.sqrt(2.)
    mesh.quad4 = np.zeros(ind_.shape[0], dtype=mesh.QUAD4_t)
    mesh.quad4["index"][:] = ind_

    jigsawpy.savevtk("obs_.vtk", mesh)




