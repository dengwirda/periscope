
import time
import sys
import os
import copy
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import geopandas as gpd

import argparse

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

HERE = os.path.abspath(os.path.dirname(__file__))

def discrepancy(Ao, Am, Po, Pm):
    return np.sqrt(np.maximum(
        0., .5 * (Ao ** 2 + Am ** 2 - 
        2. * Ao * Am * np.cos((Po - Pm) * np.pi/180.0)
        ) ) )

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
        
    parser.add_argument(
        "--show-plot", dest="show_plot",
        type=lambda x: bool(strtobool(str(x.strip()))),
        default=False,
        required=False, help="TRUE to display plots.")
    
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
    print("-"+zt_tide)

    try:
        ke_diss = np.asarray(
        data ["dk_ave_"][:], dtype=np.float32).squeeze()
        ke_diss = \
            1027. * np.sum(ke_diss * mesh.vert.area)
        print("Dissipation =", ke_diss)
    except:
        print("Dissipation: not found")

    zt_data = np.asarray(
        data["hh_cell"][:, near], dtype=np.float32)
    zb_cell = np.asarray(
        data["zb_cell"][:      ], dtype=np.float32)
    zt_data = zt_data[:, :, 0] + zb_cell [near]

    zt_step = int(data.dimensions["Time"].size)
    zt_time = np.linspace(0., zt_step - 1, zt_step)
    zt_time*= data.time_step * data.save_freq / 3600.  # hrs
    zt_time+= data.timestart / 3600.

    t0_time = datetime.fromisoformat(data.datestart)
    times = Tide._times(t0_time, zt_time)

    keep = zt_time > args.skip_days * 24.;

    mod_ = copy.deepcopy(obs_)
    err_ = copy.deepcopy(obs_)

    err_[:]["m2_amp"] = 0.; err_[:]["m2_phs"] = 0.
    err_[:]["s2_amp"] = 0.; err_[:]["s2_phs"] = 0.
    err_[:]["n2_amp"] = 0.; err_[:]["n2_phs"] = 0.
    err_[:]["k1_amp"] = 0.; err_[:]["k1_phs"] = 0.
    err_[:]["o1_amp"] = 0.; err_[:]["o1_phs"] = 0.

    used = np.zeros(obs_.size, dtype=bool)

    nobs = 0
    for s in range(obs_.size):
        tide_decomp = Tide.decompose(
            heights=zt_data[keep, s], 
                t=times[keep], constituents=primary_5)

        mod_[s]["m2_amp"] = tide_decomp.model["amplitude"][1]
        mod_[s]["m2_phs"] = tide_decomp.model["phase"][1]

        mod_[s]["s2_amp"] = tide_decomp.model["amplitude"][2]
        mod_[s]["s2_phs"] = tide_decomp.model["phase"][2]

        mod_[s]["n2_amp"] = tide_decomp.model["amplitude"][3]
        mod_[s]["n2_phs"] = tide_decomp.model["phase"][3]

        mod_[s]["k1_amp"] = tide_decomp.model["amplitude"][4]
        mod_[s]["k1_phs"] = tide_decomp.model["phase"][4]

        mod_[s]["o1_amp"] = tide_decomp.model["amplitude"][5]
        mod_[s]["o1_phs"] = tide_decomp.model["phase"][5]

        cell = near[s]

        if (zb_cell[cell] > -1000.):
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

        err_[s]["m2_amp"] = mod_[s]["m2_amp"] - \
                            obs_[s]["m2_amp"]

        err_[s]["m2_phs"] = discrepancy(
            obs_[s]["m2_amp"], mod_[s]["m2_amp"],
            obs_[s]["m2_phs"], mod_[s]["m2_phs"])
            
        err_[s]["s2_amp"] = mod_[s]["s2_amp"] - \
                            obs_[s]["s2_amp"]

        err_[s]["s2_phs"] = discrepancy(
            obs_[s]["s2_amp"], mod_[s]["s2_amp"],
            obs_[s]["s2_phs"], mod_[s]["s2_phs"])

        err_[s]["n2_amp"] = mod_[s]["n2_amp"] - \
                            obs_[s]["n2_amp"]

        err_[s]["n2_phs"] = discrepancy(
            obs_[s]["n2_amp"], mod_[s]["n2_amp"],
            obs_[s]["n2_phs"], mod_[s]["n2_phs"])

        err_[s]["k1_amp"] = mod_[s]["k1_amp"] - \
                            obs_[s]["k1_amp"]

        err_[s]["k1_phs"] = discrepancy(
            obs_[s]["k1_amp"], mod_[s]["k1_amp"],
            obs_[s]["k1_phs"], mod_[s]["k1_phs"])

        err_[s]["o1_amp"] = mod_[s]["o1_amp"] - \
                            obs_[s]["o1_amp"]

        err_[s]["o1_phs"] = discrepancy(
            obs_[s]["o1_amp"], mod_[s]["o1_amp"],
            obs_[s]["o1_phs"], mod_[s]["o1_phs"])

        used[s] = True; nobs = nobs + 1

    print("Number of stations:", nobs)
    print("M2")
    print(np.nanmin(err_["m2_amp"]),
          np.nanmax(err_["m2_amp"]), 
          np.nansum(err_["m2_amp"]) / nobs, 
          np.sqrt(np.nansum(err_["m2_amp"] ** 2) / nobs))
    print(np.nanmin(err_["m2_phs"]),
          np.nanmax(err_["m2_phs"]), 
          np.nansum(err_["m2_phs"]) / nobs, 
          np.sqrt(np.nansum(err_["m2_phs"] ** 2) / nobs))

    print("S2")
    print(np.nanmin(err_["s2_amp"]),
          np.nanmax(err_["s2_amp"]), 
          np.nansum(err_["s2_amp"]) / nobs, 
          np.sqrt(np.nansum(err_["s2_amp"] ** 2) / nobs))
    print(np.nanmin(err_["s2_phs"]),
          np.nanmax(err_["s2_phs"]), 
          np.nansum(err_["s2_phs"]) / nobs, 
          np.sqrt(np.nansum(err_["s2_phs"] ** 2) / nobs))

    print("N2")
    print(np.nanmin(err_["n2_amp"]),
          np.nanmax(err_["n2_amp"]), 
          np.nansum(err_["n2_amp"]) / nobs, 
          np.sqrt(np.nansum(err_["n2_amp"] ** 2) / nobs))
    print(np.nanmin(err_["n2_phs"]),
          np.nanmax(err_["n2_phs"]), 
          np.nansum(err_["n2_phs"]) / nobs, 
          np.sqrt(np.nansum(err_["n2_phs"] ** 2) / nobs))

    print("K1")
    print(np.nanmin(err_["k1_amp"]),
          np.nanmax(err_["k1_amp"]), 
          np.nansum(err_["k1_amp"]) / nobs, 
          np.sqrt(np.nansum(err_["k1_amp"] ** 2) / nobs))
    print(np.nanmin(err_["k1_phs"]),
          np.nanmax(err_["k1_phs"]), 
          np.nansum(err_["k1_phs"]) / nobs, 
          np.sqrt(np.nansum(err_["k1_phs"] ** 2) / nobs))

    print("O1")
    print(np.nanmin(err_["o1_amp"]),
          np.nanmax(err_["o1_amp"]), 
          np.nansum(err_["o1_amp"]) / nobs, 
          np.sqrt(np.nansum(err_["o1_amp"] ** 2) / nobs))
    print(np.nanmin(err_["o1_phs"]),
          np.nanmax(err_["o1_phs"]), 
          np.nansum(err_["o1_phs"]) / nobs, 
          np.sqrt(np.nansum(err_["o1_phs"] ** 2) / nobs))

    countries = gpd.read_file(HERE + "/../ne_50m_land.zip")
    
    countries.head()
    countries.plot(color="lightgrey")
    plt.scatter(x=obs_[used]["lon"], y=obs_[used]["lat"], 
        c=err_[used]["m2_phs"], cmap="PuRd", vmin=-0.0, vmax=0.250)
    plt.colorbar()
    plt.title("M2-phs")
    plt.savefig(os.path.splitext(args.soln_file)[0] + "_M2_phs.png", 
                bbox_inches="tight", dpi=300)

    countries.head()
    countries.plot(color="lightgrey")
    plt.scatter(x=obs_[used]["lon"], y=obs_[used]["lat"], 
        c=err_[used]["m2_amp"], cmap="coolwarm", vmin=-0.20, vmax=0.20)
    plt.colorbar()
    plt.title("M2-amp")
    plt.savefig(os.path.splitext(args.soln_file)[0] + "_M2_amp.png", 
                bbox_inches="tight", dpi=300)

    countries.head()
    countries.plot(color="lightgrey")
    plt.scatter(x=obs_[used]["lon"], y=obs_[used]["lat"], 
        c=err_[used]["s2_phs"], cmap="PuRd", vmin=-0.0, vmax=0.125)
    plt.colorbar()
    plt.title("S2-phs")
    plt.savefig(os.path.splitext(args.soln_file)[0] + "_S2_phs.png", 
                bbox_inches="tight", dpi=300)

    countries.head()
    countries.plot(color="lightgrey")
    plt.scatter(x=obs_[used]["lon"], y=obs_[used]["lat"], 
        c=err_[used]["s2_amp"], cmap="coolwarm", vmin=-0.10, vmax=0.10)
    plt.colorbar()
    plt.title("S2-amp")
    plt.savefig(os.path.splitext(args.soln_file)[0] + "_S2_amp.png", 
                bbox_inches="tight", dpi=300)

    countries.head()
    countries.plot(color="lightgrey")
    plt.scatter(x=obs_[used]["lon"], y=obs_[used]["lat"], 
        c=err_[used]["n2_phs"], cmap="PuRd", vmin=-0.0, vmax=.0625)
    plt.colorbar()
    plt.title("N2-phs")
    plt.savefig(os.path.splitext(args.soln_file)[0] + "_N2_phs.png", 
                bbox_inches="tight", dpi=300)

    countries.head()
    countries.plot(color="lightgrey")
    plt.scatter(x=obs_[used]["lon"], y=obs_[used]["lat"], 
        c=err_[used]["n2_amp"], cmap="coolwarm", vmin=-0.05, vmax=0.05)
    plt.colorbar()
    plt.title("N2-amp")
    plt.savefig(os.path.splitext(args.soln_file)[0] + "_N2_amp.png", 
                bbox_inches="tight", dpi=300)

    countries.head()
    countries.plot(color="lightgrey")
    plt.scatter(x=obs_[used]["lon"], y=obs_[used]["lat"], 
        c=err_[used]["k1_phs"], cmap="PuRd", vmin=-0.0, vmax=0.125)
    plt.colorbar()
    plt.title("K1-phs")
    plt.savefig(os.path.splitext(args.soln_file)[0] + "_K1_phs.png", 
                bbox_inches="tight", dpi=300)

    countries.head()
    countries.plot(color="lightgrey")
    plt.scatter(x=obs_[used]["lon"], y=obs_[used]["lat"], 
        c=err_[used]["k1_amp"], cmap="coolwarm", vmin=-0.10, vmax=0.10)
    plt.colorbar()
    plt.title("K1-amp")
    plt.savefig(os.path.splitext(args.soln_file)[0] + "_K1_amp.png", 
                bbox_inches="tight", dpi=300)

    countries.head()
    countries.plot(color="lightgrey")
    plt.scatter(x=obs_[used]["lon"], y=obs_[used]["lat"], 
        c=err_[used]["o1_phs"], cmap="PuRd", vmin=-0.0, vmax=.0625)
    plt.colorbar()
    plt.title("O1-phs")
    plt.savefig(os.path.splitext(args.soln_file)[0] + "_O1_phs.png", 
                bbox_inches="tight", dpi=300)

    countries.head()
    countries.plot(color="lightgrey")
    plt.scatter(x=obs_[used]["lon"], y=obs_[used]["lat"], 
        c=err_[used]["o1_amp"], cmap="coolwarm", vmin=-0.05, vmax=0.05)
    plt.colorbar()
    plt.title("O1-amp")
    plt.savefig(os.path.splitext(args.soln_file)[0] + "_O1_amp.png", 
                bbox_inches="tight", dpi=300)

    if args.show_plot: plt.show()

