
import time
import sys
import os
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt

import argparse

sys.path.insert(
    1, os.path.join(sys.path[0], "..", ".."))

from stb import strtobool

from msh import load_mesh
from map import find_cell

# Analysis for Tohoku tsunami
# Authors: Darren Engwirda

HERE = os.path.abspath(os.path.dirname(__file__))

if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "--mesh-file", dest="mesh_file", type=str,
        required=True, help="Path to user mesh file.")
        
    parser.add_argument(
        "--show-plot", dest="show_plot",
        type=lambda x: bool(strtobool(str(x.strip()))),
        default=False,
        required=False, help="TRUE to display plots.")
    
    args = parser.parse_args()
    
    # March 11, 05:46:23
    # Add 3-mins, which is 1/2 of the 6-min earthquake
    init = 31. + 28. + 11. + \
        (5. * 60. * 60. + 
        46. * 60. + 
        23. + 
        180.
        ) / 24. / 60./ 60.
    
    print("Loading the mesh file...")
    
    mesh = load_mesh(args.mesh_file)
    
    print("Creating the analysis...")
    
    dart = np.loadtxt(os.path.join(HERE,
        "dart21413_20110301to20110320_meter.txt"))
    xlon = 152.123 * np.pi / 180.
    ylat = 30.5280 * np.pi / 180.
    time = dart[:, 0]
    zlev = dart[:, 9]
    zlev[np.abs(zlev) >= 9999.] = np.nan  # not signal
    zlev-= np.nanmean(zlev)  # correct any const. bias
    mask = np.logical_and.reduce((
        time >= init + 1. / 24.,  # skip seismic waves
        time <= init + 8. / 24. ))
       
    near = find_cell(
        mesh, np.array([xlon]), np.array([ylat]))
        
    data = nc.Dataset(args.mesh_file, "r")    
    hh_data = np.asarray(
        data["hh_cell"][:, near, 0], dtype=np.float32) 
    hh_data+= data["zb_cell"][near]
    
    hh_step = int(data.dimensions["Time"].size)
    hh_time = np.linspace(0., hh_step, hh_step)
    hh_time*= data.time_step * data.save_freq / 60. / 60.
    data.close()
    
    plt.figure(1)
    plt.plot((time[mask] - init) * 24, zlev[mask])
    plt.plot(hh_time, hh_data)
    plt.xlim([0, 8])
    plt.legend(("DART-21413", "SWE"), loc="upper right")
    plt.grid(True, linestyle="-.")
    plt.ylabel("Sea Surface Height [m]")
    plt.xlabel("Hours since event")
    
    plt.savefig(os.path.splitext(args.mesh_file)[0] + "_21413.png", 
                bbox_inches="tight", dpi=300)
    
    dart = np.loadtxt(os.path.join(HERE,
        "dart52402_20110301to20110320_meter.txt"))
    xlon = 154.111 * np.pi / 180.
    ylat = 11.8820 * np.pi / 180.
    time = dart[:, 0]
    zlev = dart[:, 9]
    zlev[np.abs(zlev) >= 9999.] = np.nan  # not signal
    zlev-= np.nanmean(zlev)  # correct any const. bias
    mask = np.logical_and.reduce((
        time >= init + 3. / 24.,  # skip seismic waves
        time <= init + 8. / 24. ))
       
    near = find_cell(
        mesh, np.array([xlon]), np.array([ylat]))
        
    data = nc.Dataset(args.mesh_file, "r")    
    hh_data = np.asarray(
        data["hh_cell"][:, near, 0], dtype=np.float32) 
    hh_data+= data["zb_cell"][near]
    
    hh_step = int(data.dimensions["Time"].size)
    hh_time = np.linspace(0., hh_step, hh_step)
    hh_time*= data.time_step * data.save_freq / 60. / 60.
    data.close()
    
    plt.figure(2)
    plt.plot((time[mask] - init) * 24, zlev[mask])
    plt.plot(hh_time, hh_data)
    plt.xlim([0, 8])
    plt.legend(("DART-52402", "SWE"), loc="upper right")
    plt.grid(True, linestyle="-.")
    plt.ylabel("Sea Surface Height [m]")
    plt.xlabel("Hours since event")
    
    plt.savefig(os.path.splitext(args.mesh_file)[0] + "_52402.png", 
                bbox_inches="tight", dpi=300)
    
    dart = np.loadtxt(os.path.join(HERE,
        "dart21416_20110301to20110320_meter.txt"))
    xlon = 163.486 * np.pi / 180.
    ylat = 48.0420 * np.pi / 180.
    time = dart[:, 0]
    zlev = dart[:, 9]
    zlev[np.abs(zlev) >= 9999.] = np.nan  # not signal
    zlev-= np.nanmean(zlev)  # correct any const. bias
    mask = np.logical_and.reduce((
        time >= init + 2. / 24.,  # skip seismic waves
        time <= init + 8. / 24. ))
       
    near = find_cell(
        mesh, np.array([xlon]), np.array([ylat]))
        
    data = nc.Dataset(args.mesh_file, "r")    
    hh_data = np.asarray(
        data["hh_cell"][:, near, 0], dtype=np.float32) 
    hh_data+= data["zb_cell"][near]
    
    hh_step = int(data.dimensions["Time"].size)
    hh_time = np.linspace(0., hh_step, hh_step)
    hh_time*= data.time_step * data.save_freq / 60. / 60.
    data.close()
    
    plt.figure(3)
    plt.plot((time[mask] - init) * 24, zlev[mask])
    plt.plot(hh_time, hh_data)
    plt.xlim([0, 8])
    plt.legend(("DART-21416", "SWE"), loc="upper right")
    plt.grid(True, linestyle="-.")
    plt.ylabel("Sea Surface Height [m]")
    plt.xlabel("Hours since event")
    
    plt.savefig(os.path.splitext(args.mesh_file)[0] + "_21416.png", 
                bbox_inches="tight", dpi=300)
    
    if args.show_plot: plt.show()
    
