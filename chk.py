
import numpy as np
import netCDF4 as nc
import argparse

""" Checking and validation on a simulation output file  
"""
#-- Part of the PERISCOPE solver
#-- Darren Engwirda
#-- d.engwirda@gmail.com
#-- https://github.com/dengwirda/

if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "--soln-file", dest="soln_file", type=str,
        required=True, 
        help="Path to user solution file.")

    args = parser.parse_args()

    soln = nc.Dataset(args.soln_file, "r", format="NETCDF4")

    if (not np.all(np.isfinite(np.asarray(
            soln.variables["uu_edge"])))):
        raise ValueError("Invalid uu_edge found")

    if (not np.all(np.isfinite(np.asarray(
            soln.variables["hh_cell"])))):
        raise ValueError("Invalid hh_cell found")

    if (np.min(np.asarray(soln.variables["hh_min_"])) <= 0):
        raise ValueError("Tangled hh_cell found")


