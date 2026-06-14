
import os
import numpy as np
import geojson

import argparse

from inpoly import inpoly2

# Filter tide obs by spatial mask

if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--mask-file", dest="mask_file", type=str,
        required=True, help="Path to user mask file.")

    parser.add_argument(
        "--tide-file", dest="tide_file", type=str,
        required=True, help="Path to tide data file.")
    
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

    ppos = np.vstack((np.asarray(obs_["lon"]),
                      np.asarray(obs_["lat"])
                    ) ).T

    okay = np.full(ppos.shape[0], False, dtype=bool)

    with open(args.mask_file) as f:
        gj = geojson.load(f)

    poly = np.squeeze(np.asarray(
        gj["features"][0]["geometry"]["coordinates"]))

    mask, __ = inpoly2(ppos, poly); okay[mask] = True

    poly[:, 0]+= 360.

    mask, __ = inpoly2(ppos, poly); okay[mask] = True

    poly = np.squeeze(np.asarray(
        gj["features"][1]["geometry"]["coordinates"]))

    mask, __ = inpoly2(ppos, poly); okay[mask] = True

    np.savetxt("test.csv", obs_[okay], delimiter=",", 
        header="Station,Lon,Lat,Source," + 
                "M2_amp,M2_phs," +
                "S2_amp,S2_phs," +
                "N2_amp,N2_phs," +
                "K2_amp,K2_phs," +
                "K1_amp,K1_phs," +
                "O1_amp,O1_phs," +
                "P1_amp,P1_phs," +
                "Q1_amp,Q1_phs,Location", 
        fmt='"%s",%f,%f,%s,%f,%f,%f,%f,%f,%f,%f,%f,' + 
            '%f,%f,%f,%f,%f,%f,%f,%f,%s', comments="")

