
import math
import numpy as np
import netCDF4 as nc
import argparse
import matplotlib.pyplot as plt

from stb import strtobool

from msh import load_mesh, load_flow

# Numerical error metrics
# Authors: Darren Engwirda

def norm_cell(mesh, func):

    return (math.fsum(mesh.cell.area * func) / 
            math.fsum(mesh.cell.area))


def norm_edge(mesh, func):

    return (math.fsum(mesh.edge.area * func) / 
            math.fsum(mesh.edge.area))


def norm_dual(mesh, func):

    return (math.fsum(mesh.vert.area * func) / 
            math.fsum(mesh.vert.area))


def one_step_error(args, m1st, m2nd, step=-1, show=True):

#-- compute error wrt. one time step

    f1st = load_flow(args.test_file, m1st, step=step)
    f2nd = load_flow(args.base_file, m2nd, step=step)

    h1_cell = np.asarray(f1st.hh_cell, dtype=np.float64)
    h2_cell = np.asarray(f2nd.hh_cell, dtype=np.float64)

    eps_ = np.finfo(h1_cell.dtype).eps * 2

    h_mag = np.max(np.abs(h2_cell)) + eps_

    h_two = np.sqrt(
        norm_cell(m1st, (h1_cell - h2_cell)**2))/ h_mag
    
    h_inf = np.max(np.abs((h1_cell - h2_cell))) / h_mag

    u1_edge = np.asarray(f1st.uu_edge, dtype=np.float64)
    u2_edge = np.asarray(f2nd.uu_edge, dtype=np.float64)

    eps_ = np.finfo(u1_edge.dtype).eps * 2

    u_mag = np.max(np.abs(u2_edge)) + eps_

    u_two = np.sqrt(
        norm_edge(m1st, (u1_edge - u2_edge)**2))/ u_mag

    u_inf = np.max(np.abs((u1_edge - u2_edge))) / u_mag

    r1_dual = np.asarray(f1st.rv_dual, dtype=np.float64)
    r2_dual = np.asarray(f2nd.rv_dual, dtype=np.float64)

    eps_ = np.finfo(r1_dual.dtype).eps * 2

    r_mag = np.max(np.abs(r2_dual)) + eps_

    r_two = np.sqrt(
        norm_dual(m1st, (r1_dual - r2_dual)**2))/ r_mag

    r_inf = np.max(np.abs((r1_dual - r2_dual))) / r_mag

    p1_dual = np.asarray(f1st.pv_dual, dtype=np.float64)
    p2_dual = np.asarray(f2nd.pv_dual, dtype=np.float64)

    eps_ = np.finfo(p1_dual.dtype).eps * 2

    p_mag = np.max(np.abs(p2_dual)) + eps_

    p_two = np.sqrt(
        norm_dual(m1st, (p1_dual - p2_dual)**2))/ p_mag

    p_inf = np.max(np.abs((p1_dual - p2_dual))) / p_mag

    if (show):
        print(f"l_two(hh): {h_two:.5E}")
        print(f"l_two(uu): {u_two:.5E}")
        print(f"l_two(rv): {r_two:.5E}")
        print(f"l_two(pv): {p_two:.5E}")
        print(f"l_inf(hh): {h_inf:.5E}")
        print(f"l_inf(uu): {u_inf:.5E}")
        print(f"l_inf(rv): {r_inf:.5E}")
        print(f"l_inf(pv): {p_inf:.5E}")

    return h_two, h_inf, u_two, u_inf, \
           r_two, r_inf, p_two, p_inf
    
    
def all_step_error(args, m1st, m2nd):

#-- compute error wrt. all time step

    data = nc.Dataset(args.base_file, "r")
    try:
        N = int(data.dimensions["Time"].size)
    except:
        N = 0
    data.close()
    
    if (N<= 0): raise ValueError("No time steps found.")

    h_two = np.zeros(N, dtype=np.float64)
    h_inf = np.zeros(N, dtype=np.float64)

    u_two = np.zeros(N, dtype=np.float64)
    u_inf = np.zeros(N, dtype=np.float64)

    r_two = np.zeros(N, dtype=np.float64)
    r_inf = np.zeros(N, dtype=np.float64)

    p_two = np.zeros(N, dtype=np.float64)
    p_inf = np.zeros(N, dtype=np.float64)

    for step in range(N):

        h_two[step], h_inf[step], \
        u_two[step], u_inf[step], \
        r_two[step], r_inf[step], \
        p_two[step], p_inf[step]= \
            one_step_error(
                args, m1st, m2nd, step=step, show=False)

    plt.figure(1)
    plt.semilogy(h_inf)
    plt.semilogy(h_two)
    plt.legend(("h_inf", "h_two"), loc="best")
    plt.grid(True, linestyle="-.")
    plt.ylabel("Error norm.")

    plt.figure(2)
    plt.semilogy(u_inf)
    plt.semilogy(u_two)
    plt.legend(("u_inf", "u_two"), loc="best")
    plt.grid(True, linestyle="-.")
    plt.ylabel("Error norm.")

    if (not args.show_plot):
        print(f"l_two(hh):", h_two)
        print(f"l_inf(hh):", h_inf)

        print(f"l_two(uu):", u_two)
        print(f"l_inf(uu):", u_inf)

    else: plt.show()


if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--test-file", dest="test_file", type=str,
        required=True, help="File containing approx.")

    parser.add_argument(
        "--base-file", dest="base_file", type=str,
        required=True, help="File containing *exact.")

    parser.add_argument(
        "--time-step", dest="time_step", type=str,
        default="end",
        required=False, 
        help="Time-step(s) used in error calculation.")

    parser.add_argument(
        "--show-plot", dest="show_plot",
        type=lambda x: bool(strtobool(str(x.strip()))),
        default=False,
        required=False, help="TRUE to display plots.")

    args = parser.parse_args()

    print("Loading input assets...")

    m1st = load_mesh(args.test_file)
    m2nd = load_mesh(args.base_file)

    print("ref-length:", np.mean(np.sqrt(2.0 * m1st.edge.area)))

    if ("end" in args.time_step.lower()): 
        one_step_error(args, m1st, m2nd)
        
    if ("all" in args.time_step.lower()):
        all_step_error(args, m1st, m2nd)

