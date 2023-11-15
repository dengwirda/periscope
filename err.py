
import math
import numpy as np
import argparse

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


def one_step_error(m1st, f1st, m2nd, f2nd):

#-- compute error wrt. one time step

    h1_cell = f1st.hh_cell[-1, :, 0]
    h2_cell = f2nd.hh_cell[-1, :, 0]

    eps_ = np.finfo(h1_cell.dtype).eps * 2

    h_mag = np.max(np.abs(h2_cell)) + eps_

    l_two = np.sqrt(norm_cell(m1st, (h1_cell - h2_cell) ** 2))/ h_mag

    l_inf = np.max(np.abs((h1_cell - h2_cell))) / h_mag

    print("l_two(hh):", l_two)
    print("l_inf(hh):", l_inf)

    u1_edge = f1st.uu_edge[-1, :, 0]
    u2_edge = f2nd.uu_edge[-1, :, 0]

    eps_ = np.finfo(u1_edge.dtype).eps * 2

    u_mag = np.max(np.abs(u2_edge)) + eps_

    l_two = np.sqrt(norm_edge(m1st, (u1_edge - u2_edge) ** 2))/ u_mag

    l_inf = np.max(np.abs((u1_edge - u2_edge))) / u_mag

    print("l_two(uu):", l_two)
    print("l_inf(uu):", l_inf)

    r1_dual = f1st.rv_dual[-1, :, 0]
    r2_dual = f2nd.rv_dual[-1, :, 0]

    eps_ = np.finfo(r1_dual.dtype).eps * 2

    r_mag = np.max(np.abs(r2_dual)) + eps_

    l_two = np.sqrt(norm_dual(m1st, (r1_dual - r2_dual) ** 2))/ r_mag

    l_inf = np.max(np.abs((r1_dual - r2_dual))) / r_mag

    print("l_two(rv):", l_two)
    print("l_inf(rv):", l_inf)

    p1_dual = f1st.pv_dual[-1, :, 0]
    p2_dual = f2nd.pv_dual[-1, :, 0]

    eps_ = np.finfo(p1_dual.dtype).eps * 2

    p_mag = np.max(np.abs(p2_dual)) + eps_

    l_two = np.sqrt(norm_dual(m1st, (p1_dual - p2_dual) ** 2))/ p_mag

    l_inf = np.max(np.abs((p1_dual - p2_dual))) / p_mag

    print("l_two(pv):", l_two)
    print("l_inf(pv):", l_inf)
    
    
def all_step_error(m1st, f1st, m2nd, f2nd):

#-- compute error wrt. all time step

    h1_cell = f1st.hh_cell[:, :, 0]
    h2_cell = f2nd.hh_cell[:, :, 0]

    eps_ = np.finfo(h1_cell.dtype).eps * 2

    l_two = np.zeros(h1_cell.shape[0], dtype=float)
    l_inf = np.zeros(h1_cell.shape[0], dtype=float)

    for step in range(0, h1_cell.shape[0]):
        h_mag = np.max(np.abs(h2_cell[step, :])) + eps_
    
        l_two[step] = \
            np.sqrt(norm_cell(m1st, (h1_cell[step, :] - 
                                     h2_cell[step, :]) ** 2)) / h_mag

        l_inf[step] = np.max(np.abs((h1_cell[step, :] - 
                                     h2_cell[step, :]))) / h_mag

    print("l_two(hh):", l_two)
    print("l_inf(hh):", l_inf)

    u1_edge = f1st.uu_edge[:, :, 0]
    u2_edge = f2nd.uu_edge[:, :, 0]
    
    eps_ = np.finfo(u1_edge.dtype).eps * 2

    l_two = np.zeros(u1_edge.shape[0], dtype=float)
    l_inf = np.zeros(u1_edge.shape[0], dtype=float)
    
    for step in range(0, u1_edge.shape[0]):
        u_mag = np.max(np.abs(u2_edge[step, :])) + eps_
    
        l_two[step] = \
            np.sqrt(norm_edge(m1st, (u1_edge[step, :] - 
                                     u2_edge[step, :]) ** 2)) / u_mag
                                     
        l_inf[step] = np.max(np.abs((u1_edge[step, :] - 
                                     u2_edge[step, :]))) / u_mag

    print("l_two(uu):", l_two)
    print("l_inf(uu):", l_inf)


if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "--test-file", dest="test_file", type=str,
        required=True, help="File containing approx.")

    parser.add_argument(
        "--base-file", dest="base_file", type=str,
        required=True, help="File containing *exact.")

    args = parser.parse_args()

    m1st = load_mesh(args.test_file)
    m2nd = load_mesh(args.base_file)

    print("delta(m):", np.mean(m1st.edge.clen))

    f1st = load_flow(args.test_file, m1st)
    f2nd = load_flow(args.base_file, m2nd)
    
    if (f2nd.hh_cell.shape[0] == 1):
        one_step_error(m1st, f1st, m2nd, f2nd)    
    else:
        all_step_error(m1st, f1st, m2nd, f2nd)

