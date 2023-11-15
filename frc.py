
hh_cell_input(Time, nCells)
hh_cell_relax      (nCells)

uu_edge_input(Time, nEdges)
uu_edge_relax      (nEdges)

hs_cell_input(Time, nCells)  # hh source term

us_edge_input(Time, nEdges)  # uu source term: body forces, stresses, etc...


# drags should go into init_file, because they need state + implicitness
# ld_edge(nEdges)  # linear ld * uu / hh, also --linlaw-cd?
# z0_edge(nEdges)  # spatially dependent roughness for --loglaw

