
### `Formulation`

`PERISCOPE` provides a nonlinear rotating shallow water model, solved in so-called 
vector-invariant form using an unstructured C-grid discretisation. Thickness, velocity 
and vorticity DoF are staggered at the cells, edges and vertices (duals) of a given (orthogonal) mesh.

$$\begin{gather}
\frac{\partial h}{\partial t} + \nabla \cdot (u h) = \nu_{k}^{h}\nabla^{k}g(h + z_{b}) + S_{h}\ ,
\\\\\\
\frac{\partial u}{\partial t} + q (u h)^{\perp} = 
  -\nabla \Big(g(h + z_{b}) + \xi_{u}\Big) - \nabla \frac{1}{2} |u|^{2} 
  -c_{d} u + \Big(\nu_{k}^{u} + \nu_{k}^{t}\Big) D^{k} u + \frac{1}{h} \tau_{u} + S_{u}\ ,
\\\\\\
c_{d} = c_{1} + (c_{2} + c_{l} + c_{m}) \frac{1}{h} \|u\|\ , 
\\\\\\
c_{l} = \kappa^{2}\ \log^{-2}\Big(1+\frac{h}{2 z_{0}}\Big)\ , \quad c_{m} = n_{0}^{2} g h^{-\frac{1}{3}}\ ,
\\\\\\
D^{2} u = \nabla \Big(h^{-1} \nabla \cdot (h u)\Big) - h^{-1} \nabla^{\perp} \Big(h \nabla \times u\Big)\ ,
\\\\\\
D^{4} u = \nabla \Big(h^{-1} \nabla \cdot (h \nabla^{2}u)\Big) - h^{-1} \nabla^{\perp} \Big(h \nabla \times \nabla^{2}u\Big)\ .
\end{gather}$$

- $u$ (`uu_edge`) is the horizontal velocity, staggered at edge normals.
- $h$ (`hh_cell`) is the fluid thickness, integrated over primal cells.
- $\zeta$ (`rv_dual`) is the relative vorticity $\nabla \times u$, computed on dual cells.
- $q$ (`pv_dual`) is the potential vorticity $h^{-1}(\zeta + f)$, computed on dual cells.
- $K$ (`ke_cell`) is the kinetic energy $\frac{1}{2}|u|^{2}$, integrated over primal cells.
- $\delta$ (`du_cell`) is the velocity divergence $\nabla \cdot u$, evaluated on cells.
- $f$ (`ff_cell`, `ff_edge`, `ff_dual`) is the Coriolis parameter.
- $z_{b}$ (`zb_cell`) is the height of the bottom topography, integrated over primal cells.
- $g$ is the acceleration due to gravity, $\kappa = 0.4$ is the von Karman parameter.
- $c_{d} u$ is a composite drag law:
  - $c_{1} =$ `--linlaw-cd` is a linear drag coefficient.
  - $c_{2} =$ `--sqrlaw-cd` is a quadratic drag coefficient.
  - $c_{l}$ is a log law-of-the-wall formulation with roughness $z_{0} =$ `--loglaw-z0`.
  - $c_{l}$ is bracketed by `--loglaw-lo` and `--loglaw-hi`.
  - $c_{m}$ is a Manning's drag formulation with roughness $n_{0} =$ `--manlaw-n0`.
  - $c_{m}$ is bracketed by `--manlaw-lo` and `--manlaw-hi`.
- $\nu_{k}^{u} D^{k} u$ and $\nu_{k}^{h} \nabla^{k} g(h + z_{b})$ are dissipative operators:
  - $\nu_{2}^{u} D^{2} u$ (with $\nu_{2}^{u} =$ `--uu-visc-2`) is a Laplacian dissipation.
  - $\nu_{4}^{u} D^{4} u$ (with $\nu_{4}^{u} =$ `--uu-visc-4`) is a biharmonic dissipation.
  - $\nu_{k}^{\delta} (\nabla \nabla \cdot)^{\frac{k}{2}} u$ (with $\nu_{k}^{\delta} =$ `--du-visc-2` or `--du-visc-4`) 
    are equivalent divergence damping terms.
  - $\nu_{2}^{h} \nabla^{2} g(h + z_{b})$ (with $\nu_{2}^{u} =$ `--hh-diff-2`) is a Laplacian diffusivity.
  - $\nu_{4}^{h} \nabla^{4} g(h + z_{b})$ (with $\nu_{4}^{u} =$ `--hh-diff-4`) is a biharmonic diffusivity.
  - $\nu_{2}^{t} D^{2} u$ is an eddy viscosity closure, with $\nu_{2}^{t}$ determined by a sub-grid model. Presently 
    the Leith closure is supported, where $\nu_{2}^{t} = \big(\chi_{l} \delta_{l}\big)^{3} |\nabla \nabla \times u|$
    with $\chi_{l} =$ `--leith-chi` and $\nu_{2}^{t}$ bounded below `--leith-max`. $\delta_{l}$ is a measure of the 
    local mesh spacing.
  - Dissipation coefficients are scaled with the mesh, such that 
    $\nu_{2} = \big(\frac{\delta}{\Delta}\big)^{1} \nu_{2}$ and
    $\nu_{4} = \big(\frac{\delta}{\Delta}\big)^{3} \nu_{4}$.
  - $\Delta =$ `--ref-scale` is the reference length-scale. $\Delta \leq 0$ disables the scaling.
- $S_{h}, S_{u}$ are source terms, $\tau_{u}$ is an external stress, and $\xi_{u}$ is an applied geopotential.
- Radiative inflow / outflow boundary conditions are defined by the external velocity and thickness forcing `uE_edge`, `hE_edge`.

### `Model Output`

`PERISCOPE` outputs a number of default dynamical variables, saved in the `out_<your-case-name>.nc` 
file. Output can be customised using the `--save-vars` cmd-line argument, which should be a 
comma-delimited list that can include:

- `uu_edge`, `vv_edge`: edge-aligned normal and tangential velocities.
- `ux_cell`, `uy_cell`, `uz_cell`: cell-centred cartesian velocity components.
- `hh_cell`, `hh_edge`, `hh_dual`: fluid thickness, integrated on various staggered control volumes.
- `zt_cell`: cell-centred fluid upper surface.
- `du_cell`: cell-centred divergence $\nabla \cdot u$.
- `uh_cell`: cell-centred flux divergence $\nabla \cdot (u h)$.
- `ke_cell`: cell-centred kinetic energy $\frac{1}{2} |u|^{2}$.
- `rv_dual`, `pv_dual`: dual-centred vorticity $\nabla \times u$ and $h^{-1}(\zeta + f)$.
- `rv_cell`, `pv_cell`: cell-remapped vorticity $\nabla \times u$ and $h^{-1}(\zeta + f)$.
- `pv_bias`, `ke_bias`, `hh_bias`: upwind bias in advection scheme, if appropriate upwind scheme is selected.
- `nu_turb`: time varying turbulent eddy viscosity $\nu_{2}^{t}$.

