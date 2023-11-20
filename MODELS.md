
`PERISCOPE` provides a nonlinear rotating shallow water model, solved in 
so-called vector-invariant form using an unstructured C-grid discretisation. Thickness, 
velocity and vorticity DoF are staggered at the cells, edges and vertices (duals) of a 
given (orthogonal) mesh.

$$
\frac{\partial h}{\partial t} + \nabla \cdot (u h) = \nu_{k}^{h}\nabla^{k}g(h + z_{b}) + S_{h}\ ,
$$

$$
\frac{\partial u}{\partial t} + q (u h)^{\perp} = 
  -\nabla \Big(g(h + z_{b}) + \phi_{u}\Big) - \nabla \frac{1}{2} |u|^{2} - c_{d} u + \nu_{k}^{u} \nabla^{k} u + \frac{1}{h} \tau_{u} + S_{u}\ ,
$$

$$
c_{d} = c_{1} + (c_{2} + c_{l}) \frac{1}{h} \|u\|\ , 
$$

$$
c_{l} = \kappa^{2}\ \log^{-2}\Big(1+\frac{h}{2 z_{0}}\Big)\ .
$$

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
- $\nu_{k}^{u} \nabla^{k} u$ and $\nu_{k}^{h} \nabla g(h + z_{b})$ are dissipative operators:
  - $\nu_{2}^{u} \nabla^{2} u$ (with $\nu_{2}^{u} =$ `--uu-visc-2`) is a Laplacian viscosity.
  - $\nu_{4}^{u} \nabla^{4} u$ (with $\nu_{4}^{u} =$ `--uu-visc-4`) is a biharmonic viscosity.
  - $\nu_{k}^{\delta} (\nabla \nabla \cdot)^{\frac{k}{2}} u$ (with $\nu_{k}^{\delta} =$ `--du-visc-2` or `--du-visc-4`) are equivalent divergence damping terms.
  - $\nu_{2}^{h} \nabla^{2} g(h + z_{b})$ (with $\nu_{2}^{u} =$ `--hh-diff-2`) is a Laplacian diffusivity.
  - $\nu_{4}^{h} \nabla^{4} g(h + z_{b})$ (with $\nu_{4}^{u} =$ `--hh-diff-4`) is a biharmonic diffusivity.
  - Dissipation coefficients are scaled with the mesh, such that $\nu_{2} = \Big(\frac{\Delta x}{\overline{\Delta x}}\Big)^{1} \nu_{2}$ and
    $\nu_{4} = \Big(\frac{\Delta x}{\overline{\Delta x}}\Big)^{3} \nu_{4}$.
  - $\overline{\Delta x} =$ `--ref-scale` is the reference length-scale. $\overline{\Delta x} \leq 0$ disables the scaling.
- $S_{h}, S_{u}$ are source terms, $\tau_{u}$ is an external stress, and $\phi_{u}$ is an applied geopotential.
