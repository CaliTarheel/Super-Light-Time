# Finite collision laws on total material velocity

This extends the [joint mechanics primitive](SHARED_CONTACT_CONTRACT.md) with
the existing local opening weld and adjacent-interface viscous shear laws.
`joint_resistance.py` assembles those laws on plate **and residual sheet**
unknowns; `shared_contact.solve(..., resistance=...)` balances their reciprocal
reactions in one constrained nonlinear solve. Geometry and supplied contact
topology remain fixed. There is no native callsite or saved-policy change.

## Force and measure ownership

For a face pair, each side's velocity at the same unit point `c` is the radial
trace `P_c sum(alpha_i V_i)`, with `alpha = [r0 r1 r2]^-1 c`. It uses that
material face's nodal field, including its residual velocity. Rigid blocks are
assembled as equal and opposite common-point rotation maps. Therefore a common
Euler motion creates no relative velocity and pair torque sums to zero.

**Weld:** `weld_arc` takes an actual first-face free edge lying within both
finite material faces. Its constant outward normal `n` is perpendicular to
`c(theta)=b cos(theta)+t sin(theta)`. The closing rate

```
q(theta) = n dot (V_first(c)-V_second(c))
         = (B0 y) cos(theta) + (B1 y) sin(theta)
```

is still exactly two trigonometric modes with deforming sheets. Positive `q`
advances the first free edge into the overlap; negative `q` peels it away. The
existing `weld_geometry.integrate` law contributes zero for closure, quadratic
opening below the explicit SI smoothing speed, and capped opening traction
beyond it. Its physical potential is

```
capacity_N_per_m * measure_weight * R_m * integral(phi(q) dtheta)
```

All inputs are explicit. The native two-perimeter convention uses weight1/2
on each represented side; the helper does not silently choose that weight.
Actual resisting power is `y dot gradient`, which is generally **not** the
potential. Opposing opening/closing regions are evaluated before integration.

The whole arc is checked against both triangles using the extrema of every
homogeneous coordinate along the arc. Its first edge must be a boundary edge
of the supplied connected mesh and have the supplied outward orientation.
Internal mesh edges do not become welds. Native canonical arcs currently merge
face-pair breakpoints; a future adapter must split them at every change of
either material interpolation face. Merging them through changing residual
fields would lose the actual law even when the geometric line is straight.

**Buried shear:** `interface_patch` takes a supplied adjacent-layer polygon and
the existing explicit viscosity/thickness ratio. Its Rayleigh potential is

```
1/2 * (eta_interface / h_interface) * integral(|V_first - V_second|^2 dA)
```

Both tangent components contribute. Same plate ownership does not remove
relative residual motion between distinct stacked sheets. The caller must
provide the actual adjacent interface: the helper neither determines layer
order nor authorizes direct coupling through an intervening sheet. Existing
native region builders skip same-owner pairs; that behavior needs an explicit
adapter change before using this total-velocity law there. An internal suture's
continued finite weld strength is likewise a policy question, not an automatic
consequence of plate ownership.

Both terms replace their rigid-only contributions when integrated. They must
not be added on top of those same native plate terms. A permanent normal
equality is a different mechanical law; the finite-resistance tests use no
bilateral contact constraints. These terms do not provide hard nonpenetration,
continental intake, the remaining native compression/megathrust laws, or a
finite attachment failure criterion.

## Positive area integration and frozen operators

Each convex interface polygon is triangulated from an existing corner. For a
fan triangle `(a,b,c)`, radial Duffy quadrature uses

```
p = a + u(b-a) + (1-u)v(c-a)
point = p/|p|
dA = R_m^2 * det(a,b,c) * (1-u) / |p|^3 du dv
```

The existing `convex_partition.Edge` filtered exact predicate checks original
binary64 determinant signs and winding. No absolute determinant, negative
weight clipping or area floor repairs invalid geometry. Every quadrature node
uses both original material interpolation faces, not the integration fan's
shape functions. Positive factors give a passive shear Gram matrix.

Successive8/16/32/64 Gauss rules compare the complete Gram matrix in the **same**
first-rule diagonal energy scaling. Acceptance also compares area, the full
rigid rotation metric, and the weaker centroid-spin work against the independent
analytic polygon metric. An explicit allowance covers binary64 cancellation in
representing the latter3x3 metric. These are a posteriori integration estimates,
not rigorous quadrature error certificates. Exhausted supplied rules reject.
The accepted operator remains fixed during Newton evaluation; changing a rule
inside objective calls would destroy derivative consistency.

Weld Hessian factors use exact active angular subintervals and stable positive
trigonometric second-moment axes. The solver checks those declared factors
against the callback Hessian without clipping its eigenvalues. Records bind to
the joint geometry, owner map, tangent basis and radius. Constructing or solving
with a different same-size geometry rejects rather than reusing stale maps.

## Checks and limits

Portable tests cover absolute SI opening power and rigid-law parity, zero
traction on closure, local peeling with zero signed mean rate, full derivative
and positive-factor agreement, same-owner inter-sheet shear, opposite pair
torques, common rotation, polygon/arc subdivision and invalid/stale geometry.
The actual nonlinear solve with both physical laws changes the solved plate
velocities and passes full KKT and true power acceptance. Those are frozen
component results; they do not establish a converged evolving collision.

Native integration still requires an explicit material-mantle/remaining-domain
partition, a consistent physical sheet/basal coefficient policy, finite contact
admission and breakpoints, remaining force terms, scalable algebra, and coupled
history/remesh/timestep validation. The current gravity sheet settings imply
beta=2e18 Pa s/m, versus native plate beta=1e15 times keel factors; reusing both
would silently preserve inconsistent drag. No calibration choice is made here.
The reduced inherited-suture interpretation follows the existing model's
Scotese Rule XI discussion; this change corrects velocity/force coupling, not
the empirical strength or rupture assumptions.
