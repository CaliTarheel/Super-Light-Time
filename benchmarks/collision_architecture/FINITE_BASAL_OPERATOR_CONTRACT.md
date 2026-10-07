# Finite bottom-material basal operator

## Implemented scope

`finite_basal_operator.build` integrates caller-supplied positive allocations
from `basal_partition.partition_cell` on one frozen joint geometry. Bottom
material uses its original finite-face total-velocity trace. Upper material
receives no direct mantle contact. Uncovered control-domain pieces retain the
explicit fractional plate measure. The component can replace the old nodal
basal term and be solved with the actual finite weld and buried-interface laws.

This is research assembly and frozen solve integration. It neither discovers
contacts nor supplies global allocation completeness, native policy migration,
admission, release, remeshing, an evolving mantle, or an accepted timestep.
The caller must supply disjoint control cells and every relevant material
candidate to the partitioner. Repeated represented polygons reject; this
duplicate check is not a proof that arbitrary supplied cells cannot overlap.

## Required API choices

```python
component = build(
    system, allocations, face_id_to_index,
    owner_to_plate_slot=owner_ids,
    mantle_omega_rad_s=omega_m,
    quadrature_relative_tolerance=tolerance,
    max_order=64,
)
updated = replace_basal(
    system, component,
    other_drag_excludes_allocated_basal=True,
)
```

Every allocation retains its named `preserve-native` or explicit
`resolved-material-bottom` policy, cell beta, radius, and saved-area density.
The latter policy changes per-owner legacy basal allocation on covered mixed
cells; integrating it does not make that policy an equivalent conversion.
No coefficient, mantle velocity, accuracy tolerance, or policy is calibrated by
this module. Face IDs and plate IDs map explicitly to the system's slots.
Every mapped owner must already exist in the joint space; this module does not
add an ocean-only plate that the supplied joint assembly omitted.

`mantle_omega_rad_s` is one explicit global Euler vector. Its reference velocity
at unit position c is M(c) = (R omega_m) cross c. General non-Euler mantle fields
are unsupported. In particular, nodal mantle values are not extrapolated over
uncovered areas. A zero vector explicitly selects a stationary reference.

`replace_basal` removes the old nodal basal Hessian, mantle load, and constant.
It retains sheet strain, applied force/torque loads and independent additional
drag. Because the allocations already include the uncovered basal complement,
the retained `other_plate_drag_factor` must exclude that complement. The
mandatory `True` declaration records caller responsibility; code cannot prove
the physical provenance of an arbitrary matrix. Obsolete nodal-work fields are
removed so old work routines fail instead of silently double-counting drag.
An already-replaced system rejects a second replacement.

## Common trace and SI measure

Unknowns are y = [a,z], with a_p = R omega_p in m/s and two tangent residual
coordinates z per material vertex. For original face vertices r_i and tangent
bases E_i, solve sum(alpha_i r_i) = c. These homogeneous radial coordinates are
not normalized barycentrics. The material trace is

    V(c) = a_owner cross c + (I - c c^T) sum(alpha_i E_i z_i).

This reproduces a common Euler velocity whether represented in the plate block
or in the nodal residuals. Surface reactions use the transpose of this same
trace. The generalized gradient is in N; multiplying a plate component by R
converts it to torque in N m.

For bottom polygon P, beta has units Pa s/m, and the measure is
dmu = native_measure_density * dA_m2. Its Rayleigh term in W is

    Phi_P = 1/2 integral_P beta |V(c) - M(c)|^2 dmu.

For uncovered polygon U and native support fraction s_p, its exact finite rigid
metric J_U = integral_U (I - c c^T) dmu gives

    Phi_Up = 1/2 beta s_p (a_p - R omega_m)^T J_U (a_p - R omega_m).

The metric here multiplies equatorial speed, so its Hessian is beta s_p J_U.
Using the allocation's rotational drag beta R^2 s_p J_U directly would add an
erroneous factor R^2. Positive pieces are added; a covered matrix is never
subtracted from a fractional full-cell matrix to manufacture a remainder.

## Positive integration and finite arithmetic

Bottom polygons use positive radial Duffy-Gauss quadrature on every positive
fan. Orders 8,16,32,64,128 are tried only through the explicit supplied maximum.
There is no polygon-area floor and no discarded positive fan. Convexity,
orientation, represented measure and finite-face trace checks reject unresolved
inputs. Tiny signed linear-solve roundoff is retained as in the reviewed radial
trace; it is not a contact-admission tolerance. The caller's PR210 allocation
establishes the finite physical piece, and this integrator does not enlarge it.

Acceptance compares successive complete local Grams in one fixed first-rule
diagonal energy scaling, positive quadrature area against the analytic area,
and the full analytic rigid metric in Cholesky-scaled coordinates. The weak
centroid-spin direction is therefore tested too. The rigid block of the final
compressed factor must pass the same analytic check. Quadrature convergence is
an a posteriori estimate, not a directed integration-error certificate.

Thin QR compresses positive factors into a 9x9 block per piece without clipping
eigenvalues, dropping null directions, or subtracting Grams. A local normalized
Gram check bounds observed compression change. Remaining rigid pieces use a
positive 3x3 Cholesky factor padded to the same local format. Unresolved
arithmetic, including overflow and underflow during construction, fails closed.
All retained factor arrays are detached and marked read-only.

No material-sized dense Hessian or nullspace is allocated. Retained factor
storage is O(number of supplied pieces), with at most nine columns per piece;
geometry, vectors and metadata retain ordinary linear costs. Quadrature scratch
holds one fan rule and a 9x9 previous factor. `to_dense` is restricted to at most
256 unknowns for audits. This is not a native full-world performance benchmark.

## Loads, forces and physical work

Let F be the frozen assembled local factor map. Its reference vector is formed
from the SAME rigid columns, m = F [R omega_m,0], piece by piece. Thus

    Phi = 1/2 |F y - m|^2,
    H = F^T F, load = F^T m, constant = 1/2 |m|^2,
    gradient = F^T (F y - m).

Actual basal dissipation is |F y - m|^2. The prescribed moving-mantle actuator
power is -(F y - m) dot m. Consequently y dot gradient is basal dissipation
minus mantle input. At equilibrium, applied physical power plus mantle input
balances basal, sheet-viscous, additional-drag and finite-contact dissipation.
The reduced load-work identity is not mislabeled as slip dissipation.

The component supplies its Hessian action, diagonal, absolute coefficient-product
action, load, constant, `evaluate`, `work`, and geometry binding. Replacement
publishes the sparse solver's `physical_work`, `physical_work_absolute_scale_w`,
Hessian diagonal/action and `validate_system` hooks. The latter binds points,
faces, owners, tangent bases, radius and unknown layout. The caller must keep
the assembled system frozen throughout the solve. The arithmetic work diagnostic
uses |F| |y| before summing, so cancellation of plate and residual velocities
does not erase the constituent scale. It is an arithmetic diagnostic for the
assembled operations, not a formal bound for ill-conditioned QR/Cholesky or
quadrature assembly.

## Focused evidence

`tests.test_finite_basal_operator` contains 12 tests:

- Independent analytic octant rigid metric and SI force scale.
- Moving-mantle load, actuator sign, dissipation and common-motion zero work.
- Euler reproduction through residual material coordinates.
- Control subdivision preserving the whole residual/plate Gram and load.
- Upper-sheet shadowing and fractional uncovered rigid-domain work.
- Saved-area density and explicit changed-policy retention.
- Positive work, virtual-work gradient, diagonal/action bounds and a cancelling
  plate/residual field that retains nonzero arithmetic constituent cost.
- Actual replacement independent of obsolete nodal drag strength, full physical
  ledger, system binding, duplicate/malformed/arithmetic rejection and local
  storage with more than 256 unknowns.
- An actual nonlinear sparse solve with finite basal, weld and interface laws,
  opposite applied torques, no independent additional drag, and exactly zero
  direct basal Euler diagonal on the covered upper plate. Its resistance comes
  through the interface. Original stationarity and physical-power gates pass.

These checks establish the stated frozen operator and integration contracts.
They do not establish native event admission, timestep convergence, full-world
coverage, or realistic calibrated evolution. The passive resisting role agrees
with the project's application of Scotese Rule I. A nonzero prescribed mantle
field is an explicit test/reference choice with actuator power accounted for;
it is not a new mantle conveyor added to make the simulation move.
