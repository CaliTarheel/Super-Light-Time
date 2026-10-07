# One basal measure on a finite control cell

`basal_partition.partition_cell` is an integration component for joint plate and
sheet mechanics. It is not called by the native engine and selects no saved
policy or physical coefficient. It partitions a supplied control triangle by
all supplied material triangles, selects the bottom material layer, and builds
remaining plate drag from positive uncovered pieces. It never obtains a
remainder by subtracting a material matrix from a global matrix.

## Native measure and the constraint on ownership

The native plate operator treats each support fraction as a constant density
throughout its control cell. For plate p and cell C,

```
d mu_p = support[p,C] * rho_C * dA
rho_C = saved_cell_area / geometric_cell_area
K_p = beta_C * integral (I-r r^T) d mu_p
```

The existing saved integration weight is therefore retained explicitly, even
if it differs from current geometric area. This density is numerical measure
provenance, not a statement that physical area changed. `K` has N s/m units for
the unknown `a=R*omega`; returned rotational drag is `R^2*K`, in N m s. Beta,
radius and saved area are mandatory positive inputs. No new beta or viscosity
calibration is chosen here.

A resolved bottom material footprint has unit density on its own owner. It
can replace part of the native measure without changing that law only in a
pure matching-owner cell. A half-cell footprint cannot consume a half support
fraction throughout the full cell: that would leave negative density under the
footprint and positive density elsewhere. The tests independently exhibit an
indefinite matrix from that tempting area-only subtraction.

The default `preserve-native` allocation rejects every positive material piece whose
bottom owner has less than unit support, or where any other owner has nonzero
support. It reports the face, owner and actual fractions. It does not snap tiny
fractions, normalize support, choose a dominant plate, or infer subcell geometry.
Uncovered pieces retain the original fractional homogenization. This is a
representation limitation, not a claim that a real collision cannot occur.

The separately selected **`resolved-material-bottom`** research policy assigns
covered pieces to their unique physical bottom owner, even where native support
disagrees. Uncovered pieces retain the original fractions. This is an explicit
geometric reconstruction of the basal footprint and **changes each affected
plate's rigid force law**. It is never selected as an automatic fallback. All
bottom owners must be present in the supplied active owner map, including an
explicit zero entry when they have no native support in this cell.

The result reports policy, reassigned area/piece count and per-owner legacy
metric differences. Those differences may be indefinite because they compare
two different positive allocations; they are diagnostics and are never used
as drag. The actual material and remaining operators are assembled only from
positive pieces. Summing all owners still closes the full common-rotation
metric. Tiny native normalization roundoff is retained and checked; fractions
are never silently normalized. This policy does not reconcile transported
support with other native processes, decide the correct beta, or authorize a
production migration. Native selection needs explicit saved-policy provenance
and a complete force/domain allocation review.

## Finite geometry and physical order

For one control triangle, its homogeneous radial chart is the reference
triangle `(0,0),(1,0),(0,1)`. Each original material edge defines an exact linear
halfspace there using rational arithmetic on the represented binary64 inputs.
Successive inside/outside partitions share the same exact crossing. Every
positive chart piece is retained; exact total chart area remains1/2. There is
no inside tolerance band or positive-area floor.

Each piece carries all supplied covering face IDs. Persistent upper/lower
sheet order must give a unique bottom; same-owner sheets are still distinct
layers. Positive overlap within one sheet, ambiguous bottoms and cycles reject.
This resolves only the bottom layer. It does not prove a complete adjacent
ordering of the upper layers or authorize interface traction between them.

Casting rational chart pieces back to spherical binary64 polygons can lose
geometry. Collapsed corners, unresolved positive area, bad convex winding or
negative computed resisting work fail closed. Analytic area/rotation integrals
use the existing Decimal implementation. The summed area and complete3x3
rotation metric must agree with the original control cell, including its weak
axis, in the fixed full-cell energy scaling. This is a numerical acceptance
estimate with an explicit relative tolerance, not an interval-certified bound.
No negative metric is projected to positive eigenvalues.
Unrepresentable derived SI coefficients, areas, metrics and drag also reject;
finite input scalars alone do not establish finite assembled output.

For accepted `preserve-native` allocations, every owner's material rigid metric plus
remaining metric recovers the original native finite-cell metric. An upper
sheet never acquires its own second mantle contact. The joint operator must
integrate **total material velocity** on the returned bottom polygons; reusing
full-face nodal basal drag at the same time would double-count it again.
Moving-mantle linear/constant terms and deforming trace integration are not
provided by this geometry/rigid-metric component.

## Evidence and boundaries

Run `python -B -m unittest tests.test_basal_partition -v`.

Independent tests compare against the native finite-cell mean and analytic
octant moments, verify partial material/complement positivity, same-owner and
different-owner upper-layer screening, saved-weight provenance, rotation and
subdivision, tiny finite footprints and rejected ownership/geometry cases.
These checks concern a supplied control cell and supplied candidate set.

The caller must establish candidate completeness and global control coverage.
This function cannot discover omitted material faces, prove a full control
mesh has no holes/overlaps, resolve oceanic subcell ownership, or certify a
native evolving trajectory. Source geometry, order, support and coefficients
must be immutable over any force solve using its output.

This follows the project's Rule I force-balance intent while exposing where
the inherited fractional discretization lacks a needed physical interface.
Rejecting unsupported input is a computational limitation, not a geological
rule or evidence that the different native plate/sheet drag values are correct.
