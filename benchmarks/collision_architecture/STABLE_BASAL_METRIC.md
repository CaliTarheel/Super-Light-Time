# Positive local basal metric representation

This is a numerical representation change for the existing integral
`J = integral (I-r r.T) dOmega`. It adds no force, resistance coefficient,
geological mechanism or saved-policy transition. The native geometry is not
moved. It is an offline research component; a successful partition does not
by itself establish successful finite trace integration or a whole native solve.

## Geometry, metric and API

`stable_basal_metric.build(control_triangle, chart_polygon)` accepts the original
represented binary64 control vertices and exact Fraction/integer chart corners.
The control is positively oriented in a common open hemisphere. The chart is
checked exactly for positive simple convex winding within the control triangle.
Boolean/object/complex inputs and out-of-chart geometry reject.

Each original ray is formed by exact rational barycentric interpolation before
normalization in Decimal arithmetic. No intermediate globally rounded spherical
polygon becomes the geometry source. This matters in saved cell36312: rounding
the old spherical corners changes both tiny-region area and centroid-spin work
by several parts in a billion, exceeding the existing full-work limit.

The analytic area and second-moment boundary integral use the same formula as
`collision_interface._precise_rotation_integral`, retaining Decimal values.
A local orthonormal frame `Q=[t1,t2,n]` uses the normalized vertex-sum direction
and a deterministic least-aligned Cartesian reference axis. The metric is
transformed into that frame BEFORE any binary64 matrix cast. A diagonal-scaled
Cholesky produces the positive factor `C`, with `C.T C=J_local`; no eigenvalue
is projected and no positive polygon is deleted.

Returned `StableMetric` fields:

- Original `control_triangle`, exact `chart_polygon` and source signature.
- `frame_hi`, `frame_lo`: two binary64 parts of each global frame entry.
- `local_polygon`: the exact-ray normalized polygon transformed before rounding.
- `metric_local`, `factor_local`, `area_unit` on the unit sphere.
- `polygon` and `metric_global`: globally rounded diagnostics only.
- Selected Decimal `precision` and full-metric/area `convergence_error`.

`localize(values, normalize=False)` preserves original material-vector lengths
and transforms them using the retained Decimal frame. `globalize(values)` is
the matching high-precision frame transpose before the final binary64 cast.
`metric_in(other)` transforms retained analytic values into another metric's
frame before rounding. It is useful for aggregate closure, not a requirement
that an individual tiny piece remain Cholesky-resolvable in a distant frame.
`validate()` checks the complete typed provenance and derived arrays; changing
geometry, frame or factors invalidates the record.

## Precision and acceptance

The default precision sequence is48,80,128,192 decimal digits. Precision is not
accepted just because a rounded eigenvalue is positive. Two successive resolved
analytic evaluations must agree in area and in every local work direction:

```
norm(C^-T (J_previous_in_current_frame-J_current) C^-1, 2) <= 2e-12.
```

The precision-agreement target is separate from, and stricter than, the unchanged
`2e-10` physical partition/full-work gate. Unresolved signs or pivots escalate;
exhausting the explicit schedule rejects. Callers may supply a bounded increasing
schedule (32 through4096 digits). No universal precision is claimed sufficient.
Agreement is a numerical convergence check, not a directed-rounding proof of
the Decimal formula or a guarantee against all correlated algorithmic errors.
Independent positive radial quadrature provides a different oracle in tests.

## Partition integration

Every returned piece retains `stable_metric`; the allocation retains
`control_stable_metric`. Native saved/geometric area density remains unchanged.
Piece metrics are transformed analytically into the common CONTROL frame before
summation and full factor whitening. Per-owner and common-rotation closure use
that representation. Existing global metrics remain diagnostic outputs; they
are not positivity or weak-axis oracles. Exact clipping, positive area retention,
unique bottom order (including same-owner distinct sheets), strict native-support
rejection and explicit resolved-bottom policy remain unchanged.

SI diagnostics use one canonical multiplication by `radius_m**2 * density`.
The same factor scales both local metric and area, preserving source-bound
comparisons with finite integration. No global-minus-material remainder is used.

## Operator requirement and limits

Consumers must retain the local representation through factor action, transpose,
diagonal and work. Precomposing `C Q.T` into a single globally rounded block can
lose the weak mode again. Runtime compensated frame transforms or an equivalent
validated representation are required; this module provides retained Decimal
build-time transforms and split frame entries, not a new runtime sparse solver.
A final binary64 global force can lose weak-direction work when later contracted
against an almost-normal global vector. Compare physical work in factor space
and account for final-cast arithmetic; do not claim a naive global dot product
is relatively accurate in a cancellation-dominated weak direction.

The focused tests include the saved exact cell36312 chart, independent100-digit
positive quadrature for actual represented normal/weak-oblique vectors, generic
proper rotations, exact chart subdivision, both frame transforms, tampering,
typed inputs and insufficient-precision rejection. Rotating and then rounding
a weak-balanced vector changes its represented physical work; each rotated
case is checked against its own exact-input oracle. Partition/native basal
regressions retain original policy and measure tests. Full saved cell337/36312
finite operator acceptance remains the integration requirement after this
primitive, not a conclusion from positive local eigenvalues.

## Exactly certified frame axes

A symmetric polygon can have an exactly zero component of its normalized
vertex-sum direction. Recomputing `n cross unit(n cross e_j)` in finite Decimal
precision can nevertheless produce a tiny residual around the exact component
`-1` (the saved control31 example produced a1e-80 low word). Multiplying this
artifact by a small, otherwise valid runtime vector can exceed the compensated
binary64 product range. Disabling that range check would accept unresolved
products and is not the repair.

The frame builder now forms the original Fraction chart rays before Decimal
conversion and groups them by **exact squared length**. Coordinate `j` is
certified zero only when its exact component sum is zero separately within
every group. Every group shares the same normalization denominator, so the sum
of normalized rays has exactly zero component `j`; further normalization keeps
it zero. A raw component sum across unequal-length rays, a Decimal zero, or a
small magnitude does not prove this condition.

Only certified components are set to exact Decimal zero before normalizing the
centroid. If the least-aligned reference axis is a certified coordinate axis,
`n dot e_j=0` and `||n||=1` imply

`unit(n cross e_j) = n cross e_j`,

`n cross unit(n cross e_j) = -e_j`.

The second frame column is then constructed as exactly `-e_j`. All other cases
retain the original cross-product construction. This is a sufficient algebraic
certificate, not an exhaustive test of radical cancellations or a tolerance
for nearby geometry. The analytic area/metric, original polygon, convergence
thresholds, positive factor and runtime compensated-product gates are unchanged.

`test_basal_exact_axis` preserves the original control31 triangle and exact full
chart, proves the saved low-word failure is gone in frame action/transpose,
keeps the genuine nonzero out-of-range product rejected, and checks unequal-norm
raw cancellations,1e-30/minimum-subnormal nonzeros, asymmetric exact charts and
proper coordinate rotations. An actual uncovered finite factor still matches
independent positive radial quadrature for rigid/weak-axis work. This does not
claim completion of the global basal store or a native coupled solve.

## Exceptional exact products

The ordinary compensated Split/TwoProduct path and its conservative operand/
exponent envelope are unchanged. A lane outside that fast envelope is now
masked out before Split and evaluated separately using the exact Fraction
product of its original finite binary64 operands. Let `hi` be its nearest
binary64 value and `lo` the nearest binary64 residual. The lane is accepted only
when both words are finite, the exact product magnitude is within finite
binary64 range, and `Fraction(hi) + Fraction(lo) == exact_product` exactly.

This is an exact representability gate, not a relaxed physical error budget.
An overflow, a true underflow, or a residual that cannot be represented in the
two words still rejects. Zero products remain exact without splitting extreme
operands. The saved native48..63 exceptional products at exponent sum-925 pass;
the earlier1e-80 frame artifact times its tiny operand still fails because its
remaining fraction of the minimum subnormal is nonzero.

No whole-dot or whole-operator exceptional fallback is introduced. Subsequent
sum/product/finite-work checks retain their existing acceptance rules. A large
finite projection that has exact product expansions is now accepted and checked
against an independent Decimal projection; squared physical work at that scale
may still overflow and is not claimed supported. Basal measures, coefficients,
contact laws and native state remain unchanged.
