# Native basal domain: offline complete assembly adapter

This research adapter constructs a globally complete allocation **only when
every cell and finite operator check succeeds**. It does not change a saved
policy, update native support, install an engine, or advance a timestep. A
successful global control-mesh certificate is not a successful global basal
operator. The preserved 642 Myr checkpoint currently has documented unresolved
finite-piece arithmetic, so no global native operator has been demonstrated.

## API and explicit policy

`prepare(state, allocation_policy=...)` returns a detached `NativeBasalDomain`
bound by content fingerprint to the source geometry, native support, saved
areas, crust-dependent coefficient, active owner slots/UIDs and sheet order.
The derived candidate index, geometry arrays and owner/face mappings are also
content-bound; swapping two ocean-only slots cannot reuse a valid certificate.
The required policy is `preserve-native` or `resolved-material-bottom`.

The former rejects any positive bottom material on a mixed or mismatched
native control measure. The latter explicitly changes the allocation on
covered pieces: only the bottom sheet gets total-velocity basal drag; uncovered
pieces retain saved native fractions. It preserves the selected cell measure
and common-rotation work, and reports per-owner legacy metric changes. It does
not claim unchanged per-owner native work or infer missing subcell ownership.

* `domain.probe_cells(unique_control_ids)` returns an explicitly partial
  diagnostic. It cannot be supplied back as a global build certificate.
* `domain.build(system, mantle_omega_rad_s=..., quadrature_relative_tolerance=...,
  max_order=...)` visits **every** control ID exactly once and returns
  `NativeBasalBuild(component, certificate)` only after all checks pass. There
  is no subset argument. A failure publishes no operator or native state.
* Use `finite_basal_operator.replace_basal` to replace an existing research
  system's nodal basal term. Its explicit declaration that other drag excludes
  the allocated complement remains necessary. This adapter never adds a
  duplicate nodal term and never calls a native solver itself.

All active native slots, including ocean-only plates, map in the exact order
`np.flatnonzero(state.active)`. Material owner IDs remain native slots until
the explicit `owner_to_plate_slot` mapping at trace assembly. The associated
sparse API must retain `plate_count=len(active_native_slots)` and use
`preserve_represented_points=True`; re-normalizing stored coordinates changes
the represented geometry and is refused by exact system binding. Fixed
residual coordinates and their kinematic meaning belong to the state/solver
adapter, not this allocation component.

## Complete finite measure

Each control face is a positive minor spherical triangle. Its directed indexed
edges must occur exactly twice in opposite directions. This forms a closed,
positively oriented spherical 2-chain: its integer coverage multiplicity is
constant away from edges because the oriented boundary cancels. One point
strictly off every control edge is tested by filtered exact original-edge
predicates and must have multiplicity one. This establishes one global cover
up to zero-area edges, including exclusion of a duplicated entire sphere.
Holes, reversed faces and repeated edge incidence reject. This is stronger
than checking only a total-area sum.

Candidate discovery builds a **fresh** native cap/bin locator, including the
native `global_faces` list. The locator's inverse-map admission normally
rejects determinants below 1e-14. Candidate enumeration needs no inverse:
these valid positive faces are retained in an additional always-considered
global list, and regular subset indices are mapped back to original faces.
No cached positive coverage or its 1e-10 km² area floor is used. Exact original
edge halfspaces may reject separated pairs; exact rational control-chart
intersection may reject only zero area. Every positive rational intersection
is passed to PR210's exact atomic partition.

The adapter's local broadcast predicate uses the existing `convex_partition.Edge`
translated determinant and conservative64-eps arithmetic filter, delegating
ambiguous original-edge signs to `Edge.distances` and its exact Fraction fallback.
Exact endpoints remain zero and unrepresentable nonzero determinants reject.
This avoids an unrelated unpublished helper dependency; it introduces no
physical inside band, area threshold or different halfspace convention.

The native locator uses its existing 1e-12 outward box padding. Inputs are
restricted to represented unit coordinates within 64 binary64 eps and a
resolved open hemisphere. The geometric cap argument and exact narrowphase
are conservative within this documented binary64 arithmetic model; the
transcendental/norm computation is not a formally directed interval proof.

Every atomic cover must contain distinct sheets and have a unique bottom.
Additionally, every simultaneous layer pair must be comparable in the saved
persistent sheet-order graph. A bottom shared by two otherwise unordered upper
layers is not accepted as a complete stack. Same-owner layers are kept
separate. Owner, height, age and triangle ID never choose order. Current native
GPE's averaged-order exception below 1 km² does **not** supply a physical basal
order and is not inherited here. Missing order returns control, face and sheet
IDs and an exact positive chart witness when available.

PR210 supplies the native saved-area/geometric-area density and positive pieces.
Beta exactly matches native `plate_balance._viscous`'s selected asthenosphere
constant times clipped crust-kind keel coefficient. All pieces retain that
coefficient and density. Every cell has unchanged area and full rotational
metric closure gates; global closure and presence of every positive material
face are checked before publication. No matrix subtraction, PSD projection,
area omission, owner-total reassignment or tolerance relaxation is used.

## Finite total-velocity operator and arithmetic scope

PR213 integrates the original material-face homogeneous radial trace over each
bottom polygon, including residual motion; uncovered fractions use analytic
rigid metrics. A single explicit global Euler mantle reference is integrated on
the same measure. Forces remain the transpose of these positive local factors.
Moving-reference work, actual dissipation and external power retain PR213's
sign and SI conventions. This adapter does not choose an arbitrary mantle
motion, drag coefficient, remaining force or contact law.

Partition scratch is batched by 16 control cells. Output factors remain at most
9 local columns per piece/fraction and scale with the actual number of pieces;
no global unknown-by-unknown dense array is formed. A heavily intersected
control can still create many exact atomic pieces and take substantial time.
This is not a promised full-world build-time optimization.

The accompanying numerical repair replaces `q @ inv(points.T).T` with
`solve(points.T, q.T).T`. Both represent the identical homogeneous trace.
The positive-weight, barycentric sign, reconstruction, quadrature, compression
and full weak-axis metric acceptance gates are unchanged.

## Evidence and remaining limitation

The focused suite includes global spherical coverage, duplicated full-sphere
rejection, strict mixed-support rejection, explicit changed policy, ocean-only
slot mapping, triple-layer order, same-owner order, exhaustive candidate parity,
native global-face fallback, control subdivision, rotation, stale source/system
binding, positive geometry below the old cache floor, and a saved native trace
regression. The 12 pre-existing finite basal tests also pass after the direct
solve repair.

Read-only probes of the immutable 642 Myr checkpoint (`d63d6cf3c4668991...`) are
saved under `reviews/native-joint-integration-20260929` in the task workspace.
All 81,920 native controls pass the global control certificate; all 11 active
plates are retained. The complete material mesh has 52,025 faces. Cell337
admits 24 exact positive candidates and 388 positive pieces under the explicit
resolved policy, with full metric closure about 9.65e-12. Its original trace
failed reconstruction at 9.14e-14 against the unchanged 5.68e-14 gate; direct
solve gives 1.59e-16 on that same regression.

That fix advances the complete cell337 attempt to its unchanged weak-axis
quadrature gate, which still rejects. Across orders8/16/32/64 the area error is
at most2.9e-16 and the full operator changes by about1e-13, but the globally
rounded weak-axis comparison fluctuates between1.26e-9 and4.48e-9 against the
2e-10 gate. This is not reported as a successful full-cell integration.

Collision cell36312 contains a positive 0.002163635894 m² region. Its correctly
integrated but globally rounded 3×3 metric has an apparent negative eigenvalue
of -1.54e-33 alongside two 5.33e-17 unit-sphere eigenvalues. The weak centroid
rotation is below the resolution of that global matrix representation. It
remains rejected and its geometry is saved. A future repair needs a common
local-frame positive metric/factor representation throughout partition,
finite integration and independent weak-axis work checks. Removing the
eigenvalue gate or dropping the region would not be that repair. No full-world
operator, coupled solve, timestep accuracy or physical collision closure is
claimed by these partial probes.

An audit-only100-digit evaluation puts the same36312 metric into a local frame
**before** binary64 rounding. Its centroid-axis work is positive
5.61962089936e-34 on the unit sphere; independent positive8×8 Duffy quadrature
agrees within1.74e-15 relative. This identifies a viable representation target,
not an implemented repair. Rotating the already-rounded global matrix cannot
recover the lost mode. The eventual factor/action and its work oracle must
retain that representation end-to-end, including the effect of frame rounding.
