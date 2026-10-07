# Intact finite attachment preflight and remap primitive

`attachment_transaction.py` implements **research infrastructure**, not native
attachment admission, a rupture law, or a collision trajectory. No production
module calls it; no saved policy, checkpoint schema or engine behavior changes.
The caller must explicitly author an intact finite bond with event provenance.
Overlap, owner identity and old accretion counters do not create one implicitly.

## Durable material object

A store has version1, scope`explicitly-authored-intact-finite-bonds`, and a list
of uniquely identified bonds. Each bond carries a creation event/time and two
distinct birth-root IDs. Each endpoint has three **homogeneous root-chart rows**.
Both triples parameterize the same reference triangle: its corners correspond
pairwise. This common domain, rather than two unrelated area totals, defines
which finite material portions are attached. Root identity merely narrows the
candidate search; it never binds every descendant of that root.

This primitive supports positive-area triangles only. General polygons can be
represented by separately authored paired triangles, but this version does not
validate a cross-record tiling or de-duplicate overlapping records. It rejects
zero-area records, missing provenance and same-birth-root endpoint pairs. It
does not construct a witness from a line/point contact, choose an attachment
strength, or verify that a caller's creation event physically authorized a bond.
Those remain admission responsibilities. An explicit record is an input
assertion, not proof of its physical history.

For a live leaf with homogeneous root chart`H` and endpoint corners`P`, reference
coordinates`lambda=(1-x-y,x,y)` reach that leaf where

```
lambda P inverse(H) >= 0
```

All three inequalities are intersected with the common reference triangle.
Every matching leaf footprint is found, including a leaf reached only by the
patch interior. Source and receiver partitions are then intersected in this
same domain. The result is a list of paired current leaf IDs and finite shared
subcells. No normalized-row approximation, nearest-neighbor fill, root-wide
union, world-space overlap test or1 km² area threshold participates.

## Conservation and arithmetic contract

The durable endpoint charts never change during remapping. Only derived current
leaf incidence changes. Every endpoint must cover its whole common domain once;
positive-area duplicate coverage and missing coverage are independently
rejected. Every paired subcell must have the same owner UID on both endpoints.
A valid breakup can carry corresponding portions together to different owners;
an owner change that separates any intact paired portion is unsupported.

Chart coordinates must have real integer or binary16/32/64 floating dtypes.
Object, complex, boolean, string, datetime and wider floating dtypes are
rejected before conversion, consistently for ledger endpoints and live leaf
charts. Integer values and represented floating values are converted to exact
rationals without first rounding integers through binary64. Authored arrays
retain their original accepted dtype; signatures encode dtype, shape and value
bytes, never object pointers. Independently allocated equal representations
have identical signatures and survive the typed checkpoint serialization test.

Exact rationals are used for3×3 chart inversion and two-dimensional convex
clipping. Reported areas are exact fractions of the
**event reference triangle**, not present-day physical area, crust volume,
traction or strength. Fractions and polygon vertices are emitted as decimal
numerator/denominator strings, so diagnostics are serializable. No arithmetic
area floor can silently release a narrow bond.

This exactness concerns the *represented chart data*. It does not prove that
native rounded charts always form an exact partition of their intended
material. A represented sliver gap or overlap fails closed; this primitive
provides no rounding repair, interval uncertainty model or general numerical
chart-conformity guarantee. The measured native refinement cases below passed,
but broader charts and operations still need validation before integration.
Runtime/scaling on the production mesh has not been measured.

## Transaction boundary

`prepare_transaction(store, current_leaves, proposed_leaves)` validates current
and proposed representations, returns an unchanged deep-copied durable store,
derived incidence and exact-input SHA256 signatures. It never changes an input
or a simulation object. `validate_prepared` recomputes the preflight and rejects
stale or modified records, coordinates, IDs, owners or staged results. Returned
data remain ordinary mutable Python values; callers must use this revalidation
immediately before a serialized native commit.

This is **preflight**, not a commit/rollback framework. In particular, it does
not intercept existing native owner writes or protect them against concurrent
mutation. Future integration must stage it before the first allocation/owner
change and commit its ledger atomically with the accepted material transaction.
It may not silently delete a record to make a split succeed. Unsupported cuts
raise; no detachment criterion, cooldown or permanent transfer ban is supplied.

## Focused evidence

Run:

```sh
python -B -m unittest tests.test_attachment_transaction -v
```

The tests exercise small footprints without remote equal-root siblings,
interior-only child coverage, actual native homogeneous charts through three
refinement levels (4/16/64 descendants), partial footprint retention under parent
return and checkpoint serialization, nontrivial paired endpoint correspondence,
matched versus unmatched owner changes, missing/duplicate coverage, tiny finite
patches, stale staged data, and legacy/degenerate rejection. The checkpoint test
uses the actual typed encoder/decoder and NPZ round trip; it does not add this
ledger to native checkpoint/frame schemas or prove a native restart trajectory.
Parent return checks the proposed material representation, not native coarsening
registry or conserved phase-field integration.

The executable test`test_real_transfer_plans_cannot_cut_an_explicitly_authored_intact_bond`
also exercises the existing unmodified24-face, three-body fixture
in`tests/test_terrane_attachment.py`:

- The ordinary first docking at102 Myr transfers the eight-face terrane A→B.
- After that accepted event, the diagnostic explicitly authors one intact
  paired finite witness on engaged source face0 and receiver face8. Its measured
  birth footprint is1.892308879020243 km². The footprint was selected strictly
  inside their positive overlap; this selection demonstrates material identity,
  **not a physical attachment-admission law**.
- The real planner next proposes B→A at104,106 and108 Myr. Applying this
  primitive externally as a preflight guard rejects each unsupported cut before
  `apply_accretion`. Owners, omega, material arrays and gross-accreted counter
  are unchanged by each guard. There remains one actual accretion event and
  gross accreted area100722.0970845402 km².
- Native planner calls still update their ordinary clocks/maturity before the
  guard. This diagnostic is consistency enforcement around existing functions,
  not an integrated atomic native transaction, force-balanced trajectory,
  physical rupture solution or repair of the historical26 transfers.

## Exact integration gaps and next measurable gate

The larger lifecycle still requires accepted finite-contact admission, actual
receiving-material identity, conservative remapping, intact attachment closure
in transfer eligibility, physically justified finite release, atomic native
rollback and explicit saved-policy provenance. None is implied merely by this
preflight. Native`local_accretion.apply_accretion:705` changes
receiver omega before owners at about line754 and currently lacks both a
receiving finite witness and this preflight. The next integration experiment
must capture/revalidate both material witnesses in a real accepted plan and
stage the bond before those mutations, without inferring old-world bonds.

`native_material_adaptivity._apply:203–286` stages lineage/column remaps before
committing from287. A future native test must combine the finite bond preflight
with the actual refinement/coarsening transaction and independently check mass,
phase, heat, registry, IDs and rejected-trial rollback. This primitive never
alters those inventories, but their atomic native coupling is not yet tested.

`native_topology._commit:111–133`, `backarc._commit:134–162` and buried-sheet
consolidation all require preflight before identity allocation or owner writes.
Topological isolation explicitly claims no fracture; it cannot authorize bond
release. Stage a proposed split through only one endpoint and prove rejection
before **all** native allocations/counters; stage a split carrying both matched
finite portions and prove exact inventory/history preservation. That is the
next measurable integration gate, not a claim already established here.

The physical release gap remains concrete: `native_rift_material.refresh:65–86`
creates mechanical contacts from shared indexed edges, and`:141–160` allows only
those current edges to transmit loading. Separate overlapping/accreted sheets
have no finite attachment traction or damage edge there. Existing rift damage
therefore cannot justify releasing the new inter-sheet record. A reciprocal
traction/strength/damage law and finite cut witness must be supplied and tested;
until then native transitions crossing an intact bond must remain unsupported.
Scotese Rule XI motivates inherited sutures and later rifting but supplies none
of these missing coefficients or failure criteria. This file does not declare
those physical questions solved.
