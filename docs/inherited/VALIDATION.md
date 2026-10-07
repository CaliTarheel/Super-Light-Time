# Validation scope

## Opt-in overriding response and plate connectivity (2026-09-16)

On the frozen PR20-based source, **219 distinct local tests passed**: 112 focused
force, slab, profile and primordial-connectivity checks plus 107 geometry,
material, collision, capture, ownership and native-engine regressions. The new
suites are `test_moving_hinge_response`, `test_subduction_response_profile`,
`test_primordial_connectivity`, and `test_small_spherical_faces`; the latter
reuses PR17's high-precision cases and adds captured thin-overlap/endpoint checks.
The new suites are included in the existing CI jobs.

Seed-37 independent-ocean, mixed fixed-trench, and mixed moving-hinge worlds
completed 0.02 Myr native intervals and reproduced physical state bitwise after
checkpoint reload. Persistent markers, rather than changing material-face
indices, verify continental displacement through remeshing. Slab mass/area
ledgers close, and the original land inventories and cooling-age fields are
preserved at initialization. These short preflights do not establish a stable
production-length trajectory or terrestrial calibration.

The captured clipping pair is checked against independent 60-digit geometry in
both orders. Sixty saved overlap pairs agree between differentiated and ordinary
area evaluations within `7.41e-10 km2`, below the unchanged `1e-8 km2` acceptance
floor. Existing exact shared-edge coverage, gravity virtual work, burial and
collision checks pass. See [OVERRIDING-RESPONSE.md](OVERRIDING-RESPONSE.md) for the
equations, configuration, initial-speed comparisons and scientific limits.

## Disjoint native ocean complements

The spreading/clipping, subduction, native processes, coupled rupture,
persistent-entry and phase-source selection passed **72 tests in 72.115
seconds**. Five new robust-partition/native-continuation tests plus the five
locality tests passed **10 tests in 11.216 seconds**, for 82 distinct regressions.
The locality suite now includes its original coast fixture in the repository.
The new capture case completes the previously stuck third interval, consumes
the full requested time, and exactly reproduces it after a typed checkpoint.
Independent decimal predicates and chart-tile area oracles check the geometry.
See [CONVEX_PARTITION.md](CONVEX_PARTITION.md). PR #26's eight GitHub checks passed
at `a8d7818`; no long-world collision validation is claimed.

## Local entry-depth pressure and thermal sources

The combined entry-depth, retained-phase, persistent-entry, dense-crust, burial,
native material evolution and structure-engine selection passed **70 tests in
138.532 seconds**, including eleven new entry-depth regressions. Independent
reference moments and refined composite integrals check the depth measure,
nonlinear conversion and heating. The latter caught an endpoint layer missed
by interior-only sampling; the endpoint-inclusive rule resolves it. Actual
native source coupling, checkpoint replay and atomic source failure pass.
PR #24's eight GitHub checks passed at `71d9d4b`.

A longer third-interval native probe encountered excessive ocean-capture union
fragmentation before phase sources. Duplicate runs were interrupted and a
bounded isolated geometry was saved. That trajectory is **not** counted as a
passing validation; NEXT_WEEK_PRIORITY.md records its reproduction. This source
extension does not establish complete burial/exposure or inherited polarity.

## Explicit persistent entry regions

Eight `tests.test_entry_regions` cases exercise the actual native transport and
remeshing paths: categorical refinement/coarsening, distinct-history separation,
assigned-volume conservation, once-only upper-plate hinge transport, source/birth
accounting, end-of-motion sheet-stage hinges and shared conserved inventory,
typed checkpoint continuation, and rollback after actual native advancement.
The tests caught independently reconstructed volumes disagreeing by `2.21e-13`
on a newborn face; both potentials now read the same reference-volume inventory
without changing their consistency tolerance.

The combined entry-region, entry-sheet, entry-force, coupled rupture, native
adaptivity/evolution, material mechanics, density and retained-phase selection
passed **88 tests in 198.977 seconds**. The eight new tests also passed separately
in 13.261 seconds and are included in slab-conservation CI. PR #23's eight GitHub
checks passed at `3052369`. This verifies the explicit whole-face experiment,
not automatic admission, physical burial, inherited polarity or a complete
collision trajectory. See [ENTRY_REGIONS.md](ENTRY_REGIONS.md).

## Initial publication regression suite

The README command runs the focused collision/transport suite also selected by `.github/workflows/tests.yml`. It covers:

- Contact refresh at the actual production pre-force boundary after ownership changes.
- Independent disconnected contact patches, refinement invariance, and common rigid rotation.
- Contact state, clocks, geometry, and diagnostic behavior.
- Plate-balance numerical and finite-volume transport safeguards.
- Suture state/schema and conservative, guarded material consolidation.

Local verification on 2026-09-15: **57 tests passed in 138.452 seconds**, run from the separate publication checkout with the existing Python 3.12 environment. All 106 application Python modules matched the active application's source files before Git line-ending normalization.

Python syntax checks passed, and broader test discovery collected 1,279 tests without import errors. Discovery checks imports only; those 1,279 tests were not all executed for this publication. A scan of the packaged text found no matches for the checked credential formats, personal email addresses, or machine-specific user/project paths.

## Slab mass conservation repair

The version-2 change adds 16 focused inventory/integration tests in `tests.test_slab_mass_inventory`. They pass locally and include the actual force assembler, trench split/join/transfer lifecycle, finite native capture, and explicitly typed checkpoint serialization. They use small fixtures and do not load the running world's checkpoint.

The CI conservation job runs the following broader group. From PowerShell:

```powershell
$env:PYTHONPATH = "$PWD/tests"
python -m unittest tests.test_slab_mass_inventory tests.test_trench_history tests.test_native_subduction_oracle tests.test_native_subduction_schema tests.test_subduction_force_consistency tests.test_native_engine -q
```

The broader run exposed a pre-existing snapshot test that indexed absent, disabled LIP fields. The same error was reproduced on unchanged initial `main`. The test now compares field presence and values, and validates both frames with the production schema checker.

Final local verification on 2026-09-15: **121 tests passed in 372.860 seconds**. This combined the 64 conservation/trench/native integration tests above with the existing 57 collision/transport regressions. Changed Python files passed syntax checks, and `git diff --check` passed. This is not a long-world physical validation or deployment test.

## Crust composition conservation repair

Version 2 adds 15 tests in `tests.test_crust_inventory`. The original foundering
tests explicitly retain their version-1 oracles. The new tests use physical
volume balances through compression, erosion, magma sources, production
deformation, marker histories, actual arc geometry changes, refinement and
coalescence, migration, and checkpoint continuation.

The `crust-conservation` CI job runs these tests together with the existing
crustal structure, foundering, material adaptivity, LIP, arc and interior
structure suites. Local verification on 2026-09-15: the 119-test group passed in
118.093 seconds; two additional integration cases then passed with all 15 new
inventory tests. These are numerical conservation and lifecycle checks, not
validation of stratified composition or retained eclogite physics.

The first CI run exposed an existing checkpoint test that assumed the ignored
repository `tmp/` directory already existed. It now uses the system temporary
directory, preserving all continuation assertions on clean checkouts.

## Local burial-depth integration

Ten new analytic/integration tests in `tests.test_burial_depth` cover the
zero-versus-10 km threshold failure, refinement invariance, triple stacks,
rigid rotation, independent heating gates, local contact attribution, stale
geometry rejection, explicit migration, actual deformation and checkpoint
continuation. Existing foundering fixtures keep the legacy depth law.

Local verification on 2026-09-15: **73 tests passed in 3.238 seconds**, combining
the new depth tests with the conservative crust, legacy foundering, spherical
coverage and collision-surface acceptance suites. The new cases are included
in the crust-conservation CI job. This verifies geometric integration of the
existing depth law; it does not calibrate a pressure-temperature transition.

## Local weld traction and solver verification

Twelve tests in `tests.test_local_weld_traction` exercise connected-patch
cancellation, contained material, internal-edge exclusion, triangulation and
rotation invariance, equal/opposite torque, tangential slip, nonnegative work,
energy/force/Hessian derivatives, analytic arc integration, exact absence of
closing/rest attraction, the actual Newton solve, and versioned restart.

Local verification on 2026-09-15: **68 tests passed in 133.466 seconds**, combining
these cases with the existing 57 collision, force-balance and consolidation
regressions. The solver tests include convex megathrust curvature, explicit
residual acceptance, exhausted-solver rejection, and roundoff-limited line
search checks. These are the numerical defects identified by review point 6;
their source repair already predates the initial GitHub snapshot.

A subsequent production-assembly regression also passes: material welds remain
assembled when fractional plates have no dominant control-mesh boundary edge.

The local-opening tests are included in collision CI. This is not validation
of a complete collision model or full buried-interface shear coupling.

## Buried-interface shear

Twelve tests in `tests.test_collision_interface` cover independent analytic
spherical moments, refinement and rotation invariance, full and partial triple
stack screening, contained relative spin, units/parameter scaling, production
viscous assembly without control edges, explicit migration and restart, and a
solved force/work balance. They are included in collision CI.

Local verification on 2026-09-15: **92 related tests passed in 29.984 seconds**.
The subsequently added solved-power case passed with all twelve interface and
ten burial-depth cases (**22 tests in 1.191 seconds**). These establish the
reduced Newtonian closure's geometry and mechanical accounting, not calibration
of interface rheology or resolved thermal/fluids feedback.

## Oblique convergence partition

Fourteen tests in `tests.test_normal_partition` check independent finite spherical
area under 5 km/Myr closure plus 40 km/Myr sliding, refinement, the review's
500 km / 100 Myr history, maturity and nonclosing gates, native classification,
conservative slab feeding, arc/shortening loading, back-arc selection and restart.
They also check a 375 m foreland target across display labels, finite-budget
rift inversion against an analytic 5 km/Myr compression case with 40 km/Myr
sliding, both collision owners versus only the subduction overrider, stale-speed
rejection, and the attachment diagnostic. Nonclosing, owner and budget gates
remain covered. The suite and the existing inversion suite are in slab CI.

Follow-up verification: **64 normal-partition, inversion, structure and sink
tests passed in 6.029 seconds**. Legacy tests retain their original policy.

Local verification on 2026-09-15: **136 related tests passed in 62.702 seconds**.
After adding slab-feed, arc and restart cases and removing a remaining forearc
obliquity gate, **43 partition/trench/retreat tests passed in 4.342 seconds**.
These verify removal of classification gates; initiation, thermal evolution and
buoyancy-dependent continental polarity are separate physical closures.

## Density-dependent column gravity

Eight tests in `tests.test_column_density` independently integrate vertical
mass moments for dense basal and ordinary upper crust, then check fixed-volume
energy derivatives, layer order, mesh/frame invariance, actual plate torque,
relaxation energy acceptance and explicit profile admission. Local verification:
**35 gravity, force and native-material tests passed in 9.763 seconds**.
The new suite is included in collision CI. This validates the mechanical
component; reaction, thermal and detachment state are separate remaining work.

## Retained dense-phase column/lifecycle component

Sixteen tests in `tests.test_dense_crust` check independently stated sequential
reaction solutions, equal-rate limits, time partitioning, cold-root heating and
cooling, mass-conserving volume contraction, subsidence/rebound, upper-first
erosion, sensible-heat closure, retained roots without mechanical detachment
admission, exact checkpoint continuation and actual material/arc lifecycle paths.
Local verification: **108 related tests passed in 83.319 seconds**; after the
prospective arc-growth update, **37 phase/arc tests passed in 27.589 seconds**.
The final explicit no-detachment gate passes all **16 phase tests in 1.627 seconds**.
The suite is included in crust CI. Spatial forcing, detachment mechanics and
full timestep integration remain separate acceptance work.

## Retained-phase timestep integration

Thirteen tests in `tests.test_phase_evolution` cover hydrostatic eligibility,
local partial-cover integration, viscosity/strength admission, substep convergence,
coeval parcel/marker forcing, conversion loading and detachment rebound, signed
dense support through saved sampling and foreland input, transactional first
contacts (including water loading after strain), atomic explicit migration,
phase/heat frame validation, and full native checkpoint continuation.

The native case runs a 0.25 Myr step, checks independently closed mass and physical
volume budgets, saves and reloads a typed checkpoint, then reproduces the next
step's columns and plate motion exactly. A separate multi-step stack test
checks conserved mass plus mantle return, conversion contraction, local marker
history, and that detachment is not misreported as surface erosion.

Local verification on 2026-09-15: **63 integration, density, gravity, collision
surface, structure and native sampling tests passed in 90.466 seconds**. The
subsequently added independent water-loading contact test passed separately
in 0.028 seconds. All thirteen phase-integration cases are included in crust CI.
These checks validate the declared reduced closure and its accounting, not its
parameter calibration, long-run mesh convergence or a historical reconstruction.
No live checkpoint has been migrated or restarted with this policy.

## Mechanical slab-attachment component

Twelve tests in `tests.test_slab_tether` independently solve the two-body Stokes
matrix, check the eliminated functional's virtual-work derivative, signed-load
power balance and nonnegative dissipation, and integrate the damage ODE with a
separate fourth-order Runge-Kutta oracle. They verify rupture-event localization,
time partitioning, no healing under compression, zero traction after detachment,
relative resolution for tiny steps/stiff necks, and width subdivision of distinct
parallel neck histories. A deliberately averaged-damage join is shown to change
the transmitted load and is not used by the component.

Eight tests in `tests.test_slab_tether_history` exercise actual slab split/join,
overriding-owner transfer, exact feed/retirement, typed checkpoint continuation,
rupture retirement and subsequent identity changes. Corrupt or mixed histories
are rejected, and detached inventory remains accounted without reattachment.
An 80-channel sequential rupture case verifies exactly zero final retained
inventory without a spurious floating-point remnant.

Local verification: **73 attachment, channel-history, slab-inventory, trench,
force-consistency and plate-balance regressions passed in 1.754 seconds**.
Both new suites are included in slab CI. This is component/lifecycle evidence
only; no native force solve yet consumes the attachment law, and no result here
demonstrates limited continental entry or polarity reversal.

The complete slab CI selection also passed locally: **95 tests in 195.281
seconds**, including native engine integration. All six GitHub checks on PR #13
passed for commit `7a39709`.
## Slab depth-window review repair

Local verification on 2026-09-15: **129 tests passed** in the existing Python
environment. The slab/native group below passed 80 tests in 251.545 seconds;
the collision/contact/suture suites (excluding the plate-balance suite already
in that group) passed another 49 tests in 149.358 seconds.

```powershell
$env:PYTHONPATH = "$PWD/tests"
python -m unittest tests.test_slab_depth_window tests.test_slab_mass_inventory tests.test_plate_balance_repair tests.test_trench_history tests.test_native_subduction_oracle tests.test_native_subduction_schema tests.test_subduction_force_consistency tests.test_native_engine -q
python -m unittest tests.test_collision_force_continuity tests.test_collision_local_fronts tests.test_collision_contacts tests.test_suture_weld.WeldStateTests tests.test_suture_weld.WeldSchemaTests tests.test_suture_weld.ConsolidationGuardTests tests.test_suture_weld.ConsolidationTests -q
```

The eight new tests include fast old-ocean feeding for 200 Myr, conserved
inventories, current-trace geometry, subdivision-invariant assembled forces and
resistance, dip/work limits, bootstrap equivalence and saturated isolated terminal
speed. Syntax checks and `git diff --check` passed. These are reduced-model
regressions; neither a long-world calibration nor a live-world migration was run.
See [the review](CLAUDE_REVIEW.md) for assumptions and deferred findings.

## Frozen-state slab force coupling

Nine cases in `tests.test_slab_tether_forces` exercise the actual plate-balance
constructor and solver. An independent expanded matrix retains two separate
slab velocities at differently oriented edges and matches the condensed plate
solution. Other checks cover power and Euler-coordinate work, subdivision and
reversed edge orientation, collinear split/join with unequal damage, conservative
rupture, the shared depth window, unmatched geometry and required histories,
neutral-buoyancy resistance, and a native mesh solve whose velocities and full
diagnostics reproduce exactly after a typed checkpoint round trip.

Local verification: **61 attachment, force-coupling, channel-history, depth-window,
slab-inventory and plate-balance tests passed in 2.938 seconds**. The new suite is
included in slab CI. PR #14's six GitHub checks also passed at commit `95644b3`.
These are frozen-state force checks, not a rupture timestep or continental-entry
experiment. No running world was migrated or advanced with the new attachment.

## Local neck histories and damage

Eleven new cases in `tests.test_slab_tether_local` check conservative explicit
localization; force location through unequal-damage split/join; detached gaps;
non-cancelling local opening; first-event retirement and returned time; actual
owner transfer and persistent separation/rejoining with 80/20 local inventories;
per-patch retention and rejected unlocalized feed; actual finite capture with
cooling mass and saved source validation; and anchor advection/checkpoint restart.
An independent RK4 integration verifies that equal signed opening and closing
inside one finite-volume cell cannot cancel its positive damage rate.

Local verification: **127 local-neck, attachment, force, history, depth, capture,
trench, normal-partition and solver tests passed in 6.128 seconds**. The new suite
runs in slab CI. PR #15's six checks passed at commit `b4178e7`.
These checks do not validate spatial-resolution convergence, coupled rupture
timesteps or continental-entry/reversal behavior. The live world is unchanged.

## Other tests and fixtures

### Continental-entry residual sheet work

Eleven `tests.test_entry_sheet_work` cases exercise the actual viscous sheet
stage with the entry potential. Independent checks cover Newtonian force/drag
units, weighted rigid/residual work, vertical column mass moments and clipped
entry depth. The rigid torque matches the actual plate balance, with the
overrider's hinge reaction included. Signed neutral/dense behavior, input mass
consistency, new-stack rejection, total-energy descent, temporal refinement,
stage reconstruction and common-rotation covariance are covered.

Two false-descent probes verify that both entry and column-only paths reject a
motion whose projected-force work is negative but whose actual potential-energy
derivative is positive. An active-bound case resolves `1e-5`, `0.001` and `0.1`
Myr stages with the original physical bounds and `1e-10` force/complementarity
requirements. The endpoint solver starts at the physical bound, removing an
unconditional inward offset whose reaction gap otherwise grew as `1/dt`.

The combined entry, gravity, density, sheet/deformation, resistance, coupled
rupture, native rupture, enhanced-rifting and parallel constrained-solve
selection passed **97 tests in 69.012 seconds**. The 23 entry-sheet and ordinary
gravity tests also passed with no PYTHONPATH override, matching collision CI.
An additional **25 phase, material-mechanics/loading-epoch and native material
evolution tests passed in 84.360 seconds**, for 122 distinct local regressions.
PR #22's eight GitHub checks passed at `7f3cd02`. This explicit fixed-hinge
mechanical stage does not establish native entry transport, local physical
burial, inherited polarity or a complete collision trajectory.

### Continental-entry energy and frozen force coupling

Twelve `tests.test_continental_entry` cases check the explicit thin-sheet
potential against independent clipped-area integration, a 90-digit thin-sliver
integral, finite-difference virtual work and spherical refinement quadrature.
They cover reciprocal hinge reactions, signed buoyancy, existing phase mass,
stale identity/stack guards, an actual native checkpoint round trip, and the
effect of entry resistance on the coupled slab-neck damage solve. A sequence of
frozen plate solves slows, stalls and reverses the same fixed-inventory body;
this is not a native collision trajectory or a calibrated entry distance.

The combined entry, density, slab-force/local-history, passive-resistance,
plate-balance, coupled-evolution and native-rupture selection passed **76 tests
in 65.355 seconds**. The entry suite is included in collision CI. The parent
PR #21's eight GitHub checks passed at `d2b4609`. Full entry-region transport,
residual deformation work, local burial and inherited polarity remain pending;
see [CONTINENTAL_ENTRY.md](CONTINENTAL_ENTRY.md). No live world was modified.

### Coupled rupture and native advancement

Eight `tests.test_slab_tether_evolution` cases compare coupled force/damage
trajectories and rupture times with an independently integrated two-body Stokes
system, test tolerance refinement, global event ordering across trenches,
checkpoint continuation, source immutability and force-budget rejection.
Five `tests.test_slab_tether_native` cases advance the real native source path
through rupture and its remainder, check exact checkpoint replay, inject a
failure after source/RNG advancement to verify complete rollback, reproduce
the short-interval newborn-crust case, and check partially failed coincident
histories under both capture geometry policies. Passive resistance version and
nonnegative local resisting work are checked in the actual native experiment.

The final aligned resistance, force-coupling, coupled-event and native selection
passed **45 tests in 67.697 seconds**. The six affected native/invalid-control
cases passed again in 23.404 seconds after the final explicit-version assertions.
Earlier broad integration checks passed 84 collision tests, 139 slab/native tests
and 160 crust/phase/material tests; the subsequent shared-law alignment is
covered by the focused checks and GitHub CI. No full-world calibration, spatial
convergence or native source-step convergence is claimed. PR #17's six GitHub
checks passed at `456c1dc`. This work does not change the running world.

### Metre-scale spherical area precision

Three regressions in `tests.test_small_spherical_faces` compare arbitrarily
oriented small faces with a 60-decimal-digit determinant evaluated on the exact
stored coordinates, check scalar/batched/differentiated contained overlap and
its directional derivative, and advance a tiny gravity-driven column through
the full requested interval with decreasing energy. The precision check avoids
using the production area formula as its own oracle. Together with existing
gravity, material mesh, coverage and density checks, **45 tests passed locally**.
The new tests run in collision CI. The gravity acceptance rule is unchanged;
no running simulation was modified.

The repository retains broader Python and JavaScript regression tests plus small numerical fixtures. They include historical contracts and are not all covered by the focused CI job. Some replay checks need archived runs that are not distributed here. A full `unittest discover` run is not claimed to pass for this snapshot.

Tests for excluded local audit/email/launch programs are omitted; [PUBLICATION.md](PUBLICATION.md) lists them. The historical suture checkpoint inventory test is opt-in through `DEEP_TIME_SUTURE_CHECKPOINT` and skips when no checkpoint is available.

Passing tests establishes the listed numerical properties only. See [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) for unresolved model physics. Historical test totals in retained design documents describe earlier source versions and separate local artifacts.


Earlier installed-world checks are retained separately in [LOCAL-VALIDATION-HISTORY.md](LOCAL-VALIDATION-HISTORY.md). These historical results do not establish that the combined PR stack has passed.

## Combined PR integration, September 16, 2026

The integration joins PR #25 and PR #27 and retains all earlier PR stacks.
The final focused slab, entry, moving-hinge, resistance and precision selection
passed 144 tests in 110.883 seconds. The contact repair separately passed 32
contact, sheet-work, gravity and small-face tests with the original physical
bounds and KKT acceptance. PR #27's 40-test capture selection includes the
previously stalled third native interval and exact saved continuation. PR #26's
72 selected depth, retained-phase, material, density and burial tests also pass.
These selections overlap; their counts must not be summed as distinct tests.

The new retained-phase surface-erosion closure and its historical guards pass
29 tests with existing relief and profile suites in 42.886 seconds. Tests cover
water loading and sea crossing, ordinary/dense interface and crustal-floor caps,
mass/heat/area conservation, and exact native checkpoint continuation from 2 to
4 Myr. An independent density-based oracle checked 512 varied columns, including
116 capped requests, without a material discrepancy (maximum inferred removal
error `1.82e-11 m`). Historical erosion versions are unchanged.

Viewer syntax, advanced-physics configuration round trips, and all ten recorded
display-resolution cases pass. Two broader historical UI harnesses retain their
pre-existing mock failures (missing `source-transition-note` DOM support and
`URLSearchParams`); those failures also reproduce on the previous installed
source. They are not represented as passing checks.

The final combined source passed a full-size original-world preflight through
6 Myr in 388.328 seconds: 1024-by-512 display, control level 5, two continental
ocean attachments, moving-hinge response, and retained thermal phases. Each
2 Myr coupled interval completed without an adaptive rejection or shortened
contact interval. Solved control mean speeds were 0.541, 0.623 and 0.672 cm/yr
at 2, 4 and 6 Myr. Slab, ocean, crust and craton inventory checks passed, and
reported boundary speeds agreed with relative Euler motion. The 2 Myr checkpoint
round-tripped all saved state exactly; its continuation to 4 Myr matched the
uninterrupted physical state exactly. Python source and every historical saved
run file remained unchanged throughout this isolated check. This establishes
startup and short continuation, not geological calibration or long-duration
stability.

## Retained-phase marker containment repair

Five new containment regressions cover the actual 14 Myr failure after the
last completed 12 Myr frame, a synthetic short-edge vertex, genuine outside
rejection, deterministic shared-edge ties, unchanged parcel quadrature, and
source immutability. All 28 captured region memberships agree with an independent
80-digit determinant oracle on the represented coordinates. Only the correct
uncovered region contains the failing marker. The old arithmetic rejects it.

Together with existing phase-evolution, retained-profile and entry-depth suites,
35 distinct tests pass. Reconstructing the parent run's 12 Myr state from its
durable 10 Myr checkpoint reproduces all 133 saved frame arrays and all frame
metadata exactly. That verifies the reconstructed observable state; an original
complete 12 Myr checkpoint was never saved. A numerical-source continuation
records both source versions and preserves the original experiment. See
[PHASE-MARKER-REPAIR.md](PHASE-MARKER-REPAIR.md) for the failure and arithmetic.
