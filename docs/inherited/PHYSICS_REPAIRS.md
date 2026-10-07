# Physics repair status

## Coverage of the September 14 review

These are proposed changes in the PR stack, not a deployment record. Numerical
repairs and reduced physical extensions have different validation limits.

### Small-face geometry prerequisite

Native rupture experiments exposed cancellation in spherical area evaluation
for metre-scale newborn arc faces. A one-bit vertex change could appear as an
energy increase of about 1.46e-8 km4 against a 3.61e-13 km4 rounding allowance,
causing gravity backtracking to fail even as its interval approached zero.
The triangle determinant now uses `a dot ((b-a) cross (c-a))`, algebraically
equal to the original `a dot (b cross c)`. Material areas, scalar/batched overlap
areas and differentiated overlap areas use this stable expression. This improves
arithmetic precision; it changes neither the gravity energy law nor its
acceptance tolerance. It does not resolve every spherical clipping degeneracy
or establish native source-step convergence.

| Review point | Proposed repair and status |
| --- | --- |
| 1. Local opening cancelled by closing elsewhere | [#7](https://github.com/CaliTarheel/Deep-Time/pull/7) local traction; [#8](https://github.com/CaliTarheel/Deep-Time/pull/8) buried-interface shear. |
| 2. Slab load manufactured by joins | [#1](https://github.com/CaliTarheel/Deep-Time/pull/1) extensive inventories. Separate depth/work correction: [#6](https://github.com/CaliTarheel/Deep-Time/pull/6). |
| 3. Foundering exceeds its compositional cap after strain | [#4](https://github.com/CaliTarheel/Deep-Time/pull/4) conservative removable inventory. |
| 4. Oblique convergence excluded from subduction | [#9](https://github.com/CaliTarheel/Deep-Time/pull/9) normal partition; this follow-up completes inversion, legacy foreland and attachment-diagnostic consumers. |
| 5. Threshold applied after averaging burial | [#5](https://github.com/CaliTarheel/Deep-Time/pull/5) local overlap-region integration. |
| 6. Nonconvex megathrust and inadequate solve acceptance | Specific defects fixed before this stack; regression coverage retained in [#7](https://github.com/CaliTarheel/Deep-Time/pull/7). [#18](https://github.com/CaliTarheel/Deep-Time/pull/18) fixes nonpassive smoothing; [#19](https://github.com/CaliTarheel/Deep-Time/pull/19) carries that same law into this stack and adds per-plate force acceptance and numerical precision checks. The [entry sheet work](CONTINENTAL_ENTRY.md) also checks the actual full-energy derivative and resolves short constrained stages without an artificial reaction gap. |
| 7. Missing retained dense phase, composition and temperature | [#10](https://github.com/CaliTarheel/Deep-Time/pull/10), [#11](https://github.com/CaliTarheel/Deep-Time/pull/11), [#12](https://github.com/CaliTarheel/Deep-Time/pull/12): conservative state and explicit reduced evolution policy; limits below. |
| 8. Instant reversal at continental arrival | **Incomplete.** [#13](https://github.com/CaliTarheel/Deep-Time/pull/13), [#15](https://github.com/CaliTarheel/Deep-Time/pull/15), [#16](https://github.com/CaliTarheel/Deep-Time/pull/16) supply attachment, force coupling and local histories; [#21](https://github.com/CaliTarheel/Deep-Time/pull/21) supplies an explicit coupled native rupture experiment. [#22](https://github.com/CaliTarheel/Deep-Time/pull/22) and [#23](https://github.com/CaliTarheel/Deep-Time/pull/23) add continental-entry plate/sheet energy. [Explicit persistent regions](ENTRY_REGIONS.md) now connect whole-face histories and stage-consistent hinges to native mechanics and remeshing. Automatic finite admission, local burial, owner transfers, inherited polarity and reversed initiation remain outstanding. |

### Follow-up: explicit entry-depth phase sources

The persistent whole-face experiment can explicitly select local uniform-mantle
pressure and thermal forcing through `entry_phase_depth.upgrade`. It evolves
separate local histories before conservative projection, with pointwise marker
depth and an adaptive endpoint-inclusive quadrature. The independent thermal
oracle exposed a moving-cap interval missed by an interior-only rule; that
numerical defect is covered separately. See [ENTRY_PHASE_DEPTH.md](ENTRY_PHASE_DEPTH.md)
for the reduced physical closure, validation and unresolved exposure/geometry.
Review point 8 remains incomplete. The native continuation also exposed an
ocean-capture union-fragmentation prerequisite. The subsequent
[disjoint partition repair](CONVEX_PARTITION.md) resolves that short trajectory
without copying tolerance-band area into both complement halves.

## 1. Conserved slab excess mass

The review's two 1,000 km trench example exposed a bookkeeping defect: joining slabs with different down-dip extents increased represented excess mass by 44.444%, even though no ocean area entered the system. The old code averaged load per metre using retained area, then multiplied it by the new trace length. Changing trace length without feeding had the same conservation problem.

### Version 2 behavior

`slab_memory_version = 2` stores four extensive inventories in kg:

- `slab_initial_excess_mass_kg`: the explicit baseline when an existing world is converted; zero for newborn trenches.
- `slab_fed_excess_mass_kg`: cumulative admitted excess mass after that baseline.
- `slab_retained_excess_mass_kg`: mass still in the model's retained slab window.
- `slab_retired_excess_mass_kg`: mass that has left that window.

Every record enforces **initial + fed = retained + retired**. Splitting and joining transfer all four inventories and the existing area inventories together. Trace growth, shortening, and overriding-plate transfer preserve mass. The force calculation allocates retained mass and area over the currently matched attached edges, then couples only the uniform column's portion above 660 km depth; subdivision does not increase that force. Deeper inventory is preserved. A slab with no matched edge exerts no force, while its inventory remains recorded and continues to retire. See [the depth-window review](CLAUDE_REVIEW.md) for this subsequent coupling correction.

Feeding and retirement retain the existing cooling-age proxy and exact constant-input exponential integration. The mass feeding calculation no longer divides by trench length. The recorded `slab_line_load_kg_per_m` becomes a derived diagnostic on the stored trace. Very short traces can produce large line densities: the old absolute density bound is retained for version 1 only, since imposing it on version 2 would discard conserved mass. This change does not validate extreme local loads or replace the slab geometry model.

The reproduced join now retains **3.94650228645 × 10¹⁸ kg** instead of increasing to 5.70050330265 × 10¹⁸ kg. Tests also exercise the actual trench lifecycle and plate-force assembly, not only the join helper.

### Existing worlds and compatibility

New native worlds use version 2. Version-1 worlds retain their historical law until explicitly converted. A version-2 state with missing mass fields is rejected; the engine does not fill an old loaded history with zeros.

`slab_memory.upgrade_mass_inventory(simulation)` stages a conversion and commits it only after validating every record. It initializes retained mass as **saved line load × saved trace length**, records that amount as the baseline, and preserves all other world state. It is idempotent and records its provenance in `slab_mass_migration`, which is included in snapshots and checkpoints. This conversion preserves the mass represented at the selected boundary; it cannot reconstruct prior join errors or unrecorded thermal history.

The helper alone is not a checkpoint-deployment tool. A real continuation still requires preserving the original run, recording a source-version boundary, and writing a checkpoint bound to the changed sources under the existing strict compatibility checks. This PR does not migrate, restart, or rewind the running simulation.

### Validation

`tests.test_slab_mass_inventory` covers the reported join, unequal multiway split/rejoin, length changes, zero-length traces, current-edge allocation, integrated plate-force invariance under collinear subdivision, feeding/retirement, timestep subdivision, migration rejection and idempotence, checkpoint round trips, overriding transfer, and actual finite native capture provenance.

The dedicated CI job adds existing trench-history, native-subduction, force-consistency, and native-engine integration suites. See [VALIDATION.md](VALIDATION.md) for measured results.

### Remaining slab limitations

Mass is distributed uniformly within each tracked trench. Joining differently loaded, differently oriented segments can still redistribute the location of their forces; conservation of total excess mass is not conservation of every local torque. The thermal feed still uses an aggregated cooling-age proxy. Initiation, attachment/shutdown, retained-window timing, dip and depth remain reduced modeling assumptions.

## 2. Conservative removable-crust inventory

The compression probe starts with 40 km3, founders 10 km3, halves the footprint,
then continues removing eligible crust. Version 1 removes 26 km3 (65%) because
it mixes current thickness with a historical removal measured in metres.
Version 2 removes **24 km3 (60%)**, independent of the intervening footprint.

`crust_inventory.py` stores baseline, added, eroded, returned and remaining
removable volumes **per reference km2**. Multiplying by a parcel's reference
area gives physical km3. Every column checks **baseline + added = eroded +
returned + remaining** and bounds remaining inventory by its actual volume.
Strain leaves these fields unchanged. Refinement and coalescence conserve their
reference-area-weighted sums. Arc growth rescales every field when reference
area changes, then books the admitted magma. Arc births use the final realized
profile. Surface, ridge and LIP additions supply 60% of their admitted volume;
erosion exports the current well-mixed removable fraction. Historical
`foundered_m` remains a point-history diagnostic, never a version-2 allowance.

This is a conservative implementation of an explicit **bulk composition
assumption**, not a resolved lower-crust or phase-transition model. In
particular, well-mixed erosion differs from removing a specifically felsic
upper layer. Dense-phase retention, temperature and depth integration remain
separate work below.

### Migration and saved state

`eclogite_sink.upgrade_inventory(simulation)` explicitly stages and validates
both material and marker columns before committing version 2. Its baseline is
the **remaining version-1 allowance at the chosen boundary**, converted to
volume per reference area. It neither grants a fresh 60% allowance nor claims
to recover prior excess losses. Physical columns, existing mantle returns and
existing clocks are preserved. Checkpoints and frames include the inventory;
incomplete version-2 states are rejected. The optional historical combined
migration now calls this helper. The running world is not migrated by this PR.

### Validation

`tests.test_crust_inventory` exercises the reported compression case, changing
area histories, erosion, capacity-limited additions, unequal-weight
coalescence, production deformation, actual refinement/coarsening, LIP
feeding, arc birth/growth, explicit migration, frame corruption and checkpoint
continuation. Frozen version-1 oracles remain covered separately.

## 3. Local burial-depth integration

The review's lower 20 km column has 10 km of overburden on one half and 50 km
on the other. The legacy law averages that cover to 30 km, then obtains zero
thickness below its 50 km threshold. The local integral correctly obtains
**10 km of face-mean eligible thickness**: half the 20 km lower column.

`burial_depth.py` partitions each lower spherical triangle into disjoint convex
regions using its covering triangles' great-circle planes. Within each region,
the actual upper thicknesses are added once, then the depth threshold is
applied. Uncovered regions retain the own-root branch. Triple overlaps form a
vertical stack over a geometric union; pair areas are never mistaken for that
union. The partition must conserve its lower footprint and reproduce the
cached pair intersections, or the operation is rejected.

The contact ledger now attributes covered removal using eligible regional
volume, with shares proportional to each locally contributing upper thickness.
This is an explicit provenance convention, not a traction or temperature law.
It gives zero share to shallow cover that contributes no eligible material.

`eclogite_sink.upgrade_depth_integration(simulation)` computes and validates the
new geometry before selecting `foundering_depth_version = 1`. It leaves all
material, clocks and return inventories unchanged, records the boundary, and
persists the selection in frames/checkpoints. Existing worlds retain version 0
until explicitly converted. The optional combined migration selects both the
conservative inventory and local depth integration.

Tests cover the reported case, refinement of either/both covers and the lower
sheet, triple stacks, heating gates, rigid rotation, contact-ledger closure,
stale geometry rejection, actual deformation, and checkpoint continuation.
The face-level heating clocks and uniform thickness/composition within a face
remain approximations. This PR does not add retained eclogite or thermal state.

## 4. Local collision-opening traction

The disconnected-patch repair still allowed opening and closing inside one
connected patch to cancel before resistance was evaluated. Local coordinate
version 1 instead integrates the opening law over the clipped **free edges of
the physical sheets**. It removes internal triangulation edges and merges
collinear pieces before integration. Equal and opposite plate blocks give zero
response to common rigid rotation and balance interface torque.

On a great-circle edge, normal relative velocity is `a*cos(theta) + b*sin(theta)`.
The implementation finds its zero and yield crossings analytically, then
integrates the potential, gradient and Hessian using exact trigonometric
moments. There is no triangle-size or sampling-grid traction scale. The existing
per-metre capacity is retained with half weight on each side: two opposing
straight free edges represent one contact. A contained sheet's closed perimeter
can therefore resist local peeling even when its net overlap-area rate is zero
and no single convergent front direction is available.

The new unilateral Huber density is zero for closing speed `v >= 0`,
`v^2/(2*delta)` for `-delta < v < 0`, and `-v-delta/2` below yield. It is convex
and continuously differentiable, with nonnegative resisting work and exactly
zero traction at rest or during closing. This also removes the old smooth
law's small artificial attraction during closing. Tangential slip along an
edge is not charged as normal opening.

`weld_geometry.upgrade(simulation)` explicitly selects
`suture_weld_coordinate_version = 1`, records the source boundary and leaves
material and weld history unchanged. Existing worlds keep coordinate version 0
until converted; frames/checkpoints preserve the selection. The optional
combined migration selects the local law.

Tests cover a connected rotating contact with cancelling net area rate,
contained material, refinement, rigid rotation, action/reaction, independent
energy/force derivatives, positive curvature, resisting work, analytic versus
dense numerical integration, the actual Newton solve, and restart.

### Physical scope

This repairs the reduced opening law, not all collision mechanics. It is a
plan-view edge-cohesion model, not resolved shear traction over the full buried
interface, a no-penetration constraint or a thermomechanical contact solver.
In particular, motion everywhere tangent to a contained body's perimeter
requires an area-based shear law. Capacity and cohesion/annealing remain
uncalibrated reduced closures. Thermomechanical studies show that interface
coupling depends on fluids, melts and shear heating; perimeter geometry alone
cannot establish it ([Faccenda et al., 2009](https://jupiter.ethz.ch/~tgerya/reprints/2009_EPSL_Manuele.pdf)).

## 5. Shear over buried interface areas

Optional `collision_interface_version = 1` closes the perimeter law's remaining
kinematic gap with a thin-layer viscous closure: **traction = viscosity /
thickness × relative velocity**. Both horizontal components contribute,
including spin that is tangent to the perimeter. It acts on actual adjacent
material interfaces. In triple stacks the middle sheet screens direct coupling
of the outer sheets; where that sheet ends, the exposed outer interface can
couple directly.

The integrated rotational resistance uses the exact spherical polygon moment
`integral(I - r r^T) dA`. A boundary-integral identity evaluates it without
triangle-count or sampling dependence. Symmetric plate-pair blocks enter the
same viscous force solve, preserve common rotation and action/reaction, and
give nonnegative work. This applies even without dominant control-mesh trench
edges. Consolidated same-owner interfaces contribute no relative drag.

The explicit reference parameters are viscosity **10^20 Pa s** and weak-layer
thickness **13.2 km**, following the constant weak zone in section 2.1 of
[Bottrill, van Hunen & Allen (2012)](https://se.copernicus.org/articles/3/387/2012/se-3-387-2012.pdf).
Using that pair in this reduced spherical model is a modeling assumption, not
a calibration or a claim to reproduce that paper's full thermomechanical model.
The law has no plastic yield or resolved fluids/temperature feedback. Shear
power is reported as dissipation; it is not silently converted to a fabricated
temperature rise. Full viscous power and basal power are now separately named
in diagnostics (the old basal field had included all viscous resistances).

`collision_interface.upgrade` validates geometry and positive parameters before
selecting the new model. It preserves material, configuration and existing
history, records the boundary, and retains parameters in snapshots/checkpoints.
Existing worlds remain unchanged until this explicit conversion. The optional
combined migration selects it alongside the local weld and inventory repairs.

Tests use independent whole-sphere and octant moment integrals, mesh and frame
rotation invariance, contained spin, partial/full triple stacks, parameter and
unit scaling, actual viscous assembly, force/power balance and restart.

## 6. Oblique normal convergence

Explicit `normal_partition_version = 1` separates normal convergence from the
boundary's display classification. A margin closing at 5 km/Myr while sliding
at 40 km/Myr can now develop a trench, consume incoming ocean and feed its slab
inventory while retaining a transform map label. Tangential resistance already
acts on every contact in the force solver. Positive-length closing contacts
are considered without an obliquity or 2 km/Myr cutoff; finite capture still
uses the actual rotated front, incoming-owner water, local polarity and maturity.

The same selection is used by trench history, arc sources and orientation,
material shortening, hinge retreat, back-arc eligibility and ridge/trench
interaction. Additional 0.35 obliquity gates in back-arc and forearc processing
are removed only under this explicit policy. Pure sliding and opening do not
consume ocean. Initiation still requires the existing 10 Myr / 100 km maturity
closure; this repair does not invent an established slab at first contact.

Rift-scar inversion also uses actual normal motion under this policy, including
when the display says transform. It recomputes motion from the current Euler
velocities; stale cached speeds cannot create loading. Collision loads both
continental sides, while a selected subduction polarity loads the overrider.
The finite inherited-extension budget, owner locality and cross-rift direction
still limit this kinematic uplift prescription. The older foreland model admits
the same closing continental contacts, and the contact attachment diagnostic
uses that selection too. Regional foreland version 1 continues to use physical
column support. All legacy gates remain for normal-partition version 0.

Local accretion's separate low-shear/coherence gate is an ownership-transfer
criterion, not subduction capture or a normal-motion force. This follow-up does
not replace that declared merger policy or the legacy raster force law.

`normal_partition.upgrade(simulation)` records the boundary without changing
material, velocity or maturity. Snapshots/checkpoints preserve it; the optional
combined migration selects it. Existing worlds retain the old policy until
explicitly converted. This fixes the normal-component loss, not the separate
ridge-opening policy, retained-slab depth approximation or inherited continental
polarity limitation.

Tests check the review's 500 km / 100 Myr example, independently calculated
spherical capture area under simultaneous sliding, finite-front subdivision,
legacy behavior, opening/pure sliding, maturity, actual native classification,
accepted feed into the conservative slab inventory, arc/shortening eligibility,
back-arc selection, metadata validation and restart.

## 7. Density-dependent gravity for the retained-phase extension

The gravity functional can now represent an ordinary upper crustal layer and a
retained dense basal layer. It directly integrates each layer's `rho*g*z`
relative to displaced mantle under local Airy compensation. For basal dense
thickness D, ordinary thickness B and mass per area M, the unbalanced self term
is `g/2 * (rho_d D^2 + rho_c B^2 + 2 rho_c B D - M^2/rho_m)`.
For lower column i and upper column j, the cross term is
`g * (M_j T_i - M_i M_j/rho_m)`. Thus changing which material is below changes
the energy; unequal densities cannot use the old exchange-symmetric coefficient.
The existing balanced-reference background is retained explicitly in the self
term. The new coefficients and exact geometric derivative are shared by plate
torque, sheet relaxation, acceptance checks and source-work audits.

`column_density.options` requires an explicitly versioned retained dense-volume
inventory and persistent stack order. It never seeds a phase from depth or
silently assigns missing layer order. Phase-free worlds preserve the old law.
The reference densities remain 2800, 3450 and 3300 kg/m3 for ordinary crust,
dense crust and mantle. They are model assumptions, not universal eclogite
properties: composition and metamorphic conditions materially affect density
([Warren et al., 2008](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2008TC002292)).

This is the **mechanical component**, not yet the phase-evolution model. It adds
no reaction, temperature, detachment, reservoir migration or production flag.
A sufficiently dense basal layer can have negative gravitational stiffness;
this physical density inversion is not clamped into buoyant crust. It is
distinct from a nonconvex dissipative resistance law. The sheet model still
needs the phase/detachment component to resolve loss of that load vertically.
New contacts require a recorded layer order before density gravity can admit
them. Local compensation, planar vertical layers and the balanced-reference
background remain reduced closures; flexure and mantle flow are not resolved.

Eight tests independently integrate vertical mass moments, differentiate energy
at fixed volumes, check refinement/frame invariance, zero common-rotation work,
ordered stacking, plate-torque assembly, density-dependent relaxation and
explicit inventory admission. The retained-phase and thermal work remains open.

## 8. Retained phase, mass contraction and sensible-heat inventories

`dense_crust.py` supplies a conservative column transaction for unconverted
mafic material -> retained dense material -> detached material. Conversion keeps
mass fixed and contracts volume by the density ratio 2800/3450. The composition
ledger therefore continues in ordinary-crust-equivalent volume (mass/2800),
while the dense reservoir records actual dense volume. Conversion, retained
phase, erosion and mantle-return inventories close independently.

Sensible heat is extensive: ordinary-equivalent mass volume times temperature,
with a fixed 1200 J/kg/K heat capacity for conversion to joules. Explicit bath,
magma, erosion and mantle-export entries close the heat ledger. The initial
temperature is required; an old deep root is never inferred to be hot. Analytic
relaxation can heat or cool a column. Reaction uses only the interval actually
above the explicit 600 C threshold, with a 10 Myr reference reaction time. No
phase is removed at conversion. The sequential first-order reaction/drainage
solution handles equal rates and long intervals without cancellation.

Detachment is **disabled by default** and requires explicit mechanical admission
by the caller. Its 20 Myr reference timescale is a scenario parameter, not a
prediction that every dense root must detach. Long-lived dense lower crust is
observed; density alone does not establish that rheology permits its loss
([Buntin et al., 2021](https://www.nature.com/articles/s41467-021-26878-5)).
The future geometric/rheological adapter must supply that admission and pressure
eligibility. Temperature, fluid access and reaction times are composition- and
setting-dependent; this kernel is not a phase-equilibrium calculation.

The shared column elevation now includes the dense load. At these reference
densities, mass-conserving conversion subsides; detaching a kilometre of retained
dense material subsequently rebounds by 45.45 air-equivalent metres (with water
loading applied below sea level). Neutral reference updates preserve the prior
strain ratio and do not erase this response. Ordinary upper crust erodes before
the dense basal layer. Strain, coalescence, refinement/coarsening, arc birth and
growth, magma heat admission, and checkpoint continuation preserve the phase and
heat fields. Arc slope admission reads the same full prospective state as growth.

This section describes the **column and lifecycle component** beneath the
production policy in the next section. Spatial pressure/thermal forcing,
detachment admission, frame schema and explicit migrations are supplied by that
connection. The old instantaneous sink rejects retained-phase state instead of
deleting it. Its bath is prescribed, reactions are athermal (latent heat
omitted), and export heat uses endpoint temperature under explicit operator
splitting. Mantle return is limited to the ordinary-equivalent reserve above
the 8 km floor; the unreleased dense phase stays in the column and is reported
(`withheld_equivalent_km`, `reserve_limited`). Kinetics are never rescaled and
material is never dropped; see [PHASE-ADJUSTED-COLUMN-FLOOR.md](PHASE-ADJUSTED-COLUMN-FLOOR.md).

## 9. Explicit retained-phase timestep policy

`phase_evolution.upgrade(world, initial_temperature_c, constitutive_parameters=...)`
connects the retained-phase components to the native timestep. This is an
explicit migration requiring an initial temperature and the shared plate/sheet
gravity and persistent collision surface. It preserves current mass, geometry,
relief and motion, initializes zero retained dense material, and does not invent
historical metamorphism. Existing active checkpoints keep their saved policy;
the phase-adjusted floor requires its own explicit, recorded migration.

Pressure eligibility is integrated on disjoint spherical overlap regions before
averaging. Hydrostatic overburden includes the actual ordinary/dense mass of
each upper column. An explicit reference pressure of 1.5 GPa gates reaction in
the ordinary material above the retained dense base. The column temperature
relaxes toward a prescribed geotherm (15 C/km, capped at 1300 C) on the first-mode
diffusion time `L^2/(pi^2*kappa)`, with reference L=70 km and kappa=1e-6 m2/s
(about 16 Myr). Its representative depth is 0.75 of the column thickness below
its overburden. These are configurable, saved constitutive assumptions, not
recovered thermal history or a phase-equilibrium calculation.

The reference L was 20 km until 2026-09-22. That is the thickness of one buried
sheet, and it gave a 1.3 Myr heating time: underthrust crust crossed the 600 C
reaction temperature almost as soon as it was buried, then converted, softened
and drained within about 10-20 Myr (one production run returned 810,000 km3 to
the mantle by 88 Myr). Burial displaces the geotherm of the whole thickened
column, so its thickness sets the relaxation length; for a doubled 35 km crust
the first-mode time is ~16 Myr, consistent with the 20-30 Myr that thickened
crust takes to approach peak metamorphic temperature (England & Thompson 1984).
This changes only the reference supplied to new migrations; a saved world keeps
the parameters recorded at its own migration until it is explicitly changed.

Dense material drains only under positive overstress. The reduced Bingham law
uses `rate = max((rho_d-rho_m)*g*D - yield_stress, 0)/(3*eta(T))`, in inverse Myr.
Reference strength is 5 MPa; reference viscosity is 1e21 Pa s at 1000 C, with
Arrhenius activation energy 200 kJ/mol. Strong roots remain attached, and cooling
reduces drainage through increasing viscosity. The dependence on buoyancy,
strength and viscosity is motivated by instability models; this particular
bulk drainage law is our explicit reduced closure, **not** a calibrated result
from [Beall et al. (2017)](https://academic.oup.com/gji/article/210/2/671/3813427).
It does not resolve a drip's shape, detachment depth, fluids, or mantle flow.

Pressure and rates are refreshed within substeps of at most 0.25 Myr. Each local
region keeps its own phase/heat state through the source interval, followed by
conservative projection to face means. Markers sample their actual region.
Projection loses unresolved heterogeneity between outer steps: mesh refinement
and timestep convergence remain necessary. Upper-column forcing is frozen at
the common parcel/marker source epoch. Reaction latent heat, adiabatic heating
and thermal expansion of this heat reservoir are omitted; existing rift thermal
support is a separate reduced component. No claim of a full energy equation is
made by the sensible-heat ledger.

The production column floor is applied to current mass-equivalent thickness, not
directly to physical thickness after density conversion. The exact contraction
of dense material still retained in the column is restored when testing the 8 km
mechanical reserve. Mantle return and erosion remove mass and still spend that
reserve. This lets hot buried material complete the declared conversion kinetics
without inventing drainage or silently trapping unconverted crust at a numerical
bound. Erosion and volume-conserving strain use the same restored-thickness
invariant, and juvenile gravitational references follow the phase-adjusted
physical floor. Physical positivity remains mandatory; complete parcel
retirement or oceanization is not implemented. See
[PHASE-ADJUSTED-COLUMN-FLOOR.md](PHASE-ADJUSTED-COLUMN-FLOOR.md).

The native physical-volume audit includes conversion contraction, and an
independent mass audit includes dense erosion and dense mantle return. Surface
support includes the retained lower layer's density, so conversion loads the
stack and detachment unloads it. Signed support survives sampling and foreland
input. Saved frames carry phase, composition, temperature, heat ledger and all
constitutive parameters; validators reject missing or inconsistent state.

New overlap during prescribed motion or gravitational relaxation stages a
contact order using the existing contacting-height policy. Rejected trial
geometries leave no records. Accepted geometry uses the same density functional
for gradients and energy acceptance; the world receives the staged records only
after successful relaxation. This fixes missing order at contact formation;
the separate inherited-subduction-polarity work below remains necessary.

## 10. Mechanical slab-attachment constitutive component

`slab_tether.py` supplies a reduced connection between a sinking slab and its
incoming plate. Per metre of trench it minimizes
`Phi(u,v) = Cs*u^2/2 + Cn*(u-v)^2/2 - W*u`, where u is down-dip slab speed,
v is inlet speed, W is signed down-dip weight, Cs is mantle drag, and Cn is neck
resistance. Eliminating u transmits `alpha*(W-Cs*v)` to the plate, with
`alpha=Cn/(Cs+Cn)`. Thus both the driving load and resisting coefficient change
together. It is not valid to multiply only slab pull, or damp an already solved
plate velocity. The resulting plate functional is convex in velocity.

The intact reference resistance is `4*eta*h/L`, the plane-strain Newtonian
extensional coefficient. That relation is given in
[Ribe and Xu (2020), equation 7a](https://academic.oup.com/gji/article/220/2/910/5613957).
The remaining damage law is **our explicit reduced assumption**, not that
paper's resolved instability: resistance is multiplied by `(1-d)^2`, where d
is accumulated positive relative opening divided by a required failure opening.
Compression does not heal damage. Reference h and L set intact resistance;
the component does not claim to evolve a volume-conserving neck shape. Viscosity,
reference geometry and failure opening must be supplied explicitly. They have
no hidden default calibration.

Under frozen weight, drag and inlet speed, damage evolution has a monotone
cubic time integral. The component solves it without dropping time or stepping
past failure, returns the exact event time and unadvanced remainder, and requires
the coupled caller to re-solve plate motion after rupture. Detached plate traction
and resistance both vanish while slab weight is supported by mantle drag.
Instantaneous weight power equals plate work plus the two nonnegative viscous
dissipations. Different damaged channels are summed separately; averaging their
damage or geometry would change their mechanical response on a trench join.

`slab_tether_history.py` explicitly initializes versioned neck channels against
a current conservative slab inventory. Existing slab split/join, owner transfer,
exact feeding/retirement and typed checkpoints carry these records without
averaging their damage. New feed is distributed by continuing channel reference
width, an explicit within-trench uniform-feed assumption. Rupture atomically
transfers a failed channel's retained area and excess mass into the retirement
ledger, preserving a detached-material subtotal. A ruptured channel cannot
reattach by receiving new feed. Mixed initialized/uninitialized joins and
inconsistent channel inventories are rejected.

These are **constitutive and lifecycle components**, not an enabled force or
polarity policy. They introduce no production flag or live-world migration.
Integration must still map channels into the plate force solve, use the shared
depth window and work projection from PR #6, resolve continental-entry buoyancy
without double-counting column gravity, retain attached polarity, and split
accepted timesteps at rupture before starting a new reversed system. The present
ocean representation has no explicit overriding material column; that missing
coupling must be addressed rather than simply removing the hard polarity override.

## 11. Frozen-state attachment forces in the plate solver

`Balance(simulation, dt, slab_tethers=True)` explicitly selects the attachment
operator for a frozen-state experiment. Ordinary `plate_balance.solve` does not
select it. The force development branch incorporates PR #6's shared depth window
and fixed-hinge work projection; no remote PR is merged by that local combination.

Each channel's current width is its reference-width share of the matched trace.
Its retained area and mass define its own uniform down-dip column and the fraction
inside the 660 km vertical window. Integrated neck resistance uses the saved
reference width, preserving neck resistance on rematching. Integrated mantle
drag uses current width and effective down-dip extent. Each local down-dip slab
velocity is eliminated against its own absolute plate inlet velocity before
assembling plate torques. Both transmitted drive and drag change together; the
legacy down-plate Stokes term is replaced, not added a second time.

The component retains existing overriding anchoring, bending and interface
closures. It is not a three-dimensional mantle/interface solve. Its power audit
recovers slab velocities and checks gravity power equals plate work plus mantle
and neck dissipation. The condensed velocity quadratic alone is not the full
physical slab dissipation. A zero-excess-mass slab can still supply resistance;
ruptured and retired material supplies neither attachment drive nor drag.

Different channel damage values remain separate through assembly and collinear
split/join. Their allocation is still uniform within a trace: joining differently
oriented regions can redistribute the location of forces, as documented for slab
memory. Spatially varying damage must be carried on persistent local channel
geometry before this can drive timesteps. In particular, the independent local
slab velocities must not be averaged into one trench-wide rupture decision.

This is tested through the actual nonlinear Balance constructor/solver and a
native mesh checkpoint continuation. **No event integration or continental-entry
policy is enabled.** Remaining work includes local damage geometry, finite-step
rupture/re-solve transactions, continental-entry buoyancy and inherited polarity.

## 12. Constrained-gravity inverse budget

Run `20260917-0012-nonpenetration` stopped in its 122 to 124 Myr step with
"Gravity could not advance within its exact energy/geometry/column bounds;
reduce the outer timestep." The interval was not the cause. When a relaxation
trial pushes a column past its area bound, `bounded_gravity.solve_endpoint`
builds the constrained motion from viscous inverses solved to 1/100 of the
caller's stationarity tolerance, but it gave them the caller's own iteration
budget. On the 122 Myr geometry the unconstrained sheet solve needed 897 of its
1,024 iterations and the identical inverse needed 1,051, converging steadily at
about 80 iterations per decade; earlier steps needed about 590 and 750. The
inverse depends only on the current geometry, so all 32 trial lengths repeated
the same failed solve. The search advanced only through trials of about
3e-8 Myr that touched no bound, and ran for about seven hours before one substep
exhausted every trial. The adaptive outer timestep retries only contact
failures, so the message's advice could not be followed either.

The harder solve coincided with suture-weld consolidation re-parenting one
1.75 km² face at the start of the step (contact 11, block 08 over block 02). It
was the only block 02 face at that contact both buried and welded, and it became
a one-triangle body of block 08 resisted only by its own basal drag. Gravity
moved it 84 km in the step; no other vertex moved more than 10 km. Every earlier
consolidation in the run moved faces of 179 to 8,700 km².

The repair changes solver budgets and error handling only:

- The inverses receive `PRECISE_ITERATION_FACTOR` (4) times the caller's
  budget, and the unconstrained relaxation solve 2,048 instead of 1,024
  iterations. Conjugate gradients stop at their stationarity gate, so every
  solve that converged under the old budgets is bit-for-bit unchanged and saved
  histories are unaffected.
- An unconverged inverse raises `InverseSolveError`, a `ConstraintSolveError`.
  `relax` passes it to the caller at once instead of repeating it over shorter
  trials; contact redistribution still sees an ordinary constraint failure.
- When no trial is accepted, the error names the last rejection: constrained
  endpoint, geometry bound, entry geometry or energy increase.

Tolerances, area bounds, the energy functional, acceptance tests and
constitutive laws are unchanged. For scale, on the failing geometry a sheet
velocity converged to 1e-8 instead of 1e-12 moves any vertex by at most 0.55 mm
over the 2 Myr step; 1e-10 moves it by 9 µm.

The captured failing relaxation now completes its 2 Myr in one substep with no
backtracks: 28 active area bounds, 7 SQP iterations, largest inverse 1,054
iterations, KKT residual 9.9e-11, area bounds held to 4e-13. Regression tests
cover the inverse budget, the immediate propagation of an unconverged inverse,
continued halving for interval-dependent infeasibility and the named failure.

Consolidation still moves individual faces, so a transfer can be as small as a
seam sliver that the sheet mechanics treat as a free particle. Transferring
coherent welded patches instead is a separate design question and is not
changed here.

## 13. Ocean crust across a spreading axis

A spreading birth transaction should credit new crust to its producing plate
without reassigning pre-existing ocean support. Explicit ridge jumps, capture,
and other ownership handoffs are separate processes. For an ideal ridge with
symmetric accretion and no such reorganization, migration through the mantle
does not by itself bias what either side receives. Run
`20260917-0012-nonpenetration` did not
behave that way. On the Primordial ocean 01 / Continental block 11 ridge near
7°N 108°E, crust created during the run reaches about 1,160 km west of the axis
and 590 km east, where a symmetric ridge would put 877 km on each side. The
western surplus of 283 km matches the eastern deficit of 282 km, which is the
signature of a conserved transfer rather than unequal production. The split is
0.50 at 20 Myr and then holds between 0.67 and 0.70 for the next 130 Myr, with
no isochron gap on either flank, so it is not a ridge jump. Axis migration is
correct: −3.6 km/Myr measured against −4.07 predicted from the Euler poles.

The transfer is directly visible. On every step, 12 to 24 raster cells lying
within 39 km of the axis and carrying crust aged 19 to 61 Myr were relabelled
from block 11 to ocean 01, totalling 467,278 km² over 40 Myr.

`native_spreading.advance` builds each step's new ocean from congruent finite
half-stage polygons, and that pairing is symmetric: the p shore maps to the axis
and the axis to the q shore, and the two halves are checked for equal area. The
accretion was never at fault. Ownership was. Any control cell receiving new
crust had its entire existing ocean support re-split between the two plates by
`fraction_q`, a support contrast `b/(a+b)` sampled at the back-rotated point.
The cell's total support was preserved, so the operation created and destroyed
nothing; it moved established crust from one plate to the other, after which
ordinary transport carried it away at the receiving plate's velocity. Old crust
taken from just inside one flank and handed to the other is carried back across
the axis, cramming the receiving flank's age structure against the axis and
dilating the donor's. That is the measured 10.7 km/Myr against 3.7 km/Myr in the
first 300 km.

The reported symmetry could not witness any of this. `side_p_area_km2` and
`side_q_area_km2` were `math.fsum(paired)` twice, one value under two names, so
their equality held by construction.

The repair credits each half of a congruent patch to the plate whose shore
produced it. The pairing already resolves that side geometrically — `pc` comes
from the p shore strip and `qc` from the q shore strip — so no field sampling is
needed. Only the newly created fraction of a receiving cell is placed:

    transported[:, at] *= 1 - birth[at]
    transported[:, at] += total * created / cell_area[at]

where `created[plate, cell]` is the new area that plate's own shore made. Older
crust in the cell is diluted by the new crust sharing it and keeps its plate.
Support per cell is conserved exactly, as before. The originally staged repair
summed two lists containing the same paired areas, so its claim of independent
side measurements was incorrect. The reconciliation keeps the area measured
from the rotated q polygon for `side_q_area_km2`, rather than discarding it after
the congruence check. The p and q diagnostics now sum separate polygon-area
measurements. Deposition still uses the shared congruent area and the existing
congruence tolerance is unchanged; measurement roundoff can make their sum
differ slightly from the deposited total. This checks congruence of the admitted
pair, not whether the model has resolved every real-world spreading asymmetry.

The reconciliation tests both shore identities against the fixture's geometric
axis, so swapping the two producing owners fails even if total support and
old-support dilution remain correct. A separate diagnostic test perturbs the
q-side area measurement within the existing congruence tolerance and requires
the reported q area to change while birth and ownership arrays stay bit-exact.

On the bench fixture the original hands 1.65 units of support across the axis
per step, 0.33 of it from a single cell; the repair moves none. In the original
staged ownership benchmark, `generated_area_km2` and the two reported side areas
were identical before and after in every configuration tested. The later
diagnostic correction can change the q-side report by measurement roundoff;
five fixture comparisons retain bit-exact birth and ownership arrays and exact
production totals. The ownership change decides which plate receives the crust
without changing how much is made. Eligibility masks, the water capacity guard,
the pairing geometry, boundary geometry and Euler poles are unchanged.

The contrast field also smoothed the ownership interface inside a cell.
Ownership now emerges from accumulated support, which is only as sharp as the
control mesh. Diluting old support by the new area fraction is still a mixed-cell
approximation; it does not track every ocean parcel or resolve subcell isochrons.
This changes ridge behaviour in
every new run; saved experiments continue under their own preserved engine, and
the asymmetric record already written before the migration epoch is not repaired
retroactively.

## 14. Constrained-gravity active-set budget

Run `20260917-0012-nonpenetration` stopped in its 160 to 162 Myr step with
"Contact region 3 accepted only 0 of the 0.001953125 Myr interval; the complete
coupled timestep must be rejected. Constrained gravity exceeded its explicit
active-set budget." At the time `solve_endpoint` capped its SQP active set at a
flat default `max_active` of 128, and contact region 3 held 1,385 faces, so the
default was under a tenth of the region it had to describe. The local repair
below was written against that constant. It has not been shown to be the cause
of that stop: the message came through contact accommodation, which supplied its
own `max_active=32` (section 16), and the exact active set at the crash was
never captured. [The active-set capacity review](ACTIVE-SET-CAPACITY-REVIEW.md)
records that qualification. This section is a capacity change, not a validated
repair of the 160 Myr failure.

The run did not hit a cliff; its contact load grew steadily. Limited faces were
31 at 80 Myr, 71 at 120, 103 at 144, 132 at 156 and 134 at 160, while the number
of contact regions went from 147 to 394. Rejected trials were zero for the whole
run until 154 Myr, then one, then six. The 158 to 160 Myr step completed only by
halving to 1/64 Myr, and its recorded `last_rejection` already names the same
region and the same budget. These are aggregate counts. They do not show more
than 128 simultaneous local reactions in any one solve.

In the failing step the adaptive timestepper halved ten times, down to
2/1024 Myr, without completing. The local diagnosis read this as the same shape
of mistake as the 122 Myr inverse budget in section 12: bound violations present
in the start-of-stage geometry do not depend on the interval, so a
step-independent budget was retried by shrinking a timestep that was never the
problem. The review qualifies that reading. An entry-violation count is fixed
only for identical solver inputs, and a whole coupled retry need not present
identical inputs.

The change has three parts:

- `max_active` now defaults to `max(MINIMUM_ACTIVE_BUDGET, int(ACTIVE_SET_FACE_FRACTION*len(faces)))`,
  that is `max(128, faces/4)`: 346 for a 1,385-face region. The budget is a
  work bound, since each active bound costs one preconditioned inverse solve per
  inner iteration and one row of the dense Schur matrix, so it has to scale with
  the solve instead of sitting at a constant. The quarter-face rule is chosen
  headroom, not a measured requirement.
- Step independence is measured rather than assumed. `entry_violations` counts
  the start-of-stage areas already outside their bounds. Only when that count
  alone exceeds the budget is `ActiveSetBudgetError` raised, and
  `gravitational_relaxation` re-raises it instead of halving. Exhaustion driven
  by the motion, with an admissible entry geometry, remains an ordinary
  `ConstraintSolveError` and is still retried over a shorter interval. This
  routing is inside gravitational relaxation only: contact accommodation still
  catches `ConstraintSolveError`, and the coupled timestepper still retries
  `IncompleteContactStepError`.
- `active_set_budget`, `entry_bound_violations` and `maximum_active_bounds` are
  reported by both the moving active-set solve and the zero-velocity
  certificate; the budget and the maximum active count are aggregated into the
  relaxation diagnostics, and the entry-violation count stays on the
  per-substep rows. Nothing recorded
  the binding number before, which is why this load stayed invisible until the
  run stopped. `maximum_active_bounds` comes from accepted SQP history after
  pruning, so it can understate the largest set attempted at the capacity check.

Unlike section 12 this is not bit-identical for every previously converging
solve. `current_stationary_certificate` skips itself when its tight-bound count
exceeds `max_active`, so a stage with between 128 and the new budget tight
bounds may now take the zero-velocity stationarity certificate where it
previously fell through to the SQP. Both paths certify the same physics and
both are valid; their numbers need not agree bit for bit. That regime needs
more than 128 faces within 128 eps of a bound, and is the same regime that
previously risked the failure. Tolerances, area bounds, the energy functional,
acceptance tests, the line search and the constitutive laws are unchanged, and
any solve whose active set stayed under 128 is unaffected.

Raising the budget buys headroom; it does not explain why so many faces reach
their bounds. The per-stage bounds are only about −7.7% to +8.3%, and a region
that keeps pinning faces may want `native_material_adaptivity` to refine it
before it pins rather than a larger budget to tolerate it. That is a separate
question and is not addressed here.

The local package recorded 56 passing focused tests, including three new
gravity tests, on Python 3.12.10 and NumPy 2.3.5. Those tests did not replay the
crash. The local record is `claude-repair/gravity-active-set-20260920/` in the
desktop installation; it is not part of this repository.

## 15. Generated large igneous provinces erupted in the wrong place

`lip_events.initialize` chooses each generated source by drawing an original
continental parcel, weighted by area, and freezing that parcel's longitude and
latitude as the eruption centre. It then draws a start time uniformly over the
whole run. Those two decisions are inconsistent, because plates move between
them. The centre belongs to the geography of t = 0; the eruption happens
whenever the start time says.

The 500 Myr SEP20T configuration shows the scale of it. Of its twelve generated
events, **seven fire more than 100 Myr after their site was chosen**, and the
median start time is 255 Myr. At this model's plate speeds — 10 to 30 km/Myr in
the September run, locally more — 255 Myr is thousands of kilometres of drift.
The generator refuses to run without continental material ("Generated
continental LIPs need original continental material") and then, for most of its
events, deposits into whatever happens to occupy a point the intended continent
left long ago. Only the earliest few land where they were aimed.

A generated source now chooses its host when it opens, in `_place_generated`,
called once per event immediately before its first deposit so the 48-sided
footprint is built around the erupting centre rather than the original one. The
draw uses the same eligibility test as `initialize` — original continental
material, excluding accreted arcs — weighted by area, from a generator seeded on
the run's own LIP seed and a checksum of the event id. That makes the placement
reproducible and independent of the order in which events arrive. The event
records `placed_myr`, and generated events carry `generated: true` so they are
distinguishable from authored ones; **authored events keep the coordinates they
were given**, which is the whole point of authoring them.

Placement is triggered by the step that reaches the event's `start_myr`, not by
its first positive supply. The engine splits steps at `next_transition`, so the
step that ends exactly at onset supplies nothing yet marks the source started.
The first locally installed version keyed placement on supply and therefore
almost never fired through the engine: runs started under it most likely still
erupted their generated sources at the t = 0 site.
`test_generated_placement_survives_a_step_that_ends_exactly_at_onset` covers
that case.

This is a worldbuilding closure, not plume physics. A real plume is fixed in the
mantle and the plate drifts across it, which produces a hotspot track and
sometimes erupts through ocean floor rather than continent. This model has no
plume reference frame, and the module header already says these are "not
separate resolved magma chambers, lava layers or mantle plumes". The correction
makes the behaviour match the generator's stated intent — continental flood
basalts on continental crust — for every event instead of only the early ones.

Existing runs are unaffected while they keep their own engine: each freezes its
own copy of `lip_events.py`, so run `20260920-223931-00c45f` keeps the original
placement for all twelve of its events. The change applies to runs started
after it. A checkpoint restored under this source is different: rows written
before this change have no `generated` flag and fall back to the `generated-`
id prefix, so generated events that have not started yet are re-placed, and
such a continuation is not bit-identical to its frozen-source continuation.

## 16. Contact-accommodation active-set budget

Run SEP21T (`20260921-011257-b2f85b`) did not stop, but it slowed badly. From
62 Myr the 2 Myr step began rejecting coupled trials: 8 rejections down to
0.0625 Myr at 62 Myr (67 min wall time against about 7 min before), then 2 per
step, then a 68 Myr step still unfinished after more than 2 hours. Every
`last_rejection` named contact region 3 and "Constrained gravity exceeded its
explicit active-set budget".

That message came from `contact_response.redistribute`, not from gravitational
relaxation. `redistribute` called `bounded_gravity.solve_endpoint` with a
hard-coded `max_active=32`, which overrode the scaled default section 14
introduced. It was the same kind of flat cap, in a call that repair did not
reach.

Replayed from the 60 Myr checkpoint with the run's frozen sources, region 3
(931 faces, 6.2e6 km2, mean face about 6,600 km2) measured:

| trial | faces past bound | overshoot km2 | share of region | bounds needed | budget 32 | scaled budget 232 |
|---|---|---|---|---|---|---|
| 2 Myr | 39 | 7,283 | 0.12% | 38 | refused | solved |
| 1 Myr | 31 | 2,614 | 0.04% | 34 | refused | solved |
| 0.5 Myr | 32 | 1,671 | 0.03% | 34 | refused | solved |

Here halving does help: shorter trials pin fewer faces, and the solve slips
back under 32. So the timestepper did not fail, it slowed down, repeating whole
coupled steps to stay two to six bounds under a constant.

The repair is `contact_response.ACTIVE_SET_BUDGET = None`, which defers to
`bounded_gravity`'s `max(MINIMUM_ACTIVE_BUDGET, ACTIVE_SET_FACE_FRACTION*faces)`.
Area bounds, feasibility tolerance (1e-12), the endpoint acceptance checks
in `deforming_regions`, and every physics gate are unchanged. Every accepted
endpoint still satisfies the same bounds. As in section 14, a region with more
than 32 tight bounds may now take the zero-velocity certificate or finish its
SQP where it previously refused, so solves in that regime are not bit-identical.
Below 32 active bounds nothing changes. With the repair, the 60 to 62 Myr step
replayed from the checkpoint in 11.6 minutes with no rejected trials and at most
38 active bounds, and the same repair has run in SEP21T since 62 Myr.

`tests/test_contact_accommodation_budget.py` replays the captured 60 Myr solve
(`tests/fixtures/contact_accommodation_budget_sep21t_60myr.npz`). It fails
against the run's original `contact_response.py` with the live error and
passes with the repair. The local package and in-place migration are in
`claude-repair/contact-accommodation-budget-20260921/` in the desktop
installation; they are not part of this repository.

## 17. Unordered overlap floor

Run SEP21T stopped in its 84 to 86 Myr step with "Density-dependent overlap
requires a unique persistent layer order." The 2 Myr trial was rejected in the
ordinary way. In the 1 Myr retry, arc emplacement at 85 Myr left two juvenile
arcs on block 02 (arc 38, born 46 Myr, and arc 93, born 70 Myr; faces of 208
and 143 km2, sheets 49 and 104) overlapping by **1.83e-8 km2, which is 0.018 m2**.
The reference-energy audit that runs right after emplacement then refused the
pair. The error is not retried, so the whole run stopped.

Three tolerances disagreed:

- `arc_emplacement_geometry.inspect` admits new overlap up to
  `max(1e-9, 1e-12*area)` km2, measured by polygon clipping;
- `mesh_coverage.material_overlaps` counts any overlap above `1e-8` km2, by a
  different intersection routine;
- only `column_density.ContactAdmission`, which runs during motion, records a
  layer order, so an overlap created by emplacement never gets one.

An overlap of that size is roundoff. It is eight orders of magnitude below one
2 km map cell (4 km2), and no process over a billion years resolves it.

The repair is `column_density.UNORDERED_OVERLAP_FLOOR_KM2 = 1`. A distinct-sheet
overlap with no recorded order, at or below 1 km2, takes the mean of its two
stack-order weights. `gravitational_relaxation.reference_energy` and
`energy_gradient` pass the measured overlap areas so both use the same
weights. An unordered overlap above 1 km2, or an order that contradicts
itself, still raises, so a genuinely missing contact still stops the run.
Ordered pairs use the same arithmetic as before and are bit-identical; the
floor changes only pairs that previously raised. Of 1,213 overlaps at the
failure, 336 were under 1 km2, but every one of those was an ordered
collision contact and is untouched.

`tests/test_unordered_overlap_floor.py` uses the two captured triangles
(`tests/fixtures/unordered_arc_overlap_sep21t_85myr.npz`). Its energy test
fails against the run's original sources with the live error and passes with
the repair. The local package and in-place migration are in
`claude-repair/unordered-overlap-floor-20260921/` in the desktop installation;
they are not part of this repository.

Not addressed: emplacement still admits roundoff overlap without recording an
order. A cleaner fix would have emplacement record first-contact order the way
ContactAdmission does, which is an arc-emplacement design change.

## Continental rupture measured in kilometres, not mesh strain

`progressive_rifting.commit` removed a damaged, opening link only when its
accumulated strain passed a hand-set number: 0.35 in enhanced mechanics
(realized logarithmic link strain) and 0.15 in legacy mechanics (the
quarter-partitioned loading proxy). In both modes that strain is a length change
divided by the link's own length, and link length is set by the
`mechanics_nodes` budget, not by geology. So the kilometres of extension a rift
needed before breakup changed about fourfold across the budgets the UI offers:
enhanced 0.35 asked for 413 km at 512 nodes, 289 km at 1,024 and 103 km at
8,192; legacy 0.15 asked for 148, 103 and 37 km of realized-equivalent
extension. Neither number was derived from a physical quantity
([RUPTURE-CALIBRATION.md](RUPTURE-CALIBRATION.md) says so).

Rupture criterion 1 gates each link on its accumulated realized extension,
`BREAKUP_EXTENSION_KM = 100` km, together with the unchanged damage of at least
0.95 and current opening above 0.02 km/Myr. Enhanced links use their own
`extension_km` (the sum of positive realized lengthening). Legacy links use a
quarter of theirs, because `extension_km` holds the full proxy opening and the
model already counts that quarter (`PROXY_REALIZED_FRACTION`) as realized for
damage strain and column thinning; legacy links therefore need 400 km of proxy
opening. That quarter is itself an uncalibrated reduced-model partition, so the
legacy gate is only as physical as it. Strain still accumulates and still
drives damage. Every later guard
(coherent cut, protected hosts and cratons, projection cleanup, daughter
viability and opening) is unchanged.

100 km is a round value between magma-assisted breakup (Main Ethiopian Rift,
of the order of 60-80 km; the figure is not yet confirmed against Corti, 2009)
and magma-poor breakup (Iberia-Newfoundland, 229-256 km; Sutra et al., 2013).
A pure-shear neck that thins crust to zero conserves crustal area, so its
extension equals its original width, and two necking domains of about 55 km
each (Chenin et al., 2025) need at least about 55 km. The value is an open
choice for the user (100, about 70-80 or about 230 km). That both old gates
implied about 103 km where each was set (0.419 x 246 km for enhanced at
SEP21T's links, 0.15 x 687 km for legacy at the calibration workload) is a
consistency check, not the justification. On the 2 km/pixel map, 100 km is
about 50 pixels of total opening spread over two tapering margins. Typical
links run from about 246 km (8,192 nodes) to about 1,970 km (128 nodes), longer
than an Earth neck, so one link spans the neck and its own extension is
compared directly; a series rule for links shorter than the neck would only be
needed above about 59,000 nodes. In SEP21T such short links are 3-15% of the
damaged, opening links, and a read-only check found the series rule changed no
candidate cut.

Effect: SEP21T's links average 246 km in every checkpoint from 60 to 208 Myr
(the 324 km quoted in the gap list does not match the run). Applied read-only
to its bond state on the timeline before the 2026-09-25 rewind, criterion 1
admits 85 links and 1 candidate cut at 208 Myr, against 113 links and 2 cuts
under 0.35; on the current timeline at 268 Myr it admits 106 links and 3
candidate cuts, none two-component. The 74 and 86 Myr breakthroughs had
cut links of about 93-211 km and would still have been admissible at about the
same time. Enhanced worlds at coarse budgets now break up after 100 km instead
of 289-413 km; legacy worlds at the UI default are unchanged (103 to 100 km);
legacy worlds at 8,192 nodes need about 2.7 times more. The breakup count in
SEP21T is limited by missing internal tension and current opening, not by this
gate.

Versioning: `physics_profile.initialize` sets `rupture_criterion_version = 1`
on fresh `reviewed_v1` worlds and refreshes the rift mechanics labels. Saved
worlds and checkpoints lack the attribute, read 0 and keep their criterion-0
strain gate exactly, with identical frame labels: 0.15 for legacy mechanics,
and for enhanced mechanics the run's recorded `realized_rupture_strain_threshold`
(SEP21T: 0.30) or else 0.35. That per-run behaviour comes from PR #188, which
this change is built on and which must be merged first; a recorded per-run
strain selects nothing under criterion 1, which has no strain threshold. Pending mechanics work records its criterion and
extension and is never committed under a different criterion. Frames carry
`rupture_criterion_version` (when 1) and `breakup_extension_km`,
`realized_extension_fraction`, `rupture_extension_measure` and
`max_bond_realized_extension_km`; `progressive_rifting.validate_frame` checks
them from `mesh_history.arrays` and `native_frame_sampling.prepare`.

`tests/test_rupture_criterion.py` checks that the gate compares the same
kilometres at link lengths of 250, 400, 700 and 1,100 km in both modes (exactly
100 km passes and the next float below fails; this shows the gate is length
independent by construction, not that the model's physics is the same across
meshes), that the version 0 gate gave different outcomes at
the same 150 km, that strain no longer decides, that damage and opening still
gate, pending compatibility, fixed-rate step-subdivision additivity in both
modes and frame validation (including that the realized fraction and extension
measure match the mechanics mode). `tests/test_reviewed_physics_profile.py`
checks the profile wiring, frames and a checkpoint round trip, including an
enhanced world whose pending work with `extension_km` survives a checkpoint and
reaches the same km decision. The physics profile block of a saved reviewed
world's frames gains no key (the version is written only when it is 1). `tests/test_rupture_calibration.py` passes
unmodified.

Limits and not addressed:

- The model's realized deformation localizes over deforming belts of
  `deformation_width_km` (default 400 km) with 100 km smoothing, not over a
  55 km neck. Whether the kilometres a rift reaches are the same across node
  budgets and belt widths is not yet shown; SEP21T at one resolution shows
  extension nearly uncorrelated with unstretched link length (correlation
  0.06). The multi-resolution run harness (G121) is the real test and remains
  open.
- The gate reads gross positive lengthening, never reduced by shortening, so
  an inverted rift keeps its earlier kilometres and the gate is looser than
  the net geological extension wherever a link has a compression history (as
  criterion 0's strain was). A net-extension measure is a follow-up.
- Regrouped nodes copy a donor bond's whole `extension_km` to each child link,
  so two links in series across one neck can both pass; damage and the
  coherent cut bound this.
- Per-margin breakup extension from rift heat or large igneous provinces, and
  resolving necks narrower than a link on coarse meshes, where distributed
  wide-rift stretching can also meet the km gate.

## Trench identity across plate splits and remnant absorption

Trench records are matched to boundary edges by plate-ID pair. Before this
repair only back-arc separation and terrane accretion rehosted a record across
an ownership change (`transfer_overriding`). Two other changes stranded it:

- **A split of the subducting plate** (`native_topology._commit`). The slab
  hangs from the leading edge, which now belongs to the daughter, but the
  record lost contact ("contact_disappeared"), its slab stopped pulling and the
  margin restarted as a new initiating trench. In a 2026-09-21 production run a
  1.3e6 km2 retained slab (~25 GW) went quiet this way at 74 Myr and the three
  plates involved slowed from ~16 to ~5.5 km/Myr.
- **Absorption of a tiny remnant plate** (`native_topology._retire_remnants`).
  Every record of the absorbed plate was shut down as `plate_owner_lost`, so a
  subducting microplate's slab vanished just as it was captured.

`trench_history._transfer` now moves either role, and `transfer_split` carries
both roles for a split and for whole-plate absorption. A transfer onto the plate
that already holds the other side cannot create self-subduction: a wholly
absorbed trench ends as `boundary_absorbed`, and a partial move keeps the record
on the faces that still separate the two plates. A record whose edge moves wholly
to the daughter keeps its identity, maturity and slab; one that spans both pieces
becomes a child with inherited maturity and a length-proportional share of the
slab (and of its tether channels), exactly as for overriding transfers. The
transfer runs after the membership move, while the pre-split boundary cache is
still valid. The split transfer ports a
locally installed repair; its production A/B from 70 Myr kept the record mature
on the daughter at 76 Myr with 28.9 GW of slab power and 18.4 km/Myr, against
0 GW and 5.8 km/Myr without it.

The locally installed run record, SEP21T (`20260921-011257-b2f85b`), adds
detail. The stranded record was trench 15/122, block 08 over Continental block 02
since the initial condition: mature, 1,441 km consumed. When the 74 Myr rift
gave block 02's subducting edge to Rift plate 01, the record went quiet with
`contact_disappeared` and its slab retired on an unused record. The same margin
restarted as new initiating trenches with no slab (maturity 0.25 and 25 km of
shortening by 84 Myr). The map then flickered between "subduction" and
"transform" there, because closing speed hovered around the 2 km/Myr label
threshold. Replayed from the 70 Myr checkpoint with the transfer, 72 Myr was
identical to the original run. At 76 Myr trench 122 was block 08 over Rift
plate 01, still mature, with 1.30e6 km2 of slab, and block 08 moved at
12.1 km/Myr; in the original all three plates moved at 5.8 km/Myr. Runs diverge
from the first split of a subducting plate, so SEP21T was rewound to 70 Myr to
continue under the repair. The local package is
`claude-repair/trench-split-transfer-20260921/` in the desktop installation; it
is not part of this repository.

## Accretion plate-age policy

`local_accretion` refused every ownership transfer while either plate was
younger than `MIN_PLATE_AGE_MYR = 70`. The value has no recorded derivation.
It blocked all docking for the first 70 Myr of a world, and because rift- and
back-arc-born plates reset their birth time, each new microplate or arc waited
another 70 Myr before any contact could weld it. In one production run
(2026-09-21, 88 Myr) accreted area was still zero.

Root-local welding already requires every root of a terrane to be engaged and
to reach loading 18 at no more than one unit per Myr of full engagement, keyed
by plate UID. The only protection the age gate can justify is that a plate may
not use engagement older than itself. Age policy 1 therefore sets the minimum
plate age to that threshold (18 Myr). In a two-terrane fixture a plate born at
the start of convergence now docks after the same 36 Myr as an old plate, and a
plate reborn with mature exposure is still refused.

Fresh reviewed worlds with root-local welding select policy 1
(`accretion_age_policy_version`). Older worlds keep the recorded 70 Myr gate.
The whole-component terrane rule, its 400 km process zone and the retired
suture path are unchanged.

## Back-arc loading under the moving hinge (sea anchor)

PR #164 removed the heuristic forearc sweep from moving-hinge worlds
(`subduction_response_version = 1`, G21). `NativeSimulation._migrate_trenches`
now sets `trench_retreat_speed` to zero and returns. That part is correct: the
hinge is carried by the overriding plate by construction. But `backarc.update`
still took its only loading driver from that array. With the speed always zero,
the `speed > 0.4` test never passed, no prospective basin was created or
advanced, and existing rows only decayed. SEP21T, migrated to the moving hinge
at 156 Myr, had 21 loading rows at 0.4-2.8 km/Myr before the migration. Every
loading rate was 0.0 by 158 Myr, and by 180 Myr only the two basins that had
already opened (94 and 132 Myr) remained. A rigid-carrier hinge has no
"relative trench retreat", and nothing replaced it.

Back-arc driver 1 (`backarc_driver_version`) replaces it with a quantity read
from the force balance's solved motion. In the balance, the slab anchor
(`SLAB_ANCHOR_PA_S`) resists the overriding plate's own trench-normal velocity
in the stationary-mantle frame. When the solved upper plate moves away from its
trench, the anchored slab holds the hinge back. The moving-hinge closure
removes that lag by making the rigid plate drag the hinge along, and it
dissipates the anchor work. A deformable back-arc would take the lag up as
extension. This is the sea anchor of Uyeda & Kanamori (1979). Heuret &
Lallemand (2005, PEPI 149:31-51) found that back-arc strain follows the upper
plate's absolute trench-normal motion: moving away gives Mariana-type
extension, advancing gives Andean-type compression. The loading rate is
therefore

    rate = max(0, (omega_over x r) . n_inland)    km/Myr

on attached subduction edges, with `n_inland` the unit trench normal from the
downgoing plate into the overriding plate, and 0 elsewhere. It has the same
units and meaning as the old swept retreat: km per Myr of hinge lag the back-arc
would have to accommodate. So the existing km loading ledger keeps its meaning.

An edge counts as anchored only where the attached slab reaches below the
overriding lithosphere, because mantle resistance acts only on slab that
extends below it:

    slab_length x sin(50 deg) > overriding lithosphere thickness

Slab length is `slab_memory.line_load`'s retained length, the same attachment
the force balance uses, with its fixed 50 degree dip. Thickness is
`plate_balance.plate_thickness_m` of the overriding cell's age for oceanic
crust, and the same law's 100 km cap for continental and arc crust. The hinge
bending term already uses that law and cap. For a continental or old oceanic
upper plate the threshold is 130.5 km of slab, i.e. 100 km of depth, which
coincides with `primordial_subduction`'s default initial slab depth. A 25 Myr
oceanic upper plate (65 km thick) is anchored by 85 km of slab. No new number
is introduced.

This depth condition belongs to the back-arc driver only; it is not in
`plate_balance`. The balance applies its anchor to every attached slab, scaled
by slab length / 862 km, including slabs shallower than the overriding
lithosphere. The driver's condition is a step: a 129 km slab under a
continental plate gives no loading, while a 132 km slab gives the full away
speed, even though the balance's anchor there is only about 0.15 of full
strength. Past the condition the whole kinematic lag counts, because a hinge
held by slab below the plate is left behind at the plate's own speed; the
anchor-force scale sets how hard the slab resists, not how far the hinge lags.
Scaling the rate by the anchor factor instead is listed under the rejected
alternatives below and remains an open question for the user.

Frame dependence. The speed is measured in the stationary-mantle frame the
balance solves in, not a no-net-rotation frame, because the balance's drag and
anchor resist absolute velocities; mixing frames would break the causal chain.
The rules of thumb (RULES-OF-THUMB.md) agree: Rule VII says margin style
depends on the absolute motions of the plates (Western Pacific margins are in
net divergence, rolling back 1-2 cm/yr), and Rule X says hotspots give a
good-enough absolute frame. How much of the result the frame carries, from
SEP21T frames 84-95 (168-190 Myr): the solved net rotation is
0.011-0.014 deg/Myr, 1.3-1.5 km/Myr at the equator. It makes up to about a
quarter of trench 40's away speed: 3.9 km/Myr in the mantle frame against 2.9
with net rotation removed at 168-176 Myr, and 2.6 against 2.2 at 180 Myr. The
share of trench length moving away faster than 0.4 km/Myr barely changes
(17-20% against 15-20%). So the one rupture is not a net-rotation artefact,
but net rotation is the same order as the 0.75 km/Myr crossover the
80 Myr / 60 km gate needs. A common rotation of all plates that moves an upper
plate away from its anchored slab does load it under driver 1. That is
deliberate, since the balance resists net rotation through drag; driver 0's
invariance to common rigid rotation does not carry over.

Nothing else in the loading gate changes. The trench-edge filter (closing,
oceanic downgoing side, maturity > 0), the 0.4 km/Myr edge and sector tests,
the 2,200 km sector, the 800 km minimum, the 80 Myr memory, the 60 km rupture
gate, the 8 Myr waits, the host and driven-pair exclusions, `_commit` and
`choose_arc_sliver` are the same code. After a rupture the new arc plate's
motion comes from the force re-solve, as it did for SEP21T's two earlier basins
(1-12 km/Myr of opening). `trench_retreat_speed` stays zero, so arc supply
(`material_forcing`, `native_processes`) and the balance's guard against
independent retreat keep their meaning. A moving hinge consumes exactly the
relative convergence already in `normal_speed`, and feeding the lag into arc
supply would count ocean that is never swept.

Rejected alternatives:

- Writing the anchor speed into `trench_retreat_speed`. This double counts
  consumption and reopens the guard.
- Netting in the slab's horizontal mantle-frame motion. That makes almost every
  trench compressive, which contradicts Heuret & Lallemand.
- Scaling the rate by the anchor factor (slab length / 862 km). SEP21T's slabs
  are mostly 150-400 km long, so this gives zero back-arcs in 208 Myr (maximum
  loading 12.7 km).
- Unscaled speed on every attached trench. This ruptures a trench whose slab is
  about 10 km long.
- A stress-against-strength test now. The anchor traction is 0.04-0.25 of
  `F_REF`, while the suction reaction exceeds it at mature trenches, so doing
  it properly needs arc thermal weakening and an intraplate-stress path
  (G28/G01/G47).

Versioning:

- Fresh `reviewed_v1` worlds with `subduction_response = moving_hinge_v1` get
  driver 1 in `physics_profile.initialize`.
- Fixed-trench and legacy worlds get driver 0, and every saved checkpoint
  without the field reads as 0 and continues bit for bit. That includes SEP21T
  and the 146 Myr production restart migration, which is unchanged.
- Driver 1 refuses to run with a fixed trench, since the sweep there already
  supplies retreat.
- New frames carry `backarc_driver_version` and `backarc_driver_diagnostics`:
  anchored edges and length, loading length, maximum and length-weighted mean
  speed, and the declared frame and law. Both frame validators
  (`mesh_history`, `native_frame_sampling`) check them. A driver-0 frame may
  not carry diagnostics.
- The diagnostics are measured at the step's back-arc update, which runs
  before that step's topology commit and post-split force re-solve. In a
  rupture step they therefore describe the plates and motion before the split,
  while the frame's plates and omega are after it. They are not recomputed in
  `snapshot()`, so a frame never depends on whether a snapshot was taken, and
  the checkpoint carries the same dict.
- Under driver 1 the back-arc update calls `slab_memory.line_load`, which
  validates every trench row, so a corrupt slab ledger is reported there
  rather than at the next force solve. The depth test uses `line_load`'s
  attachment even under the experimental slab-tether path, whose balance
  anchor uses tether owners and extents; the two are not reconciled (no saved
  world enables tethers). A non-finite age on an oceanic overriding cell is
  refused rather than read as "not anchored".
- `loading_rate_km_myr` now reports whichever driver the world uses. The UI
  labels it "Back-arc loading rate".
- The event wording names the overriding plate moving away from its anchored
  slab.

Expected effect, from replaying SEP21T's own solved motions over 156-208 Myr
with frames rather than the engine:

- About 17-19% of the trench length (62,000-69,000 km) has the upper plate
  moving away faster than 0.4 km/Myr in the mantle frame, and 5-10% is also
  anchored. The p10/p50/p90 speeds are -14/-7/+2 km/Myr, so most upper plates
  advance and never load.
- Four trenches load. Trench 40 is intra-oceanic (Primordial ocean 02 over 04,
  a Mariana analogue). It reaches the 60 km gate at about 180 Myr, at
  2.6 km/Myr over 1,830 km. Trenches 201 and 3, continental margins, reach
  35-40 km, and trench 322 reaches 27 km.
- Over 0-208 Myr the law gives one rupture (trench 40, 172 Myr). The older
  150 Myr / 45 km policy would give two (trench 40 at 166 Myr and trench 3,
  Continental block 07's margin, at 198 Myr).
- Rates are 1-4 km/Myr (0.1-0.4 cm/yr, 0.5-2 px/Myr at 2 km/px). The 60 km
  gate is about 30 px of accumulated hinge lag.
- An Earth-like count of active back-arcs would need Earth-like plate speeds.
  SEP21T's mean is 0.9 cm/yr against Earth's 4-5 (G03/G24).
- The same replay on the frames SEP21T regenerated after its 2026-09-25 rewind
  to 170 Myr (156-266 Myr, loading carried over from 156 Myr) gives the same
  single rupture: trench 40 at 180 Myr. Trenches 3 and 322 then reach 41 km
  and trench 201 40 km. With the 150 Myr / 45 km policy it gives three
  (trench 40 at 174 Myr, trench 3 at 204 Myr, trench 322 at 238 Myr). Over
  170-266 Myr, 17-24% of the trench length has the upper plate moving away,
  but only 4-10% is also anchored, and the median upper plate slows from
  7 km/Myr of advance to 2. The replay does not feed a rupture back into the
  motion.

Tests: `tests/test_backarc.py` (`SeaAnchorDriverTests`,
`SeaAnchorFrameTests`) and `tests/test_subduction_response_profile.py`. The
last includes a native-engine run: the moving-hinge startup world with its
longest margin's upper plate prescribed 5 km/Myr inland of the trench loads,
ruptures an arc plate through the unchanged sliver path, and is re-solved by
the force balance in the rupture step. The same world under driver 0 creates
no basin.

Interaction with force-limit rifting (PR #189, not yet merged here):

- #189 adds an opt-in stress test (`force_limit_rifting`): rigid-plastic limit
  analysis of each plate carrying slab pull or the moving-hinge reaction,
  against depth-integrated rock strength. It checks overriding plates too, and
  its loads include the balance's slab anchor drag at the solved velocity. So
  it can tear a plate behind its arc: the stress-based counterpart of this
  driver, which measures the same anchor's kinematic lag.
- Nothing is summed twice. Driver 1 writes no force, no velocity and no
  `trench_retreat_speed`; it only advances back-arc rows. #189 reads the force
  balance and never reads back-arc rows. Neither ledger feeds the other.
- Splits are serialized. In #189's `native_topology.update_topology`, a pending
  back-arc rupture commits first and ends the step's topology stage; a
  force-limit tear can commit only in a step where no back-arc rupture
  committed. Each split goes
  through `native_topology._commit` and a fresh force solve.
- A realized split resets the other ledger for that cut. After a back-arc
  rupture, #189's accumulated opening on the parent restarts unless its
  mechanism's smaller side still overlaps the previous one by 60%, and the
  sliver is no longer part of the parent. After a #189 tear, a back-arc row
  continues only if the same plate UID still overrides that trench. If the
  trench-side piece is the new plate, the old row decays and a new row starts
  from zero.
- What remains is two failure criteria for one cause: with both on, a margin
  ruptures when either criterion is met first. That is a policy choice, not a
  double count. #189 does not respect back-arc host protection
  (`backarc.protected_hosts`), and a #189 tear behind an arc is not recorded
  as a back-arc basin. Once #189 lands, the user should decide whether its
  arc-margin tears should register as back-arc basins, or whether driver 1
  should stand down in worlds with `force_limit_rifting` on. Code for either
  belongs with #189's merge, not here.

Not addressed:

- Tension from the v1 slab suction reaction is not counted (G28/G01).
- Inherited primordial slabs are declared 100 km deep by default, which is
  exactly the anchoring depth under a continental or old oceanic upper plate.
  At time zero whether such an edge counts is therefore decided by rounding. In
  a small test world 8,600 of about 40,000 km of attached trench counted at startup and 24,000 km
  after one 2 Myr step, once feeding had lengthened the slabs. No back-arc can
  rupture within the 8 Myr minimum loading time anyway.
- Saved moving-hinge worlds can adopt the driver only through an explicit
  opt-in migration, which has not been written. Worlds that reached the moving
  hinge by migration (the production restart migration, SEP21T's 156 Myr
  path) therefore still have dead back-arc loading, and their frames carry no
  marker of it: `backarc_driver_version` is simply absent.
- Anchoring inherits G24's slab-length model. Retained slab mass is retired on
  a fixed 50 Myr clock, so steady-state slab length is about convergence x
  50 Myr, and a trench converging slower than about 2.6 km/Myr can never hold
  more than 130.5 km of slab. It then never anchors under a continental or old
  oceanic upper plate. When G24 is fixed and slabs lengthen, more trenches
  will anchor and back-arcs will become more frequent under driver 1.
- Continental, cratonic and arc columns all take the 100 km thickness cap.
  Real cratonic lithosphere is about 200-250 km thick (the balance itself
  gives cratons a 3.5x keel drag) and arc lithosphere is hot and thin. So a
  cratonic margin anchors as easily as an ordinary continental one, which is
  too easily, and an arc-crust upper plate anchors too hard. There is no
  existing continental geotherm to derive a better thickness from.
- Loading does not guarantee a rupture. The unchanged sliver path can refuse
  every attempt. In the moving-hinge startup world at mesh level 3, with an
  upper plate prescribed 5 km/Myr inland, three margins loaded past 60 km
  (one to 240 km) and none of their attempts had committed by 80 Myr. The
  same world at mesh level 2 ruptured at 34 Myr. A traced rerun names the
  check: every refusal (14 and 22 Myr) is `choose_arc_sliver`'s connectivity
  test (`child_count != 1` in backarc_geometry.py). That test world has 1,280
  cells of about 400,000 km2 (about 630 km across), wider than the nominal
  550 km strip, so the strip picks up 8 cells in two disconnected pieces
  (2.3 and 0.9 million km2). This looks like a coarse-mesh artefact: SEP21T's
  level-5 mesh has 20,480 cells of about 25,000 km2 (about 160 km across), so
  the same strip is 3-4 cells wide. It still needs checking in a fresh level-5
  moving-hinge world before driver 1 is recommended for production, since a
  refusal that depends on resolution would quietly suppress arc microplates.
- The loading policy numbers (0.4 km/Myr, 800 km, 2,200 km, 80 Myr, 60 km,
  8 Myr) remain the existing hand-set values (N2/G28/G30).
- A solved independent hinge velocity (G22) could later replace this measure
  of hinge lag.

## Welded stacks are one accretion body

In SEP21T (`20260921-011257-b2f85b`) terranes passed back and forth between
plates. Block 07 and Primordial ocean 02 exchanged material 31 times between
96 and 174 Myr, and block 08 and Primordial ocean 04 12 times. Every one of the
60 accretion events from 76 to 208 Myr moved one 24-face arc sheet, about
200 km2 of reference area (620-820 km2 now, roughly 25 x 25 km, or 12 x 12 px
on the 2 km map). Six were the same rock going back 20-44 Myr after it
arrived. Sheet 125, for example, moved from Primordial ocean 02 to block 07
at 124 Myr and back at 144 Myr.

The planner did not see stacked material as attached:

1. `local_accretion._native_material_components` joined faces into a terrane
   only by shared edges and patch or craton IDs. Two overlapping sheets on one
   plate stayed separate bodies, although `collision_contacts` labels exactly
   that pair 'accreted' ("an accreted pair is already one plate").
2. When one sheet T of a stack {T, A} touched another plate B, T alone was the
   smaller body and moved to B.
3. T's overlap with A, still on the old plate, became a new active contact
   between the two plates at once.
4. Root-local welding loads both directions every step. The new direction
   started at 0 and reached the threshold of 18 after 18 Myr of full
   engagement; with the 2 Myr step, T went back 20 Myr later. The loading T had
   built toward its first host was only decayed (35 Myr e-fold), never cleared,
   so after a return it could re-mature toward that host within a step or two.

In 34 of the 60 events the moved sheet was torn out of a stack of two or more
sheets on its plate; in 11 it was part of the larger stack (at 144 Myr sheet
125 sat in a 33-sheet, 1,706-face block 07 stack against a 48-face group).
Suture-weld consolidation moved nothing (`moved=False` throughout).

Welded-stack policy 1 (`accretion_welded_stack_version = 1`):

- `collision_contacts.same_owner_overlap_pairs` returns the welds: the
  pairs `refresh` labels 'accreted' (current positive-area overlap between
  distinct sheets of one plate), restricted to faces with mass and a buoyant
  kind on both sides, and to sheet pairs whose summed overlap exceeds the
  existing `column_density.UNORDERED_OVERLAP_FLOOR_KM2` (1 km2, §17). At or
  below that floor two sheets merely abut. Arc emplacement leaves overlaps of
  that kind as roundoff (0.018 m2 stopped this run at 85 Myr), and whether two
  abutting sheets overlap or leave a gap by that much is an accident of
  clipping, not geology; without the floor a 0.024 km2 sliver between a sheet
  and a 1,100 km neighbour on the same plate stopped a dockable stack from
  ever docking. The floor is summed over the whole sheet pair, so a genuine
  thrust contact welds however finely its faces are cut or however narrow it
  is: a 1 km overlap along a 240 km edge (240 km2) welds. No new number is
  introduced. In SEP21T frames at 120, 200 and 260 Myr, 34-41% of same-plate
  'accreted' contacts overlapped by 1 km2 or less (27 of 80, 75 of 182 and
  160 of 407); those no longer weld. The function checks the
  ledger's signature (shared with `_persistent_contact_geometry` through
  `current_overlap`) and raises on a stale ledger.
- `_native_material_components` also joins those pairs, so a component is a
  whole welded stack. Overlap never joins material of different plates.
- `localized_accretion.process_weights` lets the process-zone search step
  across those pairs inside the component. It still starts only at observed
  contact faces, still admits only faces wholly within the 400 km process
  zone, and keeps the cos^2 weight. `qualify` is unchanged: every root of the
  whole stack must be engaged and mature.
- After a transfer, `apply_accretion` removes the welding rows keyed on the
  source plate for roots that no longer have any face with mass on it; a root
  still partly on the source keeps its row. The welding diagnostics' counts are
  recomputed, and the event records `welded_stack_sheets`,
  `released_welding_roots` and, for a stack, `welded_stack_min_contact_km2`
  (the smallest sheet-pair overlap holding the body together). The release
  applies to accretion transfers only. Rock that changes owner another way
  (suture-weld consolidation, trench-split transfer, remnant absorption) still
  keeps rows keyed on its old plate.

A docked terrane overlaps its new host, so it is part of the host's stack. It
can leave only with the whole stack, or after a later rift or fault gives it a
different owner (`ensure_fields` then makes daughter sheets and the pair stops
being same-owner). Whether a whole stack can move is set by the existing
process zone. A face gains loading at cos^2(pi d / 800 km) per Myr of
engagement, where d is the distance of its farthest vertex from the contact, so
the farthest face of a stack needs 18 Myr / cos^2(pi d / 800 km) to mature: about
21 Myr at d = 100 km, 32 Myr at 185 km, 49 Myr at 235 km, 94 Myr at 285 km and
280 Myr at 335 km. In the test fixture, stacks reaching 185, 235 and 285 km
behind the contact docked after 34, 50 and 94 Myr; one reaching 335 km had not
docked after 140 Myr. A stack reaching roughly 300 km or more behind the front
effectively stays, whatever its composition. A compact microcontinent stack
lying well inside that distance can still dock, as a small welded arc cluster
does. Typical SEP21T stacks, 36-75 km across, are well inside. A re-docking in
the other direction has to build its loading from zero, at most one unit per
Myr, so at least 18 Myr. No timer, cooldown or new constant is introduced, and the size rule,
one plan per step, the 18 Myr age policy, `THRESHOLD`/`MEMORY_MYR`,
consolidation and the raster path are unchanged.

`tests/test_accretion_welded_stack.py` reproduces the return under policy 0
(T docks, then goes back 20 Myr later and flickers on its stale memory; the
test passes on the unmodified sources), and checks under policy 1 that a
stack within 400 km docks once as one body, that a stack reaching beyond
the zone keeps its terrane, that a free terrane docks at the same step with the
same plan as before, that an owner split (of a whole sheet, and through one
sheet into daughter sheets) releases the weld, that a roundoff sliver to a
remote sheet does not weld while a 1 km thrust overlap does, that a checkpoint
round trip keeps the policy and the same course, and the memory release
(including a leftover face with no mass), frame schema and validation.

Fresh reviewed native worlds with root-local welding select policy 1 in
`physics_profile.initialize`; frames carry `accretion_welded_stack_version`
when it is set, and `localized_accretion.validate_frame` accepts it only as the
integer 1 on a root-local welding frame. SEP21T and every saved checkpoint have
no field (policy 0) and keep their recorded behaviour; applying it to a resumed
run would need a recorded migration. Expected effect on a fresh world: fewer,
larger dockings (a typical 2-7 sheet stack is roughly 1,300-5,700 km2, 36-75 km
across, 18-38 px on the map) and no returns of docked rock. The G70 polarity
choice (which side of a front moves) and one plan per step are left open.
Because the size rule still ignores polarity, the body it moves is now a whole
welded stack, up to about 300 km deep, rather than a 25 km sliver: an
upper-plate forearc and its welded backstop can be handed to the downgoing
plate whenever that plate's buoyant body is larger. For arc-continent collision
this is the Taiwan-style outcome and plausible; for a forearc over a plate
carrying a larger terrane it moves more material the contested way than before.
The alternating block 02 / Rift plate 01 exchange (176-206 Myr) is not
ping-pong: nine different arcs met across one convergent front and each moved
once. It is a front-geometry and G70 matter, and about 3 of its 9 events
(190, 194, 196 Myr) would change direction under this policy; the rest are
unchanged.

## Reviewed fresh-world law selection

New worlds made in the app missed most of the recent reviewed laws. Nothing
carried them into a new world:

- The form listed `fixed_trench` and `disabled` first (`web/index.html:25-26`),
  and `fillConfig` fell back to them for any config with a seed
  (`web/app.js:219-220`). The heating length of #166 is active only with
  `retained_phases: "thermal_v1"`.
- `readConfig` never sent `enhanced_rifting` and the page had no control for
  it, so `enhanced_rifting.normalize` saved `enabled: false`. That selects the
  0.15 loading-proxy rupture gate instead of #162's gate on realized material
  links (`progressive_rifting._rupture_calibration`).
- `GET /api/config` returned the engine's `DEFAULT_CONFIG`
  (`physics_profile: "legacy"`), and the page's startup filled the form from it,
  so a fresh install switched the page's own reviewed selection back to Legacy.
- `SimulationManager.start` (used by `POST /api/run` and `run_simulation.py`)
  passed a request straight to `validate_config`, so `{"physics_profile":
  "reviewed_v1"}` alone got immediate foundering and the proxy rupture gate.

The choice is now made once, where a world is created.
`fresh_world.fresh_world_request` fills `retained_phases: "thermal_v1"` and
`enhanced_rifting: {"enabled": true}` into a `reviewed_v1` request that leaves
them unspecified. An explicit `disabled` or `{"enabled": false}` always wins,
and legacy or profile-less requests are returned unchanged.
`SimulationManager.start` applies it before validation, so `config.json`
records every choice explicitly. `GET /api/config` now serves
`server.fresh_world_defaults()`, the reviewed profile with these laws. The page
marks Retained dense crust selected and adds an enhanced-rifting switch with
its four existing parameters; `readConfig` and `fillConfig` carry them. Legacy,
or the switch off, sends only `{"version": 1, "enabled": false}`, so a hidden
parameter input can never block a start. The command line gains
`--physics-profile` and, without `--config`, uses `reviewed_v1` like the app.
A bare `POST /api/run {}` stays legacy.

`fresh_world.py` is imported only by `server.py`. The engine helper sources
that `server.capture_auxiliary_sources` hashes into every checkpoint start from
`tectonics.py`, so this module is outside them, and every engine file is
byte-identical to before. Installing this change therefore needs no helper-hash
restamp: saved checkpoints resume and branch exactly as before. (A first draft
put the function in `physics_profile.py`, which the engine imports; that would
have changed its recorded hash and made every existing checkpoint refuse
resume until restamped.)

A `--config` file is reproduced as saved. Its profile is never defaulted, so a
config written before `physics_profile` existed (for example
`20260915-180344-b93be1`) stays legacy. Every law key the selection would fill
is first made explicit with the engine's missing-key meaning
(`validate_config({})`), and a missing or flagless `enhanced_rifting` becomes
`enabled: false`, so the four key-less reviewed runs listed below re-run with
their recorded laws. `--physics-profile` on a file changes only the profile.

### What enhanced rifting does, in map units

No constitutive law, threshold or coefficient changes. The enhanced-rifting
parameters are `normalize`'s existing values, which SEP21T
(`20260921-011257-b2f85b`) runs.

- **Inherited weakness.** Ordinary continental parcels (not cratons, not ocean)
  start with a smooth seeded weakness, `strength * (1 - weakness)`, of up to
  the amplitude: 0.25, so up to 25% weaker, 12.5% on average (measured over the
  sphere for seed 37). Rift damage ratchets it up to half the peak damage,
  at most `RIFT_SCAR_FRACTION` = 0.5, a durable scar up to 50% weaker that
  outlasts the damage itself (`enhanced_rifting.accumulate_scar`).
- **Pattern size.** `correlation_km` is the length scale of the pattern's
  waves (each wave is `sin(6371 km / L * ...)`, a 2,200 km wavelength at
  L = 350 km), not its decorrelation distance. Measured for seed 37 with 40,000
  point pairs: correlation 0.98 at 100 km, 0.82 at 350 km, 0.43 at 700 km and
  zero at about 1,100 km (about 550 px on a 2 km/px map), turning slightly
  negative at 1,400 km. Weak and strong regions therefore alternate over
  roughly 1,000 to 1,500 km. The default seed is fixed, so every world gets the
  same pattern unless its weakness seed is changed.
- **Craton rim.** Craton interiors stay rigid behind a 150 km deformable rim
  (about 75 px).
- **Rupture.** A rift breaks through when material links reach 0.35
  logarithmic strain (about 42% extension). This is a numerical breakup gate,
  hand-set, as `_rupture_calibration` itself says; N6 and #188 review it. The
  interface text avoids hard-coding it: the recorded-physics note reads it from
  the frame's `rift_mechanics.rupture_strain_threshold`.

Enhanced rifting does not by itself cure starved continental breakup. The
comment above `RIFT_SCAR_FRACTION` records that the seeded field alone
correlated at -0.02 with later rift damage; the scar ratchet, not the seed, is
what draws later rifts back to broken crust. SEP21T runs exactly these laws,
and its frame at 266 Myr has only 2 rift breakthroughs (74 and 86 Myr) and 47
failed rifts, while `max_bond_extension_strain` reaches 2.45. See the
continental breakup gap.

No new saved field is needed: `config.json` holds the explicit choices, and the
existing `retained_dense_crust_version` and `enhanced_rifting_version` record
them in state and frames. There is no frame, checkpoint or schema change.

### Why the engine defaults are unchanged

The engine's missing-key defaults (`NATIVE_DEFAULTS`, `validate_config`,
`enhanced_rifting.normalize`) are deliberately unchanged:

- `SimulationManager.branch` re-validates the saved simulation config.
  Four saved `reviewed_v1` runs from 2026-09-15 (`20260915-190512-aea1cd`,
  `-191810-653c92`, `-195005-47ec8d`, `-205221-8d24f3`) have no
  `subduction_response` or `retained_phases` key. A profile-dependent default
  would relabel a branch of one as thermal (or moving hinge) while its state
  still records version 0.
- `enhanced_rifting.enabled` reads the config every step, so a default of
  `enabled: true` would switch rifting on mid-run for a key-less config.
- Resume restores checkpoints without `validate_config` and is unaffected.

### Moving hinge withheld (N1)

`subduction_response: "moving_hinge_v1"` (#164, which removes the invented
forearc-sweep retreat rule, G21) is **not** a fresh-world default yet. Under it
`trench_retreat_speed` is always zero, which switches off two things:

- back-arc nucleation, which needs a trench retreat speed above 0.4 km/Myr
  (0.4 mm/yr)
  (`backarc.py:256-266`), and
- the arc-supply retreat term, which adds retreat to the convergence that feeds
  arc loading and volcanic supply (`material_forcing.py:32-34`,
  `native_processes.py:357-359`).

Back-arc basins are the main source of Southeast-Asia-style microplates, and
the project rule is never to suppress microplate behaviour. SEP21T, which runs
the moving hinge, shows the cost: its frame at 266 Myr (after the rewind to
170 Myr) has 46 back-arc loading and 4 back-arc rifting events, the last of each
at 132 Myr, and none since. Its later back-arc spreading events (160 and
252 Myr) belong to basin 2, which opened before 132 Myr, while island arcs kept
forming (44 events, the latest at 252 Myr).

The moving hinge stays an explicit choice in the form, the API and the command
line. Making it a default is one entry in `fresh_world.REVIEWED_FRESH_LAWS`,
to add in the same change as the N1 back-arc repair
(`fix/backarc-upper-plate-motion`) or when the user accepts the trade-off.

*Update 2026-09-28:* with the N1 repair installed (#193,
`backarc_driver_version` 1, "Back-arc loading under the moving hinge" above),
`subduction_response: "moving_hinge_v1"` is now a reviewed fresh-world default,
so a new app, API or command-line reviewed world uses the same subduction
response as SEP21T. It affects only newly created worlds: saved configs,
resume, branch and `--config` reproduction keep their recorded law (a key-less
saved config still reads `fixed_trench`). `fixed_trench` stays selectable.

Open PR #190 (`trench_persistence: "attached_slab_v1"`, `slab_allocation:
"fed_v1"`) and PR #191 (the opt-in `supercontinent_ring` initial condition) are
not made defaults here either. If #190's laws are accepted as reviewed
fresh-world laws, each is one entry in the same table.

`tests/test_fresh_world_defaults.py` covers the request filling, explicit
overrides, the moving hinge default (fixed trench still selectable), the selection staying outside the
hashed engine sources, the unchanged engine contract and branch path, the
served defaults, the saved `config.json`, command-line reproduction of
profile-less and key-less saved configs, and a fresh reviewed world that
records the law versions and steps. `tests/test_primordial_config_ui.cjs`
covers the form round trip.
## Accretion no longer vetoes continental breakup

Within a step, `progressive_rifting.update` gathers the rift loading, local
accretion then moves terranes, and `progressive_rifting.commit` runs later in
the topology stage. Commit compared the whole world's rift mesh between the two
calls: if any node or link key differed anywhere, it refused every continental
cut of the step. Any terrane transfer on any plate therefore vetoed breakup
worldwide. In SEP21T accretion happened in all 59 steps from 158 to 274 Myr
(median transfer 231 km2, about 15 by 15 km, or 8 by 8 pixels at 2 km per
pixel), there has been no breakthrough since 86 Myr, and both breakthroughs
(74 and 86 Myr) fell in accretion-free steps.

Commit policy 1 (`rift_commit_version`) keeps the pending solve and carries it
to the current mesh by the persistent keys `rift_bonds` already use: a link by
its sorted base pair and owner UID, a node by its base and owner UID. A current
link with no pending counterpart cannot fail that step. Only cuts whose own
loading was gathered on material that has since changed are deferred; they are
judged again next step on freshly gathered loading, with no timer. Every other
check is unchanged: the coherent cut, protected back-arc hosts, initial ocean
plates, cratons, projection cleanup, daughter viability, current differential
opening and one breakthrough per step.

Keys alone cannot see every transfer. A node is (base, owner UID), and one base
can hold several patches (raster) or faces (native coarse sites). When
accretion moves only part of a node, the donor keeps its key with less
material and only the receiver shows a new key, or no new key at all if it
already held a node on that base. A policy-1 loading update therefore also
records its material image: each continental patch ID (persistent material
identity; a raster patch never spans two owners, a native patch is one face)
with the node key it was loaded in. Commit compares it with the current image,
and every node that gained or lost a patch counts as changed, on both sides of
the transfer. Patches that appear or disappear between the two calls (new arc
crust, consumed material) are not transfers between nodes, and node area is not
compared, because arc crust is added between the two calls too. A pending
record without its image (older work) is refused exactly as under policy 0.

The unit whose change defers a cut follows from what each mechanics mode
measures, not from a new rule:

- Realized (enhanced) links measure their own lengthening, and nodes their own
  non-rigid motion, during material transport. `rift_material` links only
  same-owner nodes, so nothing couples two connected components. A cut is
  deferred only if its connected material component gained, lost or relinked
  any node or link, or one of its nodes gained or lost a patch.
- The legacy membrane assigns every boundary face of a plate to that plate's
  nearest node within reach, placed at its material centroid, so a node or
  node material gained or lost anywhere on the plate can move load between its
  components. There a cut is deferred if its plate gained or lost any node,
  link or patch.

At 170 Myr in SEP21T (realized links), a 196 km2 terrane moved from
Continental block 08 to Primordial ocean 04. It was a one-node island of
block 08 with no links; block 08 had 112 nodes in 8 components. Policy 0
refused a block-08 cut (rift system 26) that passed every other check. Re-run
read-only on the saved 170 Myr state, without taking a step, policy 1 carries
all 4,636 links over, leaves 2,732 of 2,733 nodes in unchanged components, and
accepts that cut, opening at 3.0 km/Myr (3 mm/yr). The daughter holds 1.8%
of the block's material. Its mechanics reference area (mass-weighted node area)
is 0.11 Mkm2, but the microcontinent actually created in memory has 0.081 Mkm2
of continental crust on the map, about 285 by 285 km or 140 by 140 pixels at
2 km per pixel, inside a 0.52 Mkm2 plate that includes its ocean support. The
terrane was 24 native faces, all moved as one node, so the material image
changes nothing here. A plate-wide unit would have deferred it, because the
terrane left block 08 itself. In six more saved states (166, 180, 190, 200,
208 and 278 Myr) every change between loading and commit was one new unlinked
node, with no partial transfer, and nothing was deferred.

Frames of policy-1 worlds carry `rift_commit_version: 1` and rift mechanics
labels `commit_policy_version`, `commit_policy` and `commit_loading_unit`. On a
step whose pending links met the breakup gate they also record
`commit_mesh_changed` (a node key, link key or node's material changed; a pure
reordering of the arrays does not count), `commit_changed_plate_uids`,
`commit_carried_links`, `commit_new_links`, `commit_reassigned_patches`
(patches whose node changed), `commit_deferred_cuts` and
`commit_deferred_plate_uids`.
`progressive_rifting.validate_frame` checks that labels, version and mode
agree.

Versioning: `physics_profile.initialize` sets `rift_commit_version = 1` for
fresh `reviewed_v1` worlds, next to the rupture criterion. Saved worlds read 0
and keep the whole-world refusal exactly, with no new labels or frame keys.
Tests: `tests/test_rift_commit_policy.py` and
`tests/test_reviewed_physics_profile.py`.

## Remaining review work

### Local slab-neck history extension

`slab_tether_history_version = 2` carries spherical material anchors, explicit
support radii and complete area/excess-mass ledgers for every neck patch. The
existing trench advection rotates anchors with the overriding plate. Actual
separation and owner transfer split each patch by its locally allocated trace;
they do not use one global length fraction for differently loaded patches.
Joins preserve separate histories and positions. Version-1 unlocalized histories
remain available for earlier component experiments.

`slab_tether_local.localize` requires explicit anchors, width fractions and radius.
It creates a current-state baseline, not a reconstructed past: existing channel
damage is copied into the supplied spatial partition; previously unlocalized
retirement is apportioned by reference width above each detached subtotal, and
initial/fed mass retains its containing row's proportions. Geometry and complete
local ledgers are validated before this conversion commits.

Force allocation uses nearest spherical anchors at the current boundary
quadrature points. Coincident channels share local width without averaging
rheology; failed patches remain in that partition, so neighboring force does not
fill their gaps. Unmatched geometry supplies no force and retains its recorded
inventory. This is a finite-volume spatial approximation: within-patch damage
is uniform and boundary quadrature/refinement still requires convergence study.
Radius and anchor resolution are explicit inputs, not calibrated geological scales.

Accepted finite native capture now records local neck provenance after its
existing uniqueness, maturity and availability caps. Local feeds use the same
parent-front anchor allocation as forces and preserve the parent incoming ocean's
actual cooling-age mass rather than inverting a trench-wide buoyancy average.
Each patch applies exact constant-input retention; its feeding, retained and
retired ledgers close independently. The last source epoch's patch indices are
recorded in saved capture diagnostics and validated there. Missing local feed or
capture into a failed patch is rejected, never silently redistributed.

The staged damage helper integrates positive local opening before averaging
within each finite-volume patch. Under frozen plate inlet velocities, geometry,
weight and mantle drag, it advances to the earliest rupture in the supplied row,
retires that patch's material conservatively, and returns unadvanced time. It
rejects a stale force assembly. This helper does **not** implement a coupled
timestep: a driver still must find events across rows, re-solve motion as damage
changes, and commit ocean/material/RNG evolution once per accepted interval.

The local source and lifecycle paths are integrated, but no production force
flag, new-world initialization or live migration selects the replacement policy.
Continental-entry buoyancy and inherited polarity remain required before that
selection is appropriate.

### Coupled rupture and native source intervals

`slab_tether_evolution.integrate_frozen_geometry` solves the actual nonlinear
plate balance at each damage derivative evaluation. An adaptive embedded
third/second-order Runge-Kutta pair controls damage error. Bracketed first
crossing across all attached patches localizes rupture; conservative retirement
then removes its transmitted pull/resistance, and the remaining requested time
is integrated with the new force balance. Trial stages use a one-sided
pre-rupture continuation rather than retiring material during event search.
This component holds geometry, feeding, thermal state and geological time fixed.

`slab_tether_native.advance` explicitly couples accepted mechanical intervals to
the real native transport, slab feeding, material and topology paths. It applies
the interval's integrated Euler-vector motion, books incoming slab inventory,
then commits damage and retires the actual inventory at rupture before trench
identities can change. Detached patch shares cannot capture further ocean into
the failed attachment. Newly empty trenches receive an explicitly supplied
undamaged law; existing loaded trenches require an explicit localization baseline.
The complete requested interval runs on a private world. Any failure leaves the
caller's geometry, ledgers, clock and RNG untouched. Checkpoints retain the local
histories and diagnostics. Both coupled APIs require explicitly selected passive
`plate_resistance_version = 1`; they reject the historical nonpassive law.

Finite ocean capture applies the intact patch fraction before geometric union
and maturity selection, without changing the saved maturity. Coincident intact
and failed histories therefore admit only the surviving width share. That
already reduced feed is attributed conditionally to intact histories, avoiding
both reattachment and applying the width fraction twice. This local-history
capture protection also applies outside the experimental stepping wrapper.

Source and geometry updates are split between accepted mechanical intervals.
The explicit maximum source step bounds that additional discretization; damage
error control alone does not bound transport error. Averaging Euler-vector
components is a finite-step rotation approximation when the rotation axis changes.
Native source-step and spatial convergence are not established by the tests.
The existing continental polarity/shutdown policy still applies. Ordinary
`Simulation.step` and live-world startup do not select this experimental API.

The force audit also exposed nonphysical motion driven by the old one-sided
square-root smoothing itself. [The resistance repair](RESISTANCE_REPAIR.md)
ports PR #18's versioned passive C1 law, exactly inactive at rest/opening,
retains tiny potential changes without cancellation and verifies
each plate's force balance independently. Residual acceptance uses backward error
against assembled contributions, so a strong plate cannot hide a weak plate's
imbalance. The same tolerance and recomputed final force gate remain; the raw
external-torque-relative norm is retained only as a diagnostic. Finite rounding
still requires convergence testing; it is not the exact plastic limit.

| Issue | Required next change and acceptance checks |
| --- | --- |
| Missing retained dense phase and thermal/compositional state | Implemented as an explicit reduced timestep policy, with the constitutive and resolution limits above. Full native continuation/frame acceptance results are recorded in VALIDATION.md. |
| Continental arrival causes unconditional polarity reversal | Attachment, local histories, force assembly, explicit native rupture and persistent whole-face entry plate/sheet mechanics are implemented. Still resolve automatic/partial-face admission, oceanic/stack subregions, overriding-owner transfers and relative trench migration, local burial, attached-polarity retention and a new reversed system's initiation, then validate the complete collision experiment. |

The specific nonconvex megathrust term and small-step acceptance defects were repaired before this PR; existing regression tests remain in CI. None of these numerical fixes establishes physically complete continent-continent collision.
