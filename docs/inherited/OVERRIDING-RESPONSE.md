# Overriding response and connected mixed plates

This is an **opt-in, reduced moving-hinge energy closure**, not a solved mantle
suction model or a measured force split. The unchanged default is
`subduction_response: "fixed_trench"`, for saved configs and for new worlds
alike: new `reviewed_v1` worlds select the other reviewed laws but not the
moving hinge, because it currently stops back-arc basins from nucleating (N1;
"Reviewed fresh-world law selection" in PHYSICS_REPAIRS.md). A separate, explicit initial-condition option can join
an ocean region to a neighboring continental plate. Neither option changes an
existing checkpoint or a running world.

## Why these two changes belong together

In the independent-ocean startup, the inherited coastal slabs pull the ocean
plates. The continental plates override them and the historical force law
supplies those plates only with resistance at the trench. Increasing slab weight
mostly speeds the oceans. There are two distinct paths to continental motion:

1. Connect a continent to incoming oceanic lithosphere on the **same rigid plate**.
   Existing slab and ridge forces then enter that plate's ordinary torque budget.
2. Allow the overriding plate and its trench to respond to sinking of the slab.
   This PR supplies an explicit constrained approximation for that response.

The options are independent. Attaching an ocean does not implicitly select the
new force law, and selecting the force law does not reassign any material.

## One slab energy source

Take a trench-normal coordinate toward the overriding plate. Let `u_d` be the
incoming surface speed, `u_o` the overriding speed, `theta` the retained fixed
slab dip, and `W` the attached slab excess weight per metre of trench. Assume an
inextensible slab profile translated with a trench carried by the upper plate:

```text
u_trench = u_o
q = u_d - u_o
v_slab,h = u_o + q cos(theta)
v_slab,z = q sin(theta)
P_gravity = integral W sin(theta) (u_d - u_o) dl
```

Differentiating this one power expression gives generalized line drives
`+W sin(theta)` on the incoming plate and `-W sin(theta)` on the upper plate.
Their equal magnitudes follow from the **imposed geometric constraint**; they
are not evidence that Earth's slabs distribute their force 50/50. Plate speeds
depend on all torques and resistances, not a prescribed movement percentage.
The fixed-upper limit recovers the old slab work. Common rigid rotation supplies
zero slab driving work in version 1.

The model assembles consistent positive quadratic resistance:

```text
D_slab = 1/2 integral C_s (v_slab,h^2 + v_slab,z^2) dl
D_anchor = 1/2 integral C_a u_o^2 dl
```

`C_s` and `C_a` retain the existing coefficients and retained-length scaling.
The anchoring term is an additional unresolved mantle resistance, separate from
the slab's Stokes drag. These are phenomenological closures, not calibrated
predictions. Common translation still has mantle drag. Bending, megathrust,
basal, collision, and ridge terms retain their existing laws. Version 1 requires
passive plate resistance and rejects independent heuristic trench-retreat
speeds, which would violate the declared trench kinematics. Native evolution
carries trench traces with the overriding plate and skips the legacy forearc
remapping stage for saved response version 1. Missing/version-0 state retains
the historical remapping law.

Earlier source also ran inherited heuristic forearc remapping after the
moving-hinge solve. Final boundary reconstruction zeroed the retreat-speed
array, so checking that array at a completed step missed intermediate changes
to ocean support and the swept-area ledgers. The corrected path preserves both
through this stage; regression tests also require the same geometry to produce
nonzero legacy remapping. This corrects version 1's declared kinematics rather
than adding another constitutive policy. Historical behavior remains reproducible
with the checkpoint's saved source; source-hash compatibility requires an
explicit source migration to continue an old checkpoint with this correction.
It does not rewrite previous support, trench inventory, or swept-area history.

`plate_balance_diagnostics.slab_power_w` reports downgoing, overriding and total
work. `slab_power_by_plate_w` reports the two components on each plate. The
components explain the single `slab` driver; **do not add them to its total**.
`slab_stokes_dissipation_w` and `slab_anchor_dissipation_w` likewise are portions
of the reported viscous dissipation, not extra energy sinks.

With symmetric opposing trenches, the central ocean can have zero net Euler
motion while the two upper plates move toward it. Both fronts can then consume
ocean. This is a permitted outcome, not a guaranteed intake or mandatory split.
Native consumption still requires available incoming water, actual relative
convergence and finite swept overlap; buoyant material is excluded. The PR adds
no minimum velocity, ocean supply controller, slab-neck rupture, or internal
tensile deformation law.

This is an **instantaneous work balance**, not a closed evolving slab potential
energy ledger. On opening (`q < 0`), the signed gravity term absorbs work as if
the constrained slab were lifted. Native capture admits only positive finite
ocean consumption; the inventory does not implement reverse slab withdrawal.
It therefore cannot validate the energy history of sustained trench opening.

## Explicit continental attachments

`primordial_ocean.continental_attachments` is a list of objects containing a
zero-based `region_index` and an existing `continental_plate_uid`. Region indices
refer to the deterministic water partition for the exact starting world and
partition settings. UIDs identify continental plates, not material classes or
reusable storage slots. Inspect the initial partition and plate UID inventory
before selecting an attachment for a different starting map.

Each selected region must share a finite source-mesh coast with its continental
owner and form a connected combined footprint. Targets and regions must be
unique. At least one region remains an independent ocean plate; this focused
initializer does not retire the original ocean identity. New identities are
allocated only for independent ocean regions. Mixed carriers are recorded
separately from the pure-ocean UID list used by historical identity guards.

The land/craton geometry and inventory, source-face materials, water region
geometry, ridge locations and cooling ages are preserved. The shared coast
becomes an internal margin; distant coasts against another plate remain eligible
for inherited subduction with water locally incoming. Owner interfaces carry
the material kind on **each** side. Plate membership alone no longer determines
whether a source is water or which side may subduct. The existing source-to-control
length check rejects contacts that mix ridge and coast or incompatible polarities;
refine the control mesh to resolve them.

Example for the tested seed-37 generated world (not a universal map recipe):

```json
{
  "width": 48, "height": 24,
  "mesh_level": 4, "coast_geometry_level": 2,
  "plate_count": 4, "mechanics_nodes": 128, "seed": 37,
  "physics_profile": "reviewed_v1",
  "subduction_response": "moving_hinge_v1",
  "primordial_subduction": {"enabled": true},
  "primordial_ocean": {
    "enabled": true,
    "continental_attachments": [{"region_index": 1, "continental_plate_uid": 3}]
  }
}
```

Use this in a fresh configuration JSON accepted by `run_simulation.py --config`.
The app also exposes both under Physics setup: Subduction response (Moving
trench, opt-in) and the ocean-attachment controls.
The saved `subduction_response_version` selects the constitutive law on resume;
missing state means version 0 even if someone edits a config field later. There
is no implicit checkpoint migration.

## Scientific status and relation to Scotese

C. R. Scotese is the project's patron saint; [the supplied rules](RULES-OF-THUMB.md)
guide interpretation. Rules I-II motivate slab/ridge gravitational driving and
explicit resistance; Rule V motivates a continent attached to ocean floor that
is subducting at its far edge. Neither the reciprocal moving-hinge reaction nor
its magnitude is asserted to be Scotese's own force law. It is a computational
constraint that makes the existing native trench-relative intake kinematics and
the chosen slab power consistent.

Real slabs and mantle flow can couple the plates in more than one way.
[Conrad and Lithgow-Bertelloni (2004)](https://www.clintconrad.no/papers/Conrad_JGR2004.pdf)
distinguish slab suction from flow induced by direct pull; the two can move upper
plates in opposite directions. This closure computes neither flow field.
[Holt, Becker and Buffett (2015)](https://adamfholt.github.io/documents/papers/holt_et_al_gji2015.pdf)
show that overriding-plate properties affect rollback and stress. A rigid upper
plate, fixed dip, and no independent trench degree of freedom omit that behavior.
[Kiraly et al. (2018)](https://adamfholt.github.io/documents/papers/kiraly_et_al_g3-2018.pdf)
provide a qualitative double-subduction comparison, but omit explicit upper
plates and do not validate a quantitative reaction coefficient here.

Other retained limits: no slab-shape or mantle-pressure solution; no force
transfer from local trench motion to deformable far-field upper crust; no new
plate-breakup mechanism; uniform trace load allocation; prescribed slab
retirement and trench shutdown timers. In particular, Rule IV's persistent
subduction is not guaranteed: the existing quiet/extension shutdown can turn
off attached pull while retained inventory remains. That proxy is not physical
detachment. PRs #13-23 address separate attachment, rupture and continental-entry
experiments; this PR targets the installed #20 stack and does not duplicate or
enable them. Combining the energy closures needs a new audit to avoid counting
the same work twice.

## Acceptance evidence

The seed-37 independent-ocean baseline and mixed setup both exposed the existing
differentiated-overlap precision failure at a 0.02 Myr step. The numerical
prerequisite is reused from [PR #17](https://github.com/CaliTarheel/Deep-Time/pull/17):
evaluate the spherical determinant as `a dot ((b-a) cross (c-a))` in the face-area
and differentiated-overlap kernels. The equivalent coverage numerators were
already present on the #20 base. The upstream high-precision tests are retained;
no conservation tolerance is loosened and no rupture-stack behavior is imported.

The captured preflight also exposed a distinct clipping defect: the `2e-13`
angular duplicate filter deleted different vertices of a thin overlap polygon.
The reverse clipping order then lost `1.57e-7 km2` relative to independent
60-digit geometry. Scalar, batched, and differentiated clippers now remove only
exact coordinate duplicates, and the differentiated clipping normals use the
same stable difference-vector formula as coverage. Inside, area, conservation,
and force acceptance tolerances are unchanged. The regression preserves the
actual failing pair and checks both clipping orders against the high-precision
oracle, rather than accepting a larger mismatch.
Exact shared endpoints retain their plane incidence and original coordinates
at endpoint crossings, so retaining real short edges does not create false
overlap area between adjacent triangles. Coincident-edge coverage remains tested.

Frozen initial-state comparison on the documented seed-37 world, with all force
coefficients unchanged (continental-marker speed weighted by reference area):

| Ocean ownership | Slab work closure | Continental mean (cm/year) | UID 3 continent (cm/year) |
| --- | --- | ---: | ---: |
| Independent | Fixed trench | 0.04749 | 0.04440 |
| Independent | Moving hinge | 0.39442 | 0.47650 |
| Selected region attached | Fixed trench | 0.07323 | 0.07688 |
| Selected region attached | Moving hinge | 0.42230 | 0.42623 |

All four solves have scaled force residual below `8.9e-11` and no negative
resistance elements. Attachment gives UID 3 direct incoming-slab torque of
`9.80e25 N m` and ridge torque of `1.98e26 N m`, both absent on that continental
plate in the independent-ocean case. The competing vectors matter: UID 3's
incoming-slab work is slightly negative in the attached/fixed case, so direct
slab torque does not guarantee motion along its pull. These comparisons isolate
the two paths to motion; they are not terrestrial calibration or trajectory tests.

The focused tests exercise actual force assembly and solves, virtual work,
passive slab drag, common rotation, fixed-upper equivalence, symmetric opposing
trenches, rotated/swapped/subdivided geometry, strict version selection,
unchanged source material/ages, explicit mixed ownership, local polarity,
unresolved-contact rejection, real native consumption, and checkpoint agreement.
They test the stated reduced equations and conservation contracts. They do not
calibrate terrestrial plate speeds or validate a long production history.
