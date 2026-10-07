# Native attachment transaction experiment

This is an **external, opt-in research experiment** around the existing native
operators. No production module imports it, and it changes no saved policy,
engine callsite, migration, material admission, traction or rupture law.
It does not repair a historical trajectory or make finite attachments ready
for deployment. Its limited result is atomic enforcement of an **explicitly
authored intact paired material witness** during real native owner mutations.

The prerequisite is the [finite material preflight contract](ATTACHMENT_TRANSACTION_CONTRACT.md).
All exact represented-chart coverage and paired-correspondence limitations in
that contract apply here. Current world-space overlap never maintains, removes
or replaces an existing bond. A material root does not attach its remote siblings.

## Explicit research input

The simulation must already have complete current material lineage and a
`_research_finite_attachment_store` with the prerequisite schema, even if its
bond list is empty. Absence fails; no historical bond or ancestry migration is
inferred. The field is checkpoint-safe ordinary data, but its name is deliberately
outside any production policy. The wrapper may only be used with this experiment.

`accrete(simulation, dt, authored_witness)` executes contact refresh and the actual
`local_accretion.plan_accretions` / `apply_accretion` at the caller's current
epoch. It does not advance global simulation time. Positive `dt` advances the
native contact/loading histories. The planner must return at most one plan.
If it has no plan, its normal history updates are accepted and no bond is born.

A new witness contains exactly:

- `bond_id`, `source_face_id`, `receiver_face_id`: nonnegative integer IDs;
- `world_corners`: three corresponding unit directions strictly inside both
  actual material triangles, with positive oriented area;
- `asserted_intact: True`: the caller's explicit research assertion.

The donor must be an engaged face in the transferring plan. The receiver must
belong to its actual target. Their pair must have positive overlap in the freshly
computed native contact and belong to the plan's persistent contact sheets.
Each world corner is solved in each current spherical triangle's homogeneous
basis, then mapped through that leaf's existing birth-root chart. The same three
world points define paired source/receiver coordinates. Object, complex and
other unsupported numeric coordinate dtypes fail before conversion. The small
unit-vector roundoff allowance is an arithmetic validation only; strict interior
and positive area have no admission tolerance or area floor.
Original-edge `convex_partition.Edge` filtered exact signs determine the world
triangle's positive orientation and strict inclusion against both original
material triangles before the floating homogeneous solve. The solve must then
produce finite positive coordinates and reconstruct the original points within
an operation-scaled floating arithmetic bound; unresolved results fail closed.

These checks establish **finite material identity and geometric eligibility**.
They do not establish that a physical attachment should form or how strong it is.
The author of this input, not overlap or this wrapper, asserts its intact status.

## Transaction order

Every operation runs on a fully detached `deepcopy` of the supplied simulation.
The original is published only after the complete native operation and all
postconditions succeed. This includes planning clocks, maturity, caches, RNG,
events, counters, registries, IDs, arrays and nested phase/heat columns. Errors
and native refusal discard the candidate. External references to original arrays
also remain unchanged on rejection; there is no mutate-then-restore window.

For accretion:

1. Validate the existing intact ledger against current material leaves.
2. Refresh native contacts and run the actual native planner on the candidate.
3. Predict the plan's new owner UIDs and preflight every existing finite bond.
4. Validate the explicit receiving witness and prospective new bond on the
   proposed same-owner leaves, then revalidate existing staged signatures.
5. Only then call native application, whose first physical write changes the
   receiver's omega, followed by owners, histories and gross-accreted counter.
6. Run actual material-surface owner reassignment. Verify that the native owner
   result exactly matches the proposed leaves and that all instantaneous material
   geometry, reference mass, ancestry, trace positions/IDs and column arrays are
   unchanged. Thus retained phase, composition and heat fields are checked as
   arrays, not merely a potentially cancelling global total.
7. Bind the new record to the actual identified native accretion event. Rehosting
   may emit earlier events, so its ID is never guessed. A provisional provenance
   value used during preflight is never published. Revalidate and publish the
   unchanged old ledger plus the explicit new bond with accepted native state.

No callbacks are dispatched during publication. The candidate's entire state is
copied into a ready dictionary first; instance-bound fixture methods are rebound
to the original simulation. Then the original dictionary is replaced. This is
single-threaded in-memory transactional publication, not a concurrent lock or
crash-atomic disk write. Accepted publication invalidates prior references to
state arrays; callers must reacquire them. Normal class methods and the included
fixture's instance-bound methods are tested. Arbitrary callbacks with external
side effects, custom deepcopy behavior or mutable external closures are outside
this experiment's supported state contract; the production operators used here
have no such external writes. Whole-world memory/runtime has not been measured.

## Native topology gate before allocation

`split_continent(simulation, parent_slot, chosen, loading=None)` uses the ordinary
native material-side proposal. It predicts the fresh `next_plate_uid` without
allocating it and preflights the paired finite material owners **before** calling
`native_topology.split_continent`, and therefore before `_allocate`, capacity
growth, identity counters, owner/omega writes and events. A cut separating any
intact paired portion raises; the wrapper provides no release decision.

Corresponding attached portions can move together to a new owner. On native
acceptance, exact owner prediction and unchanged physical columns/geometry are
checked before publication. The ledger survives intact. Thus this is not an
unconditional permanent transfer/split ban. Native viability/craton checks remain
in force and can still refuse an otherwise bond-consistent proposal.

## Evidence and remaining scope

Run `python -B -m unittest tests.test_native_attachment_transaction -v` with
BLAS/OpenMP threads set to one. The small three-body fixture uses real native
contact, planner, application, event and surface operators, plus actual initialized
and evolved composition, dense phase and heat columns. Fixed geometry and authored
mature directed exposure are diagnostic inputs, not a geological trajectory.

The tests compare first docking to the unwrapped native control; reject the real
104/106/108 Myr return plans before application while preserving planning state;
check invalid/missing/degenerate witnesses; and inject a failure after actual
native application and RNG/ID/registry/heat mutations, checking the original
complete state and external aliases. A second negative control corrupts heat
after native application and requires the physical postcondition to reject it.

The split tests use the actual native `Simulation`: a proposal separating two
authored finite endpoints is rejected with `_allocate` never called; a proposal
carrying both together commits real geometry-owner separation, ID allocation and
native events while preserving columns and the finite ledger. The prior preflight
tests separately cover descendants, partial footprints and finite remapping.
The complete native simulation, including the authored ledger, also passes the
real typed disk checkpoint writer/reader and repeats the same split with identical
state. This is a fixture transaction restart using its recorded source identity;
it is not a deployed source manifest or a subsequent coupled-world trajectory.
Additional controls verify exact-boundary rejection before a misleading solve,
failed solve reconstruction, interval type rejection and the documented no-plan
history advance. The latter accepts planning history while reporting no docking;
it must not be confused with rollback of an unsupported transfer.
This experiment does **not** yet wrap actual adaptive refinement/coarsening or
other owner-changing paths such as backarc and buried-sheet consolidation.

The physical limitations remain: an accepted finite-contact admission law,
reciprocal attachment traction/damage, finite failure/rupture criteria and native
coupled force balance are still needed. Current native indexed-edge rift damage
cannot by itself authorize release between separate overlapping sheets. Scotese
Rule XI motivates inherited sutures and later rifting; it supplies no missing
strength or failure coefficients. Explicit unsupported-cut rejection is a research
consistency boundary, not a claimed physical solution for the observed 26 transfers.
