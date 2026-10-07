# Finite collision laws on local joint columns

This research component implements the same fixed-geometry weld and finite
Couette interface laws as `joint_resistance.py`, on the joint SI velocity space
of `sparse_shared_contact.py`. It does not change a constitutive coefficient,
contact admission, attachment, damage, rupture or live saved policy.

## Representation and physical work

Each paired material face uses the actual plate Euler columns and the two
tangent residual columns at each vertex. A pair therefore touches at most
18 unique coordinates. Same-owner faces share Euler columns but retain distinct
material residuals. Their shear must not disappear merely because ownership
matches. The common-point radial trace reproduces a rigid rotation exactly in
the represented functional, up to floating-point arithmetic.

`Geometry` takes one detached geometry snapshot and builds full-mesh free-edge
incidence once. A pair extracted from that mesh cannot invent a free edge by
omitting its neighboring face. Whole-arc containment, outward normal and finite
polygon checks reuse the reviewed dense law. Callers must still supply physically
admitted edges and adjacent buried interfaces with the intended measure weights.
This module does not discover that topology or eliminate duplicate physical
contacts supplied under different records.

The opening-only weld is evaluated in its two trigonometric modes. The gradient
is pulled back to local velocity columns; the existing exact active-interval
positive factors supply the Hessian action and diagonal. Closing at one end
cannot cancel peeling at the other before resistance is evaluated. Weld power
is actual velocity times its generalized resisting force, not its potential.

For buried interfaces, the existing positive quadrature converges its full local
Gram matrix and checks independent area and rigid rotation moments. Thin QR
compresses its factor to at most18 rows. Compression checks the reconstructed
Gram in diagonal energy scaling; it neither truncates eigenmodes nor replaces
negative eigenvalues. Successive quadrature agreement is an error estimate,
not a rigorous integration certificate.

The global Hessian is a sum of gathered local `F.T F` actions and scattered
reactions. Only result vectors have global length. The gradient arithmetic scale
sums absolute products actually used in the local reaction, including both
factor actions `abs(F).T abs(F) abs(y)` for interface shear. This preserves the
cost of modal cancellation; using only `abs(F y)` would erase it. It is not a
static yield-strength bound that could conceal a small force imbalance. Potential,
gradient, actual power and Hessian all refer to one fixed functional.

## API and bounds

1. Create `Geometry(system)` once.
2. Build `weld_arc(...)` and `interface_patch(...)` records using the same
   physical parameters as the dense law.
3. Construct `Resistance(system, welds=..., interfaces=...,
   smoothing_speed_m_s=...)`.
4. Pass it to `sparse_joint_solver.solve(..., resistance=...)`.

Records and resistance inputs are detached copies with read-only array flags.
This is an ordinary in-process numerical API, not a hostile-object sandbox or
concurrent transaction mechanism. The solver validates the geometry signature;
callbacks only read trial velocities. Callers must rebuild after geometry,
ownership or tangent-coordinate changes and must not mutate a system during a
solve. Loads may change on unchanged geometry.

Per-contact retained storage is bounded by the two-face columns, irrespective
of the world's vertex count. Geometry preprocessing and output-vector storage
still scale with world size. Quadrature assembly temporarily retains a local
rule; it does not allocate material-sized dense matrices. Python-loop overhead
per contact remains, and no native timestep speedup is claimed.

## Evidence and remaining integration

`tests.test_local_joint_resistance` checks dense potential/gradient/Hessian-action
and diagonal parity; local opening/closing and subdivision; same-owner shear;
common-spin work; a real nonlinear solve against the dense joint solver; local
storage above900unknowns; full-mesh internal-edge rejection; and geometry/input
binding. Solver tests additionally exercise its force, constraint and power
acceptance. These frozen-functional checks do not establish trajectory accuracy,
mesh convergence, geological calibration or an actual mountain-building outcome.

This supports Scotese's force-balance and explicit-resistance principles (RuleI)
and consequential continental interaction (RuleXI). Keeping the existing
regularized yield and viscous sheet/interface laws is a declared reduced model,
not a claim that those rheologies follow from the reference. Full native domain
allocation, admission, conservative state evolution and validated time integration
remain required before production selection. No engine call site is added here.
