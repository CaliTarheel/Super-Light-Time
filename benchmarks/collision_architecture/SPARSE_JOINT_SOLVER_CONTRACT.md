# Matrix-free joint solver

`sparse_joint_solver.solve` solves the frozen SI functional described in
[SHARED_CONTACT_CONTRACT.md](SHARED_CONTACT_CONTRACT.md), using the local maps
from [SPARSE_SHARED_CONTACT_CONTRACT.md](SPARSE_SHARED_CONTACT_CONTRACT.md).
It is a research solver with no native call site, timestep, material update,
new coefficient policy, contact admission or production deployment.

## Equations and algorithm

The unknown is `y=[R omega,z]` in m/s. The problem is

```
min Phi(y) = 1/2 y.K.y - f.y + constant + Psi(y)
subject to C y = 0,
K y - f + grad Psi(y) + C.T lambda = 0.
```

`K` retains each explicitly allocated basal, sheet-viscosity and remaining-domain
resistance exactly once. A callback may supply a fixed convex passive potential
`Psi` for finite contact. The implemented local weld/interface callback preserves
the reviewed physical laws; the solver does not infer weld membership or a new
contact law. Equalities are the supplied bilateral normal rows plus the declared
residual rigid-moment gauge. Gauge reactions are reported separately from normal
contact reactions. Explicit anchored frames, ocean-only Euler slots and exact
fixed-coordinate elimination are specified in
[NATIVE_JOINT_DOF_CONTRACT.md](NATIVE_JOINT_DOF_CONTRACT.md).

Each Newton system uses a positive diagonal `D=diag(K+Hess Psi)` and coordinates
`y=D^(-1/2) x`. The original constraint rows are normalized in this metric.
The solver constructs only their small Gram matrix

```
A = diag(row_norm)^(-1) C D^(-1/2)
G = A A.T
P x = x - A.T G^+ A x.
```

Projected conjugate gradients act on `P D^(-1/2) H D^(-1/2) P`. There is no
material-sized dense Hessian, SVD, QR or explicit nullspace. True operator
residuals replace the CG recurrence every 40 iterations and before inner
acceptance. The projected residual is only an inner solver criterion; final
acceptance requires all the independent gates below.

Newton steps use the total physical potential in a backtracking line search.
The quadratic change is evaluated in difference form. The reported arithmetic
allowance addresses cancellation in this calculation and is not a new physical
tolerance. A small step does not count as convergence. Nonpositive visited
curvature, exhausted iteration/backtracking budgets or nonfinite arithmetic
raise an error.

## Rank, constraints and conditioning

The default limit is **512 nonzero restricted constraint rows**, including gauge
rows. Structurally zero rows after fixed-coordinate elimination are recorded and
checked in original outputs but do not enter the Gram.
Exceeding it fails before allocating the Gram. Building it requires one pair of
local constraint actions per row, storage `O(N+m^2)`, and dense eigendecomposition
in `m` only. The limit is an explicit research-resource bound, not evidence of
scalability to an arbitrarily dense collision network.

The Gram rank threshold is `64 eps m max(lambda_max,1)`. Forming a Gram squares
row conditioning. Exact redundant rows are supported; nearly dependent rows may
be unresolved. Every original row is checked after solving, and a regression
with a near-parallel independent normal rejects rather than accepting its lost
slip. Passing this load-specific residual test does **not** prove every discarded
direction is exact algebraic redundancy or remains harmless under other loads.
The eigenvalues, threshold, rank and original row residuals are reported.

Multipliers are recovered through the same normalized row transpose, then
projected to minimum Euclidean norm in the original multiplier coordinates.
This latter projection is a row-space QR of at most `m` dimensions. Its nullspace
includes gauge and contact rows; it is not a uniquely calibrated traction law.
For duplicate identical contact samples, the `sqrt(length/1 m)` convention makes
their resultants split in proportion to length while aggregate reaction stays
unchanged. The final original stationarity/row gates also check this recovery.

Diagonal preconditioning is a starting point, not a mesh-independent iteration
guarantee. Strong viscosity contrasts, narrow material patches, many nearly
dependent contacts and remaining unconstrained null modes can make it fail its
resource or curvature gates. No extra pin, drag floor or physical diagonal is
added to force solvability. The base diagonal may contain zeros, for example an
upper plate resisted only by an explicitly supplied buried interface. Only the
combined base-plus-law diagonal on free coordinates must be positive at each evaluated Newton solve;
a zero or unresolved combined diagonal is rejected.

## Acceptance and power

A successful return requires finite quantities and, independently:

- Global generalized stationarity relative to the actual force terms.
- Componentwise stationarity relative to the local absolute coefficient-product
  sums of those terms. A fast unrelated plate cannot increase a distant local
  contact's acceptance scale.
- Every original gauge and normal row, using the absolute products of that row
  and its participating velocity coordinates. No row is accepted solely because
  its Gram mode was discarded.
- Physical work: applied nodal/plate power plus prescribed mantle actuator power
  equals actual basal, viscous, remaining-domain and nonlinear resisting power,
  subject to the explicit solver-error envelope and arithmetic estimate.
- Separate contact and gauge virtual work are zero within that same arithmetic
  and residual accounting.

The component force scale is a factorized backward-error measure, and can be
larger than `abs(K) abs(y)` because it retains cancellation within local factors.
The separate global force gate remains active. The power arithmetic estimate is
the documented sixteen-pass `gamma_n` absolute-product estimate inherited from
the dense research primitive; it is **not** a proven forward-error certificate
through geometry construction, Gram eigendecomposition or ill-conditioned solves.
Physical work tolerance remains unchanged. A deliberately corrupted basal-power
ledger fails despite converged algebraic stationarity.

Nonlinear callback inputs are detached, read-only trial arrays. Outputs must
provide `potential_w`, `gradient_n`, `resisting_power_w`,
`hessian_n_s_m` (operator supporting `@`), and `diagonal_n_s_m`.
`gradient_absolute_scale_n` can provide the actual local absolute-product bound;
otherwise `abs(gradient)` is used and can fail more conservatively at cancellation.
The optional `validate_system(system)` binds prepared geometry. Finite, aligned,
nonnegative potential/power/diagonal and `y dot gradient == resisting_power` are
checked at evaluated trials. Callback purity, derivative consistency, symmetry
and global convexity remain API assumptions checked independently for each law;
visited positive CG curvatures alone do not prove them.

## Basal replacement adapter

An adapter may replace the original nodal basal approximation with a reviewed
finite-domain basal operator, while retaining sheet/other/external terms once.
It must publish `hessian_diagonal_n_s_m`, `hessian_absolute_action(abs_y)`, and
`physical_work(y)`. The last returns basal/viscous/other dissipation,
`external_power_w`, `basal_reference_input_power_w`, and `objective_w`.
`physical_work_absolute_scale_w(y)` supplies the corresponding absolute-product
work estimate. The solver uses these hooks; it does not reconstruct a removed
nodal basal ledger. The adapter remains responsible for geometry binding,
positive allocation and replacing the old mantle load and constant exactly once.

## Evidence and limits

Run focused checks with one BLAS/OpenMP thread:

```
python -B -m unittest tests.test_sparse_joint_solver tests.test_local_joint_resistance -v
python -B -m tests.test_sparse_joint_solver --benchmark-output solve-benchmark.json
```

The ordinary suite covers dense velocity/reaction parity, absolute rigid-sphere
SI response, nonrigid manufactured solutions above 256 unknowns, moving-mantle
zero work, duplicate rows, an unresolved near-dependent row, finite-law nonlinear
solves and resource/power negative controls. Repeated or coincident sheets in
some tests are explicit algebra fixtures, not valid geological domain allocation.

The optional resource probe measures nonrigid known-solution loads on refined
spheres and a paired-sheet case with 32 original normal rows. It records actual
CG iterations, error gates, elapsed solve time and tracked allocation. Assembly
and input-system arrays are outside the solve measurement; `tracemalloc` is not
OS peak resident memory. Manufactured forcing is generated by the stated
operator to verify the solver, not to independently validate that physics.

This does not establish a scalable native coupled timestep, a physical beta/eta/H
calibration, complete finite-contact coverage, contact entry/rupture, remeshing,
source budgets or time-integration accuracy. Those remain separate requirements.
