# Full Euler slots and anchored residual frames

This extends the frozen joint research operator and solver. There is no native
call site, timestep, coefficient calibration or saved-policy migration. The
native adapter supplies the selected anchor mask; assembly never infers anchors
from craton incident faces, collision weights or shared vertex proximity.

## Explicit inputs

`sparse_shared_contact.assemble` accepts:

- `plate_count=P`: all active Euler slots, including slots with no material.
  Every owner must lie in `[0,P)`. Torque and other-domain drag cover all `3P`
  Euler coordinates. Omitting it preserves the previous contiguous represented
  slot inference. Ocean-only slots require their actual physical resistance;
  this extension adds no drag floor or arbitrary pin.
- `fixed_residual_dofs`: Boolean `(vertices,2)` mask, default all false. True
  entries prescribe those tangent residual coordinates to exactly zero. They
  do not freeze total world velocity: material follows its owning Euler frame.
  Partial masks refer to the returned tangent basis; rotating a partial mask
  requires transforming its physical constrained direction.
- `preserve_represented_points=True`: copy supplied vectors after the unchanged
  unit-norm validation, without another normalization. This keeps native binary
  geometry identical to the finite basal ledger. Default false preserves the
  previous normalization. This option neither repairs geometry nor relaxes the
  unit-vector tolerance.

The full order stays `y=[a_0,...,a_(P-1),z_0,...]`, where `a=R omega`, in m/s.
Total material velocity, finite contact traces, strain factors, physical loads
and the functional remain unchanged. Finite basal replacement and nonlinear
finite laws use this full space. Fixed coordinates are removed only from the
solve; their forces remain available for reaction recovery.

## Gauge after anchoring

Let `B_i a = a cross r_i`, with tangent basis `T_i`. A representation change
preserving total material velocity has `delta z_i=-T_i.T B_i delta a`.
For each plate collect the fixed rows of `-T_i.T B_i` into `J`. Let `N` span
`null(J)`. Only those remaining exchange directions receive the moment gauge
`N.T G z=0`, with `G z=sum_i (A_i/A_plate) r_i cross (T_i z_i)`.

- No anchors: `N=I`, preserving the original three rows.
- One fully anchored vertex: one exchange direction, parallel to its position.
- Two nonparallel fully anchored vertices: no exchange direction.
- One fixed tangent component: two exchange directions.
- Ocean-only plate: no residual representation and no gauge.

The first `dim(N)` of each plate's three gauge output slots hold the projected
rows; remaining slots are structurally zero. Basis, dimension, anchor rank,
singular values and numerical rank threshold are returned.

This is an **anchored Euler reference-frame choice**, not equivalence to imposing
every legacy full moment gauge together with anchors. Other-domain Euler loads
and drag remain assigned to that frame and are not invariant under arbitrary
compensating Euler/residual exchange.

Anchor rank is computed exactly on represented binary entries using rational
arithmetic. A thin three-column QR produces a `3x3` SVD core. If a genuinely
independent direction lies below `256 eps max(rows,3) sigma_max`, assembly fails
closed rather than dropping a physical anchor direction. Remaining gauge moment
resolvability is checked on free columns. No full material matrix or nullspace
is built. A mask digest prevents changing anchors after their frame gauge was
prepared; changed anchors require reassembly.

## Elimination, reactions and acceptance

With `E` injecting free coordinates, solve the same functional at `y=E x`, with
`C E x=0`. Fixed entries of diagonal congruence scaling are zero; division is
masked and every trial keeps fixed velocities exactly zero. Positive combined
Hessian diagonals are required only on free coordinates. No synthetic anchor
rows consume the bounded constraint Gram.

Only rows with exactly zero local absolute coefficient support on free
coordinates are omitted from that Gram. All original gauge/contact outputs
remain in acceptance and diagnostics. Nonzero rows with unresolved Gram norms
fail. The default 512 limit applies to nonzero restricted rows; original and
active counts are reported. Existing near-dependent contact-rank limitations
remain; load-specific residual checks do not prove universal row independence.

After equality multiplier recovery, define

```
q = K y - f + grad Psi(y) + C.T lambda
R_fixed = q on fixed residual coordinates; zero elsewhere
stationarity = q - R_fixed.
```

The attachment force is `F_i=T_i R_fixed_i`; its reciprocal torque on the same
owning Euler frame is `tau_p=-R sum_(i on p) r_i cross F_i`. Thus

```
sum F_i dot V_i + sum tau_p dot omega_p
    = sum R_fixed_i dot z_i = 0.
```

These nodal forces, Euler torques, pre-reaction imbalance and separate powers
are returned, not counted as external work. Free-component stationarity keeps
the original local product scales. Its global scale excludes fixed coordinates,
so large anchor reactions cannot hide a free imbalance. The original physical
work gate remains, plus exact fixed-velocity checks and a separate finite-product
arithmetic check on the material/frame reaction power cancellation.

## Evidence and limits

`python -B -m unittest tests.test_native_joint_dofs` covers independent full-space
KKT equivalence, partial/full anchors, force/torque signs, ocean-only Euler
balance, all-fixed rigid response beyond 512 residual coordinates, unresolved
anchor rank, geometry preservation, original normal rows, finite basal replacement
and actual nonlinear finite weld/interface integration. Prior sparse/dense
parity and solver/law/basal regressions remain required.

A read-only 642 Myr snapshot probe assembled 52,025 faces/35,357 vertices with all
11 active plates, including one ocean-only slot. Its explicit 221 static fixed
vertices (220 craton cores plus one point-only component junction) remove 442
residual coordinates, leaving 70,305 free coordinates from 70,747 and 21 active
gauge rows. The probe uses declared manufactured assembly coefficients; it does
not solve or validate native calibration. Transient protections, global physical
basal allocation, loads, contact admission and time integration remain separate
requirements. No production state was changed.
