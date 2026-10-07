# Matrix-free joint SI assembly

`sparse_shared_contact.py` assembles and evaluates the same fixed-geometry
physical functional as [the dense joint primitive](SHARED_CONTACT_CONTRACT.md).
It is an assembly and linear-operator experiment, **not a scalable velocity
solver or a native simulation update**. There is no native call site, physical
coefficient migration, contact admission, nonlinear law integration or new
saved-state policy. NumPy remains the only numerical dependency.

## Representation and equations

With default inputs, the unknown `y=[a,z]`, SI coefficients, total material
velocity, force ownership, gauge and finite common-point trace are unchanged.
Explicit all-active Euler slots and anchored frames are specified in
[NATIVE_JOINT_DOF_CONTRACT.md](NATIVE_JOINT_DOF_CONTRACT.md). Every physical
coefficient/load remains required. In particular, material basal drag acts on
total `V-M` once, the remaining-domain plate drag excludes that integral, and
GPE nodal forces must not also be included in the plate torque.

Only the numerical representation changes:

- Each vertex stores a `3 by 5` map: three Euler-speed coordinates and two
  tangent residual coordinates to its total Cartesian velocity.
- Each face stores a `10 by 9` strain factor: the same tensor's nine entries
  and divergence, against one owning plate's three Euler speeds and its three
  vertices' six residual coordinates. The factor includes `sqrt(2 eta H A)`
  and the required per-kilometre to per-metre conversion.
- Each vertex contributes a `3 by 2` residual moment block to its plate gauge.
- Each supplied contact contributes two `1 by 9` blocks, one per face, using
  the same homogeneous radial interpolation as the dense primitive. Positive
  quadrature length scales the normal row by `sqrt(ell/1 metre)`.

`BlockMap` is a small indexed gather/scatter object, not a general sparse-matrix
library. Its transpose uses exactly the same local coefficients and indices.
The joint Hessian action is

```
K y = B.T W B y + F.T F y + K_other y
```

where `B` is total velocity, `W` is positive nodal basal drag-area weight and
`F` is the local sheet strain factor. The full unknown-by-unknown Hessian is
never formed. The strain factors reuse actual `viscous_sheet.prepare` and
`strain_rate` on disjoint face copies in bounded chunks, preserving the existing
P1 geometry/projection convention; chunk size does not alter the results.

`constraint_action` and `constraint_transpose` give original gauge and
length-weighted normal-row actions without rank truncation. They do not solve
for multipliers or claim a force-balanced velocity. The remaining-domain plate
matrix is still dense in **three times the number of plates**, with its existing
PSD validation; this is not a promise of scalability to arbitrarily many plates.
Gauge resolvability uses only a `3 by 3` per-plate moment, so exceptionally
ill-conditioned narrow material patches can fail closed. No material-sized SVD,
nullspace, preconditioner or factorization is performed. Explicit anchor-frame
reduction uses a thin three-column QR and a `3 by 3` SVD, never a square material
matrix.

`evaluate` returns Rayleigh potential in watts and gradient in newtons.
`work` reports physical basal, viscous and remaining-drag powers, external work,
moving-mantle actuator work and their **unsolved** defect. At an arbitrary
supplied velocity that defect equals `y dot gradient` to arithmetic precision;
it is generally nonzero and is not presented as an equilibrium residual.

## Complexity and bounded evidence

Retained material arrays grow as `O(vertices + faces + contact samples)`, plus
the explicit `O(plates^2)` remaining-domain matrix. Local gathers/scatters also
use linear temporary storage. The viscous-sheet preparation scratch is bounded
by `face_chunk_size`; other geometry and gather temporaries remain linear in
mesh size. Small `to_dense` conversions are guarded: more than 256
unknowns or a large map conversion raises rather than allocating a hidden dense
matrix. The assembly itself has no 256-unknown limit.

Run the focused checks:

```sh
python -B -m unittest tests.test_sparse_shared_contact -v
```

They compare all small local maps, Hessian actions, loads, objective and gradient
against the reviewed dense reference; check transpose/reciprocal work, common
rotation, rotated physical fields, moving-mantle accounting, finite/positive
input guards and chunk invariance; and exercise more than 256 unknowns plus 320
manufactured paired contact rows. The duplicate spherical sheets in that last
test are an algebra/storage fixture, not a physical mantle allocation or
attachment-admission scenario.

The explicit performance probe is reproducible with one BLAS thread:

```sh
python -B -m tests.test_sparse_shared_contact --benchmark-output sparse-assembly-benchmark.json
```

One measured run on 28 September 2026, NumPy 2.3.5, reported:

| Faces | Unknowns | Assembly | Mean Hessian action | Retained NumPy arrays | Peak tracked assembly allocation | Hypothetical dense Hessian alone |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,280 | 1,287 | 0.053 s | 0.342 ms | 1.46 MB | 5.34 MB | 13.25 MB |
| 5,120 | 5,127 | 0.108 s | 2.230 ms | 5.82 MB | 13.39 MB | 210.29 MB |
| 20,480 | 20,487 | 0.386 s | 7.632 ms | 23.27 MB | 33.19 MB | 3.36 GB |
| 81,920 | 81,927 | 1.643 s | 33.819 ms | 93.06 MB | 121.33 MB | 53.70 GB |

MB/GB are decimal bytes. Retained storage deduplicates array backing buffers;
`tracemalloc` reports tracked allocations during assembly, **not OS peak resident
memory**. The first small run includes ordinary first-use allocations. Input mesh
generation is outside assembly timing. Ten Hessian actions are averaged per
row. The large dense Hessians were never allocated; their sizes are simply
`8*unknowns^2`. The benchmark uses one complete spherical sheet/plate and no
contact rows. It measures operator storage/action, not solver iterations,
full-run performance, native topology, contact-search or collision refinement.

## Exact next limitations

The dense nonlinear `joint_resistance` callback is not automatically made sparse
by this module. Its finite arc/interface factors must be exposed as local
operators, retaining their own physical quadrature and derivative gates.
A scalable constrained solver still needs tested rank/redundancy handling,
preconditioning and original-row/local-force/work acceptance equivalent to the
dense oracle. None is claimed here.

The native coefficient and domain-allocation mismatch also remains: current
plate basal resistance and gravity-sheet inferred resistance differ, and global
plate support already includes material footprints. This representation does
not choose a replacement beta/eta/H policy, assign mantle drag to stacked layers,
or bypass the required conservative domain partition. A faster matrix action
alone does not establish a physically correct native coupled timestep.
