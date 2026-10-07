# Compensated framed basal map

This component changes numerical representation only. It does not choose a
basal allocation, physical coefficient, quadrature rule, sheet order or native
policy. Its caller supplies local positive factors and their frames.

For each piece let Q be the global frame whose columns are stored as the exact
sum of two binary64 arrays `frame_hi + frame_lo`. For the local nine-coordinate
vector `[a, z]`, where a is the three-component global Euler speed and z contains
six residual coordinates, the forward map is

`F [Q.T a, z]`.

The transpose first applies `F.T`, then maps its first three components by Q.
It retains a normalized two-word expansion through the factor dot, all four
hi/lo products in the frame pullback, and the indexed sum across pieces. Each
global component rounds to one binary64 value only after its complete sum.
The frame and factor are never precomposed into a stored rounded
global block. Temporary ordinary-coordinate columns are constructed only for
the diagonal diagnostic; dense conversion is bounded and uses basis actions.

## API

`FramedBlockMap(blocks, rows, columns, shape, frame_hi, frame_lo)` accepts blocks
of shape `(pieces,9,9)`, index arrays `(pieces,9)` and frames `(pieces,3,3)`.
Frames must represent proper orthonormal matrices within the existing 2e-12
binary64 input tolerance. All supplied arrays are validated, copied and made
read-only. Forward output rows and transpose input columns may repeat; their
contributions sum. A repeated indexed matrix entry is coalesced before its
diagonal contribution is squared.

Methods are `@`, `rmatvec`, `diagonal`, `absolute_forward`, `absolute_action`,
`storage_bytes` and bounded audit-only `to_dense`. `diagonal()` returns the
diagonal of the Gram form M.T M. `absolute_forward(y)` bounds |M| |y| using
nonnegative coefficient products; `absolute_action(y)` applies its positive
transpose bound as well. These account for both frame words. Positive products
and sums round outward by a binary64 step, with per-index summation inflation.

`project_frames(frame_hi, frame_lo, global_vectors)` computes Q.T times supplied
three-vectors, broadcasting leading dimensions. References should use this
same function rather than a rounded `Q.T @ mantle` product.

## Arithmetic model and range

The vectorized algorithm uses standard binary64 round-to-nearest arithmetic,
without relying on Windows `longdouble`, fused multiply-add or extended CPU
register precision. Split, TwoProduct and a compensated dot cascade follow
Algorithms 4–6 of [Ogita, Rump and Oishi, *Accurate Sum and Dot Product with
Applications*](https://www.tuhh.de/ti3/paper/rump/OgRuOi04a.pdf). TwoSum uses the
standard error-free `(a-(s-z))+(b-z)` form. Each factor dot has nine terms;
forward frame projection has six terms. The transpose frame uses twelve
products, including both frame words and both words of the factor result.
Two-word addition uses error-free leading and trailing sums followed by two
renormalizations; unavoidable discarded terms are of second order in unit
roundoff, rather than an intermediate binary64 rounding of each child force.

To protect the error-free transforms, a nonzero product requires each operand
between 2^-968 and 2^968 in absolute value, and the sum of its `frexp` exponents
between -900 and 900. Exact zero products need not split the other operand.
All computed overflow, inexact underflow, division and invalid events reject.
The ranges are explicit conservative arithmetic support, not physical area or
velocity tolerances. Values outside them are not silently clamped or dropped.

Compensated dots have doubled-working-precision quality with a final binary64
rounding term and a term proportional to squared unit roundoff times absolute
constituents. They do not promise uniformly small *relative* error for an
arbitrarily ill-conditioned exact cancellation. This is an arithmetic model
and tested implementation, not a formally directed interval proof for the
complete solver.

Transpose indexed accumulation uses a precomputed segmented pairwise tree of
two-word sums. Completed groups leave the tree immediately; both prepared
indices and action scratch grow linearly with the actual factor entries,
without padding by the largest group or allocating a global square matrix.
The immutable tree is included in `storage_bytes`. Forward output scatter
retains ordinary binary64 `bincount`; basal output rows are normally disjoint.
The positive absolute bounds keep their outward summation allowance, and
diagonal diagnostics still coalesce repeated entries before squaring.

Final global force components are binary64, so independently forming y.T force may lose relative digits when
large terms cancel to a very weak work value. Mathematical transpose symmetry
does not eliminate that component-rounding condition. Preserve the caller's
force and constituent-aware power checks; do not replace them with an
unjustified uniformly relative weak-work assertion.

## Validation and scope

Tests construct the actual saved cell36312 frame using independent 100-digit
Decimal geometry, then compare projection of the actually represented global
inputs against exact hi/lo-frame Decimal products. Weak obliques, centroid
motion, extreme local cancellation, reciprocal action, duplicate index
diagonals, nonnegative absolute bounds, moving-reference power, read-only
storage and unsupported arithmetic are covered. No source unit vectors or
rotated input vectors are silently reconstructed into different exact inputs.

A stored two-child cell36312 centroid regression compares each global
transpose component against an independent 140-digit evaluation of the exact
represented factors and hi/lo frames. The earlier implementation rounded
individual child forces before summation and lost about 5e-9 relatively on
this case. The regression also checks reversed order, further exact binary
factor subdivision, and indexed residual cancellation. This does not claim
uniform relative accuracy for arbitrarily ill-conditioned sums; the
constituent-based doubled-precision qualification above still applies.

This component alone does not establish analytic metric accuracy, full-cell
quadrature convergence, complete basal allocation, collision dynamics or a
successful native timestep.
