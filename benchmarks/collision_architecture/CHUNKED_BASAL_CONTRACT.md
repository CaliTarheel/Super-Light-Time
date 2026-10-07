# Chunked and restartable basal factors

This is an offline research extension of the finite basal operator. It changes
storage, accumulation and restart handling. Original candidate discovery,
exact positive partitioning, saved layer order, explicit allocation policy,
coefficients, density, finite material traces and quadrature acceptance gates
remain unchanged. It introduces no new geological approximation or departure
from the project's Scotese guidance. It does not activate a native policy,
solve a production world, merge a source branch or resume a live simulation.

## Operator and numerical contract

`FactorChunk(design, reference)` contains one existing `FramedBlockMap` and its
moving-mantle reference. Its factor rows must be exactly the consecutive local
indices `0..9B-1`; overlapping or missing output rows reject. Global output
rows are the logical concatenation of those disjoint ranges. Generalized
columns may repeat, including the zero-valued placeholders in uncovered rigid
factors. Existing local duplicate-column diagonal coalescence remains active.

`ChunkedFramedMap(chunks)` supports `@`, `rmatvec`, `gram_action`, `diagonal`,
`absolute_forward`, `absolute_action`, `storage_bytes` and bounded `to_dense`.
No operation concatenates factor or frame arrays. `@`, `absolute_forward` and
explicit `reference_factor_sqrt_w` access materialize the requested complete
factor-space vector; the physical Gram, evaluation, work and solve paths do
not use those diagnostic materializations. `to_dense` keeps the existing
small-audit dimensions limit.

Empty chunks reject before any archive or manifest write. The narrow
`FramedBlockMap.rmatvec_contributions` extension returns **unique
global indices and separate high/low force words**. Factor and frame products
retain the same compensated arithmetic as PR217. The chunked transpose keeps
one pair of global accumulators and updates only each chunk's referenced
indices; it rounds once after the last chunk. It does not round individual
chunk force vectors and then sum them. This avoids recreating the measured
split-cell cancellation defect. Absolute-action bounds use outward positive
cross-chunk additions. Disjoint factor rows justify summing chunk Gram forms
and diagonals without missing cross terms.

`ChunkedFiniteBasal(system, design)` supplies the replacement protocol used by
`finite_basal_operator.replace_basal`: bound geometry, load, constant,
diagonal, Hessian action, `evaluate`, `work`, and absolute-action diagnostics.
It computes the same positive slip dissipation and moving-mantle actuator
power. Physical evaluation and work stream one chunk at a time. Scalar work
totals use `math.fsum` on per-chunk binary64 dot results; this is not a promise
of arbitrary relative accuracy for ill-conditioned cancellations. The prior
force, virtual-work and constituent-aware arithmetic checks remain necessary.
The arithmetic constituent bound is handled separately: reference addition,
squaring, per-chunk positive summation (with its conservative gamma allowance)
and cross-chunk accumulation round outward.
The independent `other_drag_excludes_allocated_basal=True` declaration is
still required by the existing replacement function.

Memory chunks retain their existing factor arrays, copy only their reference,
and verify array content signatures before use. File-backed chunks load one
bounded archive into an immutable byte snapshot, validate it and construct a
local map. Their `storage_bytes` reports resident provider metadata, **not**
disk usage or measured process RSS; `disk_bytes` and
`maximum_chunk_file_bytes` are separately available on the disk provider.
Local loading includes the archive snapshot, decoded arrays, immutable map
copies and a reduction plan; iterator advancement may also retain the previous
chunk until the caller replaces its reference. Scratch scales with the largest loaded chunk and
the generalized-space vectors. There is no dense global square matrix.

## Durable store

`FactorStore` requires an explicit binding, exact sorted unique expected
control universe, generalized dimension, maximum archive bytes and maximum
store bytes. It begins as `BUILDING`. Each append:

1. obtains a nonblocking operating-system writer lock;
2. verifies that its manifest and prepared binding remain unchanged;
3. rejects duplicate, foreign or malformed control IDs;
4. writes a unique temporary NPZ, flushes it, records its checksum and moves it
   to its final unique filename;
5. atomically replaces the flushed manifest to commit that batch.

The OS releases the lock if the process exits; no stale PID is guessed or
deleted. A crash before the manifest commit leaves an ignored orphan, never a
completed control. Both orphan archives and interrupted temporary archives
count against the disk cap. The store never silently deletes them. Previously
committed batches remain available after a later numerical/resource failure.
Atomic file replacement and file flushing are used, but this does not claim
an independently verified power-loss guarantee for every filesystem/device.

Files use fixed safe relative names. Reads verify exact file lengths and
SHA256, then parse the **same immutable bytes**. NPZ member names, header
versions, dtype, shape, payload length and uncompressed size are checked before
array allocation; object/pickle arrays are prohibited. A source-bound opened
provider detects a changed manifest or changed chunk on every use. Checksums
protect the committed journal against corruption; they are not a signature
authenticating arbitrary externally authored scientific claims.

`seal` requires every expected control exactly once and revalidates every
committed archive. A partial store cannot return a complete factor map.
Generic store sealing establishes journal completeness, not geological
coverage; its caller supplies the certificate. Only the native builder below
establishes the additional native geometry/material/measure checks.

## Native resume entry point

`build_native(domain, system, path, *, mantle_omega_rad_s,
quadrature_relative_tolerance, max_order, max_chunk_bytes, max_store_bytes,
controls_per_batch=16, stop_after_batches=None)` uses the unchanged prepared
native domain. It has no subset-controls argument. An explicit batch stop
returns `NativeBuildProgress(global_allocation_complete=False)` with no
operator. A subsequent identical invocation reuses completed control batches.

Bindings include the exact native source/derived geometry-order-support-beta
fingerprints, joint geometry and active slot mapping, explicit allocation
policy, mantle Euler reference, integration settings, batch size, Python/NumPy
versions, and hashes of root/benchmark numerical Python source. Every retained
archive is verified before computing any remaining exact cell. Resume rejects
changed source or settings; this implementation supplies no source-compatibility
override. Source and state validate before construction and after every batch,
without adding a full-state hash to each individual `_cell` call.

Before returning `NativeBasalBuild`, the builder requires all controls, every
positive material face, individual saved cell area and full-work closure,
global saved area closure, and the unchanged strict saved layer order. It
retains per-owner allocation changes as diagnostics. A sealed reopen verifies
the saved completion certificate against reconstructed batch summaries.
Missing order, invalid geometry or unconverged quadrature remains an error;
`failure.json` records the failed controls and any existing geometry witness,
plus the exact failing finite-piece chart and quadrature records when available
from the unchanged integrator's traceback. Accepted per-region diagnostics
are retained inside their batch archive, rather than accumulated globally.
No restart loop, area omission, coefficient adjustment or alternative ordering
is introduced.

Resource limits apply to completed factor batches and store files. An exact
atomic cell partition can itself exceed a memory/time allowance before its
factor size is known. This implementation does not split that transaction,
interrupt its internal geometry computation, spill its individual pieces or
claim a strict whole-process RAM ceiling. An oversized completed batch rejects
while preserving prior batches. Choosing batch size and running a representative
resource survey remain prerequisites for a full native construction.

The manifest rewrites its compact descriptor list at each batch commit; disk
providers verify/read factor files on each action. These are deliberate
bounded-scope tradeoffs, not a claim that disk-backed solves are faster or that
all metadata I/O is asymptotically optimal. No native whole-world operator or
evolving timestep has been demonstrated by the focused tests.

## Evidence

Focused tests compare full versus chunked factors, mantle/physical work,
duplicate-column diagonals and conservative bounds; repeat the stored cell36312
split-centroid 140-digit transpose oracle across chunk boundaries; exercise
interrupted manifest commits, orphan handling, stale source/state, control
identity, altered content and resource caps; and stop/resume a complete small
native sphere without repeating accepted controls. Actual unordered native
stack geometry still fails. These tests establish the stated storage and
operator behavior, not full-world convergence or geological validity.
