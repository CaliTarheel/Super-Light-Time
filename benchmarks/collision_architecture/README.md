# Continental collision architecture benchmark

This portable **component** experiment measures the current split collision
path. It is not `Simulation.step()`, a calibrated Earth model, or a claim that
continental collision is fully solved. No private fixture, archived world,
network access or production server is required.

## Run

From the repository root, using the project's Python dependencies:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -B -m benchmarks.collision_architecture.runner \
  --case symmetric --dt 1 --end 12 --output /chosen/output
python -B -m unittest tests.test_collision_architecture
python -B -m benchmarks.collision_architecture.study --output /chosen/study
```

On PowerShell set the three environment variables with `$env:NAME='1'` before
running the same Python commands. Choose an output directory outside the source
tree or under the ignored `output/` directory. The report is atomically replaced
after each observation. A numerical gate failure stops further advancement and
is preserved with its traceback. A successful numerical run writes a typed
`state.npz` checkpoint; resume it in a fresh process with `--resume PATH` and a
later `--end`. The source-byte/NumPy compatibility check remains enforced.

The report distinguishes numerical gate success, actual contact coverage, and
physical completeness (`collision_validated` deliberately remains false).
Reaching the requested end before contact is not a collision validation.
The study command runs six cases and records comparison gates in `summary.json`.
It exits with status1 when a declared comparison fails; the baseline is expected
to do so. See [BASELINE.md](BASELINE.md) for the measured failures.

## Fixture and measurements

Two identical ordinary continental meshes start approximately150 km apart.
One is a proper rotation of the other, including triangle diagonals and winding.
Both start at35 km crustal thickness and equal column elevation. The default
mesh has64 faces. A fixed generalized load and isotropic resistance are chosen
to produce free rotations of +/-0.003 rad/Myr. They are manufactured boundary
conditions, not a new mantle force or a replacement for slab-driven startup.
This use respects the guide's force/resistance distinction (Scotese Rules I
and XI) without claiming geological calibration.

Cases:

- `rest`: zero external load; separated uniform material must remain still.
- `symmetric`: equal opposing loads and equal initial properties.
- `weak0` / `weak1`: authored inherited weakness0.6 on one ordinary sheet,
  swapped between sides. Native enhanced-rifting viscosity multipliers start
  at0.4 and1. Geometry, crust class, columns and constraints stay matched.
  Later thermal changes can alter these multipliers; each epoch records them.

The driver assembles production local opening weld, interface shear and rigid
GPE terms, solves plate motion, then calls production material advection and
geometric column evolution. Boundary-strength arrays are empty, so the weld
uses its documented default strength; this is a **sheet viscosity** experiment.
It omits normal ocean/trench/slab, entry, rupture, consolidation, dense-phase,
erosion, rendering and rift-lifecycle assemblies. Column thermal response still
operates and its height budget is recorded.

`tolerances.json` declares numerical and comparison thresholds before judging
the measurements. Inventory checks apply independently to each face, each
body and the whole fixture using areas recomputed from geometry. Additional
checks cover actual per-plate solver acceptance, local sheet convergence,
passivity, persistent IDs/owners/reference weights, triangle orientation,
column bounds and independently recounted same-sheet overlaps. Sheet solver
residuals describe its constrained solution before any subsequent geometric
limiter; recorded limiter activity must be examined separately.

Power metrics have unitsW. They are instantaneous, not integrated work. Physical
torque residuals are converted toN m; a tiny **relative plate residual** does not
prove that the separate sheet reaction balances the plates. Material-stage
timing excludes plate assembly and observation; report timing covers the whole
component invocation, not a live global timestep. BLAS thread settings and host
information are recorded so concurrent machine load can be considered.

## Known architectural gaps and next gate

See [MECHANICAL_CONTRACT.md](MECHANICAL_CONTRACT.md). GPE already has a
rigid/residual gradient split. Reciprocal constrained and distributed sheet
reactions are missing from plate balance, and interface/weld rates use rigid
velocities. Returning raw normalized sheet residuals as physical torque would
be dimensionally wrong. A shared force functional and a declared physical
sheet/basal allocation must precede that change.

Predeclared timestep comparisons use equal final physical time at1,0.5 and
0.25 Myr. A passing solver and conservative inventory do not imply trajectory
convergence. Preserve failures, contact-time brackets and limiter activity;
do not widen tolerances to make the baseline pass. Mesh refinement, native
ocean-to-continent entry, inherited polarity, weld transactions and a full
mechanical work audit remain separate, required work.

## Frozen joint mechanics prototype

`shared_contact.py` provides a small, isolated joint solve for plate rotations
and sheet residual velocities at fixed geometry. Explicit SI coefficients,
one total-velocity drag integral, a stated residual closure, and reciprocal
common-point contact reactions make its units and work accounting testable.
See [SHARED_CONTACT_CONTRACT.md](SHARED_CONTACT_CONTRACT.md) for the functional,
contact scope, numerical acceptance, and remaining integration requirements.

Run its portable manufactured and analytic checks with:

```sh
python -B -m unittest tests.test_shared_contact tests.test_shared_contact_analytic tests.test_shared_contact_nonlinear tests.test_joint_resistance -v
```

The prototype accepts already-admitted closed bilateral normal contacts and
at most256 generalized unknowns. It supplies no physical calibration, contact
birth/release, fracture law, timestep evolution, or saved-policy migration.
Neither the component runner above nor the production engine calls it.

The [finite resistance extension](JOINT_RESISTANCE_CONTRACT.md) supplies the
existing local opening-weld and buried-interface shear laws on total material
velocity. Its constrained Newton solve returns both plate and sheet reactions,
with actual resisting power checked separately from the dissipation potential.
Fixed admitted face-pair geometry and physical coefficients remain explicit;
these tests do not provide native admission, rupture or timestep validation.
