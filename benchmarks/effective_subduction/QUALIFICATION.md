# Effective subduction qualification — 2 October 2026

Reproduce from the fork root with the existing Python environment:

```powershell
& '.venv\Scripts\python.exe' -m unittest tests.test_effective_subduction_forces -v
& '.venv\Scripts\python.exe' -m experiments.effective_subduction --levels 2 --repeats 7 --steps 3 --dt .05 --mixed-carrier --output validation\effective-subduction-benchmark.json
```

To check the saved evidence without repeating the timings:

```powershell
& '.venv\Scripts\python.exe' -m experiments.effective_subduction --qualify-existing benchmarks\effective_subduction\measurements-20261002.json
```

All 21 recorded qualification checks pass. The command exits nonzero on failed
checks; fresh measurements save their source hashes before running and preserve
failure evidence before exiting. The saved qualification review adds only checks
and a review timestamp to the original measured fields. Final CLI/fresh-world
tests, three UI suites, and targeted native/force/history regressions also pass.

The eight mechanical tests pass. They independently check physical torque,
basal resistance, owner direction, edge subdivision/storage reversal, force
independence from slab mass/dip and prior velocity, persistence through rest
and opening, shared local virtual work and passive resisting work, equal
opposed pulls with internal tension, local buoyant arrival, and finite daughter
opening. The strong daughter cut stays stopped; the weak cut opens while the
generalized power including cut dissipation closes.

The first native force solve starts with input angular velocity exactly zero.
It gives 0.309568 cm/yr mean and 0.457482 cm/yr maximum surface motion without
creating slab area, slab mass, elapsed time, or initial ocean consumption.
Both the coarse native evolution and the material-preserving mixed-carrier
startup replay bitwise through checkpoints for the ten compared fields.

The 320-control benchmark uses seven warmed frozen-state samples per law and
three real 0.05 Myr steps per law. Inventory-only and effective states have
identical initial control, boundary, ownership and material geometry. The
current coupled lifecycle baseline selects actual coupled slab tethers, rather
than accidentally timing the older inventory-only law; all three have sixteen
force-bearing initial contacts. Full raw timings/configuration/runtime details
are in [measurements-20261002.json](measurements-20261002.json). This dated
measurement predates source-hash capture in the experiment script; no historical
source hashes have been inferred. The fork's final regression checks include the
reporting and configuration refinements made after these timings.
This timing receipt also predates Lite's simple upper-plate-motion back-arc
loading rule. Its cost and later basin evolution have not been measured here.

| Measured median | Inventory-only slab law | Coupled lifecycle | Effective traction |
| --- | ---: | ---: | ---: |
| Force assembly | 0.176946 s | 0.180150 s | 0.164724 s |
| Nonlinear solve | 0.017853 s | 0.015923 s | 0.015286 s |
| Assembly and solve | 0.194564 s | 0.195318 s | 0.180608 s |
| Whole native step | 5.723972 s | 8.163369 s | 4.735611 s |

Force assembly plus solve is about 1.08 times faster in this small workload.
Whole-step ratios are 1.21 and 1.72 relative to the inventory-only and coupled
baselines, respectively. Those have only three samples, changed forces and
different evolving trajectories; they are not a general production speedup or
a matched-physics accuracy comparison. The native geometry/material/source
operators remain the large cost. An earlier three-repeat smoke had wider
ratios, reinforcing the need to treat these small benchmarks cautiously.

The separate mixed-carrier qualification uses the existing seed-37 attachment
of water region 1 to continental plate UID 3, at the required level-4 control
resolution and level-2 coast geometry. Two offshore declared incoming trenches
supply 9.947453e25 N m of direct torque to this same plate. All 206 original
continental markers survive the actual 0.02 Myr native step; mean displacement
is 0.008908 km and maximum displacement is 0.018233 km. Its checkpoint replay
is bitwise identical and its slab inventory remains zero. This demonstrates
that the simplified force can move directly connected continental material;
it does not establish realistic long-term speeds or a geological history.

The coarse evolution changes material tessellation, so raw parcel-position
arrays cannot be compared as fixed material identities over those steps. The
mixed-carrier displacement uses enduring marker IDs instead. No spontaneous
subduction initiation, independent rollback, or resolved slab energetics is
qualified by these experiments.
