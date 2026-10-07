# Measured baseline, 28 September 2026

**Numerical conservation passes; trajectory convergence does not.** These are
component experiments on engine source `86a79b3`, with actual file hashes,
environment, policies and all comparison metrics in
[baseline_summary.json](baseline_summary.json). The summary records uncommitted
benchmark files because it was generated before this benchmark's first commit;
the engine was unchanged. Reproduce with the `study` command in the README.

All cases run to the same 12 Myr endpoint on a mirrored 64-face fixture. No
geometric face limit activated. Tolerances were selected before the study.

| Symmetric timestep | Mean crustal thickening | Maximum thickness | Final distinct-sheet overlap |
| --- | ---: | ---: | ---: |
| 1 Myr | 1.3806 km | 37.6820 km | 33,485.3 km² |
| 0.5 Myr | 1.4661 km | 37.8065 km | 18,259.1 km² |
| 0.25 Myr | 1.8539 km | 38.6210 km | 668.9 km² |

The medium-to-fine mean thickening difference is 0.3878 km, about 20.9% of
the fine result's increment, versus a predeclared 2% target. Compression also
fails, and these two differences grow relative to the coarse-to-medium
comparison. Maximum displacement passes its comparison; the left-quadrature
integral of passive plate power misses its 2% target. It is not a complete
mechanical work budget. No convergence order is claimed.

The first contact brackets contract from [3,4] to [3.5,4] to [3.75,4] Myr.
Contact admission is observed at endpoints; this does not prove that first
contact is localized inside an interval. The dramatic overlap change warrants
investigation of event timing and the split mechanics before calibration.

## Passing evidence

- The force-free rest case preserves geometry, thickness and inventories.
- All six runs pass per-face/body/total volume, actual plate and constrained
  sheet stationarity, passivity, orientation and same-sheet overlap gates.
  Maximum local relative volume error is below 7.5e-15.
- Symmetry agrees within 2.5e-14 radians in position and 3e-11 km in thickness.
- Swapping the authored weak side swaps the result within 1.1e-13 radians and
  1.3e-10 km. The contrast changes maximum per-face thickness by 1.953 km from
  the equal-strength result. No universal weak-side shortening fraction is
  imposed.
- Thirteen harness tests: twelve pass; one explicitly expected failure records
  the GPE sensitivity below. Fresh-process checkpoints reproduce pre-contact
  and already-contacting physical state, contact histories, RNG and policies
  bitwise. Deliberate inventory/column/policy/diagnostic failures are detected.

The complete six-case matrix took roughly 27 seconds on this machine with one
BLAS thread. These tiny fixtures do not predict whole-world step performance.

## Additional force-derivative failure

The same represented overlapping geometry, rotated as a whole, preserves the
plate solution. However, independently constructing the equivalent rotated
overlap introduces only floating-point coordinate differences yet changes
the solved rotation by up to 1.69e-5 rad/Myr. Overlap area, stack order and
interface shear agree. The discrepancy is in the differentiated GPE clipping
geometry. `test_rebuilt_rotated_contact_preserves_gpe_at_near_coincident_edges` retains the
expected failure without loosening its tolerance.

This is being isolated with energy directional differences and clipped endpoint
Jacobians. A passing covariance test alone would be insufficient: an incorrect
but covariant derivative can still produce the wrong force. No fix for this
issue, combined plate/sheet reaction, native collision entry or welding is
included in the baseline PR. Production is unchanged.
