# Stable local representation of finite basal resistance

Read the recorded validation before treating this as ready.
This research path supplies no production callsite or saved-policy migration.

## Physical law and numerical defect

The basal law remains the positive integral of beta times squared total
material-minus-mantle velocity over the selected bottom material and uncovered
plate domains. Only the bottom sheet receives basal drag. Plate and residual
material forces remain reciprocal derivatives of that same functional.
Scotese Rule I motivates passive mantle resistance against explicit plate
drivers; no mantle conveyor, added force, coefficient change or geological
departure is introduced here.

A tiny spherical patch has two large resisting rotation directions and a much
weaker rotation about its own centre. Rounding its analytic metric in global
coordinates can erase the weak positive mode. Rotating that already-rounded
matrix cannot restore it. The saved native cell36312 case demonstrates this
with a positive region of approximately0.00216 square metres. Cell337 also
exposed cancellation in a full weak-direction quadrature comparison.

The original exact rational intersection chart is part of the geometry.
Rounded global corners remain diagnostics and display data. They must not be
substituted for the chart during analytic integration or local quadrature:
an independent check of36312 measured a2.03e-9 area difference and3.59e-9
centroid-direction work difference from that substitution, above the existing
2e-10 integration gate. Retaining the chart is an arithmetic repair of the
declared geometric domain, not a new physical allocation policy.

## Representation through the operator

Each piece retains its bound original control/chart, an accurately constructed
local frame, a positive analytic local metric and a local factor. Precision
agreement is checked in the full factor-whitened metric, not only in area or
eigenvalue signs. This is a convergence check, not an interval proof. Unresolved
cases still reject without dropping regions or projecting eigenvalues.

Finite quadrature transforms the original material face and tangent bases into
the same local frame without normalizing the stored material coordinates. It
solves the original homogeneous trace directly. Cross-product factors retain
the small squared transverse terms instead of subtracting nearly equal ones.
The existing full9-by9 convergence, positive-area, trace reconstruction and
2e-10 full rigid-work gates remain required.

The local9-by9 factor is retained. Every operator application first transforms
the three Euler coordinates into the local frame using hi/lo frame values;
transpose force application uses the reciprocal transform. The implementation
does not precompose a rounded global factor. Mantle-reference work uses that
same operator. Control closure uses analytic piece metrics expressed in the
control's local frame before rounding; legacy global matrices are diagnostics.
Native batching must preserve each factor's frame alongside its columns.

The transpose retains two-word contributions through the local factor,
frame pullback and indexed accumulation, rounding a combined force component
only at the end. Independent split-piece tests reduced error against the
represented-factor transpose from5.04e-9 to below8e-17. This repairs avoidable
child-force rounding without pretending to remove all conditioning effects of
the finite binary64 factor itself.

Compensated products follow the error-free-transform approach described by
[Ogita, Rump and Oishi, Accurate Sum and Dot Product with Applications](https://www.tuhh.de/ti3/paper/rump/OgRuOi04a.pdf).
The implementation declares and checks its arithmetic range. No Windows
extended-precision assumption or new runtime dependency is required.

## Evidence required

Require both saved native cells to build under the unchanged gates; independent
high-precision analytic and positive-quadrature comparisons of actual global
input directions; generic rotations; split/recombine; moving-mantle work;
reciprocal forces; and rejection of changed geometry/frame provenance.
Final global force components are binary64: their rounding can dominate a
naive dot product along a very weak mode. Strict weak-mode checks therefore
inspect actual factor-space power; force and reciprocal-work checks must
state their constituent rounding bounds. This is not permission to conceal
a force imbalance or loosen the declared integration gates.

Passing these tests would establish a bounded numerical component repair.
It would not establish a complete native basal operator, calibrated joint
mechanics, evolving collision, timestep convergence or corrected terrain.

## Measured integration and conditioning boundary

The first integrated probe accepted all388 positive pieces of native cell337
and all2,738 pieces of36312. The maximum accepted finite-quadrature errors were
2.23e-12 and2.52e-11 respectively, below the unchanged2e-10 gate. The latter
cell retained a still smaller0.000944 square metre region. These are two
partial frozen cells within a70,747-coordinate geometry fixture, with zero
viscosity and zero applied loads; they are not a global joint solution.

Independent positive high-precision material-trace quadrature also inspected
near-cancelling Euler/residual inputs. With opposing inputs of order0.3 and a
1e-6 perturbation, relative errors in the small remaining work reached3.2e-10
and9.45e-10, while absolute differences were2.37e-23 and2.37e-22 of the
constituent work. At unperturbed represented rigid exchange, excess work was
2.77e-32 and1.72e-31 of that scale. These measurements expose conditioning of
the stored binary64 trace/factor. The rigid weak-mode gate is not a universal
relative-work guarantee for arbitrarily cancelling joint inputs. Report both
relative and constituent-scaled errors; global feasible joint residual and
trajectory gates remain necessary.

For weak split centroid inputs, the final represented-factor transpose agrees
with its high-precision oracle below8e-17, while force error against the
underlying exact physical integral can still reach7.33e-9 relatively after
cancellation of much larger contributions. The separate physical/represented
comparison must be retained. Ordinary and weak rigid/joint/residual input work
in the independent native-piece audit agrees within2.1e-15; these finite
fixtures do not certify arbitrary near-null inputs or a global force solve.
