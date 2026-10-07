# Frozen joint plate/sheet contact primitive

`shared_contact.py` is an unintegrated research operator. It solves plate Euler
velocities and tangent residual sheet velocities together for **supplied,
already admitted, closed bilateral normal contacts on frozen geometry**. It is
not a simulation replacement, calibrated Earth model, or contact birth/opening
law. None of the native engine's behavior or saved policies changes.

This addresses one missing mechanical ingredient identified in
[MECHANICAL_CONTRACT.md](MECHANICAL_CONTRACT.md): a sheet/contact reaction must
enter the plate equations through the same virtual-work map as the sheet
equations. It does not complete the remaining collision integration work.

## Explicit SI inputs and force ownership

For unit material positions `r_i` and radius `R` in metres, unknowns are

```
y = [a_p, z_i]                      all components in m/s
a_p = R * omega_p                  omega in radians/s
u_i = T_i z_i                      two tangent residual coordinates
V_i = a_owner(i) cross r_i + u_i    total material velocity
```

`assemble` requires every physical coefficient and external load; there are no
default material calibration values. The manufactured tests choose coefficients
only to test equations and units. `solve` returns the simultaneous constrained
minimum of the Rayleigh potential, in watts:

```
Phi(y) = 1/2 sum_i beta_i A_i |V_i - M_i|^2
       + sum_f eta_f H_f A_f [D_f(V):D_f(V) + div_f(V)^2]
       + 1/2 omega.T D_other omega
       - sum_i F_i dot V_i - sum_p tau_p dot omega_p
```

| Input | Units | Ownership |
| --- | --- | --- |
| `basal_drag_pa_s_per_m` | Pa s/m | Positive material basal drag, once on total `V-M` |
| `viscosity_pa_s` | Pa s | Nonnegative face viscosity |
| `sheet_thickness_m` | m | Positive face thickness |
| `basal_reference_velocity_m_s` | m/s | Prescribed tangent mantle velocity `M` |
| `other_plate_rotational_drag_n_m_s` | N m s | Symmetric passive remaining-domain rotational drag; excludes the material basal integral above |
| `external_nodal_forces_n` | N | Nodal force through total velocity, including any supplied frozen GPE force once |
| `external_torques_n_m` | N m | Additional plate torque; excludes a GPE/material load already in `F` |

Nodal areas are one third of each incident spherical face area. Face viscosity
uses the existing `viscous_sheet.prepare` and `strain_rate` gradient/tensor,
including its declared embedded-triangle discretization. Those gradients use
kilometres, so an SI m/s nodal velocity produces a tensor divided by 1000 to give
1/s. A direct Gram factor `sqrt(2 eta H A) * [D,div]` yields the Hessian and
nonnegative viscous dissipated power. Actual viscous power is twice its
quadratic Rayleigh term. In the speed coordinates, remaining rotational drag
is `D_other/R^2` and the torque load is `tau/R`.

No native plate basal term is added over the same material support. The
prototype neither calculates GPE forces nor claims finite-step conservative
energy closure. A supplied conservative force supports instantaneous power
only; the caller must provide a consistent energy gradient and force ledger.

## Gauge and paired finite trace

Each plate imposes `sum_i A_i r_i cross u_i = 0`, normalized by its material
area. This removes the duplicate Euler representation in material total-velocity
terms. With additional plate-only drag or loads it also declares the kinematic
closure equating mean material spin and plate spin. It is not a claim that
plate-only physical work is invariant under moving rotation into `u`.

A `ContactSample` specifies two finite faces, one common unit point `c`, one
tangent normal `n`, and a positive quadrature length `ell` in metres. For either
triangle, solve the homogeneous radial coordinates

```
[r0 r1 r2] alpha = c
trace(V,c) = (I - c c.T) sum_i alpha_i V_i
```

This is the derivative of the radially projected P1 interpolation. It exactly
reproduces `R omega cross c` for a common Euler rotation on unequal triangles.
Normalizing `alpha` before interpolating velocity would introduce a different
chord-length factor on each side, creating fictitious common-spin contact.
True outside-face samples are rejected. Tiny signed linear-solve roundoff is
retained to preserve this identity; it is not an admission tolerance.

The supplied fixed contact imposes

```
C0_j y = n dot [trace_first(V,c) - trace_second(V,c)] = 0
C_j = sqrt(ell_j / 1 metre) * C0_j
K y - f + C.T lambda + G.T mu = 0
```

Both reaction blocks use the transpose of this same trace. The unweighted
resultant is `t_j = sqrt(ell_j / 1 metre) lambda_j` in newtons, with first point
force `-t_j n` and second force `+t_j n`. If `n` points from first toward second,
positive `t` is compression and negative `t` is tension. **Equality contact can
carry tension**; there is no unilateral complementarity, failure strength,
friction, or automatic weld. The reported quadrature traction is `t_j/ell_j` in
N/m, not a fully resolved stress distribution.

Rigid virtual work gives each point torque `R c cross force`. Point resultants
and total contact torque cancel between sides. Cartesian sums of the pulled
back nodal forces need not cancel on curved constrained sheets; radial
constraint reactions differ. Gauge multipliers are reported separately and
must not be interpreted as contact forces.

## Rank, arithmetic and acceptance

The dense primitive is limited to 256 generalized unknowns. Plates have
contiguous slots, positively oriented owned material faces and enough geometry
to resolve the three-component residual moment gauge. Material basal drag is
strictly positive. A negative supplied remaining-drag eigenvalue is rejected
even if small compared with another mode; unresolved numerical PSD inputs fail
closed without silently changing their coefficients.

Diagonal energy scaling and row-normalized SVD determine the constraint
nullspace. Rank, singular values and redundant-row count are returned. A
Cholesky solve handles the reduced positive quadratic; no damping is added.
The original multiplier norm is minimized over the retained dual nullspace.
That convention minimizes `||mu||^2 + sum_j t_j^2/(ell_j/1 metre)`, including
gauge multipliers, not solely traction norm. Bit-identical unweighted contact
rows receive an exact final null-direction projection: their total resultant
is distributed proportional to lengths. Splitting an identical sample thus
preserves both velocity and aggregate reaction. Near-dependent rows are not
grouped or silently discarded as equivalent.

Successful return requires finite velocities, reactions, diagnostics and powers;
global and componentwise stationarity; every original unweighted normal-slip
row against its own local plate/residual trace speed; every gauge row against
its own plate speed; and the physical power identity below. A large unrelated
plate speed or resistance cannot excuse failure in a small contact/force
component. The SVD tolerance is a numerical rank threshold, not a constitutive
law. The routine rejects constraints or arithmetic range it cannot resolve.

For a moving prescribed mantle, report two distinct power identities:

```
algebraic:  f_effective dot y = y.T K y
physical:  P_external + P_mantle = P_basal + P_viscous + P_other
P_basal  = sum beta A |V-M|^2
P_mantle = -sum beta A (V-M) dot M
```

The algebraic effective right-hand side includes `beta A M`, so its power alone
is not physical external work. Contact and gauge virtual powers vanish for the
homogeneous constraints. Reported passive powers use direct squared factors
where available, avoiding spurious negative near-null quadratic evaluations.

A free common mantle/material Euler motion has exactly zero physical power.
The power gate therefore adds a reported arithmetic allowance to the relative
physical-work tolerance: the already-gated solver error propagated into work,
`sum |y_i e_i| + sum |lambda_j (C y)_j|`, plus `gamma_n` times absolute
constituent products in watts. Here `e = K y-f+C.T lambda`,
`gamma_n = n eps/(1-n eps)`, with a diagnostic estimate of sixteen passes
through the assembly/evaluation inner dimensions. Absolute Gram-factor,
basal `(|V|+|M|)^2`, matrix, load and constraint products include the terms that
cancel in the common-motion limit. This is an explicit binary64 arithmetic
estimate for reported products, not a proven bound through ill-conditioned
geometry solves, SVD, eigendecomposition or assembly. Separate force and original
constraint checks still apply; no directed interval certification is claimed.
There is no fixed watt floor or new physical budget. A deliberately inconsistent one-percent external-power
ledger is rejected, while the exact common-motion limit remains admissible.

## Evidence and remaining work

### Optional joint nonlinear passive resistance

`solve(system, resistance=operator)` accepts a pure `operator.evaluate(y)`
callback on the **same full SI generalized velocity**. It returns:

```
potential_w:             scalar Rayleigh potential R(y), W
gradient_n:              dR/dy, N
hessian_n_s_m:           d2R/dy2, N s/m
resisting_power_w:       y dot dR/dy, W
diagnostics:             dictionary of per-law geometry/work evidence
hessian_factor_sqrt_n_s_m: optional F with Hessian = F.T F
```

The callback receives a read-only trial copy. Its source geometry, topology,
coefficients and physical force ownership must remain fixed during a solve.
If it supplies `validate_system(system)`, both public entry points invoke that
binding check before evaluating it; a preassembled geometry operator should
implement this check rather than relying on unknown count alone.
The interface checks shape, finite arithmetic, symmetry, nonnegative potential
and resisting power, and equality of reported resisting power to `y dot gradient`.
A physical Gram factor is checked componentwise against the supplied Hessian,
then used as its declared positive construction. This is not eigenvalue clipping.
Without a factor, a negative computed Hessian eigenvalue fails closed. Analytic
derivatives and convexity remain the callback author's responsibility and need
independent derivative/physical tests; pointwise PSD checks cannot prove a
global constitutive law.

The final functional is `Phi_linear(y)+R(y)`. Constrained Newton operates in
the original homogeneous constraint nullspace, uses the total Hessian and
backtracks the **total potential**. A stable quadratic difference avoids
subtracting large linear objectives. A documented binary64 objective allowance
handles cancellation during line search; acceptance still requires final full
nonlinear KKT stationarity, componentwise force balance, original local
constraints and physical power. A small step or exhausted iteration budget
cannot authorize a result. Intermediate Newton systems never replace the
physical force ledger with a fabricated external load.

SVD nullspace roundoff can mix a strongly driven mode into an exact symmetry-zero
component, leaving a tiny normwise residual but an unacceptable local backward
error. Up to four refinement corrections solve the **original KKT equations**
with rank-many independent original constraint rows. The weighted minimum dual
and exact duplicate convention are restored, nonlinear derivatives are
reevaluated, and every original final gate still applies. This improves the
algebra without introducing an absolute force floor or altering the physical
tolerance. Unresolved refinement still raises an error.

At the accepted solution the same transpose maps reactions into both unknown
blocks. The return includes nonlinear generalized reaction `-gradient`, its
plate torque, per-law diagnostics and Newton history. Equality-contact and
gauge reactions remain distinct. True resisting power is reported separately
from potential; for a quartic manufactured law it is four times the potential,
whereas for a quadratic it is twice the potential.

The corresponding identities are now

```
f_effective dot y = y.T K y + y dot dR/dy
P_external + P_mantle = P_basal + P_viscous + P_other + P_resistance
```

The additional tests in `tests.test_shared_contact_nonlinear` include an
independently reduced cubic stationarity solution, derivative checks, retained
contact feasibility, incorrect work/factor/negative-curvature controls,
iteration-budget rejection and trial immutability. A callback containing a
finite weld or buried-interface law is still a **frozen admitted-contact
experiment**, not native admission, material-mantle allocation, source
integration, physical calibration or timestep validation.

Run `python -B -m unittest tests.test_shared_contact tests.test_shared_contact_analytic`.
The tests cover analytic absolute SI plate torque, spherical area/radius scaling,
an independent affine membrane-power solution, moving-mantle actuator work,
functional derivatives, reciprocal cross blocks, common Euler trace,
rotation covariance, quadrature splitting, passivity, local numerical guards,
and a residual sheet load that changes another plate through contact reaction.

The moving-contact test independently evaluates the existing finite-belt
profile on either side of its old `weight=0.9999` threshold, while keeping the
supplied positive contact patch and face topology fixed. Its smoothly varying
joint response shows that this operator contains no driven-node threshold.
It does **not** solve the observed native threshold chatter, establish smooth
contact birth as positive length tends to zero, certify swept admission, or
validate a timestep trajectory.

Before native use: define contact admission/opening and finite coverage,
integrate actual material/GPE loads and remaining-domain resistance once,
calibrate physical coefficients, handle ownership/topology and remeshing,
choose the gauge closure for real domains, implement scalable algebra, and
verify native energy/work, conservation, rollback and temporal refinement.
No saved-state migration or production deployment belongs to this prototype.
