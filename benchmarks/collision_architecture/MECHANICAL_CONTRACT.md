# Current collision mechanics contract

This benchmark describes the existing split implementation. It does **not**
implement or validate a shared plate-and-sheet force solve. The baseline source
was audited at `86a79b3e5b693f93a45e197dd0dae4b6c249e8f2`; benchmark results must
record their own actual source identity and effective policies.

## Units and conjugate quantities

Let `S = 365.25 * 86400 * 1e6` seconds/Myr, `v0 = 0.01/(365.25*86400)` m/s,
and `R_m = 6371e3` m. The plate unknown is

```text
x = omega_SI * R_m / v0
omega_rad_per_Myr = x * v0 * S / R_m
Phi_plate(x) = 0.5*x.K.x - T.x + sum(D_e(x))
```

`x` is dimensionless, numerically expressing the reference equatorial speed in
cm/year. `Phi_plate`, `x.K.x` and `T.x` have units W. Thus `T` is a power-scaled
generalized load; physical torque is `T * R_m/v0` in N m. The actual plastic
resisting power is `x.grad(D_e)`, which generally differs from its regularized
dissipation potential `D_e`.

Source: [plate_balance.py](../../plate_balance.py), lines19–39,60–64,385–407,
554–587,775–855,856–892 and1000–1019.

The sheet uses velocity `u` in km/Myr, lumped vertex area `a` in km² and strain
rate in 1/Myr. Its objective is

```text
Q(u) = 0.5*sum(a*|u-target|^2)
       + 0.5*sum(A*L^2*w*(D:D + div(u)^2))
       - sum(a*body.u)
```

`Q`, `viscous_dissipation_quadratic` and `normalized_body_work` have units
**km⁴/Myr²**, not J or W. If a physical basal-drag coefficient `beta` in Pa s/m
is supplied, the conversions are

```text
physical quadratic power [W] = normalized quadratic * beta * 1e12 / S^2
nodal reaction [N] = normalized nodal residual * beta * 1e9 / S
L_m^2 = 2 * eta_Pa_s * H_m / beta
```

These are dimensional conversions, not a calibration. The contact sheet receives
`L` and dimensionless viscosity weights without an absolute `eta/beta` pair.
Gravity declares that pair; its default `eta=1e23`, `H=100 km`, `L=100 km` imply
`beta=2e18 Pa s/m`. Plate basal drag is `1e15 Pa s/m` times a crust-class keel
factor. These operators act on different modeled velocities; a common physical
allocation has not yet been established.

Sources: [viscous_sheet.py](../../viscous_sheet.py), lines1–10,104–107,204–297;
[gravitational_relaxation.py](../../gravitational_relaxation.py), lines285–301,
401–405; [plate_balance.py](../../plate_balance.py), lines72–77,385–407.

For the gravity functional, `E_reduced` has km⁴ and the vertex gradient is with
respect to unit-sphere coordinates. With
`K = rho_c*g*(1-rho_c/rho_m)` in N/m³:

```text
E_J = K * 1e12 * E_reduced
P_W = K * 1e12 / S * sum(gradient.u) / R_km
```

Source: [gravitational_relaxation.py](../../gravitational_relaxation.py),
lines1–20,34–43,340–370,526–530 and683.

## Existing connections and missing reciprocal reactions

- **GPE already has a shared rigid/residual decomposition.** Its per-owner rigid
  torque enters the plate balance; its residual gradient drives gravity's
  internal sheet relaxation. Adding another whole GPE driver would double count
  it. This instantaneous gradient identity is not a finite-step work balance.
  See `plate_balance.py:554` and `gravitational_relaxation.py:309,497`.
- **Contact deformation follows solved Euler motion.** The target is half the
  relative Euler velocity toward the local mean, weighted over a contact belt.
  Some normal velocities and craton residual velocities are constrained.
  Final shortening need not split equally: strength, geometry, constraints and
  gravity alter it. See [deforming_regions.py](../../deforming_regions.py),
  lines364–410 and567–607.
- **The sheet reaction is not returned to the plate solve.** Its convergence
  check projects out constrained nodal residuals, and it returns velocities plus
  scalar diagnostics. Both hard-constraint reactions and the distributed
  `a*target` loading require reciprocal accounting. See `viscous_sheet.py:233–297`
  and [native_material_evolution.py](../../native_material_evolution.py),
  lines275–319.
- **Interface shear and local weld laws use rigid plate rates.** Their velocity
  maps exclude sheet residual velocities. They have plate action/reaction and
  resisting-power checks, but not a total-material-velocity closure. See
  [collision_interface.py](../../collision_interface.py), lines215–239, and
  `plate_balance.py:734–747`.
- **Force projection is not a velocity constraint.** Constraints can turn a
  zero-rigid-moment residual force into motion containing a rigid component.
  Gravity acknowledges this and checks its full energy gradient
  (`gravitational_relaxation.py:526–530`). A shared representation must define a
  gauge separating rigid motion from internal motion and include its reactions.
- **Current finite-step GPE work is an approximation.** The diagnostic at
  `native_material_evolution.py:337–353` uses first gravity-stage torque times
  omega times dt, after prescribed motion. It is dimensionally J but is not a
  quadrature of the actual torque throughout that trajectory.

## What the baseline can establish

The benchmark can measure plate-subsystem stationarity and power balance,
sheet-subsystem stationarity, gravity descent, actual spherical strain,
conserved volume/provenance, same-sheet nonpenetration, declared stack overlap,
local contact speeds and refinement/runtime behavior. Instantaneous `T.x`
metrics must be named **power**, not accumulated work; angular-speed reduction
must not be called local shortening.

The synthetic plate driver/drag must be identified as fixture boundary
conditions. The component fixture has no native slab/ocean source or geological
event transaction. In particular, its weld-only constructor has no boundary
strength samples and therefore uses the default weld-strength multiplier1
(`plate_balance.py:763–771`); changing sheet viscosity alone does not exercise
production boundary-strength mapping. Conservation and slowdown cannot certify
complete collision mechanics.

## Required next implementation

Define a physical mechanical functional on **both plate rotations and sheet
residual velocities**, with each force/resistance assigned exactly once. State
which terms use total material velocity, inter-sheet slip, or a genuinely
external prescribed load. Supply physical sheet coefficients and a basal-drag
allocation; define the residual-velocity gauge; carry distributed loading,
constraints, interface/weld operators and GPE through a symmetric block solve or
an equivalent Schur reduction.

Before evolved claims, verify frozen-geometry derivatives independently,
cross-block reciprocity, action/reaction, passive resisting power and total
virtual-power balance. Then require strength-contrast trajectories, changing
contact constraints, timestep/mesh refinement and restart agreement. This shared
solve and its acceptance gates are **future work**, not results of this baseline.
