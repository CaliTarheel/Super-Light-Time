# Native state and force port for joint mechanics

`native_joint_state.py` exports a detached, validated mapping of an existing
native world into the joint SI coordinates. It does not advance the world or
select a new saved physics policy. This is the native-input part of the coupled
mechanics integration, not an end-to-end collision validation.

## Geometry and reference frames

All active native plate slots receive three Euler coordinates, including plates
without continental material. Persistent plate UIDs and face IDs retain explicit
maps; native owners are never silently treated as contiguous solver indices.
Shared indexed material vertices have two orthonormal tangent coordinates.
The unknown is `y = [R_m * omega_rad_s, z_m_s]`, and material velocity is
`u_i = (R_m * omega_owner) cross r_i + E_i z_i`.

The adapter copies the selected enhanced-rifting craton-core mask, authored
fully protected vertices, and the native guard at corners shared by otherwise
edge-disconnected bodies. It preserves mobile craton rims. Partial authored
kinematic response has no physical joint constitutive equivalent and is rejected.
No collision-history row is converted into an invented bilateral velocity lock.

The joint solver must interpret fixed residuals in its explicitly anchored
plate frame. Remaining rigid/residual moment gauges span only exchange
directions left by those anchors; imposing all old mean-residual rows as well
can overconstrain actual motion. This reference-frame choice and recovery of
anchor reactions belong to the fixed-coordinate solver contract. Transient
same-sheet nonpenetration guards still need event-time assembly and evolution;
the static mask here is not a substitute.

Mapping arrays are detached and read-only. A typed content signature binds
geometry, persistent identities, active slots, mechanical columns, collision
order and the relevant saved policies. `validate_source` rejects a stale source
or modified descriptor. No source initialization, remesh, owner reassignment,
contact refresh or RNG update occurs.

## One gravity load and explicit units

For the existing selected native material energy `E = ENERGY_UNIT_J * E_reduced`,
the tangent force is `F_i = -ENERGY_UNIT_J * gradient_i / R_m` in newtons.
The total-material-velocity transpose already produces both the Euler torque
and the residual load. Its rigid diagnostic is `sum_i R_m * (r_i cross F_i)`
per plate. The old native `collision` driver must therefore be excluded when
this full nodal force is used; the old separately projected sheet RHS must also
be excluded. Neither is an additional physical source.

The native `Balance.drivers` values are watts conjugate to
`x = R_m * omega_rad_s / CM_YR_M_S`. `rigid_driver_torques` converts the explicitly
provided `slab` and `ridge` blocks by `R_m / CM_YR_M_S` into newton-metres and
excludes `collision`. The caller must establish that this driver ledger belongs
to the native fixed Earth radius; the conversion rejects other radii because
`Balance` uses its fixed radius in defining its input units. It must also belong
to the same frozen state. Creating a native Balance can update contact defaults,
so any future extraction must run on a detached state with provenance checks.

An active entry registry is rejected by the gravity port until the complete
entry/stack potential is available. The adapter exports the current native
density/ordered-overlap potential, including its documented small-unordered-area
approximation; it does not make that approximation a valid bottom-layer order
for the separate basal-domain partition.

## Validation and remaining integration

Tests independently compare force work against a finite difference of the
conserved-volume native energy, check torque units through virtual power, and
exercise ocean-only slots, mobile rims, point-only corner guards, stale identity,
entry rejection and velocity round trips. A separate read-only real-checkpoint
probe records actual geometry size, selected masks and GPE work without solving
or moving the production world.

Sheet coefficients, complete finite bottom allocation, finite weld/interface
provenance, other slab/hinge/transform resistance, contact admission, attachment
and conservative finite-step evolution remain required. Missing laws are not
zero forces. This follows Scotese Rule I's explicit driving/resisting force
accounting and Rule XI's consequential collisions; the reduced native
constitutive assumptions remain model choices, not geological laws.
