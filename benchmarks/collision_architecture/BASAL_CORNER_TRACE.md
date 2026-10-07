# Exact-corner homogeneous material trace

This is a numerical repair to the frozen finite basal research operator. It
does not choose a new basal policy, change a coefficient, activate a simulation
policy, or establish a complete native-world operator.

## Failure and equivalent calculation

Native control 72871, positive piece 3173, material face 44117 has area
0.002379510608176308 square metres. Solving the material-face system separately
at each rounded local quadrature point loses relative accuracy in a very small
active nodal weight. The full nine-coordinate Gram matrix disagreed with an
independent 140-digit original-coordinate reference by roughly 1.1–2.6e-9.
Increasing quadrature to order 128 passed the successive-rule test but still
had 1.77e-9 reference error. Area and rigid-only checks did not expose this
residual-coordinate error.

Let the original represented material vertices be the columns of P. The
existing velocity trace uses homogeneous coordinates alpha(q) = P^-1 q; these
coordinates are **not** normalized to sum to one. Each exact chart corner is
an unnormalized rational ray r_i. Compute

    beta_i = P^-1 r_i
    alpha_i = beta_i / |r_i|.

All Cramer determinants and beta signs use exact rational arithmetic on the
represented source vertices and chart rays. For a radial triangle rule with
positive chord weights w_i and chord length L, the same linear trace is

    alpha(q) = (sum_i w_i alpha_i) / L.

The implementation caches the corner coordinates once per positive piece and
interpolates them for each quadrature rule. The existing local frame, original
material tangent degrees of freedom, positive measure, QR factor compression,
work-conjugate transpose, and physical coefficients remain unchanged.

## Numerical guards and limits

- Negative exact corner weights reject the piece. Exact incidence stays zero;
  no small negative coefficient is clipped into the face.
- Positive ray normalization uses Decimal at at least 80 digits and again at
  32 additional digits. Both binary64 casts must agree; any exact nonzero
  coefficient that cannot survive a positive binary64 cast rejects the piece.
  Agreement is a numerical guard, not a directed-rounding certificate.
- Existing original-face containment, finite reconstruction, area, complete
  rigid metric, factor-compression and full nine-coordinate convergence gates
  remain in force. Their tolerances and maximum quadrature order are unchanged.
- The pre-existing represented-polygon compatibility path without an exact
  StableMetric chart retains its former calculation. Native positive pieces
  use the exact-chart path repaired here.

## Independent evidence

The portable fixture `tests/fixtures/basal_corner_trace_72871.json` stores all
81 entries of an independently evaluated 140-digit original-global Gram,
source provenance and tangent basis. Independent order 8 versus order 16
reference change is 6.71e-90 in the diagonal-energy-scaled matrix norm.

The repaired source's errors at quadrature orders 8, 16, 32, 64 and 128 are
1.73e-14, 2.21e-14, 1.87e-14, 2.41e-14 and 4.66e-14 respectively. The exported
piece now accepts at order 16 under the unchanged 2e-10 gate, including its
actual SI coefficient and area scaling. Tests also retain exact zero weights,
positive homogeneous weights, rejection of nonzero underflow, and rejection
of outside or modified geometry.

This establishes the captured piece's repair. It does not establish that all
8,760 pieces in control 72871, all native controls, or an evolving coupled
timestep succeed. Independent finite-order admission and whole-world resource
and completeness requirements remain separate gates.
