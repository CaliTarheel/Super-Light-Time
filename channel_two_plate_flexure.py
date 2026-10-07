"""Read-only nonlocal two-plate contact benchmark on an anchored strip.

Both descending and overriding plates pay curvature energy. The overriding
plate also pays a density-contrast foundation energy. This one-dimensional
strip has externally supplied material anchors and constitutive parameters;
it is not a native trench geometry, plastic rheology, or owner transaction.
"""

import numpy as np


def anchored_two_plate_contact(preferred_depth_m, unloaded_upper_base_depth_m,
                               buoyancy_pa, lower_rigidity_n_m,
                               upper_rigidity_n_m, upper_foundation_n_m3,
                               spacing_m, *, force_tolerance=1e-8,
                               max_iterations=400, free_upper_hinge=False,
                               contact_mask=None):
    """Minimize two bending plates subject to nonpenetration and depth >= 0.

    `z` is incoming top depth and `w` is upper basal uplift, so contact needs
    `z + w >= H`. Both ends of `z` follow the independently supplied preferred
    path. The far end of `w` is anchored; by default its hinge end is anchored
    too. `free_upper_hinge` lets that edge uplift and pay bending/foundation
    work when a thick upper plate reaches the shallow trench contact. The
    returned reaction is conjugate to the unloaded basal reference `H` only
    where `contact_mask` identifies an actual upper/lower overlap. Omitting
    the mask retains the original full-strip benchmark.
    """
    preferred = np.asarray(preferred_depth_m, float)
    base = np.asarray(unloaded_upper_base_depth_m, float)
    buoyancy = np.asarray(buoyancy_pa, float)
    dl = float(lower_rigidity_n_m)
    du = float(upper_rigidity_n_m)
    foundation = float(upper_foundation_n_m3)
    spacing = float(spacing_m)
    tolerance = float(force_tolerance)
    if (preferred.ndim != 1 or len(preferred) < 5
            or base.shape != preferred.shape
            or buoyancy.shape not in ((), preferred.shape)
            or not np.isfinite(preferred).all() or not np.isfinite(base).all()
            or not np.isfinite(buoyancy).all()
            or np.any(preferred < 0.) or np.any(base < 0.)
            or not all(np.isfinite(value) and value > 0.
                       for value in (dl, du, foundation, spacing, tolerance))
            or not isinstance(max_iterations, (int, np.integer))
            or max_iterations <= 0 or type(free_upper_hinge) is not bool):
        raise ValueError('Two-plate contact needs finite depths, rigidity, foundation and spacing.')
    n = len(preferred)
    m = n - 2
    mask = np.ones(n, bool) if contact_mask is None else np.asarray(contact_mask)
    if mask.shape != (n,) or mask.dtype.kind != 'b' or not np.any(mask):
        raise ValueError('Two-plate contact needs a boolean physical contact mask.')
    if ((mask[-1] and base[-1] > preferred[-1] + 1e-9)
            or (mask[0] and not free_upper_hinge
                and base[0] > preferred[0] + 1e-9)):
        raise ValueError('Two-plate contact is infeasible at an anchored end.')
    buoyancy = np.broadcast_to(buoyancy, preferred.shape)
    first_upper = 0 if free_upper_hinge else 1
    upper_nodes = np.arange(first_upper, n - 1)
    contact_nodes = np.flatnonzero(mask[first_upper:n - 1]) + first_upper
    upper_count = len(upper_nodes)
    contact_count = len(contact_nodes)
    if not contact_count:
        raise ValueError('Two-plate contact needs a resolved free contact node.')
    weights = np.full(n, spacing)
    weights[[0, -1]] *= .5
    curvature = np.zeros((m, n))
    rows = np.arange(m)
    curvature[rows, rows] = 1. / spacing**2
    curvature[rows, rows + 1] = -2. / spacing**2
    curvature[rows, rows + 2] = 1. / spacing**2
    lower_full = dl * spacing * curvature.T @ curvature
    upper_full = du * spacing * curvature.T @ curvature
    lower = lower_full[1:-1, 1:-1]
    upper = upper_full[np.ix_(upper_nodes,upper_nodes)] + np.diag(foundation * weights[upper_nodes])
    hessian = np.zeros((m + upper_count, m + upper_count))
    hessian[:m, :m] = lower
    hessian[m:, m:] = upper
    linear = np.r_[-lower @ preferred[1:-1]
                   + weights[1:-1] * buoyancy[1:-1], np.zeros(upper_count)]
    constraints = np.zeros((contact_count+m,m+upper_count))
    for row,node in enumerate(contact_nodes):
        if node:
            constraints[row,node-1]=1.
        constraints[row,m+node-upper_nodes[0]]=1.
    constraints[contact_count+np.arange(m),np.arange(m)]=1.
    bounds = np.r_[base[contact_nodes] - np.where(contact_nodes==0,preferred[0],0.),
                   np.zeros(m)]
    initial_base = np.where(mask[upper_nodes], base[upper_nodes], 0.)
    current = np.r_[preferred[1:-1],
                    np.maximum(initial_base - preferred[upper_nodes], 0.)]
    active = set(np.flatnonzero(np.abs(constraints @ current - bounds) <= 1e-7))
    force_scale = max(float(np.max(np.abs(linear))),
                      float(np.max(np.abs(hessian @ current))), 1.)
    feasibility_m = 1e-6
    for iteration in range(max_iterations):
        chosen = sorted(active)
        if chosen:
            a = constraints[chosen]
            kkt = np.block([[hessian, -a.T],
                            [a, np.zeros((len(chosen), len(chosen)))]])
            solved = np.linalg.solve(kkt, np.r_[-linear, bounds[chosen]])
            target = solved[:m + upper_count]
            multipliers = solved[m + upper_count:]
        else:
            target = np.linalg.solve(hessian, -linear)
            multipliers = np.empty(0)
        direction = target - current
        slack = constraints @ current - bounds
        trial_slack = constraints @ target - bounds
        blockers = [row for row in range(len(bounds))
                    if row not in active and trial_slack[row] < -feasibility_m
                    and (constraints[row] @ direction) < 0.]
        if blockers:
            fractions = np.asarray([slack[row] / -(constraints[row] @ direction)
                                    for row in blockers])
            first = int(np.argmin(fractions))
            step = float(np.clip(fractions[first], 0., 1.))
            current = current + step * direction
            active.add(blockers[first])
            continue
        current = target
        if len(multipliers) and float(np.min(multipliers)) < -tolerance * force_scale:
            active.remove(chosen[int(np.argmin(multipliers))])
            continue
        reaction = np.zeros(contact_count+m)
        reaction[chosen] = multipliers
        residual = hessian @ current + linear - constraints.T @ reaction
        force_scale = max(force_scale, float(np.max(np.abs(hessian @ current))),
                          float(np.max(np.abs(constraints.T @ reaction))))
        if (float(np.linalg.norm(residual, ord=np.inf)) > tolerance * force_scale
                or float(np.min(constraints @ current - bounds)) < -feasibility_m
                or float(np.min(reaction)) < -tolerance * force_scale):
            raise ValueError('Two-plate contact did not balance force and nonpenetration.')
        depth = preferred.copy()
        depth[1:-1] = current[:m]
        uplift = np.zeros(n)
        uplift[upper_nodes] = current[m:]
        lower_offset = curvature @ (depth - preferred)
        upper_curvature = curvature @ uplift
        lower_work = .5 * dl * spacing * float(lower_offset @ lower_offset)
        upper_work = .5 * du * spacing * float(upper_curvature @ upper_curvature)
        foundation_work = .5 * foundation * float(weights @ (uplift**2))
        buoyancy_work = float(weights @ (buoyancy * depth))
        pressure = np.zeros(n)
        pressure[contact_nodes] = reaction[:contact_count] / weights[contact_nodes]
        basal_force = np.zeros(n)
        basal_force[contact_nodes] = reaction[:contact_count]
        return dict(depth_m=depth, upper_uplift_m=uplift,
                    reaction_pa=pressure,
                    basal_reference_derivative_n_per_m=basal_force,
                    hinge_line_reaction_n_per_m=(float(reaction[0])
                                                 if free_upper_hinge
                                                 and contact_nodes[0] == 0 else 0.),
                    contact=pressure > tolerance * force_scale / spacing,
                    lower_bending_j_per_m=lower_work,
                    upper_bending_j_per_m=upper_work,
                    upper_foundation_j_per_m=foundation_work,
                    buoyancy_j_per_m=buoyancy_work,
                    energy_j_per_m=lower_work + upper_work
                                   + foundation_work + buoyancy_work,
                    free_force_residual_n_per_m=float(
                        np.linalg.norm(residual, ord=np.inf)),
                    minimum_gap_m=float(np.min(depth[mask] + uplift[mask] - base[mask])),
                    iterations=iteration,
                    scope='read-only anchored two-plate flexure; no native geometry or source commit')
    raise ValueError('Two-plate contact active set did not converge.')
