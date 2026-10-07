"""Redistribute a bounded contact response through its viscous material body.

The preferred boundary-driven velocity remains the unloaded optimum. Active
column bounds alter that response by minimizing one half (v-v*) A (v-v*),
where A is the same positive viscous operator. This is a reduced accommodation
rule, not an extra force or a change to the plate rotations. Endpoint reactions
can redirect shortening into lateral escape instead of stopping the entire
connected body at the first saturated column. Exact geometry is still checked.
"""
from __future__ import annotations
import numpy as np
import viscous_sheet
import bounded_gravity

# The SQP active-set budget for contact accommodation. It was a flat 32, the
# same shape of cap the face-scaled gravity default replaces, and it overrode
# that default here. In a production run at 60 Myr, contact region 3 (931 faces,
# 6.2e6 km2) needed 38 simultaneous area bounds at 2 Myr, 34 at 1 Myr and 34 at
# 0.5 Myr; every one of those solves succeeds with the scaled budget (232 here)
# and still satisfies every unchanged area bound. Falling one to six bounds
# short made the coupled step retry down to 0.0625 Myr. None defers to
# bounded_gravity's max(MINIMUM_ACTIVE_BUDGET, ACTIVE_SET_FACE_FRACTION*faces).
ACTIVE_SET_BUDGET = None


def redistribute(points, faces, area, preferred, rigid, minimum, maximum, dt, *,
                 radius=6371., length_km=100., viscosity_weights=None,
                 iterations=1024, tolerance=1e-8, max_speed_km_myr=100., numerical_policy=None, contact_return_context=None):
    if numerical_policy is not None:
        from numerical_accuracy import NumericalPolicy
        if not isinstance(numerical_policy,NumericalPolicy):raise ValueError('Unsupported contact numerical accuracy policy.')
        numerical_policy.validate(len(faces));tolerance=numerical_policy.outer_tolerance
    if contact_return_context is not None:
        if (not isinstance(contact_return_context, dict)
                or numerical_policy is None or not numerical_policy.has_hard_bounds
                or not np.array_equal(points, contact_return_context.get('local_points'))
                or not np.array_equal(faces, contact_return_context.get('local_faces'))
                or not np.array_equal(rigid, contact_return_context.get('local_rigid'))
                or radius != contact_return_context.get('radius') or dt != contact_return_context.get('stage_dt')):
            raise ValueError('Mapped redistribution requires exact inputs from its fixed original contact stage.')
        # Validate only the immutable geometry association before constructing
        # the unchanged physical P1 context. It supplies no force or metric.
        bounded_gravity.validate_contact_return_context(points, faces, rigid, dt, radius, contact_return_context)
    context=viscous_sheet.prepare(points,faces,area,length_km,radius=radius,
                                 viscosity_weights=viscosity_weights)
    rhs=viscous_sheet.apply(context,preferred)
    rhs[rigid]=0.
    force=np.divide(rhs,context['data'][:,None],out=np.zeros_like(rhs),where=context['data'][:,None]>0)
    initial=dict(converged=True,iterations=0,maximum_iterations=iterations,relative_residual=0.,
        tolerance=tolerance,failure_reason=None,
        formulation='preferred contact response in its original viscous metric')
    endpoint,velocity,solver,diagnostic=bounded_gravity.solve_endpoint(
        points,faces,area,force,preferred,initial,rigid,minimum,maximum,dt,
        radius=radius,length_km=length_km,viscosity_weights=viscosity_weights,
        iterations=iterations,tolerance=tolerance,max_speed_km_myr=max_speed_km_myr,
        max_sqp=32,max_active=ACTIVE_SET_BUDGET,feasibility_tolerance=1e-12,
        **({} if numerical_policy is None else dict(numerical_policy=numerical_policy,
            quality_minimum=np.minimum(.15,bounded_gravity._joint_quality(points,faces)[0]*.5))),
        **({} if contact_return_context is None else dict(contact_return_context=contact_return_context)))
    difference=velocity-preferred
    diagnostic.update(model='constrained viscous accommodation of contact motion',
        face_index_scope='local contact region faces before gravity or material adaptation',
        response_change_metric=float(np.sum(difference*viscous_sheet.apply(context,difference))),
        preferred_velocity_norm=float(np.linalg.norm(preferred)),
        accommodated_velocity_norm=float(np.linalg.norm(velocity)))
    return endpoint,velocity,solver,diagnostic
