"""Isolated finite-step constrained viscous objective; no production activation."""
from __future__ import annotations

import math
import numpy as np
import viscous_sheet as sheet
from material_surface import spherical_face_areas
from ridge_geometry import rotate

VERSION = 1

# Largest relative face-area bound violation treated as feasible.
#
# The endpoint areas are evaluated from absolute vertex coordinates on a sphere,
# so their determinant cancellation leaves a relative error floor well above
# machine epsilon. Requiring feasibility at 1e-12 demanded better than that
# floor: converged solves with a KKT residual of 1e-10 and a worst violation of
# 4e-11 were rejected, exhausted the SQP budget and dropped the whole
# relaxation interval. This bound is the shared acceptance scale for the solver
# and for the relaxation search that consumes its geometry; the two must agree
# or an endpoint accepted here is rejected there and the step cannot advance.
#
# Volume error stays negligible: 1e-9 relative per face per step is ~6e-6 km^2
# on a 6200 km^2 face, and 500 steps of systematic drift remain far below the
# area-evaluation roundoff itself.
FEASIBILITY_TOLERANCE = 1e-9

# Iteration budget of the inner viscous inverses, as a multiple of the caller's.
#
# The inverses are the caller's own operator solved to 100 times its stationarity
# tolerance, and preconditioned CG pays a roughly fixed number of iterations per
# decade on a given mesh. A shared budget left them the smaller margin: at 122 Myr
# in run 20260917-0012-nonpenetration the unconstrained gravity solve needed 897
# of 1024 iterations and the identical inverse needed 1051, so every trial of the
# relaxation interval failed. CG stops at convergence, so any solve that converged
# under the old budget is unchanged bit for bit; only failures are affected.
PRECISE_ITERATION_FACTOR = 4

# Draft proposal: scale the default active-set work capacity with face count.
# Dense Schur rows and cached viscous inverses impose a numerical work cost.
# The quarter-face factor has not been validated on the failed run.
# contact_response.redistribute used to pass an explicit max_active=32 that
# this default could not reach; it now defers to it (ACTIVE_SET_BUDGET = None,
# measured on a captured 931-face contact region). Aggregate limited-face
# counts do not establish a local solver's peak active-set size. See
# ACTIVE-SET-CAPACITY-REVIEW.md before treating this as a recovery fix.
ACTIVE_SET_FACE_FRACTION = .25
MINIMUM_ACTIVE_BUDGET = 128

# Smallest eigenvalue kept by the strictly convex working-set model, as a
# fraction of the exact viscous objective curvature (unit in the whitened
# active-support coordinates of the joint solver). It is used only by the one
# convex restart after an original joint solve failed its numerical search, a
# path that formerly raised. It bounds the model step amplification; it is not
# an acceptance tolerance. See CONVEX-RESTART.md.
CONVEX_MODEL_CURVATURE_MARGIN = .1

class ConstraintSolveError(ValueError):
    pass


def area_violation_summary(current, minimum, maximum):
    """Absolute endpoint-bound misses, without cancellation or per-face masking."""
    misses=np.maximum(minimum-current,0.)+np.maximum(current-maximum,0.)
    return dict(total_area_violation_km2=float(np.sum(misses)),
                maximum_face_area_violation_km2=float(np.max(misses,initial=0.)),
                violated_faces=int(np.count_nonzero(misses)))


def _area_failure(message, current, minimum, maximum):
    measured=area_violation_summary(current,minimum,maximum)
    return ConstraintSolveError(message+' Current endpoint area-bound mismatch: '
        f"{measured['total_area_violation_km2']:.9g} km2 total absolute, "
        f"{measured['maximum_face_area_violation_km2']:.9g} km2 worst face, "
        f"{measured['violated_faces']} faces. This is a constraint discrepancy, not lost crust.")


def _dependent_inequality_pivot(rows, target, multipliers, *, with_direction=False):
    """Retire only a dual reaction that reaches zero along a physical null row.

    Violated inequalities are a working-set guess, not simultaneous equalities.
    Their equality targets can disagree even though one bound can remain slack.
    A certified row dependency permits a dual descent direction without changing
    the represented physical reaction. Follow it to the first zero multiplier,
    then solve again. Near-singular viscous compliance alone is not a dependency.
    A one-sided null direction may instead certify linearized infeasibility; it
    never authorizes dropping a physical constraint or accepting an endpoint.
    """
    flat=np.asarray(rows).reshape(len(rows),-1)
    eps=np.finfo(float).eps
    left,singular,_=np.linalg.svd(flat,full_matrices=len(rows)>flat.shape[1])
    rank=int(np.count_nonzero(singular>16*eps*singular[0]))
    null=left[:,rank:]
    if not null.shape[1]:return None
    # Project onto the actual area-row nullspace. The Schur least-squares
    # residual also contains resolved numerical error from its PCG columns;
    # using that raw residual would need an unjustifiably looser null check.
    direction=null@(null.T@target)
    scale=float(np.sum(np.abs(direction)))
    if scale==0.:return None
    reaction=float(np.linalg.norm(direction@flat))
    if reaction>64*eps*scale:return None
    gain=float(direction@target)
    if gain<=64*eps*float(np.linalg.norm(direction))*float(np.linalg.norm(target)):return None
    negative=np.flatnonzero(direction < -64*eps*float(np.max(np.abs(direction))))
    if not len(negative):return None
    # The caller already removes significant negative reactions. Its admitted
    # arithmetic-scale negatives are the same zero reactions clamped below.
    crossing=np.maximum(multipliers[negative],0.)/(-direction[negative])
    at=int(negative[int(np.argmin(crossing))])
    proof=dict(physical_row_rank=rank,working_set_rows=len(rows),
        normalized_null_reaction=reaction/scale,dual_directional_gain=gain,
        step_to_zero=float(max(float(multipliers[at]),0.)/(-direction[at])))
    return (at,proof,direction) if with_direction else (at,proof)


class SpeedBoundError(ConstraintSolveError):
    """The candidate needs motion faster than the caller's speed bound allows.

    Distinguished from ordinary infeasibility because the two want opposite
    handling. Infeasibility over an interval is worth retrying over a shorter
    one; a speed bound is an assertion about the candidate itself, and must
    reach the caller rather than be retried or scaled quietly into range.
    """


class InverseSolveError(ConstraintSolveError):
    """A precise viscous inverse did not converge on the current geometry.

    Its operator and right-hand side depend on the start-of-stage geometry,
    never on the interval, so a shorter trial repeats the identical solve.
    Like a speed bound, it must reach the caller instead of being retried.
    """


class ActiveSetBudgetError(ConstraintSolveError):
    """More simultaneous area bounds are needed than the budget allows, and a
    shorter interval cannot reduce the count.

    Raised when the supplied entry-area array violates more bounds than the
    capacity. That count is fixed for retries of this solver on identical
    inputs; it does not prove that a complete coupled retry has identical
    entry geometry, or that every violated constraint needs a final reaction.
    Ordinary motion-driven exhaustion remains a ConstraintSolveError. The
    actual 160-to-162 Myr contact failure has not validated this classification.
    """


def _inverse_failure(info):
    return ('Constrained gravity inverse solve failed its stricter internal stationarity gate: '
            f"{info['iterations']}/{info['maximum_iterations']} iterations, relative residual "
            f"{info['relative_residual']}, required {info['tolerance']}. It depends only on the "
            'current geometry; a shorter interval repeats the same solve.')


def endpoint(points, velocity, dt, radius):
    result=rotate(points,np.cross(points,velocity)*(dt/radius))
    return result/np.linalg.norm(result,axis=1)[:,None]


def validate_contact_return_context(points,faces,rigid,dt,radius,context):
    """Validate transient, immutable full-stage identities; never a policy upgrade."""
    if not isinstance(context,dict) or context.get('kind')!='contact_final_return_geometry_context' or context.get('version')!=1:
        raise ValueError('Unsupported contact return geometry context.')
    names=('full_points','full_faces','full_owner','full_face_owner','full_omega','full_rotation_vectors',
        'full_rigid','full_rigid_vertices','full_region_id','global_vertex_indices',
        'global_face_indices','local_points','local_faces','active_mask','local_rigid')
    if any(name not in context or not isinstance(context[name],np.ndarray) or context[name].flags.writeable or not context[name].flags.owndata for name in names):
        raise ValueError('Contact return context requires owning immutable array identities.')
    p=np.asarray(points);f=np.asarray(faces);r=np.asarray(rigid,bool)
    full=context['full_points'];ff=context['full_faces'];vi=context['global_vertex_indices'];fi=context['global_face_indices']
    if (p.ndim!=2 or p.shape[1]!=3 or p.dtype.kind!='f' or f.ndim!=2 or f.shape[1]!=3 or f.dtype.kind not in 'iu'
        or np.any(f<0) or np.any(f>=len(p)) or full.ndim!=2 or full.shape[1]!=3 or full.dtype.kind!='f' or ff.ndim!=2 or ff.shape[1]!=3
        or ff.dtype.kind not in 'iu' or vi.dtype.kind not in 'iu' or fi.dtype.kind not in 'iu'
        or vi.shape!=(len(p),) or fi.shape!=(len(f),) or len(np.unique(vi))!=len(vi) or len(np.unique(fi))!=len(fi)
        or np.any(vi<0) or np.any(vi>=len(full)) or np.any(fi<0) or np.any(fi>=len(ff))
        or np.any(ff<0) or np.any(ff>=len(full))):
        raise ValueError('Contact return global/local indexing is invalid.')
    owner=context['full_owner'];fo=context['full_face_owner'];fixed=context['full_rigid'];labels=context['full_region_id'];omega=context['full_omega']
    active=context['active_mask'];lr=context['local_rigid']
    if (owner.shape!=(len(full),) or fo.shape!=(len(ff),) or owner.dtype.kind not in 'iu' or fo.dtype.kind not in 'iu'
        or labels.shape!=(len(full),) or labels.dtype.kind not in 'iu' or fixed.shape!=(len(full),) or fixed.dtype.kind!='b'
        or active.shape!=(len(p),) or active.dtype.kind!='b' or lr.shape!=(len(p),) or lr.dtype.kind!='b'
        or r.shape!=(len(p),) or np.any(lr & ~r)
        or context['full_rotation_vectors'].shape!=full.shape or context['full_rigid_vertices'].shape!=full.shape
        or any(context[name].dtype!=full.dtype for name in ('full_rotation_vectors','full_rigid_vertices','local_points','full_omega'))
        or omega.ndim!=2 or omega.shape[1]!=3 or omega.dtype.kind!='f' or np.any(owner<0) or np.any(owner>=len(omega))
        or context['local_points'].shape!=p.shape or context['local_faces'].shape!=f.shape
        or not all(np.isfinite(context[name]).all() for name in ('full_points','full_omega','full_rotation_vectors','full_rigid_vertices','local_points'))):
        raise ValueError('Contact return stage geometry/owner/mask identity is invalid.')
    region=context.get('region_id')
    if (not isinstance(region,(int,np.integer)) or isinstance(region,(bool,np.bool_)) or region<0
        or not np.array_equal(context['local_points'],p) or not np.array_equal(context['local_faces'],f)
        or not np.array_equal(full[vi],p) or not np.array_equal(ff[fi],vi[f])
        or not np.array_equal(vi,np.unique(ff[fi])) or not np.array_equal(labels<0,fixed)
        or not np.array_equal(owner[ff],np.broadcast_to(fo[:,None],ff.shape))
        or not np.array_equal(lr,fixed[vi]) or not np.array_equal(active,(labels[vi]==region)&~lr)
        or np.any(~active & ~lr) or np.any(~fixed[ff[fi]] & (labels[ff[fi]]!=region))
        or not np.isfinite(context.get('radius',np.nan)) or context['radius']!=radius
        or not np.isfinite(context.get('stage_dt',np.nan)) or context['stage_dt']<=0 or dt<=0 or dt>context['stage_dt']):
        raise ValueError('Contact return context is stale, cross-region, or misaligned with its original stage.')
    expected_faces=np.flatnonzero(np.max(labels[ff],axis=1)==region)
    if not np.array_equal(fi,expected_faces):raise ValueError('Contact return context omits or relabels this original region face set.')
    if 'full_face_ids' in context:
        ids=context['full_face_ids']
        if (not isinstance(ids,np.ndarray) or ids.flags.writeable or not ids.flags.owndata
            or ids.shape!=(len(ff),) or ids.dtype.kind not in 'iu'):
            raise ValueError('Contact return material face identities are not immutable/aligned.')
    if not np.array_equal(context['full_rotation_vectors'],omega[owner]*context['stage_dt']):
        raise ValueError('Contact return rotations differ from the fixed original owner motion.')
    rigid_raw=rotate(full,context['full_rotation_vectors']);rigid_norm=np.linalg.norm(rigid_raw,axis=1)
    if (not np.isfinite(rigid_norm).all() or np.any(rigid_norm<=0)
        or not np.array_equal(context['full_rigid_vertices'],rigid_raw/rigid_norm[:,None])):
        raise ValueError('Contact return rigid preview differs from the original full-layout owner normalization.')
    return context


def contact_return_map(q_local,contact_return_context):
    """Preview the exact full-layout final owner map; retain local response separately.

    This follows the caller's actual active-only assembly, unchanged Rodrigues
    rotation, normalization and exact rigid override before measuring the original
    global face batch. It does not move vertices after solving or alter admission.
    """
    context=contact_return_context;q=np.asarray(q_local)
    vi=context['global_vertex_indices'];fi=context['global_face_indices'];active=context['active_mask']
    if q.shape!=context['local_points'].shape or q.dtype.kind!='f' or not np.isfinite(q).all():
        raise ConstraintSolveError('Invalid proposed local contact return geometry.')
    assembled=context['full_points'].copy();assembled[vi[active]]=q[active]
    raw=rotate(assembled,context['full_rotation_vectors']);norm=np.linalg.norm(raw,axis=1)
    if not np.isfinite(norm).all() or np.any(norm<=0):
        raise ConstraintSolveError('Degenerate proposed owner-return normalization.')
    returned=raw/norm[:,None];returned[context['full_rigid']]=context['full_rigid_vertices'][context['full_rigid']]
    area=spherical_face_areas(returned,context['full_faces'],context['radius'])[fi]
    if not np.isfinite(area).all() or np.any(area<=0):
        raise ConstraintSolveError('Invalid proposed owner-return spherical areas.')
    cache=dict(assembled_local_points=assembled[vi],full_returned_points=returned,
        owner_pre_normalized=raw[vi],owner_norm=norm[vi],returned_local_points=returned[vi],
        rotation_vectors=context['full_rotation_vectors'][vi],active_mask=active,
        local_rigid=context['full_rigid'][vi])
    return returned[vi],area,cache


def _contact_unit_adjoint(raw,gradient,*,normalized=None,norm=None):
    """Continuous adjoint of the actual normalization, at its measured norm."""
    norm=np.linalg.norm(raw,axis=-1,keepdims=True) if norm is None else norm
    if not np.isfinite(norm).all() or np.any(norm<=0):raise ConstraintSolveError('Invalid contact normalization derivative domain.')
    q=raw/norm if normalized is None else normalized
    return (gradient-q*np.sum(q*gradient,axis=-1,keepdims=True))/norm


def _contact_fixed_rotation_adjoint(rotation,gradient):
    """Transpose of the unchanged fixed-owner Rodrigues point map."""
    theta=np.linalg.norm(rotation,axis=-1,keepdims=True)
    return (np.cos(theta)*gradient+np.sinc(theta/np.pi)*np.cross(gradient,rotation)
        +.5*np.sinc(theta/(2*np.pi))**2*rotation*np.sum(rotation*gradient,axis=-1,keepdims=True))


def _contact_rotation_vector_adjoint(points,rotation,gradient):
    """General Rodrigues adjoint in its rotation vector; p need not be exactly unit.

    This differentiates the retained real-valued expression, including its
    measured input p and both normalizations. It is not a derivative of rounding
    discontinuities or a binary64 feasibility guarantee.
    """
    theta=np.linalg.norm(rotation,axis=-1);s=np.sinc(theta/np.pi);c=.5*np.sinc(theta/(2*np.pi))**2
    # Series evaluate these removable singularities without subtracting nearly
    # equal O(theta^2) terms. This is a derivative kernel, not an area tolerance.
    b=np.empty_like(theta);d=np.empty_like(theta);small=theta<1e-2;z=theta[small]**2
    b[small]=-1/3+z/30-z*z/840+z*z*z/45360
    d[small]=-1/12+z/180-z*z/6720+z*z*z/453600
    t=theta[~small];b[~small]=(t*np.cos(t)-np.sin(t))/t**3
    d[~small]=(t*np.sin(t)-4*np.sin(t*.5)**2)/t**4
    gp=np.sum(gradient*points,axis=-1);gr=np.sum(gradient*rotation,axis=-1);rp=np.sum(rotation*points,axis=-1)
    triple=np.sum(gradient*np.cross(rotation,points),axis=-1)
    return (-s[...,None]*gp[...,None]*rotation+s[...,None]*np.cross(points,gradient)
        +b[...,None]*triple[...,None]*rotation+c[...,None]*(rp[...,None]*gradient+gr[...,None]*points)
        +d[...,None]*(gr*rp)[...,None]*rotation)


def _contact_endpoint_adjoint(p,v,faces,gradient,dt,radius,rigid,*,return_cache=None):
    """Pull actual restored local or returned-world gradients to original P1 DOFs."""
    pf=p[faces];vf=v[faces];scale=dt/radius;rotation=np.cross(pf,vf)*scale;g=gradient
    if return_cache is not None:
        g=_contact_unit_adjoint(return_cache['owner_pre_normalized'][faces],g,
            normalized=return_cache['returned_local_points'][faces],norm=return_cache['owner_norm'][faces,None])
        g=_contact_fixed_rotation_adjoint(return_cache['rotation_vectors'][faces],g)
        g=np.where((return_cache['active_mask'][faces]&~return_cache['local_rigid'][faces])[...,None],g,0.)
    # Use the same full local endpoint batch that geometry() evaluates, before
    # its rigid restoration; restored corners contribute exactly zero columns.
    raw=rotate(p,np.cross(p,v)*scale)[faces];g=_contact_unit_adjoint(raw,g)
    result=scale*np.cross(_contact_rotation_vector_adjoint(pf,rotation,g),pf)
    result-=pf*np.sum(result*pf,axis=2)[:,:,None]
    result[rigid[faces]]=0.
    return result


def contact_endpoint_geometry(points,velocity,faces,dt,radius,rigid,contact_return_context):
    """Actual restored local endpoint and separate full-layout owner-return witness.

    The local quality row is unchanged in meaning. Returned quality/orientation
    and original local-area admission are secondary fail-closed guards, never
    extra reactions or additional Newton rows/budgets.
    """
    q=endpoint(points,velocity,dt,radius);q[rigid]=points[rigid]
    returned,current,cache=contact_return_map(q,contact_return_context)
    q=cache['assembled_local_points'];tri=q[faces];returned_tri=returned[faces]
    signed=np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2]));quality,_=_joint_quality(q,faces)
    returned_signed=np.einsum('ij,ij->i',returned_tri[:,0],np.cross(returned_tri[:,1],returned_tri[:,2]))
    returned_quality,_=_joint_quality(returned,faces);local_area=spherical_face_areas(q,faces,radius)
    return dict(q=q,current=current,Q=quality,signed=signed,returned=returned,cache=cache,
        local_area=local_area,returned_quality=returned_quality,returned_signed=returned_signed)


def area_evaluation_roundoff(points, faces, radius):
    """Conservative cancellation scale of the unchanged spherical-area formula.

    The determinant is evaluated from nearly parallel absolute unit vectors.
    Its error scales with the absolute triple products, not with the much
    smaller resulting triangle area. This estimate only permits an inward
    target correction; it never relaxes the physical endpoint acceptance gate.
    """
    a,b,c=np.moveaxis(np.asarray(points)[np.asarray(faces)],1,0)
    absolute=np.abs(a);bb=np.abs(b);cc=np.abs(c)
    products=np.sum(absolute*(bb[:,[1,2,0]]*cc[:,[2,0,1]]
        +bb[:,[2,0,1]]*cc[:,[1,2,0]]),axis=1)
    numerator=np.abs(np.einsum('ij,ij->i',a,np.cross(b,c)))
    denominator=1+np.sum(a*b+b*c+c*a,axis=1)
    denominator_scale=1+np.sum(np.abs(a*b)+np.abs(b*c)+np.abs(c*a),axis=1)
    eps=np.finfo(float).eps;gamma=16*eps/(1-16*eps)
    scale=2*float(radius)**2/(denominator**2+numerator**2)
    return scale*gamma*(np.abs(denominator)*products+numerator*denominator_scale)


def endpoint_area_jacobian(points, velocity, faces, dt, radius):
    """Derivative d spherical face area / d initial tangent velocity."""
    from gravitational_relaxation import face_area_gradients
    q=endpoint(points,velocity,dt,radius)
    gradient=face_area_gradients(q,faces,radius)
    u=velocity*(dt/radius)
    theta=np.linalg.norm(u,axis=1)
    sinc=np.sinc(theta/np.pi)
    b=np.empty_like(theta);small=theta<1e-3
    z=theta[small]**2
    b[small]=-1/3+z/30-z*z/840+z*z*z/45360
    t=theta[~small]
    b[~small]=(t*np.cos(t)-np.sin(t))/t**3
    # Apply the adjoint Jacobian to each area gradient, then project onto
    # the original tangent plane. q is unit, so normalization adds no term
    # on the exact tangent derivative (verified independently by the reviewer).
    pf=points[faces];uf=u[faces]
    result=(sinc[faces,None]*gradient
        +(b[faces]*np.sum(gradient*uf,axis=2)-sinc[faces]*np.sum(gradient*pf,axis=2))[:,:,None]*uf)
    result-=pf*np.sum(result*pf,axis=2)[:,:,None]
    return result*(dt/radius)


def current_stationary_certificate(points, faces, area, rhs, rigid, minimum,
                                   maximum, dt, radius, tolerance, predictor, *, max_active=128,
                                   feasibility_tolerance=FEASIBILITY_TOLERANCE):
    """Certify zero velocity at current physical bounds, never at a trial point.

    A nonnegative reaction must balance the complete free tangent force. Only
    physically tight bounds at floating-point area resolution may contribute.
    This is a first-order constrained stationarity certificate, not a global
    equilibrium or a license to drop a small nonzero candidate displacement.
    """
    eps=np.finfo(float).eps
    if np.any(area<minimum*(1-feasibility_tolerance)) or np.any(area>maximum*(1+feasibility_tolerance)):return None
    keys=[(int(i),1) for i in np.flatnonzero(np.abs(area-minimum)<=128*eps*np.maximum(1.,minimum))]
    keys += [(int(i),-1) for i in np.flatnonzero(np.abs(area-maximum)<=128*eps*np.maximum(1.,maximum))]
    if not keys:return None
    # This optional equilibrium certificate must obey the same work bound as
    # the moving active-set solve. Skip it before any dense row allocation;
    # the caller still solves (or rejects) the actual constrained motion.
    if len(keys)>max_active:return None
    jacobian=endpoint_area_jacobian(points,np.zeros_like(points),faces,dt,radius)/dt
    rows=[];norms=[];retained=[]
    for face,side in keys:
        row=np.zeros_like(points);np.add.at(row,faces[face],side*jacobian[face])
        row-=points*np.sum(row*points,axis=1)[:,None];row[rigid]=0.
        norm=float(np.linalg.norm(row))
        if norm>1e-20:rows.append(row/norm);norms.append(norm);retained.append((face,side))
    if not rows:return None
    rows=np.asarray(rows);norms=np.asarray(norms);keep=list(range(len(rows)))
    while keep:
        selected=rows[keep]
        gram=np.einsum('kij,lij->kl',selected,selected)
        target=-np.einsum('kij,ij->k',selected,rhs)
        multipliers=np.linalg.lstsq(gram,target,rcond=1e-12)[0]
        if np.min(multipliers,initial=0.) < -64*eps*max(float(np.max(np.abs(multipliers),initial=0.)),1e-30):
            del keep[int(np.argmin(multipliers))];continue
        multipliers=np.maximum(multipliers,0.)
        reaction=np.einsum('k,kij->ij',multipliers,selected)
        residual=float(np.linalg.norm(rhs+reaction));rhs_norm=float(np.linalg.norm(rhs))
        if residual>tolerance*rhs_norm:return None
        active=[retained[i] for i in keep]
        gaps=np.array([side*(area[face]-(minimum[face] if side==1 else maximum[face]))/(dt*norms[i])
                       for i,(face,side) in zip(keep,active)])
        work_scale=max(abs(float(np.sum(rhs*predictor))),1e-300)
        complement=float(np.max(np.abs(multipliers*gaps),initial=0.)/work_scale)
        if complement>tolerance:return None
        return dict(version=1,formulation='zero-velocity current-geometry KKT at physical area bounds',
            stationary_current_geometry=True,endpoint_dt_myr=float(dt),input_face_count=len(faces),
            face_index_scope='input gravity-stage faces, before later source birth or adaptation',
            # Reported on this path too, so saved history records the headroom
            # whether or not the stage needed the moving active-set solve. The
            # certificate is only reached from an admissible current geometry,
            # so its entry violation count is zero by construction.
            active_set_budget=int(max_active),entry_bound_violations=0,maximum_active_bounds=len(active),
            active_faces=[face for face,side in active],active_sides=['minimum' if side==1 else 'maximum' for face,side in active],
            multipliers_normalized=multipliers.tolist(),constraint_jacobian_norm_km=norms[keep].tolist(),
            endpoint_area_km2=area[[face for face,side in active]].tolist(),
            physical_bound_km2=[float(minimum[face] if side==1 else maximum[face]) for face,side in active],
            kkt_relative_residual=residual/rhs_norm if rhs_norm else 0.,true_stationarity_norm=residual,rhs_norm=rhs_norm,
            minimum_multiplier=float(np.min(multipliers,initial=0.)),
            maximum_tight_bound_roundoff_km2=max(abs(float(area[face]-(minimum[face] if side==1 else maximum[face]))) for face,side in active),
            maximum_lower_relative_violation=float(np.max(np.maximum(minimum-area,0.)/minimum,initial=0.)),
            maximum_upper_relative_violation=float(np.max(np.maximum(area-maximum,0.)/maximum,initial=0.)),
            normalized_complementarity=complement,normalized_constraint_gap_km_myr=gaps.tolist(),
            complementarity_normalization_work=work_scale,
            complementarity_scope='Measured physical bound gaps and multipliers, normalized by unconstrained predictor work; tightness also bounded independently by128eps*area.',
            reaction_norm=float(np.linalg.norm(reaction)),inner_solves=[],inverse_basis_vectors=0,sqp_iterations=0,
            nonlinear_geometry_correction_applied=False,speed_scaled_after_constraints=False,original_physical_bounds_unchanged=True)
    return None


def solve_endpoint(points, faces, area, body_force, initial_velocity, initial_solver,
                   rigid, minimum, maximum, dt, *, radius=6371., length_km=100.,
                   iterations=1024, tolerance=1e-8, max_speed_km_myr=100.,
                   max_sqp=32, max_active=None, max_step_halvings=40, viscosity_weights=None,
                   basis_cache=None, feasibility_tolerance=FEASIBILITY_TOLERANCE,
                   numerical_policy=None, quality_minimum=None, contact_return_context=None, gravity_area_targeting=False):
    """Minimize .5 v A v - rhs v with exact endpoint area inequalities.

    A remains the original start-of-stage viscous bilinear form. Each SQP
    subproblem uses current endpoint derivatives; inverse products are cached
    per free vertex tangent basis. Every returned motion passes actual nonlinear
    bounds and recomputed endpoint KKT, not merely a tangent-cone approximation.
    """
    if type(gravity_area_targeting) is not bool:raise ValueError('Gravity area targeting requires an explicit boolean option.')
    if gravity_area_targeting and (numerical_policy is None or contact_return_context is not None):
        raise ValueError('Gravity area targeting requires an explicit no-hard, unmapped numerical policy.')
    if numerical_policy is not None:
        from numerical_accuracy import NumericalPolicy
        if not isinstance(numerical_policy,NumericalPolicy):raise ValueError('Unsupported numerical accuracy policy.')
        if gravity_area_targeting and numerical_policy.has_hard_bounds:
            raise ValueError('Gravity area targeting cannot use contact hard bounds.')
        return _solve_joint_endpoint(points,faces,area,body_force,initial_velocity,initial_solver,rigid,minimum,maximum,dt,
            radius=radius,length_km=length_km,iterations=iterations,tolerance=tolerance,max_speed_km_myr=max_speed_km_myr,
            max_sqp=max_sqp,max_active=max_active,max_step_halvings=max_step_halvings,viscosity_weights=viscosity_weights,
            basis_cache=basis_cache,feasibility_tolerance=feasibility_tolerance,numerical_policy=numerical_policy,quality_minimum=quality_minimum,
            contact_return_context=contact_return_context,**(dict(gravity_area_targeting=True) if gravity_area_targeting else {}))
    if contact_return_context is not None:raise ValueError('Contact return geometry requires an explicit hard-contact numerical policy.')
    if quality_minimum is not None:raise ValueError('Endpoint quality constraints require an explicit future numerical policy.')
    if not np.isfinite(feasibility_tolerance) or not 0 < feasibility_tolerance <= FEASIBILITY_TOLERANCE:
        raise ValueError('Endpoint feasibility tolerance must be positive and no looser than the shared physical acceptance gate.')
    if max_active is None:
        max_active=max(MINIMUM_ACTIVE_BUDGET,int(ACTIVE_SET_FACE_FRACTION*len(faces)))
    # Bounds already violated by the start-of-stage areas owe nothing to the
    # interval, so they are the part of the count a shorter trial cannot move.
    entry_violations=int(np.count_nonzero(area<minimum*(1-feasibility_tolerance))
                        +np.count_nonzero(area>maximum*(1+feasibility_tolerance)))
    context=sheet.prepare(points,faces,area,length_km,radius=radius,viscosity_weights=viscosity_weights)
    rigid=np.asarray(rigid,bool)|(context['data']==0)
    def project(v):
        value=v-points*np.sum(v*points,axis=1)[:,None]
        value[rigid]=0.
        return value
    rhs=project(context['data'][:,None]*body_force)
    rhs_norm=float(np.linalg.norm(rhs))
    if rhs_norm==0:raise ConstraintSolveError('Constrained gravity has zero force but an inadmissible trial.')
    stationary=current_stationary_certificate(points,faces,area,rhs,rigid,minimum,maximum,dt,radius,tolerance,initial_velocity,
        max_active=max_active,feasibility_tolerance=feasibility_tolerance)
    if stationary is not None:
        info=dict(initial_solver)
        info.update(converged=True,true_stationarity_norm=stationary['true_stationarity_norm'],rhs_norm=rhs_norm,
            relative_residual=stationary['kkt_relative_residual'],tolerance=tolerance,failure_reason=None,
            stationarity_recomputed=True,stationarity_formulation='zero-velocity current-geometry KKT including physical bound reactions',
            stationarity_kind='current_geometry_zero_kkt',
            iterations_scope='original unconstrained predictor; zero-motion reaction certificate uses no PCG',
            original_linear_solver=dict(initial_solver),constrained_inner_pcg_iterations=0,constrained_sqp_iterations=0,
            data_misfit_energy=0.,viscous_dissipation_quadratic=0.,normalized_body_work=0.,objective=0.,
            maximum_strain_rate_per_myr=0.,maximum_absolute_divergence_per_myr=0.,maximum_radial_velocity_km_myr=0.)
        return points.copy(),np.zeros_like(points),info,stationary
    precise_tolerance=tolerance*.01
    precise_iterations=PRECISE_ITERATION_FACTOR*int(iterations)
    inner=[]
    def inverse(force_rhs):
        body=np.divide(force_rhs,context['data'][:,None],out=np.zeros_like(force_rhs),where=context['data'][:,None]>0)
        value,info=sheet.solve(points,faces,area,np.zeros_like(points),rigid,np.zeros(len(points),bool),
            length_km,radius=radius,body_force=body,iterations=precise_iterations,tolerance=precise_tolerance,
            viscosity_weights=viscosity_weights)
        inner.append({name:info[name] for name in ('iterations','maximum_iterations','relative_residual','tolerance','converged')})
        if not info['converged']:raise InverseSolveError(_inverse_failure(info))
        return value
    base=initial_velocity.copy()
    if np.linalg.norm(project(sheet.apply(context,base)-rhs))>precise_tolerance*rhs_norm:
        base=inverse(rhs)
    value=base.copy()
    # Each entry is one full-mesh preconditioned-CG solve of the viscous operator
    # for a unit tangent force at a node, and an active constraint needs up to six
    # of them (three face corners x two tangent axes). They dominate this solve.
    #
    # inverse() closes over points, faces, area, rigid, length_km, radius and
    # viscosity_weights, and takes NO dt -- so the cache is a pure function of the
    # geometry, not of the interval. A caller retrying the same geometry over a
    # shorter interval may therefore hand its own dict in and keep every solve
    # already paid for. relax() does exactly that when it backtracks; without it,
    # each halving discarded the entire cache and re-solved from scratch.
    basis_cache={} if basis_cache is None else basis_cache
    basis_vectors={}
    def basis_at(node):
        if node not in basis_vectors:
            axis=np.eye(3)[int(np.argmin(np.abs(points[node])))]
            first=np.cross(points[node],axis);first/=np.linalg.norm(first)
            basis_vectors[node]=(first,np.cross(points[node],first))
        return basis_vectors[node]
    def prefetch_inverse_rows(rows):
        from parallel_runtime import active_runtime, share_inputs
        runtime=active_runtime()
        if runtime is None or runtime.policy.status()['requested_workers']==1:
            return
        missing={}
        for row in rows:
            for node in np.flatnonzero(np.linalg.norm(row,axis=1)>0):
                for axis,vector in enumerate(basis_at(node)):
                    key=(int(node),axis)
                    if key not in basis_cache and key not in missing:
                        missing[key]=vector
        if len(missing)<2:
            return
        stage=dict(points=points,faces=faces,area=area,data=context['data'],rigid=rigid,
            length_km=length_km,radius=radius,viscosity_weights=viscosity_weights,
            iterations=precise_iterations,tolerance=precise_tolerance)
        with share_inputs(stage) as shared:
            solved=runtime.map(sheet.independent_inverse,
                ((shared,node,vector) for (node,axis),vector in missing.items()))
        # Insertion order is exactly the serial row/node/axis encounter order;
        # retain diagnostic order and every subsequent floating-point sum.
        for key,(velocity,info) in zip(missing,solved):
            inner.append({name:info[name] for name in ('iterations','maximum_iterations','relative_residual','tolerance','converged')})
            if not info['converged']:
                raise InverseSolveError(_inverse_failure(info))
            basis_cache[key]=velocity
    def inverse_row(row):
        result=np.zeros_like(points)
        for node in np.flatnonzero(np.linalg.norm(row,axis=1)>0):
            for axis,vector in enumerate(basis_at(node)):
                key=(int(node),axis)
                if key not in basis_cache:
                    unit_rhs=np.zeros_like(points);unit_rhs[node]=vector
                    basis_cache[key]=inverse(unit_rhs)
                result+=float(row[node]@vector)*basis_cache[key]
        return result
    eps=np.finfo(float).eps
    # Start at the actual physical bounds. An unconditional inward offset gives
    # a nonzero reaction gap even at an exactly converged iterate. Since the
    # complementarity gap is divided by dt, that offset prevents short steps
    # from passing the unchanged physical KKT gate. The stable area formula
    # normally resolves the physical target directly. If measured arithmetic
    # error warrants padding, the guarded correction below still solves again
    # and still checks the ORIGINAL physical bounds and complementarity.
    width=maximum-minimum
    low=minimum.copy()
    high=maximum.copy()
    active=[];history=[];multipliers=np.empty(0);roundoff_corrections=[];dependency_pivots=[];inequality_pivots=[]
    def geometry(v):
        q=endpoint(points,v,dt,radius);q[rigid]=points[rigid]
        return q,spherical_face_areas(q,faces,radius)
    def oriented(q):
        triangle=q[faces]
        return bool(np.all(np.einsum('ij,ij->i',triangle[:,0],
                                    np.cross(triangle[:,1],triangle[:,2]))>1e-15))
    def rows_at(v, keys):
        q,current=geometry(v)
        jacobian=endpoint_area_jacobian(points,v,faces,dt,radius)
        rows=[];gaps=[];norms=[]
        for face,side in keys:
            row=np.zeros_like(points)
            np.add.at(row,faces[face],side*jacobian[face]/dt)
            row=project(row);norm=float(np.linalg.norm(row))
            if norm<=1e-20:raise ConstraintSolveError('Active area bound has no free tangent derivative.')
            bound=low[face] if side==1 else high[face]
            rows.append(row/norm);gaps.append(side*(current[face]-bound)/(dt*norm));norms.append(norm)
        return q,current,np.array(rows),np.array(gaps),np.array(norms)
    # Unsigned spherical areas hide an inside-out triangle, while the area
    # derivative assumes its original outward orientation. Stay in that
    # domain for every linearization. Damping changes only an initial guess:
    # the original force, objective and final stationarity checks remain.
    predictor_orientation_backtracks=0
    while not oriented(geometry(value)[0]):
        predictor_orientation_backtracks+=1
        if predictor_orientation_backtracks>max_step_halvings:
            raise ConstraintSolveError('No outward-oriented constrained initial iterate was found.')
        value=project(base*(.5**predictor_orientation_backtracks))
    for iteration in range(1,max_sqp+1):
        q,current=geometry(value)
        violations=[(int(i),1) for i in np.flatnonzero(current<minimum*(1-feasibility_tolerance))]
        violations += [(int(i),-1) for i in np.flatnonzero(current>maximum*(1+feasibility_tolerance))]
        for key in violations:
            if key in active:continue
            # A face cannot be below its minimum and above its maximum at once:
            # minimum<maximum holds for every face by construction. An opposing
            # entry left over from an earlier iterate is therefore stale, and its
            # row is the exact negation of this one. Keeping both made the Schur
            # matrix rank deficient (observed condition number 4e17, past the
            # point where double precision retains any digit) and its linearized
            # requirements unsatisfiable. Retire the stale side, keep the live one.
            opposed=(key[0],-key[1])
            if opposed in active:del active[active.index(opposed)]
            active.append(key)
        if len(active)>max_active:
            if entry_violations>max_active:
                raise ActiveSetBudgetError(
                    f'Constrained gravity needs more than its {max_active} simultaneous area bounds for '
                    f'{len(faces)} faces, and the start-of-stage geometry already violates {entry_violations} '
                    'of them. That count does not depend on the interval, so a shorter timestep repeats the '
                    'identical solve; the mesh needs refinement or looser stage area bounds, not a smaller step.')
            raise ConstraintSolveError('Constrained gravity exceeded its explicit active-set budget.')
        if not active:raise ConstraintSolveError('Constrained branch was requested without an active bound.')
        # Solve the complete linearized INEQUALITY QP, not just equalities for
        # the currently violated faces. Deleting a negative reaction can make
        # that face (or an unselected face) violate its linearized bound at the
        # resulting proposal. Previously that invalid proposal reached the
        # nonlinear line search, which could stagnate indefinitely on a bound
        # the QP had silently omitted (captured43Myr nine-node contact region).
        # Dual reactions remain nonnegative throughout working-set pivots;
        # crossing to a restricted optimum stops at the first zero reaction.
        # Every inactive linearized bound is then checked before globalization.
        dual_reactions=np.zeros(len(active))
        linear_jacobian=endpoint_area_jacobian(points,value,faces,dt,radius)
        solved_working_sets=set()
        while active:
            _,_,rows,gaps,_=rows_at(value,active)
            prefetch_inverse_rows(rows)
            inverse_rows=np.array([inverse_row(row) for row in rows])
            schur=np.einsum('kij,lij->kl',rows,inverse_rows)
            # The exact inverse is symmetric, but its individually converged
            # PCG columns need not be symmetric to the tighter area tolerance.
            # Solve the represented map: symmetrizing it introduces a fixed
            # linearized constraint error into the actual correction below.
            # Endpoint stationarity still uses the original viscous operator.
            linear_target=np.einsum('kij,ij->k',rows,value)-gaps
            target=linear_target-np.einsum('kij,ij->k',rows,base)
            multipliers=np.linalg.lstsq(schur,target,rcond=1e-12)[0]
            inconsistent=np.linalg.norm(schur@multipliers-target)>tolerance*max(float(np.linalg.norm(target)),1e-30)
            if inconsistent:
                # An inconsistent dependent restricted dual has no finite
                # optimum. Its pseudoinverse signs cannot choose a bound to
                # retire: that can immediately discard a newly blocking row
                # and cycle. Pivot from the current feasible dual reaction
                # along a certified PHYSICAL row-null direction instead.
                pivot=_dependent_inequality_pivot(rows,target,dual_reactions,with_direction=True)
                if pivot is not None:
                    at,proof,direction_dual=pivot
                    face,side=active[at]
                    dependency_pivots.append(dict(iteration=iteration,face=face,
                        side='minimum' if side==1 else 'maximum',**proof))
                    dual_reactions=np.maximum(dual_reactions+proof['step_to_zero']*direction_dual,0.)
                    del active[at];dual_reactions=np.delete(dual_reactions,at)
                    continue
            dual_scale=max(float(np.max(np.abs(multipliers),initial=0.)),1e-30)
            negative=np.flatnonzero(multipliers < -64*eps*dual_scale)
            if len(negative):
                direction_dual=multipliers-dual_reactions
                crossing=dual_reactions[negative]/(-direction_dual[negative])
                at=int(negative[int(np.argmin(crossing))])
                fraction=float(crossing.min())
                dual_reactions=np.maximum(dual_reactions+fraction*direction_dual,0.)
                face,side=active[at]
                inequality_pivots.append(dict(iteration=iteration,kind='negative_reaction_to_zero',
                    face=face,side='minimum' if side==1 else 'maximum',dual_step_fraction=fraction))
                del active[at];dual_reactions=np.delete(dual_reactions,at)
                continue
            if inconsistent:
                raise _area_failure('Dependent active area constraints have inconsistent linearized requirements.',
                                    current,minimum,maximum)
            dual_reactions=np.maximum(multipliers,0.)
            proposal=project(base+np.einsum('k,kij->ij',dual_reactions,inverse_rows))
            predicted=current+np.einsum('fij,fij->f',linear_jacobian,(proposal-value)[faces])
            lower_miss=np.maximum(low-predicted,0.)/low
            upper_miss=np.maximum(predicted-high,0.)/high
            if max(float(lower_miss.max(initial=0.)),float(upper_miss.max(initial=0.)))<=feasibility_tolerance:
                multipliers=dual_reactions
                break
            signature=tuple(sorted(active))
            if signature in solved_working_sets:
                raise ConstraintSolveError('Linearized inequality working set repeated without a feasible proposal; no endpoint accepted.')
            solved_working_sets.add(signature)
            lower_face=int(np.argmax(lower_miss));upper_face=int(np.argmax(upper_miss))
            key=(lower_face,1) if lower_miss[lower_face]>=upper_miss[upper_face] else (upper_face,-1)
            if key in active or (key[0],-key[1]) in active:
                raise ConstraintSolveError('Solved active area row failed its original linearized bound; no endpoint accepted.')
            if len(active)>=max_active:
                raise ConstraintSolveError('Constrained gravity exceeded its explicit active-set budget.')
            inequality_pivots.append(dict(iteration=iteration,kind='omitted_linearized_bound',
                face=key[0],side='minimum' if key[1]==1 else 'maximum',
                relative_violation=float(max(lower_miss[key[0]],upper_miss[key[0]]))))
            active.append(key);dual_reactions=np.r_[dual_reactions,0.]
        if not active:
            raise ConstraintSolveError('Nonlinear active-set linearization lost every admissible reaction; no endpoint accepted.')
        multipliers=np.maximum(multipliers,0.)
        # Globalize the step. The area bounds are nonlinear in the vertex
        # positions, so the undamped Newton step this loop used to take could
        # leave the region where its own linearization holds; the worst relative
        # violation was observed doubling every iteration (1.2 -> 2.4 -> ... ->
        # 1751) until a face crossed BOTH of its bounds at once.
        #
        # The guard is a backtracking search that refuses any step which worsens
        # the worst bound violation. A full step is tried first and taken
        # unchanged whenever it behaves, so converging solves keep their exact
        # previous behaviour.
        #
        # The step is deliberately NOT capped at max_speed_km_myr. That bound is
        # a physical assertion about the candidate, not a length to scale into:
        # a solve that genuinely needs to exceed it must raise rather than be
        # quietly shortened until it fits. test_gravity_bounds's
        # test_constrained_speed_cap_is_never_hidden_by_postsolve_scaling pins
        # that contract, and it is the reason this loop damps only on violation.
        correction=np.einsum('k,kij->ij',multipliers,inverse_rows)
        def worst_violation(candidate):
            q,areas=geometry(candidate)
            if not oriented(q):return float('inf')
            return max(float(np.max(np.maximum(minimum-areas,0.)/minimum,initial=0.)),
                       float(np.max(np.maximum(areas-maximum,0.)/maximum,initial=0.)))
        # Damp the Newton displacement from the CURRENT iterate. Scaling the
        # reaction around the original unconstrained predictor instead jumps
        # back toward an infeasible point whenever a later iteration needs
        # damping, and can exhaust the budget without making any progress.
        newton=project(base+correction)
        direction=newton-value
        step_scale=1.
        # Globalization can cross the area-evaluation noise floor while the
        # final acceptance below still enforces the caller's stricter bound.
        # Otherwise roundoff can stall the search before stationarity is
        # reached and its inward target correction can be evaluated.
        allowed=max(worst_violation(value),FEASIBILITY_TOLERANCE)
        for _ in range(max_step_halvings):
            candidate=project(value+step_scale*direction)
            if worst_violation(candidate)<=allowed:break
            step_scale*=.5
        else:
            raise _area_failure('Constrained gravity line search could not reduce its nonlinear area violation.',
                                current,minimum,maximum)
        # Deliberately no revert to the full step here. A step that never stopped
        # worsening the violation is precisely the one that must not be taken
        # undamped; reverting was what turned this search into a no-op.
        #
        # A shortened step is a move toward the solution, not a candidate answer:
        # its multipliers were solved for the full step, so the reaction, KKT
        # residual and complementarity computed below would all be evaluated
        # against a geometry that was never taken. Record it in the history and
        # re-solve from the new point instead of testing it for acceptance.
        damped=step_scale<1.
        multipliers=step_scale*multipliers
        value=candidate
        q,current,new_rows,new_gaps,norms=rows_at(value,active)
        reaction=np.einsum('k,kij->ij',multipliers,new_rows)
        residual=project(sheet.apply(context,value)-rhs-reaction)
        relative=float(np.linalg.norm(residual)/rhs_norm)
        work_scale=max(abs(float(np.sum(rhs*value))),abs(float(np.sum(value*sheet.apply(context,value)))),1e-30)
        target_complement=float(np.max(np.abs(multipliers*new_gaps),initial=0.)/work_scale)
        # Report/check complementarity against the ORIGINAL physical bounds,
        # including any inward target margin used to obtain a feasible float.
        physical_gaps=np.array([side*(current[face]-(minimum[face] if side==1 else maximum[face]))/(dt*norm)
            for (face,side),norm in zip(active,norms)])
        complement=float(np.max(np.abs(multipliers*physical_gaps),initial=0.)/work_scale)
        lower_error=float(np.max(np.maximum(minimum-current,0.)/minimum,initial=0.))
        upper_error=float(np.max(np.maximum(current-maximum,0.)/maximum,initial=0.))
        history.append(dict(iteration=iteration,active_constraints=len(active),step_scale=step_scale,kkt_relative_residual=relative,
            normalized_complementarity=complement,maximum_lower_relative_violation=lower_error,
            maximum_upper_relative_violation=upper_error,**area_violation_summary(current,minimum,maximum)))
        if not damped and max(lower_error,upper_error)>feasibility_tolerance and relative<=tolerance and complement<=tolerance:
            # An already stationary iterate may chatter across a tiny physical
            # bound because its absolute-coordinate area determinant cancels.
            # Move only affected solver targets inward, then SOLVE again. Do
            # not post-correct vertices, accept infeasibility, change volumes,
            # or increase the SQP budget. Equal-area intervals remain exact.
            roundoff=area_evaluation_roundoff(q,faces,radius)
            for face,side in active:
                bound=minimum[face] if side==1 else maximum[face]
                miss=side*(bound-current[face])
                if not (miss>bound*feasibility_tolerance and miss<=roundoff[face]):continue
                old=low[face]-minimum[face] if side==1 else maximum[face]-high[face]
                padding=min(max(old,4*miss),width[face]*.25)
                if padding<=old:continue
                if side==1:low[face]=minimum[face]+padding
                else:high[face]=maximum[face]-padding
                roundoff_corrections.append(dict(iteration=iteration,face=int(face),
                    side='minimum' if side==1 else 'maximum',observed_violation_km2=float(miss),
                    area_evaluation_roundoff_km2=float(roundoff[face]),inward_padding_km2=float(padding)))
        if not damped and max(lower_error,upper_error)<=feasibility_tolerance and relative<=tolerance and complement<=tolerance:
            speed=float(np.linalg.norm(value,axis=1).max(initial=0.))
            if speed>max_speed_km_myr:
                raise SpeedBoundError('Constrained endpoint requires an active speed bound; unsupported by this isolated candidate.')
            fields=sheet.strain_rate(context,value)
            strain_square=np.einsum('fij,fij->f',fields['D'],fields['D'])
            viscous=float(np.sum(context['face_scale']*(strain_square+fields['divergence']**2)))
            misfit=float(np.sum(context['data'][:,None]*value**2));body_work=float(np.sum(rhs*value))
            info=dict(initial_solver)
            info.update(converged=True, true_stationarity_norm=float(np.linalg.norm(residual)),rhs_norm=rhs_norm,
                relative_residual=relative,tolerance=tolerance,failure_reason=None,
                stationarity_recomputed=True,stationarity_formulation='finite-step area-constrained KKT including endpoint reactions',
                stationarity_kind='nonlinear_endpoint_kkt',
                iterations_scope='original unconstrained predictor; constrained work is listed separately',
                original_linear_solver=dict(initial_solver),constrained_inner_pcg_iterations=sum(row['iterations'] for row in inner),
                constrained_sqp_iterations=iteration,
                data_misfit_energy=.5*misfit,viscous_dissipation_quadratic=viscous,normalized_body_work=body_work,
                objective=.5*(misfit+viscous)-body_work,
                maximum_strain_rate_per_myr=float(np.sqrt(strain_square).max(initial=0.)),
                maximum_absolute_divergence_per_myr=float(np.abs(fields['divergence']).max(initial=0.)),
                maximum_radial_velocity_km_myr=float(np.abs(np.sum(points*value,axis=1)).max(initial=0.)))
            diagnostic=dict(version=1,formulation='same viscous quadratic with nonlinear spherical endpoint-area inequalities',
                endpoint_dt_myr=float(dt),input_face_count=len(faces),face_index_scope='input gravity-stage faces, before later source birth or adaptation',
                sqp_iterations=iteration,active_faces=[face for face,side in active],
                active_sides=['minimum' if side==1 else 'maximum' for face,side in active],
                multipliers_normalized=multipliers.tolist(),constraint_jacobian_norm_km=norms.tolist(),
                endpoint_area_km2=current[[face for face,side in active]].tolist(),
                physical_bound_km2=[float(minimum[face] if side==1 else maximum[face]) for face,side in active],
                inward_roundoff_padding_km2=[float(low[face]-minimum[face] if side==1 else maximum[face]-high[face]) for face,side in active],
                kkt_relative_residual=relative,normalized_complementarity=complement,
                target_normalized_complementarity=target_complement,
                complementarity_scope='original physical bounds, including inward target padding',
                area_roundoff_corrections=roundoff_corrections,
                dependent_inequality_pivots=dependency_pivots,
                linearized_inequality_pivots=inequality_pivots,
                all_linearized_bounds_checked_before_globalization=True,
                maximum_lower_relative_violation=lower_error,maximum_upper_relative_violation=upper_error,
                reaction_norm=float(np.linalg.norm(reaction)),inner_solves=inner,inverse_basis_vectors=len(basis_cache),
                nonlinear_geometry_correction_applied=False,speed_scaled_after_constraints=False,
                original_physical_bounds_unchanged=True,feasibility_tolerance=feasibility_tolerance,
                predictor_orientation_backtracks=predictor_orientation_backtracks,
                outward_orientation_preserved=True,
                active_set_budget=int(max_active),entry_bound_violations=entry_violations,
                maximum_active_bounds=max((row['active_constraints'] for row in history),default=0),
                history=history)
            return q,value,info,diagnostic
    raise ConstraintSolveError('Constrained gravity SQP failed its nonlinear/KKT convergence budget: '+str(history[-1] if history else None))


def _strictly_convex_qp(G, c, C, d, threshold, max_steps):
    """Dual active-set solution of min c.x + x.G.x/2 subject to C x >= d - threshold.

    Goldfarb and Idnani (1983). G must be symmetric positive definite. The
    iterate starts at the unconstrained minimizer and every step keeps the
    dual multipliers nonnegative: a violated row is added only by a full step
    that makes it exactly tight, and an active row is retired only when its
    own multiplier reaches zero along the move (the first such row, never the
    most negative EQP multiplier). Each full step strictly increases the
    objective, so for strictly convex G no working set repeats and the
    method terminates in finitely many steps. A violated row whose normal is
    dependent on the active rows and that admits no dual descent certifies
    an infeasible linearization; it is reported, never dropped.
    """
    eps=np.finfo(float).eps;n=len(c)
    L=np.linalg.cholesky(G);Linv=np.linalg.solve(L,np.eye(n))
    # Rows of the constraint normals in the Cholesky-whitened metric.
    B=C@Linv.T
    x=-(Linv.T@(Linv@c));active=[];u=np.empty(0);steps=0;retired=0
    while True:
        slack=C@x-d
        candidates=[int(k) for k in np.flatnonzero(slack < -threshold) if int(k) not in active]
        if not candidates:return x,active,u,dict(qp_steps=steps,qp_retired_rows=retired)
        p=min(candidates,key=lambda k:(float(slack[k]),k));up=np.r_[u,0.]
        while True:
            steps+=1
            if steps>max_steps:raise ConstraintSolveError('Convex linearized QP exceeded its bounded dual active-set budget.')
            q=len(active);w=B[p]
            if q:
                Q1,R=np.linalg.qr(B[active].T);coefficient=Q1.T@w;projected=w-Q1@coefficient
                dual=np.linalg.solve(R,coefficient)
            else:projected=w.copy();dual=np.empty(0)
            positive=np.flatnonzero(dual > 64*eps*max(float(np.max(np.abs(dual),initial=0.)),1e-300))
            if len(positive):
                ratios=up[positive]/dual[positive];at=int(positive[int(np.argmin(ratios))]);partial=max(float(np.min(ratios)),0.)
            else:at=None;partial=np.inf
            free=float(np.linalg.norm(projected))
            full=np.inf if free<=64*eps*float(np.linalg.norm(w)) else -float(C[p]@x-d[p])/(free*free)
            if not np.isfinite(partial) and not np.isfinite(full):
                raise ConstraintSolveError('Convex linearized QP is infeasible: a violated row depends on active rows without dual descent.')
            step=min(partial,full)
            if np.isfinite(full):x=x+step*(Linv.T@projected)
            up[:q]=np.maximum(up[:q]-step*dual,0.);up[q]+=step
            if full<=partial:
                active.append(p);u=up;break
            up[at]=0.;active.pop(at);up=np.delete(up,at);retired+=1


def _convex_lagrangian_model(H, margin):
    """Largest damping theta in (0,1] of multiplier curvature with G >= margin.

    H = I - M, where I is the exact objective metric in whitened active-support
    coordinates and M the multiplier-weighted constraint curvature. G = I -
    theta M is the Lagrangian Hessian at the damped multipliers theta*mu; it
    equals H unchanged (theta = 1) whenever H is already positive definite
    with the margin.
    """
    H=(H+H.T)*.5;identity=np.eye(len(H));M=identity-H
    top=float(np.linalg.eigvalsh(M).max(initial=0.)) if len(H) else 0.
    if 1.-top>=margin:return H,1.,1.-top
    theta=(1.-margin)/top
    return identity-theta*M,theta,1.-top


def _joint_norm(x):
    s=float(np.max(np.abs(x),initial=0.));return 0. if s==0 else float(s*np.linalg.norm(x/s))

def _joint_quality(q,faces):
    a,b,c=np.moveaxis(q[faces],1,0);e=b-a;f=c-a;bc=c-b;ca=a-c;cross=np.cross(e,f);N=np.linalg.norm(cross,axis=1);S=np.sum(e*e+bc*bc+ca*ca,axis=1)
    if np.any(N<=0) or np.any(S<=0):raise ConstraintSolveError('Degenerate endpoint quality domain.')
    unit=cross/N[:,None];Nb=np.cross(f,unit);Nc=np.cross(unit,e);Na=-Nb-Nc;dN=np.stack((Na,Nb,Nc),axis=1)
    dS=2*np.stack((2*a-b-c,2*b-a-c,2*c-a-b),axis=1)
    return 2*math.sqrt(3.)*N/S,2*math.sqrt(3.)*(dN/S[:,None,None]-N[:,None,None]*dS/S[:,None,None]**2)

def _joint_endpoint_adjoint(p,v,faces,gradient,dt,R):
    u=v*(dt/R);theta=np.linalg.norm(u,axis=1);sinc=np.sinc(theta/np.pi);b=np.empty_like(theta);small=theta<1e-3;z=theta[small]**2
    b[small]=-1/3+z/30-z*z/840+z*z*z/45360;t=theta[~small];b[~small]=(t*np.cos(t)-np.sin(t))/t**3
    pf=p[faces];uf=u[faces]
    result=sinc[faces,None]*gradient+(b[faces]*np.sum(gradient*uf,axis=2)-sinc[faces]*np.sum(gradient*pf,axis=2))[:,:,None]*uf
    result-=pf*np.sum(result*pf,axis=2)[:,:,None]
    return result*(dt/R)

def _contact_area_targets(minimum,maximum,stage_area,movable,*,numerical_policy):
    """Fixed inward arithmetic targets; original admission is never replaced.

    The captured spherical endpoint missed a hard lower bound by49 binary64
    area ULPs while its original KKT residual was small. As for the existing
    quality target,64*eps uses only the fixed full-stage area scale. This is a
    numerical interior aim, not a physical guard, new credit or geometry repair.
    Unsupported narrow intervals fail closed after the original stationary path;
    rigid faces are kept on their original bounds without a fabricated interior.
    """
    minimum=np.asarray(minimum,float);maximum=np.asarray(maximum,float)
    stage_area=np.asarray(stage_area,float);movable=np.asarray(movable,bool)
    if (minimum.ndim!=1 or any(a.shape!=minimum.shape for a in (maximum,stage_area,movable))
        or not all(np.isfinite(a).all() for a in (minimum,maximum,stage_area))
        or np.any(minimum<=0) or np.any(maximum<minimum) or np.any(stage_area<=0)):
        raise ValueError('Invalid fixed contact-area target inputs; original bounds cannot be widened.')
    requested_margin=64*np.finfo(float).eps*stage_area
    target_minimum=minimum.copy();target_maximum=maximum.copy()
    target_minimum[movable]=minimum[movable]+requested_margin[movable]
    target_maximum[movable]=maximum[movable]-requested_margin[movable]
    diagnostic=dict(kind='fixed_contact_area_inward_arithmetic_targets',version=1,
        scope='solver aim only; final exact original dimensional credit/hard strain predicates and raw-bound complementarity unchanged',
        roundoff_factor=64.,roundoff_epsilon=float(np.finfo(float).eps),
        margin_scale='fixed original full contact-stage area; no absolute unit floor, substep renewal or added admission slack',
        fixed_stage_area_km2=stage_area.tolist(),requested_inward_margin_km2=requested_margin.tolist(),
        raw_admitted_minimum_area_km2=minimum.tolist(),raw_admitted_maximum_area_km2=maximum.tolist(),
        solver_target_minimum_area_km2=target_minimum.tolist(),solver_target_maximum_area_km2=target_maximum.tolist(),
        applied_lower_inward_area_km2=(target_minimum-minimum).tolist(),
        applied_upper_inward_area_km2=(maximum-target_maximum).tolist(),movable_faces=movable.tolist(),
        rigid_faces_untightened=np.flatnonzero(~movable).tolist(),
        original_physical_bounds_unchanged=True,credit_or_spent_changed=False,
        endpoint_or_velocity_corrected=False,representable_target_certified=False)
    unsupported=movable&(~np.isfinite(target_minimum)|~np.isfinite(target_maximum)
        |(target_minimum<=minimum)|(target_maximum>=maximum)|(target_minimum>=target_maximum))
    if np.any(unsupported):
        error=ConstraintSolveError('Moving contact faces lack a representable strict interior at the fixed64eps stage-area target; numerical targeting unresolved, no physical infeasibility claimed.')
        error.diagnostics=dict(version=2,converged=False,failure_scope='fixed contact-area numerical target construction',
            unsupported_target_local_faces=np.flatnonzero(unsupported).tolist(),contact_area_solver_targets=diagnostic,
            original_physical_bounds_unchanged=True,no_posthoc_geometry_correction=True)
        raise error
    # Use ORIGINAL nominal policy arrays here. Applying credit to the admitted
    # or tightened interval would double-spend it. No policy.advance is called.
    nominal_minimum=numerical_policy.minimum_area_km2;nominal_maximum=numerical_policy.maximum_area_km2
    baseline=numerical_policy.baseline_miss_km2
    masks=[];predicted=[]
    for target in (target_minimum,target_maximum):
        masks.append(numerical_policy.feasible_mask(target,nominal_minimum,nominal_maximum))
        miss=np.maximum(nominal_minimum-target,0.)+np.maximum(target-nominal_maximum,0.)
        predicted.append(numerical_policy.spent_km2+np.maximum(miss-baseline,0.))
    certified=(~movable)|(masks[0]&masks[1]&(predicted[0]<=numerical_policy.budget_km2)
        &(predicted[1]<=numerical_policy.budget_km2))
    diagnostic.update(lower_target_original_policy_feasible=masks[0].tolist(),
        upper_target_original_policy_feasible=masks[1].tolist(),
        lower_target_predicted_spent_km2=predicted[0].tolist(),upper_target_predicted_spent_km2=predicted[1].tolist(),
        target_certification_scope='exact original policy and unchanged predicted spending for movable target endpoints; rigid endpoints are not used as movement targets')
    if not np.all(certified):
        error=ConstraintSolveError('Fixed inward contact-area targets fail original exact admission or predicted persistent spending; no bound, credit or strain guard changed.')
        error.diagnostics=dict(version=2,converged=False,failure_scope='fixed contact-area target original-policy certification',
            unsupported_target_local_faces=np.flatnonzero(~certified).tolist(),contact_area_solver_targets=diagnostic,
            original_physical_bounds_unchanged=True,no_posthoc_geometry_correction=True)
        raise error
    diagnostic['representable_target_certified']=True
    return target_minimum,target_maximum,diagnostic


def _restore_gravity_area_targets(current,minimum,maximum,target_minimum,target_maximum,normalization_area,movable,active,kinds,multipliers,*,numerical_policy,tolerance,iteration):
    """Measured Newton aims for an explicitly opted-in gravity endpoint.

    Only failed original-policy sides with current positive area reactions are
    eligible. Nominal inequalities, persistent credit, and raw complementarity
    remain unchanged. A certified target is not a certified computed endpoint.
    """
    from numerical_accuracy import NumericalPolicy
    if not isinstance(numerical_policy,NumericalPolicy) or numerical_policy.has_hard_bounds:
        raise ValueError('Gravity numerical targets require the original no-hard area policy.')
    current,minimum,maximum,target_minimum,target_maximum,normalization_area=(
        np.asarray(a,float) for a in (current,minimum,maximum,target_minimum,target_maximum,normalization_area))
    movable=np.asarray(movable,bool);multipliers=np.asarray(multipliers,float)
    if (current.ndim!=1 or any(a.shape!=current.shape for a in (minimum,maximum,target_minimum,target_maximum,normalization_area,movable))
            or not all(np.isfinite(a).all() for a in (current,minimum,maximum,target_minimum,target_maximum,normalization_area,multipliers))
            or np.any(current<=0) or np.any(minimum<=0) or np.any(maximum<minimum)
            or np.any(normalization_area<=0) or np.any(target_minimum<minimum)
            or np.any(target_maximum>maximum) or np.any(target_minimum>target_maximum)
            or multipliers.shape!=(len(active),) or np.any(multipliers<0)
            or not np.isfinite(tolerance) or tolerance<=0 or tolerance!=numerical_policy.outer_tolerance
            or isinstance(iteration,(bool,np.bool_)) or not isinstance(iteration,(int,np.integer)) or not 1<=iteration<=32):
        raise ValueError('Invalid measured gravity-target inputs; original bounds and accuracy cannot change.')
    numerical_policy.validate(len(current))
    if (not np.array_equal(minimum,numerical_policy.minimum_area_km2)
            or not np.array_equal(maximum,numerical_policy.maximum_area_km2)
            or not np.allclose(normalization_area,numerical_policy.current_area_km2,rtol=2e-12,atol=0.)):
        raise ValueError('Gravity target restoration must retain original nominal bounds and input area scale.')
    lookup={}
    for key,kind in enumerate(kinds):
        if (len(kind)!=2 or isinstance(kind[0],(bool,np.bool_)) or not isinstance(kind[0],(int,np.integer))
                or not 0<=kind[0]<len(current) or kind[1] not in ('minimum','maximum','quality')
                or tuple(kind) in lookup):
            raise ValueError('Ambiguous gravity-target constraint kinds.')
        lookup[tuple(kind)]=key
    reactions={}
    for key,m in zip(active,multipliers):
        if (isinstance(key,(bool,np.bool_)) or not isinstance(key,(int,np.integer))
                or not 0<=key<len(kinds) or key in reactions):
            raise ValueError('Invalid supported gravity-target reaction keys.')
        reactions[key]=float(m)
    failed=~numerical_policy.feasible_mask(current,minimum,maximum)
    if not np.any(failed) or np.any(failed&~movable):return None
    lower_failed=failed&(current<minimum);upper_failed=failed&(current>maximum)
    if not np.all((lower_failed|upper_failed)==failed):return None
    next_minimum=target_minimum.copy();next_maximum=target_maximum.copy()
    indices=np.flatnonzero(failed);details=[];base_margin=64*np.finfo(float).eps*normalization_area
    allowance=numerical_policy.allowance_km2;baseline=numerical_policy.baseline_miss_km2
    for face in indices:
        lower=bool(lower_failed[face]);side='minimum' if lower else 'maximum';key=lookup.get((int(face),side))
        defect=float(target_minimum[face]-current[face] if lower else current[face]-target_maximum[face])
        if (key not in reactions or reactions[key]<=0 or not np.isfinite(defect)
                or defect<=0 or defect/normalization_area[face]>tolerance):return None
        old=float(target_minimum[face] if lower else target_maximum[face]);increment=defect+base_margin[face]
        updated=float(np.nextafter(old+increment,np.inf) if lower else np.nextafter(old-increment,-np.inf))
        if lower:next_minimum[face]=updated
        else:next_maximum[face]=updated
        details.append(dict(local_face=int(face),side=side,measured_area_km2=float(current[face]),
            original_nominal_bound_area_km2=float(minimum[face] if lower else maximum[face]),
            old_target_area_km2=old,measured_old_target_defect_km2=defect,
            normalization_area_km2=float(normalization_area[face]),normalized_old_target_defect=float(defect/normalization_area[face]),
            positive_reaction_scaled=reactions[key],original_64eps_margin_km2=float(base_margin[face]),
            requested_inward_increment_km2=float(increment),new_target_area_km2=updated,
            applied_inward_increment_km2=float(updated-old if lower else old-updated),
            original_allowance_km2=float(allowance[face]),original_baseline_miss_km2=float(baseline[face]),
            original_budget_km2=float(numerical_policy.budget_km2[face]),original_spent_km2=float(numerical_policy.spent_km2[face])))
    diagnostic=dict(kind='measured_gravity_area_interior_targets',version=1,iteration=int(iteration),
        scope='changed-side Newton aim only; original nominal inequalities, exact policy, raw complementarity and all acceptance gates retained',
        updated_local_faces=indices.tolist(),measured_updates=details,
        original_physical_bounds_unchanged=True,credit_or_spent_changed=False,endpoint_or_velocity_corrected=False,
        unchanged_sides_retained=True,representable_target_certified=False,
        limitation='Measured numerical aiming; no geometry, convergence, roundoff-enclosure or later energy-acceptance guarantee.')
    unsupported=(~np.isfinite(next_minimum)|~np.isfinite(next_maximum)|(next_minimum<=0)
        |(next_minimum<target_minimum)|(next_maximum>target_maximum)|(next_minimum>next_maximum)
        |(lower_failed&(next_minimum<=minimum))|(upper_failed&(next_maximum>=maximum))
        |(failed&(next_minimum>=next_maximum))
        |((next_minimum-minimum)/normalization_area>tolerance)
        |((maximum-next_maximum)/normalization_area>tolerance))
    for detail in details:
        if not detail['applied_inward_increment_km2']>0:unsupported[detail['local_face']]=True
    if np.any(unsupported):
        error=ConstraintSolveError('Measured gravity target lacks a representable bounded inward interval; original admission is unchanged.')
        diagnostic['unsupported_target_faces']=np.flatnonzero(unsupported).tolist();error.diagnostics=diagnostic;raise error
    masks=[];predicted=[]
    for target in (next_minimum,next_maximum):
        masks.append(numerical_policy.feasible_mask(target,minimum,maximum))
        miss=np.maximum(minimum-target,0.)+np.maximum(target-maximum,0.)
        predicted.append(numerical_policy.spent_km2+np.maximum(miss-baseline,0.))
    certified=masks[0]&masks[1]&(predicted[0]<=numerical_policy.budget_km2)&(predicted[1]<=numerical_policy.budget_km2)
    for detail in details:
        face=detail['local_face']
        detail.update(lower_target_original_policy_feasible=bool(masks[0][face]),upper_target_original_policy_feasible=bool(masks[1][face]),
            lower_target_predicted_spent_km2=float(predicted[0][face]),upper_target_predicted_spent_km2=float(predicted[1][face]))
    if not np.all(certified):
        error=ConstraintSolveError('Measured gravity targets fail original exact admission or persistent spending; no credit is changed.')
        diagnostic['unsupported_target_faces']=np.flatnonzero(~certified).tolist();error.diagnostics=diagnostic;raise error
    diagnostic['representable_target_certified']=True
    return next_minimum,next_maximum,diagnostic



def _restore_contact_area_targets(current,minimum,maximum,target_minimum,target_maximum,stage_area,movable,active,kinds,multipliers,*,numerical_policy,tolerance,iteration,normalization_area):
    """Aim farther inward after a measured, supported tiny endpoint miss.

    This is a Newton target update, not endpoint repair or admission slack.
    It is eligible only after the caller's original KKT/complementarity gates.
    Every raw-failing side must already have a positive area reaction and a
    target defect within the existing outer scale. An unconstrained predictor
    with zero reactions therefore cannot acquire an artificial interior bias.
    An update does not certify the next computed geometry or global feasibility.
    """
    if not numerical_policy.has_hard_bounds:return None
    current,minimum,maximum,target_minimum,target_maximum,stage_area,normalization_area=(
        np.asarray(a,float) for a in (current,minimum,maximum,target_minimum,target_maximum,stage_area,normalization_area))
    movable=np.asarray(movable,bool);multipliers=np.asarray(multipliers,float)
    if (current.ndim!=1 or any(a.shape!=current.shape for a in (minimum,maximum,target_minimum,target_maximum,stage_area,normalization_area,movable))
            or not all(np.isfinite(a).all() for a in (current,minimum,maximum,target_minimum,target_maximum,stage_area,normalization_area,multipliers))
            or np.any(current<=0) or np.any(minimum<=0) or np.any(maximum<minimum)
            or np.any(stage_area<=0) or np.any(normalization_area<=0)
            or multipliers.shape!=(len(active),) or np.any(multipliers<0)
            or not np.isfinite(tolerance) or tolerance!=numerical_policy.outer_tolerance):
        raise ValueError('Invalid measured contact-target restoration inputs; original accuracy cannot change.')
    nominal_minimum=numerical_policy.minimum_area_km2;nominal_maximum=numerical_policy.maximum_area_km2
    original_minimum,original_maximum=numerical_policy.effective_bounds(nominal_minimum,nominal_maximum)
    if (not np.array_equal(minimum,original_minimum) or not np.array_equal(maximum,original_maximum)
            or not np.array_equal(stage_area,numerical_policy.hard_stage_area_km2)):
        raise ValueError('Measured target restoration must retain original raw admission and fixed full-stage anchors.')
    lower_failed=current<minimum;upper_failed=current>maximum;failed=lower_failed|upper_failed
    if not np.any(failed) or np.any(failed&~movable):return None
    reactions={}
    for key,m in zip(active,multipliers):
        if key<0 or key>=len(kinds) or key in reactions:raise ValueError('Invalid supported contact-target reaction keys.')
        reactions[key]=float(m)
    lookup={tuple(kind):key for key,kind in enumerate(kinds)}
    indices=np.flatnonzero(failed);details=[]
    base_margin=64*np.finfo(float).eps*stage_area
    next_minimum=target_minimum.copy();next_maximum=target_maximum.copy()
    for face in indices:
        lower=bool(lower_failed[face]);side='minimum' if lower else 'maximum';key=lookup.get((int(face),side))
        defect=float(target_minimum[face]-current[face] if lower else current[face]-target_maximum[face])
        if (key not in reactions or reactions[key]<=0 or not np.isfinite(defect)
                or defect<=0 or defect/normalization_area[face]>tolerance):return None
        old=float(target_minimum[face] if lower else target_maximum[face]);increment=defect+base_margin[face]
        updated=float(np.nextafter(old+increment,np.inf) if lower else np.nextafter(old-increment,-np.inf))
        if lower:next_minimum[face]=updated
        else:next_maximum[face]=updated
        details.append(dict(local_face=int(face),side=side,measured_area_km2=float(current[face]),
            original_raw_bound_area_km2=float(minimum[face] if lower else maximum[face]),old_target_area_km2=old,
            measured_old_target_defect_km2=defect,normalization_area_km2=float(normalization_area[face]),
            normalized_old_target_defect=float(defect/normalization_area[face]),positive_reaction_scaled=reactions[key],
            fixed_stage_area_km2=float(stage_area[face]),original_64eps_margin_km2=float(base_margin[face]),
            requested_inward_increment_km2=float(increment),new_target_area_km2=updated,
            applied_inward_increment_km2=float(updated-old if lower else old-updated)))
    diagnostic=dict(kind='measured_contact_area_interior_target_restoration',version=1,iteration=int(iteration),
        scope='solver target only; original exact admission, raw complementarity and every physical gate retained',
        updated_local_faces=indices.tolist(),measured_updates=details,
        solver_target_minimum_area_km2=next_minimum.tolist(),solver_target_maximum_area_km2=next_maximum.tolist(),
        applied_lower_inward_area_km2=(next_minimum-minimum).tolist(),applied_upper_inward_area_km2=(maximum-next_maximum).tolist(),
        original_physical_bounds_unchanged=True,credit_or_spent_changed=False,endpoint_or_velocity_corrected=False,
        representable_target_certified=False,
        limitation='A measured numerical restoration attempt, not a geometry, convergence or global roundoff guarantee.')
    unsupported=movable&(~np.isfinite(next_minimum)|~np.isfinite(next_maximum)
        |(next_minimum<=minimum)|(next_maximum>=maximum)|(next_minimum>=next_maximum)
        |(next_minimum<target_minimum)|(next_maximum>target_maximum))
    for detail in details:
        if not detail['applied_inward_increment_km2']>0:unsupported[detail['local_face']]=True
    if np.any(unsupported):
        error=ConstraintSolveError('Measured contact-target restoration lacks a representable strict inward interval; numerical targeting unresolved, no physical infeasibility claimed.')
        diagnostic['unsupported_target_local_faces']=np.flatnonzero(unsupported).tolist();error.diagnostics=diagnostic;raise error
    baseline=numerical_policy.baseline_miss_km2;masks=[];predicted=[]
    for target in (next_minimum,next_maximum):
        masks.append(numerical_policy.feasible_mask(target,nominal_minimum,nominal_maximum))
        miss=np.maximum(nominal_minimum-target,0.)+np.maximum(target-nominal_maximum,0.)
        predicted.append(numerical_policy.spent_km2+np.maximum(miss-baseline,0.))
    certified=(~movable)|(masks[0]&masks[1]&(predicted[0]<=numerical_policy.budget_km2)&(predicted[1]<=numerical_policy.budget_km2))
    diagnostic.update(lower_target_original_policy_feasible=masks[0].tolist(),upper_target_original_policy_feasible=masks[1].tolist(),
        lower_target_predicted_spent_km2=predicted[0].tolist(),upper_target_predicted_spent_km2=predicted[1].tolist(),
        target_certification_scope='exact original nominal policy and unchanged predicted spending at both movable target endpoints')
    if not np.all(certified):
        error=ConstraintSolveError('Measured contact-target restoration fails original exact admission or persistent spending; no bound or credit is changed.')
        diagnostic['unsupported_target_local_faces']=np.flatnonzero(~certified).tolist();error.diagnostics=diagnostic;raise error
    diagnostic['representable_target_certified']=True
    return next_minimum,next_maximum,diagnostic


def _contact_area_target_observation(returned_area,assembled_local_area,minimum,maximum):
    """Select a numerical aiming witness without replacing either geometry.

    A lower-side miss uses the smaller measured area; an upper-side miss
    uses the larger. The original restoration helper must independently
    qualify every failed side. Neither this witness nor an aimed target is
    model geometry, a ledger value or an endpoint acceptance certificate.
    """
    returned_area,assembled_local_area,minimum,maximum=(
        np.asarray(a,float) for a in (returned_area,assembled_local_area,minimum,maximum))
    if (returned_area.ndim!=1 or any(a.shape!=returned_area.shape for a in (assembled_local_area,minimum,maximum))
            or not all(np.isfinite(a).all() for a in (returned_area,assembled_local_area,minimum,maximum))
            or np.any(returned_area<=0) or np.any(assembled_local_area<=0)
            or np.any(minimum<=0) or np.any(maximum<minimum)):
        raise ValueError('Invalid aligned two-geometry contact-area target observations; no admission criterion changed.')
    returned_lower=returned_area<minimum;local_lower=assembled_local_area<minimum
    returned_upper=returned_area>maximum;local_upper=assembled_local_area>maximum
    lower=returned_lower|local_lower;upper=returned_upper|local_upper;opposed=lower&upper
    diagnostic=dict(kind='two_geometry_contact_area_numerical_target_observation',version=1,
        scope='numerical targeting only; actual returned and assembled-local geometry remain independently admitted',
        failed_local_faces=np.flatnonzero(lower|upper).tolist(),opposed_local_faces=np.flatnonzero(opposed).tolist(),
        returned_lower_failed_local_faces=np.flatnonzero(returned_lower).tolist(),
        assembled_local_lower_failed_local_faces=np.flatnonzero(local_lower).tolist(),
        returned_upper_failed_local_faces=np.flatnonzero(returned_upper).tolist(),
        assembled_local_upper_failed_local_faces=np.flatnonzero(local_upper).tolist(),
        original_physical_bounds_unchanged=True,actual_geometry_replaced=False,credit_or_spent_changed=False,
        endpoint_acceptance_certified=False,
        limitation='A worst-side aiming observation, not a local-area reaction, stationarity or binary64 feasibility guarantee.')
    if np.any(opposed):
        error=ConstraintSolveError('Opposed two-geometry contact-area misses cannot select one supported interior target; admission remains unresolved.')
        diagnostic['restoration_refused']=True;error.diagnostics=diagnostic;raise error
    observation=returned_area.copy()
    observation[lower]=np.minimum(returned_area,assembled_local_area)[lower]
    observation[upper]=np.maximum(returned_area,assembled_local_area)[upper]
    details=[]
    for face in np.flatnonzero(lower|upper):
        is_lower=bool(lower[face]);observed=float(observation[face])
        source=('both' if returned_area[face]==assembled_local_area[face] else
            ('returned' if observed==returned_area[face] else 'assembled_local'))
        details.append(dict(local_face=int(face),side='minimum' if is_lower else 'maximum',
            numerical_target_observation_area_km2=observed,observation_geometry=source,
            actual_returned_area_km2=float(returned_area[face]),actual_assembled_local_area_km2=float(assembled_local_area[face]),
            original_raw_bound_area_km2=float(minimum[face] if is_lower else maximum[face])))
    diagnostic['failed_side_observations']=details;diagnostic['restoration_refused']=False
    return observation,diagnostic


# Joint SQP rejections that are failures of the numerical search, not of the
# problem: the heuristic pivots cycled or ran out, the KKT-residual merit
# stalled, or the outer budget ended. Only these exact messages of the base
# ConstraintSolveError class earn the one convex restart below. Never retried:
# SpeedBoundError, InverseSolveError, ActiveSetBudgetError, input ValueErrors,
# capacity limits, structural failures (zero force, no outward start) and the
# reduced-curvature gate. See CONVEX-RESTART.md for the evidence per message.
CONVEX_RESTART_TRIGGERS = frozenset((
    'Joint linearized working set cycled; no endpoint accepted.',
    'Joint inequality pivots exhausted their bounded budget.',
    'Joint working set lost every admissible reaction.',
    'Joint Newton search failed to reduce actual nonlinear KKT merit.',
    'Joint Newton exhausted the original32 outer-iteration budget.',
))


def _convex_restart_record(original, restart, accepted, restart_error=None):
    history=restart.get('history',[]);pivots=restart.get('working_set_pivots',[]);inner=restart.get('inner_solves',[])
    record=dict(kind='convex_restart',accepted=accepted,trigger=str(original),trigger_class=type(original).__name__,
        original_sqp_iterations=len(original.diagnostics.get('history',[])),
        original_working_set_pivots=len(original.diagnostics.get('working_set_pivots',[])),
        original_inner_pcg_iterations=sum(x['iterations'] for x in original.diagnostics.get('inner_solves',[])),
        start='initial outward iterate of the same solve',sqp_budget_renewed=True,
        restart_sqp_iterations=len(history),restart_inner_pcg_iterations=sum(x['iterations'] for x in inner),
        model_minimum_eigenvalue_margin=CONVEX_MODEL_CURVATURE_MARGIN,
        multiplier_curvature_damping=[p['multiplier_curvature_damping'] for p in pivots if p.get('kind')=='strictly_convex_qp_working_set'],
        kkt_relative_residuals=[h['kkt_relative_residual'] for h in history])
    if restart_error is not None:
        record.update(restart_error_class=type(restart_error).__name__,restart_error=str(restart_error),
            restart_working_set_pivots=pivots,restart_last_velocity=restart.get('last_velocity'))
    return record


def _solve_joint_endpoint(*args, basis_cache=None, **kwargs):
    """Joint SQP, with one convex restart after a numerical search failure.

    The first attempt is the original solve, unchanged; whatever it returns is
    returned as is, and whatever it raises outside CONVEX_RESTART_TRIGGERS
    propagates as is. After a trigger, the solve runs once more from its own
    initial outward iterate, choosing every working set by the strictly convex
    QP, with its own fresh max_sqp budget (renewal approved by the user). The
    restart answers to the identical acceptance gate. If it fails for any
    reason, the ORIGINAL exception is raised with its original diagnostics,
    plus a 'convex_restart' summary, so caller retry behaviour is unchanged.
    """
    # One cache for both attempts: identical inverse columns are reused. A
    # fresh dict behaves exactly like the attempt's own default.
    cache={} if basis_cache is None else basis_cache
    try:return _joint_endpoint_attempt(*args,basis_cache=cache,**kwargs)
    except ConstraintSolveError as error:
        if type(error) is not ConstraintSolveError or str(error) not in CONVEX_RESTART_TRIGGERS or not hasattr(error,'diagnostics'):raise
        original=error
    known=set(cache);failure=None
    try:
        q,value,info,diag=_joint_endpoint_attempt(*args,basis_cache=cache,convex_working_sets=True,**kwargs)
    except Exception as error:
        failure=error
    except BaseException:
        for key in [key for key in cache if key not in known]:del cache[key]
        raise
    if failure is None:
        diag['convex_restart']=_convex_restart_record(original,diag,True)
        return q,value,info,diag
    # A caller-shared cache (gravitational_relaxation's attempt loop) must hold
    # exactly what the original rejection left, so later accepted solves keep
    # their bytes. Keys are only ever added, so deleting the new ones restores
    # contents and insertion order.
    for key in [key for key in cache if key not in known]:del cache[key]
    original.diagnostics['convex_restart']=_convex_restart_record(original,getattr(failure,'diagnostics',{}) or {},False,failure)
    raise original


def _joint_endpoint_attempt(points,faces,area,body_force,initial_velocity,initial_solver,rigid,minimum,maximum,dt,*,radius=6371.,length_km=100.,iterations=1024,tolerance=1e-8,max_speed_km_myr=100.,max_sqp=32,max_active=None,max_step_halvings=40,viscosity_weights=None,basis_cache=None,feasibility_tolerance=1e-12,numerical_policy=None,quality_minimum=None,contact_return_context=None,deadline_check=lambda:None,convex_working_sets=False,gravity_area_targeting=False):
    if type(gravity_area_targeting) is not bool:raise ValueError('Gravity area targeting requires an explicit boolean option.')
    if numerical_policy is None:raise ValueError('Joint contact solver requires explicit numerical accuracy policy.')
    if gravity_area_targeting and (numerical_policy.has_hard_bounds or contact_return_context is not None):
        raise ValueError('Gravity area targeting requires original no-hard, unmapped gravity geometry.')
    numerical_policy.validate(len(faces));tolerance=numerical_policy.outer_tolerance
    if quality_minimum is None:raise ValueError('Joint contact solver requires the original endpoint-quality floor.')
    if not 0<max_sqp<=32 or not 0<max_step_halvings<=40:raise ValueError('Joint solver cannot exceed original32/40 outer/search budgets.')
    p=np.asarray(points);faces=np.asarray(faces);area=np.asarray(area);minimum=np.asarray(minimum);maximum=np.asarray(maximum);quality_minimum=np.asarray(quality_minimum)
    if not np.allclose(area,numerical_policy.current_area_km2,rtol=2e-12,atol=0.):raise ValueError('Joint numerical policy is not aligned with original material geometry.')
    if (quality_minimum.shape!=(len(faces),) or not np.isfinite(quality_minimum).all() or np.any(quality_minimum<=0) or np.any(quality_minimum>.15) or dt<=0 or np.any(minimum<=0) or np.any(maximum<minimum)):
        raise ValueError('Invalid original joint endpoint constraints.')
    if not np.isfinite(initial_velocity).all() or not np.isfinite(body_force).all():raise ValueError('Nonfinite original contact force/velocity.')
    # Only the contact stage adds hard strain anchors. Gravity-only policy1
    # keeps its existing exact inequalities and unchanged numerical semantics.
    contact_hard_bounds=numerical_policy.has_hard_bounds
    solve_minimum,solve_maximum=(numerical_policy.effective_bounds(minimum,maximum)
        if contact_hard_bounds else (minimum,maximum))
    if max_active is None:max_active=max(MINIMUM_ACTIVE_BUDGET,int(ACTIVE_SET_FACE_FRACTION*len(faces)))
    ctx=sheet.prepare(p,faces,area,length_km,radius=radius,viscosity_weights=viscosity_weights);rigid=np.asarray(rigid,bool)|(ctx['data']==0)
    mapped_contact=contact_return_context is not None
    if mapped_contact:
        if not contact_hard_bounds:raise ValueError('Final return geometry is only supported for fixed hard-contact bounds.')
        validate_contact_return_context(p,faces,rigid,dt,radius,contact_return_context)
    def project(v):
        value=v-p*np.sum(p*v,axis=1)[:,None];value[rigid]=0.;return value
    rhs=project(ctx['data'][:,None]*body_force);rhs_norm=_joint_norm(rhs)
    if rhs_norm==0:raise ConstraintSolveError('Joint contact has zero force but requires an admissibility reaction.')
    # Retain the original exact-zero current-geometry certificate. Numerical
    # area credit is never a reason to declare equilibrium: only a physically
    # tight original bound and the original stricter KKT/roundoff gate qualify.
    current_quality=_joint_quality(p,faces)[0]
    tri=p[faces];current_signed=np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2]))
    stationary=(current_stationary_certificate(p,faces,area,rhs,rigid,minimum,maximum,dt,radius,
        min(float(tolerance),1e-8),initial_velocity,max_active=max_active,
        feasibility_tolerance=feasibility_tolerance)
        if np.all(current_quality>=quality_minimum) and np.all(current_signed>1e-15)
        and (not contact_hard_bounds or numerical_policy.feasible(area,minimum,maximum)) else None)
    if stationary is not None:
        if mapped_contact:
            returned,returned_area,_=contact_return_map(p.copy(),contact_return_context)
            returned_quality=_joint_quality(returned,faces)[0];returned_tri=returned[faces]
            returned_signed=np.einsum('ij,ij->i',returned_tri[:,0],np.cross(returned_tri[:,1],returned_tri[:,2]))
            if (not numerical_policy.feasible(returned_area,minimum,maximum)
                or np.any(returned_quality<quality_minimum) or np.any(returned_signed<=1e-15)):
                raise ConstraintSolveError('Original zero-motion contact certificate fails actual final-return admission; no residual motion is invented.')
            stationary['contact_final_return_geometry']=dict(kind='actual_final_owner_return_geometry',version=1,
                returned_endpoint=returned.tolist(),returned_area_km2=returned_area.tolist(),
                returned_quality=returned_quality.tolist(),returned_outward_orientation_preserved=True,
                secondary_quality_certificate_only=True,
                stationary_scope='unchanged original current-geometry zero KKT; actual returned geometry independently admitted')
            stationary['numerical_accuracy_policy']=numerical_policy.evidence(returned_area,minimum,maximum)
        info=dict(initial_solver)
        info.update(converged=True,true_stationarity_norm=stationary['true_stationarity_norm'],rhs_norm=rhs_norm,
            relative_residual=stationary['kkt_relative_residual'],tolerance=min(float(tolerance),1e-8),failure_reason=None,
            stationarity_recomputed=True,stationarity_formulation='zero-velocity current-geometry KKT including physical bound reactions',
            stationarity_kind='current_geometry_zero_kkt',
            iterations_scope='original unconstrained predictor; zero-motion reaction certificate uses no PCG',
            original_linear_solver=dict(initial_solver),constrained_inner_pcg_iterations=0,constrained_sqp_iterations=0,
            data_misfit_energy=0.,viscous_dissipation_quadratic=0.,normalized_body_work=0.,objective=0.,
            maximum_strain_rate_per_myr=0.,maximum_absolute_divergence_per_myr=0.,maximum_radial_velocity_km_myr=0.)
        if not mapped_contact:stationary['numerical_accuracy_policy']=numerical_policy.evidence(area,minimum,maximum)
        return p.copy(),np.zeros_like(p),info,stationary
    # Stationary and no-hard paths retain their exact original bounds.
    # A fixed target shifts only the Newton equations/merit, never admission,
    # reaction gradients or ORIGINAL raw-bound complementarity.
    target_minimum,target_maximum=solve_minimum,solve_maximum;area_target_diagnostic=None
    initial_area_target_diagnostic=None;area_target_updates=[];area_target_restoration_failure=None
    gravity_area_target_updates=[];gravity_area_target_failure=None
    if contact_hard_bounds:
        target_minimum,target_maximum,area_target_diagnostic=_contact_area_targets(
            solve_minimum,solve_maximum,numerical_policy.hard_stage_area_km2,
            np.any(~rigid[faces],axis=1),numerical_policy=numerical_policy)
        initial_area_target_diagnostic=area_target_diagnostic
    precise_tolerance=min(float(numerical_policy.inverse_tolerance),float(tolerance)*.01);precise_iterations=PRECISE_ITERATION_FACTOR*int(iterations);inner=[]
    def inverse(force):
        body=np.divide(force,ctx['data'][:,None],out=np.zeros_like(force),where=ctx['data'][:,None]>0)
        result,info=sheet.solve(p,faces,area,np.zeros_like(p),rigid,np.zeros(len(p),bool),length_km,radius=radius,body_force=body,iterations=precise_iterations,tolerance=precise_tolerance,viscosity_weights=viscosity_weights)
        inner.append({k:info[k] for k in ('iterations','maximum_iterations','relative_residual','tolerance','converged')})
        if not info['converged']:raise InverseSolveError(_inverse_failure(info))
        return result
    base=project(initial_velocity.copy())
    if _joint_norm(project(sheet.apply(ctx,base)-rhs))>precise_tolerance*rhs_norm:base=inverse(rhs)
    W=float(np.sum(rhs*base))
    if not np.isfinite(W) or W<=0:raise ConstraintSolveError('Original contact metric/work scale is not positive.')
    kinds=[(int(i),'minimum') for i in range(len(faces))]+[(int(i),'maximum') for i in range(len(faces))]+[(int(i),'quality') for i in range(len(faces))]
    # The quality target is a tiny inward arithmetic margin. Acceptance and
    # complementarity retain the ORIGINAL caller floor, which is never lowered.
    target_quality=quality_minimum*(1+64*np.finfo(float).eps)
    def geometry_details(v):
        if mapped_contact:return contact_endpoint_geometry(p,v,faces,dt,radius,rigid,contact_return_context)
        q=endpoint(p,v,dt,radius);q[rigid]=p[rigid]
        tri=q[faces];signed=np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2]))
        current=spherical_face_areas(q,faces,radius);Q,_=_joint_quality(q,faces)
        return dict(q=q,current=current,Q=Q,signed=signed)
    def geometry(v):
        measured=geometry_details(v)
        signed=(np.minimum(measured['signed'],measured['returned_signed']) if mapped_contact else measured['signed'])
        return measured['q'],measured['current'],measured['Q'],signed
    def secondary_geometry_admitted(v):
        # Final endpoint and target-restoration certification only. Original
        # Newton merit iterates may remain tentative/infeasible; they are never
        # returned as accepted model geometry or a spent ledger transaction.
        if not mapped_contact:return True
        measured=geometry_details(v)
        return (numerical_policy.feasible(measured['local_area'],minimum,maximum)
            and np.all(measured['returned_quality']>=quality_minimum) and np.all(measured['returned_signed']>1e-15))
    def constraints(v):
        if mapped_contact:
            from gravitational_relaxation import face_area_gradients
            measured=geometry_details(v);q=measured['q'];current=measured['current'];Q=measured['Q']
            signed=np.minimum(measured['signed'],measured['returned_signed'])
            aj=_contact_endpoint_adjoint(p,v,faces,face_area_gradients(measured['returned'],faces,radius),dt,radius,rigid,return_cache=measured['cache'])
            qj=_contact_endpoint_adjoint(p,v,faces,_joint_quality(q,faces)[1],dt,radius,rigid)
        else:
            q,current,Q,signed=geometry(v);aj=endpoint_area_jacobian(p,v,faces,dt,radius);qj=_joint_endpoint_adjoint(p,v,faces,_joint_quality(q,faces)[1],dt,radius)
        local=np.concatenate((aj/area[:,None,None],-aj/area[:,None,None],qj/quality_minimum[:,None,None]))
        local[np.tile(rigid[faces],(3,1))]=0.
        # The working set/merit g use inward numerical area targets. A bounded
        # measured restoration may change these constants within this solve.
        # original_g retains RAW admitted bounds for complementarity, including
        # target-induced slack. Evidence/spending keep original nominal column
        # bounds, so dimensional credit is never applied a second time.
        g=np.r_[(current-target_minimum)/area,(target_maximum-current)/area,(Q-target_quality)/quality_minimum]
        original_g=np.r_[(current-solve_minimum)/area,(solve_maximum-current)/area,(Q-quality_minimum)/quality_minimum]
        return g,original_g,local,q,current,Q,signed
    vectors={};cache={} if basis_cache is None else basis_cache
    def basis_at(node):
        if node not in vectors:
            axis=np.eye(3)[int(np.argmin(abs(p[node])))];first=np.cross(p[node],axis);first/=np.linalg.norm(first);vectors[node]=(first,np.cross(p[node],first))
        return vectors[node]
    support=[];T=None;z=None;support_keys=[];value=base.copy();active=[];mu=np.empty(0);history=[];pivots=[];maximum_support=0
    def build_metric(keys,current_value):
        nonlocal support_keys,T,z,maximum_support
        nodes=sorted({int(node) for key in keys for node in faces[kinds[key][0]] if not rigid[node]})
        required=[(node,axis) for node in nodes for axis in range(2)]
        if required==support_keys:return
        missing={key:basis_at(key[0])[key[1]] for key in required if key not in cache}
        from parallel_runtime import active_runtime,share_inputs
        runtime=active_runtime()
        if len(missing)>=2 and runtime is not None and runtime.policy.status()['requested_workers']!=1:
            stage=dict(points=p,faces=faces,area=area,data=ctx['data'],rigid=rigid,length_km=length_km,radius=radius,viscosity_weights=viscosity_weights,iterations=precise_iterations,tolerance=precise_tolerance)
            with share_inputs(stage) as shared:solved=runtime.map(sheet.independent_inverse,((shared,node,vector) for (node,axis),vector in missing.items()))
            for key,(v,info) in zip(missing,solved):
                inner.append({k:info[k] for k in ('iterations','maximum_iterations','relative_residual','tolerance','converged')})
                if not info['converged']:raise InverseSolveError(_inverse_failure(info))
                cache[key]=v
        else:
            for key,vector in missing.items():
                force=np.zeros_like(p);force[key[0]]=vector;cache[key]=inverse(force)
        V=np.stack([cache[key] for key in required],axis=2)
        C=np.array([[float(basis_at(node)[axis]@V[node,:,j]) for j in range(len(required))] for node,axis in required])
        asym=_joint_norm(C-C.T)/max(_joint_norm(C),1e-30)
        if asym>precise_tolerance*10:raise InverseSolveError('Active-support compliance is not numerically symmetric.')
        C=(C+C.T)*.5
        try:B=np.linalg.solve(C,np.eye(len(C)));L=np.linalg.cholesky(B)
        except np.linalg.LinAlgError as e:raise InverseSolveError('Active-support compliance is singular or non-positive; no shorter-interval inverse retry.') from e
        T=math.sqrt(W)*np.einsum('ijk,kl->ijl',V,L)
        delta=np.array([float((current_value-base)[node]@basis_at(node)[axis]) for node,axis in required])
        z=L.T@delta/math.sqrt(W);support_keys=required;maximum_support=max(maximum_support,len(required))
    def rows(local,keys):return np.array([np.einsum('ij,ijk->k',local[key],T[faces[kinds[key][0]]]) for key in keys])
    def reaction(local,keys,multipliers):
        result=np.zeros_like(p)
        for key,m in zip(keys,multipliers):np.add.at(result,faces[kinds[key][0]],W*m*local[key])
        return project(result)
    def curvature(v,keys,difference_step=2e-4):
        result=[]
        for key in keys:
            f,kind=kinds[key];tri=faces[f];localp=p[tri];localv=v[tri];localface=np.array([[0,1,2]]);localbasis=[basis_at(int(node)) for node in tri]
            def local_row(vv):
                if mapped_contact:
                    # Preserve the actual full-layout return map and original
                    # local restoration for BOTH unchanged derivative scales.
                    # A triangle-only preview would lose batching/owner/masks.
                    from gravitational_relaxation import face_area_gradients
                    fullv=v.copy();fullv[tri]=vv;measured=geometry_details(fullv);selected=faces[[f]]
                    if kind!='quality':
                        gradient=face_area_gradients(measured['returned'],selected,radius)
                        row=_contact_endpoint_adjoint(p,fullv,selected,gradient,dt,radius,rigid,return_cache=measured['cache'])[0]/area[f]
                        if kind=='maximum':row=-row
                    else:
                        gradient=_joint_quality(measured['q'],selected)[1]
                        row=_contact_endpoint_adjoint(p,fullv,selected,gradient,dt,radius,rigid)[0]/quality_minimum[f]
                elif kind!='quality':
                    row=endpoint_area_jacobian(localp,vv,localface,dt,radius)[0]/area[f]
                    if kind=='maximum':row=-row
                else:
                    qq=endpoint(localp,vv,dt,radius);qq[rigid[tri]]=localp[rigid[tri]]
                    row=_joint_endpoint_adjoint(localp,vv,localface,_joint_quality(qq,localface)[1],dt,radius)[0]/quality_minimum[f]
                row[rigid[tri]]=0.
                return np.array([row[i]@localbasis[i][a] for i in range(3) for a in range(2)])
            H=np.zeros((6,6))
            for j in range(6):
                corner,axis=divmod(j,2)
                if rigid[tri[corner]]:continue
                h=difference_step*max(1.,abs(float(localv[corner]@localbasis[corner][axis])))
                d=np.zeros_like(localv);d[corner]=h*localbasis[corner][axis];H[:,j]=(local_row(localv+d)-local_row(localv-d))/(2*h)
            H=(H+H.T)*.5
            tangentT=np.stack([localbasis[i][axis]@T[tri[i]] for i in range(3) for axis in range(2)])
            result.append(tangentT.T@H@tangentT)
        return np.array(result)
    def convex_working_set(iteration,active,mu,value,threshold):
        # Used only by the convex restart of _solve_joint_endpoint, after the
        # original attempt failed its numerical search. Solve the complete linearized inequality QP exactly with a strictly
        # convex Lagrangian model, then return its working set, primal step
        # and multiplier direction to the unchanged nonlinear search and gate.
        margin=CONVEX_MODEL_CURVATURE_MARGIN
        keys=list(active);curved=[i for i in range(len(active)) if mu[i]!=0]
        for _ in range(len(faces)+1):
            build_metric(keys,value);value=project(base+np.einsum('ijk,k->ij',T,z))
            g,physical_g,local,q,current,Q,signed=constraints(value)
            H=np.eye(len(z))-(np.einsum('k,kij->ij',mu[curved],curvature(value,[active[i] for i in curved])) if curved else 0.)
            G,theta,smallest=_convex_lagrangian_model(H,margin)
            allJ=rows(local,list(range(len(kinds))))
            dz,chosen,lam,detail=_strictly_convex_qp(G,z,allJ,-g,threshold,2*len(kinds)+2)
            covered={node for node,axis in support_keys}
            if all(rigid[node] or int(node) in covered for key in chosen for node in faces[kinds[key][0]]):break
            keys=list(dict.fromkeys(keys+chosen))
        else:raise ConstraintSolveError('Convex linearized QP support did not close.')
        if not chosen:raise ConstraintSolveError('Convex linearized QP retained no admissible reaction.')
        if len(chosen)>max_active:raise ConstraintSolveError('Joint linearized active-set budget exhausted.')
        start=np.array([float(mu[active.index(key)]) if key in active else 0. for key in chosen])
        r=z-start@rows(local,chosen)
        detail.update(iteration=iteration,kind='strictly_convex_qp_working_set',incoming_working_set=[list(kinds[k]) for k in active],
            working_set=[list(kinds[k]) for k in chosen],multiplier_curvature_damping=theta,
            lagrangian_hessian_minimum_eigenvalue=smallest,model_minimum_eigenvalue_margin=margin,support_tangent_dofs=len(z))
        return chosen,start,dz,lam-start,r,g,physical_g,local,q,current,Q,signed,value,detail
    failure=None
    try:
        orientation_backtracks=0
        while not np.all(geometry(value)[3]>1e-15):
            orientation_backtracks+=1
            if orientation_backtracks>max_step_halvings:raise ConstraintSolveError('No outward joint initial iterate.')
            value=project(base*(.5**orientation_backtracks))
        for iteration in range(1,max_sqp+1):
            deadline_check();g,physical_g,local,q,current,Q,signed=constraints(value)
            for key in np.flatnonzero(g<0):
                key=int(key)
                if key in active:continue
                f,kind=kinds[key];opposed=f+len(faces) if kind=='minimum' else (f if kind=='maximum' else None)
                if opposed in active:
                    at=active.index(opposed);active.pop(at);mu=np.delete(mu,at)
                active.append(key);mu=np.r_[mu,0.]
            if len(active)>max_active:raise ConstraintSolveError('Joint endpoint exceeded its explicit area/quality active-set capacity.')
            react=reaction(local,active,mu);Av=sheet.apply(ctx,value);residual=project(Av-rhs-react);relative=_joint_norm(residual)/rhs_norm
            work=max(abs(float(np.sum(rhs*value))),abs(float(np.sum(value*Av))),1e-30);comp=float(np.max(abs(W*mu*physical_g[active]),initial=0.)/work)
            feasible=(numerical_policy.feasible(current,minimum,maximum) and np.all(Q>=quality_minimum)
                and np.all(signed>1e-15) and secondary_geometry_admitted(value))
            row=dict(iteration=iteration,active=[list(kinds[k]) for k in active],multipliers_scaled=mu.tolist(),kkt_relative_residual=relative,normalized_complementarity=comp,endpoint_area_km2=current.tolist(),endpoint_quality=Q.tolist(),minimum_quality=quality_minimum.tolist(),area_policy_feasible=bool(numerical_policy.feasible(current,minimum,maximum)),velocity_km_myr=value.tolist(),endpoint=q.tolist())
            if mapped_contact:
                measured=geometry_details(value)
                row['contact_final_return_geometry']=dict(kind='actual_final_owner_return_geometry',version=1,
                    returned_endpoint=measured['returned'].tolist(),returned_area_km2=current.tolist(),
                    assembled_local_area_km2=measured['local_area'].tolist(),returned_quality=measured['returned_quality'].tolist(),
                    assembled_local_area_policy_feasible=bool(numerical_policy.feasible(measured['local_area'],minimum,maximum)),
                    returned_quality_feasible=bool(np.all(measured['returned_quality']>=quality_minimum)),
                    returned_outward_orientation_preserved=bool(np.all(measured['returned_signed']>1e-15)),
                    secondary_quality_certificate_only=True,
                    area_geometry_scope='actual full-layout assembled/fixed-owner/normalized/rigid-overridden selected global faces')
            history.append(row)
            if relative<=tolerance and comp<=tolerance and feasible and np.all(mu>=0):
                speed=float(np.linalg.norm(value,axis=1).max(initial=0.))
                if speed>max_speed_km_myr:raise SpeedBoundError('Joint endpoint exceeds unchanged speed bound; no scaling accepted.')
                reduced_minimum=1.;curvature_margin=0.;reduced_dimension=None
                if T is not None and np.any(mu>0):
                    Jfinal=rows(local,active);Hmain=np.eye(len(z))-np.einsum('k,kij->ij',mu,curvature(value,active))
                    Hhalf=np.eye(len(z))-np.einsum('k,kij->ij',mu,curvature(value,active,1e-4))
                    positive=mu>64*np.finfo(float).eps*max(float(mu.max(initial=0.)),1e-30)
                    _,singular,Vh=np.linalg.svd(Jfinal[positive],full_matrices=True)
                    rank=int(np.count_nonzero(singular>64*np.finfo(float).eps*max(Jfinal.shape)*singular[0]))
                    Z=Vh[rank:].T;reduced_dimension=Z.shape[1]
                    if reduced_dimension:
                        reduced_minimum=float(np.linalg.eigvalsh(Z.T@Hhalf@Z).min())
                        curvature_margin=float(np.linalg.norm(Hmain-Hhalf,ord=2))+64*np.finfo(float).eps*max(float(np.linalg.norm(Hhalf,ord=2)),1.)
                        if not reduced_minimum>curvature_margin:raise ConstraintSolveError('Joint KKT endpoint lacks positive reduced Lagrangian curvature beyond its measured derivative margin.')
                    else:reduced_minimum=None
                fields=sheet.strain_rate(ctx,value);strain_square=np.einsum('fij,fij->f',fields['D'],fields['D']);viscous=float(np.sum(ctx['face_scale']*(strain_square+fields['divergence']**2)));misfit=float(np.sum(ctx['data'][:,None]*value**2));body_work=float(np.sum(rhs*value))
                info=dict(initial_solver);info.update(converged=True,true_stationarity_norm=_joint_norm(residual),rhs_norm=rhs_norm,relative_residual=relative,tolerance=tolerance,failure_reason=None,stationarity_recomputed=True,stationarity_kind='nonlinear_area_quality_endpoint_kkt',stationarity_formulation='original viscous objective with original endpoint area and chordal-quality inequalities',original_linear_solver=dict(initial_solver),constrained_inner_pcg_iterations=sum(x['iterations'] for x in inner),constrained_sqp_iterations=iteration,data_misfit_energy=.5*misfit,viscous_dissipation_quadratic=viscous,normalized_body_work=body_work,objective=.5*(misfit+viscous)-body_work,maximum_strain_rate_per_myr=float(np.sqrt(strain_square).max(initial=0.)),maximum_absolute_divergence_per_myr=float(abs(fields['divergence']).max(initial=0.)),maximum_radial_velocity_km_myr=float(abs(np.sum(p*value,axis=1)).max(initial=0.)))
                diag=dict(version=2,formulation='joint nonlinear original area/quality contact accommodation with Lagrangian curvature',endpoint_dt_myr=float(dt),input_face_count=len(faces),face_index_scope='input local contact-stage faces before gravity/adaptation',sqp_iterations=iteration,active_faces=[kinds[k][0] for k in active],active_sides=[kinds[k][1] for k in active],multipliers_scaled=mu.tolist(),kkt_relative_residual=relative,normalized_complementarity=comp,reaction_norm=_joint_norm(react),quality_reaction_norm=_joint_norm(reaction(local,[k for k in active if kinds[k][1]=='quality'],[m for k,m in zip(active,mu) if kinds[k][1]=='quality'])),quality_reaction_source='existing caller chordal mesh-admissibility inequality; numerical accommodation, not added tectonic driving force',endpoint_quality=Q.tolist(),quality_minimum=quality_minimum.tolist(),quality_target_inward_roundoff=(target_quality-quality_minimum).tolist(),inner_solves=inner,inverse_basis_vectors=len(cache),precise_inverse_tolerance=precise_tolerance,requested_inverse_tolerance=float(numerical_policy.inverse_tolerance),inverse_accuracy_scope='Conditioning margin: at most outer_tolerance/100; original actual operator KKT independently recomputed.',active_set_budget=int(max_active),maximum_active_bounds=max(len(x['active']) for x in history),maximum_active_support_tangent_dofs=maximum_support,nonlinear_geometry_correction_applied=False,speed_scaled_after_constraints=False,original_physical_bounds_unchanged=True,outward_orientation_preserved=True,history=history,working_set_pivots=pivots,numerical_accuracy_policy=numerical_policy.evidence(current,minimum,maximum),global_optimum_claimed=False,minimum_reduced_lagrangian_eigenvalue=reduced_minimum,reduced_curvature_difference_margin=curvature_margin,reduced_tangent_dimension=reduced_dimension,curvature_scope='Numerical local sufficient curvature in active-support compliance; two derivative scales, no global optimum or rigorous interval claim.')
                if gravity_area_targeting:
                    diag['gravity_area_targeting']=dict(enabled=True,updates=list(gravity_area_target_updates),
                        original_nominal_bounds_and_credit_unchanged=True,raw_nominal_complementarity_retained=True)
                if contact_hard_bounds:
                    diag['contact_area_solver_targets']=area_target_diagnostic
                    diag['actual_area_solver_target_feasible']=bool(np.all(current>=target_minimum)&np.all(current<=target_maximum))
                    diag['original_area_complementarity_scope']='raw once-intersected admitted bounds; target-induced slack is included in unchanged complementarity check'
                    info['stationarity_formulation']='original viscous objective with once-intersected dimensional admission/hard strain area bounds and original chordal-quality inequalities'
                    diag.update(formulation='joint nonlinear admitted area/quality contact accommodation with Lagrangian curvature',
                        active_area_reaction_scope='once-intersected nominal dimensional admission and fixed stage hard strain bounds; numerical accommodation, not new tectonic force',
                        nominal_minimum_area_km2=minimum.tolist(),nominal_maximum_area_km2=maximum.tolist(),
                        effective_minimum_area_km2=solve_minimum.tolist(),effective_maximum_area_km2=solve_maximum.tolist())
                if mapped_contact:
                    diag['contact_final_return_geometry']=row['contact_final_return_geometry']
                    diag['stationarity_geometry_scope']='original-A objective; actual returned-area constraints and original assembled-local quality rows; secondary returned quality/orientation and local-area admission are fail-closed acceptance certificates only'
                    diag['constraint_count_per_face']=3
                    diag['derivative_scope']='continuous general Rodrigues and measured normalization pullbacks at actual restored/returned geometry; no derivative-of-rounding or binary64 feasibility guarantee'
                    info['stationarity_formulation']='original viscous objective with actual returned-area inequalities and original assembled-local chordal-quality inequalities'
                return q,value,info,diag
            if (contact_hard_bounds and relative<=tolerance and comp<=tolerance and np.all(mu>=0)
                    and np.all(Q>=quality_minimum) and np.all(signed>1e-15)):
                try:
                    restoration_current=current;observation_diagnostic=None
                    restoration_eligible=(not mapped_contact and secondary_geometry_admitted(value)
                        and not numerical_policy.feasible(current,minimum,maximum))
                    if mapped_contact:
                        # Area admission is an unchanged FINAL guard, not a
                        # prerequisite for observing its tiny supported miss.
                        # Both quality/orientation certificates still qualify
                        # aiming; the helper retains every reaction/scale gate.
                        restoration_eligible=False
                        if (np.all(measured['returned_quality']>=quality_minimum)
                                and np.all(measured['returned_signed']>1e-15)):
                            restoration_current,observation_diagnostic=_contact_area_target_observation(
                                current,measured['local_area'],solve_minimum,solve_maximum)
                            restoration_eligible=bool(observation_diagnostic['failed_local_faces'])
                            row['contact_area_numerical_target_observation']=observation_diagnostic
                    restored=(_restore_contact_area_targets(restoration_current,solve_minimum,solve_maximum,target_minimum,target_maximum,
                        numerical_policy.hard_stage_area_km2,np.any(~rigid[faces],axis=1),active,kinds,mu,
                        numerical_policy=numerical_policy,tolerance=tolerance,iteration=iteration,normalization_area=area)
                        if restoration_eligible else None)
                except ConstraintSolveError as error:
                    area_target_restoration_failure=getattr(error,'diagnostics',None);raise
                if restored is not None:
                    target_minimum,target_maximum,update=restored
                    if observation_diagnostic is not None:update['two_geometry_numerical_target_observation']=observation_diagnostic
                    area_target_updates.append(update)
                    area_target_diagnostic=dict(initial_area_target_diagnostic,kind='measured_contact_area_inward_arithmetic_targets',version=2,
                        initial_fixed_target_record=initial_area_target_diagnostic,measured_interior_updates=list(area_target_updates),
                        **{key:update[key] for key in ('solver_target_minimum_area_km2','solver_target_maximum_area_km2',
                            'applied_lower_inward_area_km2','applied_upper_inward_area_km2','lower_target_original_policy_feasible',
                            'upper_target_original_policy_feasible','lower_target_predicted_spent_km2','upper_target_predicted_spent_km2',
                            'target_certification_scope','representable_target_certified','limitation')})
                    row['measured_interior_target_update']=update
                    # Refresh the merit/equations for these new constants and
                    # take the ORIGINAL Newton/search in this same iteration.
                    # Never retarget repeatedly without a normal Newton step.
                    g,physical_g,local,q,current,Q,signed=constraints(value)
            if (gravity_area_targeting and relative<=tolerance and comp<=tolerance and np.all(mu>=0)
                    and np.all(Q>=quality_minimum) and np.all(signed>1e-15)
                    and secondary_geometry_admitted(value) and not numerical_policy.feasible(current,minimum,maximum)):
                try:
                    restored=_restore_gravity_area_targets(current,minimum,maximum,target_minimum,target_maximum,
                        area,np.any(~rigid[faces],axis=1),active,kinds,mu,
                        numerical_policy=numerical_policy,tolerance=tolerance,iteration=iteration)
                except ConstraintSolveError as error:
                    gravity_area_target_failure=getattr(error,'diagnostics',None);raise
                if restored is not None:
                    target_minimum,target_maximum,update=restored
                    gravity_area_target_updates.append(update);row['measured_gravity_target_update']=update
                    # The raw nominal rows and every acceptance guard are unchanged.
                    # Refresh only the aim, then take the original Newton/search step.
                    g,physical_g,local,q,current,Q,signed=constraints(value)
            if not active:raise ConstraintSolveError('Joint admissibility branch has no supported active constraint.')
            build_metric(active,value);value=project(base+np.einsum('ijk,k->ij',T,z));g,physical_g,local,q,current,Q,signed=constraints(value)
            # Local Newton equations use the constant-scaled physical rows.
            # Active-set QP pivots check every omitted linearized inequality.
            solved_sets=set()
            for pivot in range(2*len(kinds)+2):
                if convex_working_sets:
                    active,mu,dz,dm,r,g,physical_g,local,q,current,Q,signed,value,detail=convex_working_set(
                        iteration,active,mu,value,max(tolerance*.01,64*np.finfo(float).eps))
                    # The deployed solver computes dual crossing within its
                    # original pivot branch. Initialize the identical search
                    # limits here before bypassing that branch on restart.
                    crossing=np.flatnonzero(dm<0);dual_fraction=1.;crossed=None
                    if len(crossing):
                        fractions=-mu[crossing]/dm[crossing];at=int(np.argmin(fractions))
                        if fractions[at]<1.:dual_fraction=float(fractions[at]);crossed=int(crossing[at])
                    pivots.append(detail);break
                build_metric(active,value);value=project(base+np.einsum('ijk,k->ij',T,z));g,physical_g,local,q,current,Q,signed=constraints(value)
                Ja=rows(local,active);r=z-mu@Ja;H=np.eye(len(z))-np.einsum('k,kij->ij',mu,curvature(value,active))
                K=np.block([[H,-Ja.T],[Ja,np.zeros((len(active),len(active)))]])
                target=np.r_[-r,-g[active]]
                try:direction=np.linalg.solve(K,target)
                except np.linalg.LinAlgError:
                    direction=np.linalg.lstsq(K,target,rcond=1e-12)[0]
                    if _joint_norm(K@direction-target)>tolerance*.01*max(_joint_norm(target),1e-30):
                        proof=_dependent_inequality_pivot(Ja[:,None,:],-g[active],mu,with_direction=True)
                        if proof is None:raise ConstraintSolveError('Dependent joint rows lack a certified feasible dual pivot.')
                        at,detail,dual_direction=proof
                        mu=np.maximum(mu+detail['step_to_zero']*dual_direction,0.)
                        pivots.append(dict(iteration=iteration,kind='dependent_physical_row_zero_reaction',constraint=list(kinds[active.pop(at)]),**detail));mu=np.delete(mu,at)
                        if not active:raise ConstraintSolveError('Dependent joint pivot retired every reaction.')
                        continue
                else:
                    # Experimental objective-I SQP direction, corrected only
                    # within null([Ja,0]) for ORIGINAL represented-K merit
                    # descent. Neither physical rows nor acceptance change.
                    if mapped_contact and np.all(g>=0.) and np.any(mu>0.):
                        try:
                            _,singular_step,Vh_step=np.linalg.svd(Ja,full_matrices=False)
                            full_rank=(len(singular_step)==len(active) and len(singular_step)>0
                                and np.all(np.isfinite(singular_step))
                                and singular_step[-1]>64*np.finfo(float).eps*max(Ja.shape)*max(float(singular_step[0]),1e-30))
                            if full_rank:
                                Kobjective=np.block([[np.eye(len(z)),-Ja.T],[Ja,np.zeros((len(active),len(active)))]])
                                objective_direction=np.linalg.solve(Kobjective,target)
                                gradient=-K.T@target
                                projected_gradient=gradient.copy()
                                projected_gradient[:len(z)]-=Vh_step.T@(Vh_step@gradient[:len(z)])
                                projection_norm=_joint_norm(projected_gradient)
                                gradient_norm=_joint_norm(gradient)
                                target_square=float(target@target);armijo_coefficient=1e-4
                                numerator=float(gradient@objective_direction)+armijo_coefficient*target_square
                                meaningful_projection=(np.isfinite(projection_norm) and np.isfinite(gradient_norm)
                                    and projection_norm>64*np.finfo(float).eps*max(gradient_norm,1e-30))
                                if (meaningful_projection and np.all(np.isfinite(objective_direction))
                                        and np.isfinite(numerator) and target_square>0.):
                                    beta=max(numerator,0.)/float(projected_gradient@projected_gradient)
                                    corrected=objective_direction-beta*projected_gradient
                                    slope=float(gradient@corrected)
                                    original_norm=_joint_norm(direction);corrected_norm=_joint_norm(corrected)
                                    active_relative=_joint_norm(Ja@corrected[:len(z)]+g[active])/max(_joint_norm(target),1e-30)
                                    selected=bool(np.all(np.isfinite(corrected)) and np.isfinite(beta)
                                        and np.isfinite(slope) and slope<-.5*armijo_coefficient*target_square
                                        and corrected_norm<original_norm and np.isfinite(active_relative)
                                        and active_relative<=tolerance*.01)
                                    row.setdefault('objective_sqp_merit_steps',[]).append(dict(
                                        kind='objective_I_SQP_with_original_merit_descent_correction',version=1,pivot=pivot,
                                        selected=selected,original_newton_direction_norm=original_norm,
                                        objective_direction_norm=_joint_norm(objective_direction),corrected_direction_norm=corrected_norm,
                                        original_K_linear_merit_slope=slope,required_linear_merit_slope=-armijo_coefficient*target_square,
                                        predicted_active_equation_relative_residual=active_relative,
                                        active_equation_residual_normalization='original represented residual F norm',
                                        beta=beta,projected_merit_gradient_norm=projection_norm,
                                        objective_linear_merit_slope=float(gradient@objective_direction),
                                        correction_nullspace_residual=_joint_norm(Ja@projected_gradient[:len(z)]),
                                        corrected_original_newton_equation_relative_residual=_joint_norm(K@corrected-target)/max(_joint_norm(target),1e-30),
                                        exact_original_newton_equation_claimed=False,physical_objective_optimum_claimed=False,
                                        actual_nonlinear_merit_guard_unchanged=True,
                                        coordinate_scope='combined original whitened P1 primal and scaled dual coordinates; unit-metric correction to objective-SQP direction',
                                        descent_scope='represented original-K model with all current g nonnegative; no actual nonlinear or recovery guarantee'))
                                    if selected:direction=corrected
                        except np.linalg.LinAlgError:
                            # The already-resolved original Newton direction is
                            # retained; dependent-row fallback is not broadened.
                            row.setdefault('objective_sqp_merit_fallbacks',[]).append(dict(pivot=pivot,reason='objective_SQP_or_projection_linear_algebra_unresolved',original_direction_retained=True))
                dz=direction[:len(z)];dm=direction[len(z):]
                negative=np.flatnonzero((mu==0)&(dm < -64*np.finfo(float).eps*max(float(np.max(abs(mu+dm),initial=0.)),1e-30)))
                if len(negative):
                    at=int(negative[np.argmin(dm[negative])]);pivots.append(dict(iteration=iteration,kind='zero_reaction_retired',constraint=list(kinds[active.pop(at)])));mu=np.delete(mu,at)
                    if not active:raise ConstraintSolveError('Joint working set lost every admissible reaction.')
                    continue
                # Use the existing dual-admissible coupled step when checking
                # omitted bounds; the full Newton direction may cross a
                # positive reaction zero before reaching that prediction.
                # Retire positive reactions only after the unchanged actual
                # nonlinear merit search accepts the coupled primal/dual step.
                # Stop exactly at a dual zero before retiring that inequality.
                # Repeatedly halving toward zero stalls without changing the set.
                crossing=np.flatnonzero(dm<0);dual_fraction=1.;crossed=None
                if len(crossing):
                    fractions=-mu[crossing]/dm[crossing];at=int(np.argmin(fractions))
                    if fractions[at]<1.:dual_fraction=float(fractions[at]);crossed=int(crossing[at])
                allJ=rows(local,list(range(len(kinds))));predicted=g+dual_fraction*(allJ@dz)
                omitted=[k for k in np.flatnonzero(predicted < -max(tolerance*.01,64*np.finfo(float).eps)) if int(k) not in active]
                if omitted:
                    key=int(min(omitted,key=lambda k:predicted[k]));signature=tuple(active)
                    if signature in solved_sets:raise ConstraintSolveError('Joint linearized working set cycled; no endpoint accepted.')
                    solved_sets.add(signature)
                    if len(active)>=max_active:raise ConstraintSolveError('Joint linearized active-set budget exhausted.')
                    active.append(key);mu=np.r_[mu,0.];pivots.append(dict(iteration=iteration,kind='omitted_linearized_constraint',constraint=list(kinds[key]),dual_step_fraction=dual_fraction))
                    continue
                break
            else:raise ConstraintSolveError('Joint inequality pivots exhausted their bounded budget.')
            merit=.5*(float(r@r)+float(g[active]@g[active])+float(np.minimum(g,0.)@np.minimum(g,0.)));accepted=False
            for halving in range(max_step_halvings+1):
                deadline_check();alpha=dual_fraction*.5**halving;tm=mu+alpha*dm
                if crossed is not None and halving==0:tm[crossed]=0.
                if np.any(tm<0):continue
                tz=z+alpha*dz;tv=project(base+np.einsum('ijk,k->ij',T,tz));tg,tpg,tl,tq,ta,tQ,ts=constraints(tv)
                if np.any(ts<=1e-15):continue
                tJ=rows(tl,active);tr=tz-tm@tJ;tmrt=.5*(float(tr@tr)+float(tg[active]@tg[active])+float(np.minimum(tg,0.)@np.minimum(tg,0.)))
                if np.isfinite(tmrt) and tmrt<=merit*(1-1e-4*alpha):
                    z=tz;value=tv;mu=tm;accepted=True;row['step_scale']=alpha;row['step_halvings']=halving
                    if crossed is not None and halving==0:
                        pivots.append(dict(iteration=iteration,kind='positive_reaction_to_zero',constraint=list(kinds[active.pop(crossed)]),dual_step_fraction=dual_fraction));mu=np.delete(mu,crossed)
                    break
            if not accepted:raise ConstraintSolveError('Joint Newton search failed to reduce actual nonlinear KKT merit.')
        raise ConstraintSolveError('Joint Newton exhausted the original32 outer-iteration budget.')
    except BaseException as error:
        # Attach complete evidence before the caller handles rollback. Do not
        # discard the endpoint merely because certification/quality failed.
        if isinstance(error,Exception):
            try:q,A,Q,signed=geometry(value);last=dict(last_endpoint=q.tolist(),last_area_km2=A.tolist(),last_quality=Q.tolist())
            except Exception as diagnostic_error:last=dict(last_endpoint=None,last_area_km2=None,last_quality=None,endpoint_diagnostic_error=str(diagnostic_error))
            error.diagnostics=dict(version=2,converged=False,history=history,working_set_pivots=pivots,last_velocity=value.tolist(),quality_minimum=quality_minimum.tolist(),inner_solves=inner,maximum_active_support_tangent_dofs=maximum_support,original_physical_bounds_unchanged=True,no_posthoc_geometry_correction=True,**last)
            if gravity_area_targeting:
                error.diagnostics['gravity_area_targeting']=dict(enabled=True,updates=list(gravity_area_target_updates),
                    target_construction_failure=gravity_area_target_failure,original_nominal_bounds_and_credit_unchanged=True,
                    raw_nominal_complementarity_retained=True)
            if area_target_diagnostic is not None:
                error.diagnostics['contact_area_solver_targets']=area_target_diagnostic
                error.diagnostics['original_area_complementarity_scope']='raw once-intersected admitted bounds; solver inward targets never replace final admission'
            if area_target_restoration_failure is not None:
                error.diagnostics['contact_area_target_restoration_failure']=area_target_restoration_failure
            if mapped_contact:
                try:
                    measured=geometry_details(value)
                    error.diagnostics['contact_final_return_geometry']=dict(kind='actual_final_owner_return_geometry',version=1,
                        returned_endpoint=measured['returned'].tolist(),returned_area_km2=measured['current'].tolist(),
                        assembled_local_area_km2=measured['local_area'].tolist(),returned_quality=measured['returned_quality'].tolist(),
                        returned_signed_orientation=measured['returned_signed'].tolist(),
                        secondary_quality_certificate_only=True,accepted=False,
                        area_geometry_scope='actual full-layout selected global faces; tentative failed endpoint only')
                except Exception as diagnostic_error:
                    error.diagnostics['contact_final_return_geometry_error']=str(diagnostic_error)
            if isinstance(error,np.linalg.LinAlgError):
                failure=ConstraintSolveError('Joint Newton linearization is unresolved: '+str(error));failure.diagnostics=error.diagnostics;raise failure from error
        raise
