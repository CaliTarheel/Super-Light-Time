"""Tangent P1 thin-viscous-sheet velocities, with plane-stress coupling.

Coordinates are unit spherical points; radius/length and area are km/km²;
velocities and normalized body force are km/Myr. L² = 2 eta H / basal_drag.
The bilinear viscous term is sum area L² (D(u):D(v) + div(u) div(v)).
Each D uses the embedded planar triangle's P1 gradient projected into its
plane. Spherical triangle areas supply quadrature. A common Euler velocity
has zero symmetric strain on each planar triangle, including near poles.

This module only solves velocities. It does not move geometry, alter columns,
limit heights, choose viscosity, change mass, or claim a nonlinear time step.
"""
from __future__ import annotations

import numpy as np
from parallel_runtime import read_inputs

VERSION = 1


def independent_inverse(job):
    """Solve one shared immutable tangent-basis right-hand side in a worker."""
    shared, node, vector = job
    with read_inputs(shared) as data:
        points=data['points']
        rhs=np.zeros_like(points);rhs[node]=vector
        body=np.divide(rhs,data['data'][:,None],out=np.zeros_like(rhs),where=data['data'][:,None]>0)
        return solve(points,data['faces'],data['area'],np.zeros_like(points),data['rigid'],
            np.zeros(len(points),bool),data['length_km'],radius=data['radius'],body_force=body,
            viscosity_weights=data['viscosity_weights'],iterations=data['iterations'],tolerance=data['tolerance'])


def _tangent(value, points):
    return value-np.einsum('ij,ij->i', value, points)[:, None]*points


def _vectors(value, count, name):
    array=np.asarray(value, float)
    if array.shape!=(count,3) or not np.isfinite(array).all():
        raise ValueError(f'{name} must contain one finite three-vector per vertex.')
    return array.copy()


def _difference_operator(gradient, plane, scale):
    """Precompute the unchanged face law on its two velocity differences.

    Columns correspond to Cartesian components of v1-v0 and v2-v0. Keeping
    these differences explicit preserves cancellation of a common translation
    on small triangles. The six basis evaluations use the same two-sided
    projections as the original strain/stress law; no projector idempotence or
    continuum approximation is assumed. Storage is 54 float64 values per face.
    """
    result=np.empty((len(gradient),9,6),float)
    raw=np.zeros((len(gradient),3,3),float)
    for edge in range(2):
        for axis in range(3):
            raw.fill(0.)
            raw[:,axis,:]=gradient[:,edge+1]
            projected=plane@raw@plane
            strain=.5*(projected+projected.transpose(0,2,1))
            stress=strain+np.trace(strain,axis1=1,axis2=2)[:,None,None]*plane
            stress=plane@stress@plane
            force=np.einsum('fij,fkj->fki',stress,gradient)*scale[:,None,None]
            result[:,:,3*edge+axis]=force.reshape(len(gradient),9)
    if not np.isfinite(result).all():
        raise ValueError('Sheet difference operator is non-finite.')
    return result


def prepare(points, faces, area_km2, length_km, *, radius=6371., viscosity_weights=None):
    """Build the P1 operator and exact local block-Jacobi contributions."""
    points=np.asarray(points,float)
    if points.ndim!=2 or points.shape[1:]!=(3,) or not np.isfinite(points).all():
        raise ValueError('Sheet points must be finite Nx3 unit spherical vectors.')
    norms=np.linalg.norm(points,axis=1)
    if not np.allclose(norms,1.,rtol=0.,atol=2e-10):
        raise ValueError('Sheet points must have unit radius; radius is supplied in km.')
    points=points/norms[:,None]
    faces=np.asarray(faces)
    if (faces.ndim!=2 or faces.shape[1:]!=(3,) or faces.dtype.kind not in 'iu'
            or np.any(faces<0) or np.any(faces>=len(points))):
        raise ValueError('Sheet faces must be valid integer vertex triples.')
    faces=faces.astype(np.int64,copy=True)
    area=np.asarray(area_km2,float)
    if area.shape!=(len(faces),) or not np.isfinite(area).all() or np.any(area<=0):
        raise ValueError('Sheet quadrature requires positive finite spherical face areas in km².')
    for value,name,positive in ((radius,'radius',True),(length_km,'length_km',False)):
        if isinstance(value,(bool,np.bool_)) or not np.isscalar(value) or not np.isfinite(value) or (value<=0 if positive else value<0):
            raise ValueError(f'Sheet {name} must be finite and '+('positive.' if positive else 'nonnegative.'))
    weights=np.ones(len(faces)) if viscosity_weights is None else np.asarray(viscosity_weights,float)
    if weights.ndim==0:weights=np.full(len(faces),float(weights))
    if weights.shape!=(len(faces),) or not np.isfinite(weights).all() or np.any(weights<0):
        raise ValueError('Viscosity weights must be finite nonnegative face multipliers.')
    triangle=points[faces]*float(radius)
    edge1=triangle[:,1]-triangle[:,0];edge2=triangle[:,2]-triangle[:,0]
    cross=np.cross(edge1,edge2);twice_area=np.linalg.norm(cross,axis=1)
    if np.any(twice_area<=0) or not np.isfinite(twice_area).all():
        raise ValueError('Sheet triangles must have a finite nondegenerate planar gradient.')
    normal=cross/twice_area[:,None]
    gradient1=np.cross(edge2,normal)/twice_area[:,None]
    gradient2=np.cross(normal,edge1)/twice_area[:,None]
    gradient=np.stack((-gradient1-gradient2,gradient1,gradient2),axis=1)
    plane=np.eye(3)[None]-normal[:,:,None]*normal[:,None,:]
    scale=area*float(length_km)**2*weights
    if not np.isfinite(scale).all() or not np.isfinite(gradient).all():
        raise ValueError('Sheet gradient or viscosity scale overflows the requested units.')
    data=np.bincount(faces.ravel(),weights=np.repeat(area/3.,3),minlength=len(points))
    diagonal=data[:,None,None]*np.eye(3)[None]
    for corner in range(3):
        g=gradient[:,corner]
        local=(.5*np.einsum('ij,ij->i',g,g)[:,None,None]*plane
               +1.5*g[:,:,None]*g[:,None,:])*scale[:,None,None]
        np.add.at(diagonal,faces[:,corner],local)
    if not np.isfinite(diagonal).all():
        raise ValueError('Sheet operator diagonal is non-finite.')
    largest_edge2=np.maximum.reduce((np.einsum('ij,ij->i',edge1,edge1),
        np.einsum('ij,ij->i',edge2,edge2),np.einsum('ij,ij->i',edge2-edge1,edge2-edge1)))
    return dict(points=points,faces=faces,area_km2=area.copy(),gradient=gradient,plane=plane,
        face_scale=scale,data=data,diagonal=diagonal,radius_km=float(radius),length_km=float(length_km),
        difference_operator=_difference_operator(gradient,plane,scale),
        viscosity_weights=weights.copy(),minimum_planar_quality=float(np.min(twice_area/largest_edge2,initial=1.)))


def strain_rate(context, velocity):
    """Return per-face D[face,3,3] and divergence[face], in inverse Myr."""
    value=_tangent(_vectors(velocity,len(context['points']),'Sheet velocity'),context['points'])
    face=value[context['faces']];gradient=context['gradient'];plane=context['plane']
    # Differences give exact constant-velocity cancellation and avoid a large
    # common translation degrading a small triangle's affine gradient.
    raw=((face[:,1]-face[:,0])[:,:,None]*gradient[:,1,None,:]
         +(face[:,2]-face[:,0])[:,:,None]*gradient[:,2,None,:])
    projected=plane@raw@plane
    strain=.5*(projected+projected.transpose(0,2,1))
    return dict(D=strain,divergence=np.trace(strain,axis1=1,axis2=2))


def apply(context, velocity):
    """Apply lumped drag plus the symmetric plane-stress viscous operator."""
    value=_tangent(_vectors(velocity,len(context['points']),'Sheet velocity'),context['points'])
    # strain_rate performs a second tangent projection; retain it exactly.
    face=_tangent(value,context['points'])[context['faces']]
    difference=np.stack((face[:,1]-face[:,0],face[:,2]-face[:,0]),axis=1).reshape(-1,6)
    contribution=np.einsum('fij,fj->fi',context['difference_operator'],difference).reshape(-1,3,3)
    result=context['data'][:,None]*value
    for corner in range(3):np.add.at(result,context['faces'][:,corner],contribution[:,corner])
    return _tangent(result,context['points'])


def _constraints(points, target, rigid, driven, normals):
    normal_mode=normals is not None
    if normal_mode:
        normal=_tangent(_vectors(normals,len(points),'Driven normals'),points)
        lengths=np.linalg.norm(normal,axis=1)
        if np.any(lengths[driven]<=1e-12):
            raise ValueError('Every normal-driven vertex requires a nonzero tangent front normal.')
        normal[driven]/=lengths[driven,None]
        normal[~driven]=0.
        fixed=np.where(driven[:,None],normal*np.einsum('ij,ij->i',normal,target)[:,None],0.)
    else:
        normal=np.zeros_like(points)
        fixed=np.where(driven[:,None],target,0.)
    fixed[rigid]=0.
    free=~(rigid|driven)
    def project(value):
        result=_tangent(value,points)
        result[rigid]=0.
        if normal_mode:
            result-=normal*np.einsum('ij,ij->i',normal,result)[:,None]
        else:result[driven]=0.
        return result
    return fixed,free,normal,project


def _preconditioner(context, free, driven, normal):
    points=context['points'];diagonal=context['diagonal']
    # Solve each genuine tangent block. An artificial radial identity can
    # disappear beside a very stiff/slender element and corrupt a 3D inverse.
    axes=np.eye(3)[np.argmin(np.abs(points),axis=1)]
    first=np.cross(points,axes);first/=np.linalg.norm(first,axis=1)[:,None]
    second=np.cross(points,first)
    basis=np.stack((first,second),axis=2)
    block=np.einsum('nia,nij,njb->nab',basis[free],diagonal[free],basis[free])
    if len(block):
        eigen=np.linalg.eigvalsh(block)
        if not np.isfinite(eigen).all() or np.any(eigen<=0):
            raise np.linalg.LinAlgError('Projected block-Jacobi diagonal lost positive definiteness.')
        inverse=np.linalg.inv(block)
        condition=float(np.max(eigen[:,1]/eigen[:,0]))
    else:inverse=np.empty((0,2,2));condition=1.
    partial=driven & (np.linalg.norm(normal,axis=1)>0)
    along=np.cross(points,normal)
    one=np.einsum('ni,nij,nj->n',along[partial],diagonal[partial],along[partial])
    if np.any(one<=0) or not np.isfinite(one).all():
        raise np.linalg.LinAlgError('Normal-driven tangent diagonal is not positive.')
    def precondition(value):
        result=np.zeros_like(value)
        local=np.einsum('nia,ni->na',basis[free],value[free])
        result[free]=np.einsum('nia,nab,nb->ni',basis[free],inverse,local)
        result[partial]=along[partial]*(np.einsum('ni,ni->n',along[partial],value[partial])/one)[:,None]
        return result
    return precondition,condition


def solve(points, faces, area_km2, target_velocity, rigid_mask, driven_mask, length_km, *,
          radius=6371., body_force=None, viscosity_weights=None, driven_normals=None,
          iterations=400, tolerance=1e-8, extension_iterations=0):
    """Return (tangent velocity, scalar diagnostics), failing closed on failure.

    Rigid vertices fix zero residual velocity. Full driven targets are the
    default; driven_normals constrains only their prescribed normal component.
    Other tangent components minimize the same soft-target/data objective.
    A valid but unconverged solve emits all-zero velocity and converged=False.
    Invalid input raises ValueError. Per-face fields are available through
    strain_rate(prepare(...), velocity), avoiding large arrays in diagnostics.
    An optional bounded extension retains the original iteration-boundary
    recomputation/restart and every response certified within that budget.
    """
    if isinstance(iterations,(bool,np.bool_)) or not isinstance(iterations,(int,np.integer)) or iterations<1:
        raise ValueError('Sheet solver iterations must be a positive integer.')
    if isinstance(extension_iterations,(bool,np.bool_)) or not isinstance(extension_iterations,(int,np.integer)) or extension_iterations<0:
        raise ValueError('Sheet solver extension budget must be a nonnegative integer.')
    maximum_iterations=int(iterations)+int(extension_iterations)
    if isinstance(tolerance,(bool,np.bool_)) or not np.isscalar(tolerance) or not np.isfinite(tolerance) or not 0<tolerance<1:
        raise ValueError('Sheet solver tolerance must lie strictly between zero and one.')
    context=prepare(points,faces,area_km2,length_km,radius=radius,viscosity_weights=viscosity_weights)
    points=context['points'];count=len(points)
    target=_tangent(_vectors(target_velocity,count,'Target velocity'),points)
    body=np.zeros_like(target) if body_force is None else _tangent(_vectors(body_force,count,'Normalized body force'),points)
    masks=[]
    for value,name in ((rigid_mask,'rigid'),(driven_mask,'driven')):
        mask=np.asarray(value)
        if mask.shape!=(count,) or mask.dtype.kind!='b':raise ValueError(f'Sheet {name} mask must be an aligned boolean array.')
        masks.append(mask.copy())
    rigid=masks[0]|(context['data']==0);driven=masks[1]&~rigid
    fixed,free,normal,project=_constraints(points,target,rigid,driven,driven_normals)
    rhs=context['data'][:,None]*(target+body)
    reduced=project(rhs-apply(context,fixed))
    scale=float(np.linalg.norm(reduced));threshold=float(tolerance)*scale
    invalid_rhs_norm=not np.isfinite(scale) or (scale==0. and np.any(reduced!=0.))
    solution=project(target)
    residual=project(rhs-apply(context,fixed+solution))
    norm=float(np.linalg.norm(residual));initial_norm=norm
    converged=norm<=threshold and not invalid_rhs_norm
    used=0;failure='arithmetic_rhs_norm_range_failure' if invalid_rhs_norm else None;condition=None;restarts=0
    if not converged and not invalid_rhs_norm:
        try:
            precondition,condition=_preconditioner(context,free,driven,normal)
            z=project(precondition(residual));direction=z.copy();rz=float(np.sum(residual*z))
            for used in range(1,maximum_iterations+1):
                product=project(apply(context,direction));curvature=float(np.sum(direction*product))
                if not np.isfinite(curvature) or curvature<=0 or not np.isfinite(rz) or rz<=0:
                    failure='nonpositive_or_nonfinite_projected_curvature';break
                solution+=direction*(rz/curvature)
                residual-=product*(rz/curvature)
                norm=float(np.linalg.norm(residual))
                # Retain the conjugate subspace between reliability checks.
                # A claimed recursive convergence is always checked against
                # the actual operator; if it was false, restart from that
                # recomputed residual. Preserve that check at the original
                # budget boundary before any extension. The final exact
                # stationarity gate and its tolerance remain mandatory.
                recompute=norm<=threshold or used==iterations or used==maximum_iterations
                if recompute:
                    solution=project(solution)
                    residual=project(rhs-apply(context,fixed+solution))
                    norm=float(np.linalg.norm(residual))
                    if norm<=threshold:converged=True;break
                z=project(precondition(residual));new_rz=float(np.sum(residual*z))
                if recompute:
                    direction=z.copy();restarts+=1
                else:direction=z+direction*(new_rz/rz)
                rz=new_rz
        except np.linalg.LinAlgError as error:
            failure='invalid_block_preconditioner: '+str(error)
    attempted=project(solution)+fixed
    exact_residual=project(rhs-apply(context,attempted))
    norm=float(np.linalg.norm(exact_residual))
    invalid_stationarity_norm=not np.isfinite(norm) or (norm==0. and np.any(exact_residual!=0.))
    converged=bool(converged and not invalid_rhs_norm and not invalid_stationarity_norm
        and np.isfinite(attempted).all() and np.isfinite(norm) and norm<=threshold)
    if invalid_stationarity_norm and failure is None:failure='arithmetic_stationarity_norm_range_failure'
    if not converged and failure is None:failure='iteration_limit_or_true_stationarity_not_reached'
    velocity=attempted if converged else np.zeros_like(target)
    velocity[rigid]=0.
    fields=strain_rate(context,velocity)
    strain_square=np.einsum('fij,fij->f',fields['D'],fields['D'])
    viscous=float(np.sum(context['face_scale']*(strain_square+fields['divergence']**2)))
    misfit=float(np.sum(context['data'][:,None]*(velocity-target)**2))
    body_work=float(np.sum(context['data'][:,None]*body*velocity))
    diagnostics=dict(version=VERSION,model='P1 tangent thin viscous sheet with plane-stress incompressible coupling',
        converged=converged,iterations=int(used),maximum_iterations=maximum_iterations,
        initial_iteration_budget=int(iterations),extension_iterations=int(extension_iterations),tolerance=float(tolerance),
        failure_reason=failure,true_stationarity_norm=norm,rhs_norm=scale,
        relative_residual=None if invalid_rhs_norm or invalid_stationarity_norm else (norm/scale if scale>0 else (0. if norm==0 else None)),
        initial_stationarity_norm=initial_norm,residual_restarts=restarts,
        preconditioner='projected tangent block Jacobi',maximum_block_condition=condition,
        constrained_vertices=int((rigid|driven).sum()),rigid_vertices=int(rigid.sum()),driven_vertices=int(driven.sum()),
        normal_only_driven=driven_normals is not None,unused_vertices=int(np.count_nonzero(context['data']==0)),
        minimum_planar_quality=context['minimum_planar_quality'],
        data_misfit_energy=.5*misfit,viscous_dissipation_quadratic=viscous,normalized_body_work=body_work,
        objective=.5*(misfit+viscous)-body_work,
        maximum_strain_rate_per_myr=float(np.sqrt(strain_square).max(initial=0.)),
        maximum_absolute_divergence_per_myr=float(np.abs(fields['divergence']).max(initial=0.)),
        maximum_radial_velocity_km_myr=float(np.abs(np.einsum('ij,ij->i',points,velocity)).max(initial=0.)),
        emitted_zero_on_failure=not converged,length_km=float(length_km),radius_km=float(radius),
        constitutive_law='2 eta H [D(u):D(v) + div(u) div(v)]; L²=2 eta H/basal_drag',
        stationarity_recomputed=True,geometry_or_mass_changed=False)
    return velocity,diagnostics
