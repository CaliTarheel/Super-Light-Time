"""Strict saved constrained-gravity diagnostics, shared by writer and sampler."""
import numpy as np
# The saved-frame validator must accept exactly what the solver accepts. This is
# the third site that has to agree on feasibility, after bounded_gravity's own
# acceptance test and gravitational_relaxation's backtracking gate: a frame the
# solver produced and relaxation approved was still rejected here at write time,
# failing the run at 318 Myr. Import the constant rather than restate it.
from bounded_gravity import FEASIBILITY_TOLERANCE
import restart_boundary

def require(ok,message):
    if not ok:raise ValueError(message)

def _number(row, key, minimum=None, maximum=None):
    require(isinstance(row,dict),'Invalid constrained-gravity metadata object.')
    value=row.get(key)
    require(type(value) in (int,float) and np.isfinite(value)
        and (minimum is None or value>=minimum) and (maximum is None or value<=maximum),
        'Invalid constrained-gravity scalar: '+key)
    return float(value)


def _integer(row,key,minimum=0):
    value=row.get(key)
    require(type(value) is int and value>=minimum,'Invalid constrained-gravity integer: '+key)
    return value


def _solver(row, maximum_tolerance=1e-8):
    require(isinstance(row,dict) and row.get('converged') is True
        and row.get('stationarity_recomputed') is True,'Missing recomputed gravity stationarity.')
    norm=_number(row,'true_stationarity_norm',0)
    rhs=_number(row,'rhs_norm',0)
    tolerance=_number(row,'tolerance',0,maximum_tolerance)
    require(tolerance>0 and norm<=tolerance*rhs,'Gravity stationarity exceeds its declared tolerance.')
    relative=_number(row,'relative_residual',0,maximum_tolerance)
    require(np.isclose(relative,norm/rhs if rhs else 0.,rtol=1e-10,atol=1e-15),
        'Gravity relative stationarity differs from the recorded physical norm.')
    require(_integer(row,'iterations')<=_integer(row,'maximum_iterations',1),
        'Gravity solver exceeded its iteration budget.')


def validate_accepted_row(row):
    """Validate saved accepted-interval arithmetic, not a replacement for the oracle."""
    step=_number(row,'dt_myr',0)
    require(step>0,'An accepted gravity interval must advance physical time.')
    import numerical_accuracy
    policy=row.get('numerical_accuracy_policy')
    if policy is not None:numerical_accuracy.validate_evidence(policy)
    tolerance=1e-8 if policy is None else numerical_accuracy.OUTER_RELATIVE_TOLERANCE
    solver=row.get('solver');_solver(solver,tolerance)
    kind=solver.get('stationarity_kind')
    bound=row.get('bound_constraints')
    if bound is None:
        require(kind=='unconstrained_linear'
            and row.get('stationary_within_constraint_kkt') is not True,
            'Unconstrained roundoff/predictor diagnostics cannot claim constrained equilibrium.')
        return
    if isinstance(bound,dict) and bound.get('version')==2:
        require(policy is not None,'Joint area/quality constraints require declared numerical accuracy.')
        _validate_joint_row(row,bound,solver,policy)
        return
    require(isinstance(bound,dict) and type(bound.get('version')) is int and bound['version']==1,
        'Accepted constrained gravity requires its exact method version.')
    zero=bound.get('stationary_current_geometry',False)
    require(type(zero) is bool,'Invalid constrained equilibrium discriminator.')
    expected_kind='current_geometry_zero_kkt' if zero else 'nonlinear_endpoint_kkt'
    formulation=('zero-velocity current-geometry KKT at physical area bounds' if zero else
        'same viscous quadratic with nonlinear spherical endpoint-area inequalities')
    require(kind==expected_kind and bound.get('formulation')==formulation,
        'Accepted constrained gravity uses a different stationarity formulation.')
    require(_number(bound,'endpoint_dt_myr',0)==step,
        'Constraint endpoint interval differs from the actually accepted interval.')
    face_count=_integer(bound,'input_face_count',1)
    faces=bound.get('active_faces');sides=bound.get('active_sides')
    require(isinstance(faces,list) and bool(faces) and isinstance(sides,list) and len(sides)==len(faces)
        and all(type(index) is int and 0<=index<face_count for index in faces)
        and all(side in ('minimum','maximum') for side in sides)
        and len(set(zip(faces,sides)))==len(faces),
        'Accepted gravity constraint rows are duplicated, misaligned or outside their input geometry.')
    arrays={}
    for name in ('multipliers_normalized','constraint_jacobian_norm_km','endpoint_area_km2','physical_bound_km2'):
        values=bound.get(name)
        require(isinstance(values,list) and len(values)==len(faces)
            and all(type(value) in (int,float) and np.isfinite(value) for value in values),
            'Invalid aligned gravity constraint array: '+name)
        arrays[name]=np.asarray(values,float)
    require(np.all(arrays['multipliers_normalized']>=0)
        and all(np.all(arrays[name]>0) for name in ('constraint_jacobian_norm_km','endpoint_area_km2','physical_bound_km2')),
        'Gravity reactions or physical constraint dimensions are invalid.')
    sign=np.asarray([1. if side=='minimum' else -1. for side in sides])
    gap=sign*(arrays['endpoint_area_km2']-arrays['physical_bound_km2'])/arrays['physical_bound_km2']
    require(np.all(gap>=-FEASIBILITY_TOLERANCE),'Saved accepted endpoint violates its physical bound.')
    kkt=_number(bound,'kkt_relative_residual',0,1e-8)
    require(np.isclose(kkt,solver['relative_residual'],rtol=1e-10,atol=1e-15),
        'Constraint and solver stationarity refer to different accepted motions.')
    _number(bound,'normalized_complementarity',0,1e-8)
    _number(bound,'maximum_lower_relative_violation',0,FEASIBILITY_TOLERANCE)
    _number(bound,'maximum_upper_relative_violation',0,FEASIBILITY_TOLERANCE)
    require(bound.get('nonlinear_geometry_correction_applied') is False
        and bound.get('speed_scaled_after_constraints') is False
        and bound.get('original_physical_bounds_unchanged') is True,
        'Accepted constraints were followed by a geometry/speed correction or different physical bounds.')
    inner=bound.get('inner_solves')
    require(isinstance(inner,list),'Constrained solve omits its inner stationarity evidence.')
    # Inner inverse-basis solves store compact summaries; the accepted full
    # reaction-inclusive norm is checked above and independently by the oracle.
    for item in inner:
        require(isinstance(item,dict) and item.get('converged') is True,
            'A constrained inner solve did not converge.')
        tolerance=_number(item,'tolerance',0,1e-8)
        require(tolerance>0 and _number(item,'relative_residual',0)<=tolerance
            and _integer(item,'iterations')<=_integer(item,'maximum_iterations',1),
            'A constrained inner solve exceeds its own tolerance or iteration budget.')
    if zero:
        require(row.get('stationary_within_constraint_kkt') is True
            and row.get('stationary_within_energy_roundoff') is True
            and _number(row,'virtual_power_km4_myr')==0.,
            'Current-geometry constrained equilibrium must record exact zero motion/work.')
        require(np.all(np.abs(arrays['endpoint_area_km2']-arrays['physical_bound_km2'])
            <=128*np.finfo(float).eps*arrays['endpoint_area_km2']),
            'Zero-motion reactions act on a physical bound that is not currently tight.')
    else:
        require(row.get('stationary_within_constraint_kkt') is not True
            and row.get('stationary_within_energy_roundoff') is not True
            and _number(row,'velocity_safety_scale')==1.
            and _number(row,'virtual_power_km4_myr')<0.,
            'Moving constrained gravity changed its accepted velocity or mislabels stationarity.')


def _validate_joint_row(row,bound,solver,policy):
    """Separate versioned area/quality formulation; legacy schema stays exact."""
    import numerical_accuracy
    require(type(bound.get('version')) is int and bound['version']==2
        and solver.get('stationarity_kind')=='nonlinear_area_quality_endpoint_kkt'
        and bound.get('formulation')=='joint nonlinear original area/quality contact accommodation with Lagrangian curvature',
        'Invalid joint area/quality convergence formulation.')
    require(_number(bound,'endpoint_dt_myr',0)==row['dt_myr'],'Joint endpoint uses a different accepted interval.')
    count=_integer(bound,'input_face_count',1)
    faces=bound.get('active_faces');sides=bound.get('active_sides');multipliers=bound.get('multipliers_scaled')
    require(isinstance(faces,list) and isinstance(sides,list) and isinstance(multipliers,list)
        and len(faces)==len(sides)==len(multipliers)
        and all(type(i) is int and 0<=i<count for i in faces)
        and all(side in ('minimum','maximum','quality') for side in sides)
        and len(set(zip(faces,sides)))==len(faces)
        and all(type(v) in (int,float) and np.isfinite(v) and v>=0 for v in multipliers),
        'Invalid joint reaction identities or multipliers.')
    evidence=numerical_accuracy.validate_evidence(bound.get('numerical_accuracy_policy'))
    require(evidence.get('input_face_count')==count,'Joint area evidence has a different material scope.')
    quality=[]
    for name in ('endpoint_quality','quality_minimum'):
        value=bound.get(name)
        require(isinstance(value,list) and len(value)==count
            and all(type(v) in (int,float) and np.isfinite(v) and v>0 for v in value),
            'Invalid joint shape-quality evidence: '+name)
        quality.append(np.asarray(value,float))
    require(np.all(quality[0]>=quality[1]),'Accepted joint endpoint violates the unchanged shape-quality floor.')
    kkt=_number(bound,'kkt_relative_residual',0,numerical_accuracy.OUTER_RELATIVE_TOLERANCE)
    require(np.isclose(kkt,solver['relative_residual'],rtol=1e-10,atol=1e-15),'Joint KKT and velocity certification differ.')
    _number(bound,'normalized_complementarity',0,numerical_accuracy.OUTER_RELATIVE_TOLERANCE)
    _number(bound,'quality_reaction_norm',0)
    require(bound.get('nonlinear_geometry_correction_applied') is False
        and bound.get('speed_scaled_after_constraints') is False
        and bound.get('original_physical_bounds_unchanged') is True
        and bound.get('outward_orientation_preserved') is True,
        'Joint constraints changed the physical bounds or corrected geometry/speed afterward.')
    require(row.get('stationary_within_constraint_kkt') is not True
        and row.get('stationary_within_energy_roundoff') is not True
        and _number(row,'velocity_safety_scale')==1. and _number(row,'virtual_power_km4_myr')<0.,
        'Moving joint-constrained gravity mislabels work or modifies its accepted velocity.')
    inner=bound.get('inner_solves')
    require(isinstance(inner,list),'Joint constraints omit their inverse evidence.')
    precise=_number(bound,'precise_inverse_tolerance',0,numerical_accuracy.INVERSE_RELATIVE_TOLERANCE)
    require(precise>0,'Joint inverse tolerance must be positive.')
    for item in inner:
        require(isinstance(item,dict) and item.get('converged') is True,'Uncertified joint inverse response.')
        tol=_number(item,'tolerance',0,precise)
        require(tol>0 and _number(item,'relative_residual',0)<=tol
            and _integer(item,'iterations')<=_integer(item,'maximum_iterations',1),
            'Joint inverse exceeds its own tighter conditioning tolerance or budget.')


def validate_gravity_policies(frames):
    require(isinstance(frames,list) and bool(frames),'Gravity successor requires all saved native epochs.')
    for frame in frames:
        require(isinstance(frame,dict),'Invalid constrained-gravity frame.')
        require(type(frame.get('mesh_version')) is int and frame['mesh_version']==1
            and type(frame.get('material_mechanics_version')) is int and frame['material_mechanics_version']==1,
            'Constrained gravity requires native coupled material mechanics.')
        require(type(frame.get('gravity_constraint_version')) is int and frame['gravity_constraint_version']==1,
            'Every new frame, including epoch zero, requires gravity_constraint_version=1.')
        if restart_boundary.validate_frame(frame):
            continue
        gravity=_gravity(frame)
        if _number(frame,'time_myr',0)==0:
            require(not gravity.get('internal_steps'),'A time-zero frame cannot claim an integrated gravity sequence.')
            if 'gravity_constraint_version' in gravity:
                require(type(gravity['gravity_constraint_version']) is int and gravity['gravity_constraint_version']==1,
                    'Integrated gravity policy differs from its saved frame marker.')
            continue
        require(type(gravity.get('gravity_constraint_version')) is int and gravity['gravity_constraint_version']==1,
            'Integrated gravity diagnostics omit their accepted constraint policy.')
        requested=_number(gravity,'requested_dt_myr',0)
        completed=_number(gravity,'completed_dt_myr',0)
        require(requested>0 and abs(completed-requested)<=1e-12*requested
            and _number(gravity,'accepted_time_fraction')==1.,'Gravity did not complete its full physical interval.')
        rows=gravity.get('internal_steps')
        require(isinstance(rows,list) and bool(rows) and _integer(gravity,'substeps',1)==len(rows),
            'Gravity omits its complete accepted interval sequence.')
        for row in rows:
            require((frame.get('numerical_accuracy_version',0)==1)==('numerical_accuracy_policy' in row),
                'Saved gravity row differs from the frame numerical accuracy policy.')
            validate_accepted_row(row)
        require(abs(sum(row['dt_myr'] for row in rows)-requested)<=1e-12*requested,
            'Saved accepted gravity intervals do not sum to the requested physical time.')


def _gravity(frame):
    deformation=frame.get('deformation_diagnostics',{})
    require(isinstance(deformation,dict),'Invalid saved deformation diagnostics.')
    gravity=deformation.get('gravitational_relaxation',{})
    require(isinstance(gravity,dict),'Invalid saved gravity diagnostics.')
    return gravity


def validate_frame(frame):
    require(isinstance(frame,dict),'Invalid constrained-gravity frame.')
    import numerical_accuracy
    numerical_accuracy.validate_frame(frame)
    version=frame.get('gravity_constraint_version',0)
    require(type(version) is int and version in (0,1),'Unsupported gravity constraint version.')
    if restart_boundary.validate_frame(frame):
        return
    gravity=_gravity(frame)
    if isinstance(gravity,dict) and 'gravity_constraint_version' in gravity:
        marker=gravity['gravity_constraint_version']
        require(type(marker) is int and marker==version,'Integrated gravity policy differs from its saved frame marker.')
    if version==0:
        rows=gravity.get('internal_steps',[])
        require(isinstance(rows,list),'Invalid saved gravity interval sequence.')
        require(all(isinstance(row,dict) and isinstance(row.get('solver',{}),dict) for row in rows),
            'Invalid saved gravity interval metadata.')
        new_rows=any(isinstance(row,dict) and ('bound_constraints' in row
            or 'stationary_within_constraint_kkt' in row
            or row.get('solver',{}).get('stationarity_kind') in
                ('unconstrained_linear','nonlinear_endpoint_kkt','current_geometry_zero_kkt')) for row in rows)
        require(not new_rows,'Constrained gravity diagnostics require their saved policy version.')
        return
    if frame.get('time_myr')==0 and isinstance(gravity,dict):
        require(not gravity.get('internal_steps'),'A time-zero frame cannot claim an integrated gravity sequence.')
    deformation=frame.get('deformation_diagnostics',{})
    if version==1 and deformation.get('model')=='rigid material transport':
        require(type(frame.get('mesh_version')) is int and frame['mesh_version']==1
            and type(frame.get('material_mechanics_version')) is int and frame['material_mechanics_version']==1,
            'Constrained gravity requires native coupled material mechanics.')
        _number(frame,'time_myr',0)
        require(type(deformation.get('deforming_vertices')) is int and deformation['deforming_vertices']==0
            and 'gravitational_relaxation' not in deformation,
            'Rigid transport cannot claim a gravity interval or deforming vertices.')
        return
    validate_gravity_policies([frame])
