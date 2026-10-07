"""Frozen-geometry joint plate/sheet research primitive, entirely in SI units.

This is not called by the simulation. Contacts are supplied, already admitted,
closed bilateral normal constraints; this module does not choose contact birth,
opening, weld strength, slip friction, material entry, or a physical calibration.
See SHARED_CONTACT_CONTRACT.md for the force ledger and numerical limitations.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

import mesh_geometry
import viscous_sheet


@dataclass(frozen=True)
class ContactSample:
    """One common finite-trace point and its positive length quadrature weight."""
    first_face: int
    second_face: int
    point: object
    normal: object
    length_m: float


def _scalar(value, name, *, positive=True):
    if (isinstance(value, (bool, np.bool_)) or not np.isscalar(value)
            or not np.isfinite(value) or (value <= 0 if positive else value < 0)):
        raise ValueError(f'{name} must be finite and '+('positive.' if positive else 'nonnegative.'))
    return float(value)


def _field(value, count, name, *, positive=True):
    a = np.asarray(value, float)
    if a.ndim == 0:
        a = np.full(count, float(a))
    if (a.shape != (count,) or not np.isfinite(a).all()
            or np.any(a <= 0 if positive else a < 0)):
        raise ValueError(f'{name} must be finite, aligned and '+('positive.' if positive else 'nonnegative.'))
    return a.copy()


def _vectors(value, count, name):
    a = np.asarray(value, float)
    if a.shape != (count, 3) or not np.isfinite(a).all():
        raise ValueError(f'{name} must contain {count} finite three-vectors.')
    return a.copy()


def _cross_matrix(p):
    x, y, z = p
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def _trace(points, face, c):
    # alpha are homogeneous radial P1 coordinates: sum alpha_i*r_i = c.
    # They are not normalized barycentrics. The derivative of radial projection
    # is P_c sum alpha_i*V_i, which reproduces every common Euler rotation.
    alpha = np.linalg.solve(points[face].T, c)
    tolerance = 256*np.finfo(float).eps*max(1., float(np.max(np.abs(alpha))))
    if np.any(alpha < -tolerance) or alpha.sum() <= 0:
        raise ValueError('Contact sample must be in both supplied finite triangles.')
    # Keep tiny signed solve roundoff: clipping it changes the common-point
    # interpolation identity. This is not an automatic contact-admission band.
    if np.linalg.norm(points[face].T@alpha-c) > 256*np.finfo(float).eps:
        raise ValueError('Contact trace reconstruction is numerically unresolved.')
    return alpha


def assemble(points, faces, vertex_plate, *, radius_m, basal_drag_pa_s_per_m,
             viscosity_pa_s, sheet_thickness_m, basal_reference_velocity_m_s,
             other_plate_rotational_drag_n_m_s, external_torques_n_m,
             external_nodal_forces_n, contacts):
    """Assemble one SI power functional and the same contact/gauge Jacobians.

    All physical coefficients and loads are mandatory. ``other_plate`` drag is
    only the remaining domain, excluding the material basal integral here.
    A supplied nodal GPE force enters both unknown blocks through total velocity;
    the external torque must not include that same GPE a second time.

    Unknown y=[a,z] has m/s units, a_p=radius_m*omega_p and two tangent residual
    coordinates per node. Dense algebra is deliberately limited to 256 unknowns.
    Faces cannot cross plate ownership; geometry and contact topology are fixed.
    """
    radius = _scalar(radius_m, 'radius_m')
    p = np.asarray(points, float)
    f = np.asarray(faces)
    owner = np.asarray(vertex_plate)
    if (p.ndim != 2 or p.shape[1:] != (3,) or not len(p) or not np.isfinite(p).all()
            or not np.allclose(np.linalg.norm(p, axis=1), 1., rtol=0., atol=2e-12)):
        raise ValueError('Points must be finite unit spherical vectors.')
    p = p/np.linalg.norm(p, axis=1)[:, None]
    if (f.ndim != 2 or f.shape[1:] != (3,) or not len(f) or f.dtype.kind not in 'iu'
            or np.any(f < 0) or np.any(f >= len(p))):
        raise ValueError('Faces must be nonempty integer vertex triples.')
    f = f.astype(np.int64, copy=True)
    if owner.shape != (len(p),) or owner.dtype.kind not in 'iu' or np.any(owner < 0):
        raise ValueError('One nonnegative integer plate slot is required per vertex.')
    owner = owner.astype(np.int64, copy=True)
    count = int(owner.max())+1
    if not np.array_equal(np.unique(owner), np.arange(count)):
        raise ValueError('Plate slots must be contiguous and represented by vertices.')
    if np.any(owner[f] != owner[f[:, :1]]):
        raise ValueError('A material face cannot straddle plate ownership.')
    nplate = 3*count
    size = nplate+2*len(p)
    if size > 256:
        raise ValueError('This dense research primitive supports at most 256 unknowns.')
    triangle = p[f]
    signed = np.einsum('ij,ij->i', triangle[:, 0], np.cross(triangle[:, 1], triangle[:, 2]))
    if np.any(signed <= 0.):
        raise ValueError('Material faces must have positive orientation.')
    area_km2 = mesh_geometry.spherical_area(p, f, radius_km=radius/1000.)
    area_m2 = area_km2*1e6
    nodal_area = np.bincount(f.ravel(), weights=np.repeat(area_m2/3., 3), minlength=len(p))
    if np.any(nodal_area <= 0.):
        raise ValueError('Unused material vertices are not supported.')
    beta = _field(basal_drag_pa_s_per_m, len(p), 'basal_drag_pa_s_per_m')
    eta = _field(viscosity_pa_s, len(f), 'viscosity_pa_s', positive=False)
    thick = _field(sheet_thickness_m, len(f), 'sheet_thickness_m')
    mantle = _vectors(basal_reference_velocity_m_s, len(p), 'basal_reference_velocity_m_s')
    force = _vectors(external_nodal_forces_n, len(p), 'external_nodal_forces_n')
    torque = _vectors(external_torques_n_m, count, 'external_torques_n_m')
    # Only tangent external forces do work on the fixed sphere. Basal reference
    # must itself be tangent; a radial mantle speed would create fake dissipation.
    if np.any(np.abs(np.einsum('ij,ij->i', mantle, p)) > 1e-12*np.linalg.norm(mantle, axis=1)):
        raise ValueError('Basal reference velocities must be tangent to the sphere.')
    drag = np.asarray(other_plate_rotational_drag_n_m_s, float)
    if drag.shape != (nplate, nplate) or not np.isfinite(drag).all():
        raise ValueError('Remaining-domain rotational drag must be an aligned finite matrix.')
    if not np.allclose(drag, drag.T, rtol=64*np.finfo(float).eps, atol=0.):
        raise ValueError('Remaining-domain rotational drag must be symmetric.')
    drag = .5*(drag+drag.T)
    # A small negative mode cannot be excused by an unrelated huge positive
    # eigenvalue. Ambiguous roundoff in a semidefinite input fails closed.
    other_eigenvalues, other_eigenvectors = np.linalg.eigh(drag)
    if other_eigenvalues.min() < 0.:
        raise ValueError('Remaining-domain rotational drag must be positive semidefinite.')
    other_factor = np.sqrt(other_eigenvalues)[:, None]*other_eigenvectors.T/radius

    axes = np.eye(3)[np.argmin(np.abs(p), axis=1)]
    first = np.cross(p, axes); first /= np.linalg.norm(first, axis=1)[:, None]
    basis = np.stack((first, np.cross(p, first)), axis=2)
    velocity_map = np.zeros((3*len(p), size))
    for i in range(len(p)):
        velocity_map[3*i:3*i+3, 3*owner[i]:3*owner[i]+3] = -_cross_matrix(p[i])
        velocity_map[3*i:3*i+3, nplate+2*i:nplate+2*i+2] = basis[i]
    drag_weight = beta*nodal_area
    basal = velocity_map.T@(np.repeat(drag_weight, 3)[:, None]*velocity_map)
    context = viscous_sheet.prepare(p, f, area_km2, 0., radius=radius/1000.)
    # Existing gradients are per km. For SI m/s velocity, divide D by1000 to
    # obtain 1/s. Build the tensor energy Gram matrix directly, never subtract
    # two large drag+viscosity matrices to isolate the viscous part.
    strain_design = np.empty((10*len(f), size))
    factor = np.sqrt(2.*eta*thick*area_m2)
    for column in range(size):
        field = viscous_sheet.strain_rate(context, velocity_map[:, column].reshape(-1, 3))
        tensor = field['D']/1000.
        div = field['divergence']/1000.
        strain_design[:, column] = (np.column_stack((tensor.reshape(-1, 9), div))*factor[:, None]).ravel()
    viscous = strain_design.T@strain_design
    remaining = np.zeros((size, size)); remaining[:nplate, :nplate] = drag/radius**2
    hessian = basal+viscous+remaining
    mantle_load = velocity_map.T@(drag_weight[:, None]*mantle).ravel()
    nodal_load = velocity_map.T@force.ravel()
    torque_load = np.r_[torque.ravel()/radius, np.zeros(2*len(p))]
    load = mantle_load+nodal_load+torque_load

    gauge = np.zeros((nplate, size))
    for plate in range(count):
        take = np.flatnonzero(owner == plate)
        weight = nodal_area[take]/nodal_area[take].sum()
        for i, w in zip(take, weight):
            gauge[3*plate:3*plate+3, nplate+2*i:nplate+2*i+2] = w*_cross_matrix(p[i])@basis[i]
        if np.linalg.matrix_rank(gauge[3*plate:3*plate+3]) != 3:
            raise ValueError('Residual rigid-moment gauge is unresolved on a plate.')

    rows = []; contact_nodes = []; locations = []; normals = []; lengths = []; pairs = []; trace_maps = []
    for sample in contacts:
        if not isinstance(sample, ContactSample):
            raise ValueError('Contacts must be explicit ContactSample records.')
        ia, ib = sample.first_face, sample.second_face
        if (isinstance(ia, (bool, np.bool_)) or isinstance(ib, (bool, np.bool_))
                or not isinstance(ia, (int, np.integer)) or not isinstance(ib, (int, np.integer))
                or not 0 <= ia < len(f) or not 0 <= ib < len(f)
                or owner[f[ia, 0]] == owner[f[ib, 0]]):
            raise ValueError('Contact faces must belong to distinct represented plates.')
        c = _vectors(np.asarray(sample.point)[None], 1, 'Contact point')[0]
        normal = _vectors(np.asarray(sample.normal)[None], 1, 'Contact normal')[0]
        if abs(np.linalg.norm(c)-1.) > 2e-12 or np.linalg.norm(normal) == 0.:
            raise ValueError('Contact point must be unit and its normal nonzero.')
        c /= np.linalg.norm(c)
        if abs(normal@c) > 2e-12*np.linalg.norm(normal):
            raise ValueError('Contact normal must be tangent at the common point.')
        normal -= c*(normal@c); normal /= np.linalg.norm(normal)
        length = _scalar(sample.length_m, 'Contact length quadrature weight')
        pullback = np.zeros((len(p), 3))
        pair_maps = []
        for sign, face in ((1., f[ia]), (-1., f[ib])):
            alpha = _trace(p, face, c)
            trace = np.zeros((3, 3*len(p)))
            for i, a in zip(face, alpha):
                pullback[i] += sign*a*normal
                trace[:, 3*i:3*i+3] = a*(np.eye(3)-np.outer(c, c))
            pair_maps.append(trace@velocity_map)
        rows.append(pullback.ravel()@velocity_map)
        contact_nodes.append(pullback)
        locations.append(c); normals.append(normal); lengths.append(length)
        pairs.append((int(owner[f[ia, 0]]), int(owner[f[ib, 0]])))
        trace_maps.append(pair_maps)
    contact = np.asarray(rows).reshape(-1, size)
    lengths = np.asarray(lengths)
    # Numerical reference length is exactly1m; equality constraints introduce
    # no contact stiffness. Minimum-norm dual includes gauge multipliers and
    # length-weighted contact resultants; identical-sample splitting preserves
    # the aggregate resultant, but this is not a unique resolved traction law.
    weights = np.sqrt(lengths/1.)
    constraints = np.vstack((gauge, weights[:, None]*contact))
    if not all(np.isfinite(a).all() for a in (hessian, load, constraints)):
        raise ValueError('Requested SI assembly is nonfinite.')
    return dict(points=p, faces=f, vertex_plate=owner, plate_count=count, nplate=nplate,
        radius_m=radius, velocity_map=velocity_map, tangent_basis=basis,
        area_m2=area_m2, nodal_area_m2=nodal_area, basal_drag_weight_n_s_m=drag_weight,
        viscosity_pa_s=eta, thickness_m=thick, sheet_context=context,
        strain_design=strain_design, basal_hessian_n_s_m=basal,
        viscous_hessian_n_s_m=viscous, other_hessian_n_s_m=remaining,
        other_plate_drag_factor=other_factor,
        hessian_n_s_m=hessian, load_n=load, nodal_load_n=nodal_load,
        torque_load_n=torque_load, mantle_load_n=mantle_load,
        mantle_velocity_m_s=mantle, external_nodal_forces_n=force,
        external_torques_n_m=torque, gauge_matrix=gauge, contact_matrix=contact,
        constraint_matrix=constraints, contact_weights=weights, contact_length_m=lengths,
        contact_point=np.asarray(locations).reshape(-1, 3),
        contact_normal=np.asarray(normals).reshape(-1, 3),
        contact_velocity_maps=np.asarray(trace_maps).reshape(-1, 2, 3, size),
        contact_plate_pairs=np.asarray(pairs, dtype=int).reshape(-1, 2),
        contact_nodal_pullback=np.asarray(contact_nodes).reshape(-1, len(p), 3))


def _resistance(resistance, y):
    """Validate a pure SI passive-potential callback without changing its law."""
    size = len(y)
    if resistance is None:
        return dict(potential_w=0., gradient_n=np.zeros(size),
            hessian_n_s_m=np.zeros((size, size)), resisting_power_w=0., diagnostics={}, hessian_representation='none')
    argument = y.copy(); argument.flags.writeable = False
    result = resistance.evaluate(argument)
    required = {'potential_w', 'gradient_n', 'hessian_n_s_m', 'resisting_power_w', 'diagnostics'}
    if not isinstance(result, dict) or not required.issubset(result):
        raise ValueError('Passive resistance must return the complete SI functional contract.')
    potential = float(result['potential_w']); power = float(result['resisting_power_w'])
    gradient = np.asarray(result['gradient_n'], float).copy()
    hessian = np.asarray(result['hessian_n_s_m'], float).copy()
    if (gradient.shape != (size,) or hessian.shape != (size, size)
            or not all(np.isfinite(a).all() for a in (potential, power, gradient, hessian))
            or potential < 0. or power < 0. or not isinstance(result['diagnostics'], dict)):
        raise ValueError('Passive resistance values must be finite, aligned and nonnegative in power/potential.')
    if not np.allclose(hessian, hessian.T, rtol=64*np.finfo(float).eps, atol=0.):
        raise ValueError('Passive resistance Hessian must be symmetric.')
    hessian = .5*(hessian+hessian.T)
    factor = result.get('hessian_factor_sqrt_n_s_m')
    if factor is not None:
        factor = np.asarray(factor, float)
        if factor.ndim != 2 or factor.shape[1] != size or not np.isfinite(factor).all():
            raise ValueError('Passive Hessian factor must be finite and aligned.')
        represented = factor.T@factor
        absolute = np.abs(factor).T@np.abs(factor)
        tolerance = 64*np.finfo(float).eps*max(1, len(factor))*absolute
        if (not np.isfinite(represented).all() or not np.isfinite(absolute).all()
                or np.any(np.abs(hessian-represented) > tolerance)):
            raise ValueError('Passive Hessian does not match its supplied Gram factor.')
        # The declared Gram construction supplies positivity, not an eigenvalue
        # projection of an indefinite matrix. Use those same factor bytes.
        hessian = represented
    elif np.linalg.eigvalsh(hessian).min() < 0.:
        raise ValueError('Passive resistance Hessian is not resolved positive semidefinite; provide its physical Gram factor.')
    work = float(y@gradient)
    absolute_work = float(np.abs(y)@np.abs(gradient))
    allowance = 64*np.finfo(float).eps*max(1, size)*(absolute_work+abs(power))
    if (not np.isfinite(work) or not np.isfinite(absolute_work) or not np.isfinite(allowance)
            or abs(work-power) > allowance):
        raise ValueError('Passive resistance power must equal velocity dotted with its gradient.')
    return dict(potential_w=potential, gradient_n=gradient, hessian_n_s_m=hessian,
        resisting_power_w=power, diagnostics=result['diagnostics'],
        hessian_representation='supplied positive Gram factor' if factor is not None else 'checked symmetric PSD matrix')


def evaluate(system, generalized_velocity_m_s, *, resistance=None):
    """Return physical Rayleigh potential [W] and its generalized gradient [N]."""
    y = np.asarray(generalized_velocity_m_s, float)
    k = system['hessian_n_s_m']
    if y.shape != (len(k),) or not np.isfinite(y).all():
        raise ValueError('Generalized velocities must be finite and aligned.')
    if resistance is not None and hasattr(resistance, 'validate_system'):
        resistance.validate_system(system)
    constant = .5*np.sum(system['basal_drag_weight_n_s_m'][:, None]*system['mantle_velocity_m_s']**2)
    nonlinear = _resistance(resistance, y)
    objective = float(.5*y@k@y-system['load_n']@y+constant+nonlinear['potential_w'])
    gradient = k@y-system['load_n']+nonlinear['gradient_n']
    if not np.isfinite(objective) or not np.isfinite(gradient).all():
        raise RuntimeError('Joint functional is outside finite arithmetic range.')
    return objective, gradient


def solve(system, *, relative_tolerance=1e-10, resistance=None,
          maximum_newton_iterations=80, maximum_backtracks=64):
    """Solve the homogeneous constrained joint functional; verify original rows.

    SVD exposes redundant rows explicitly. Every original unweighted contact
    and gauge residual is checked afterward, so unresolved constraints are not
    accepted merely because a numerical rank threshold removed a direction.
    Optional ``resistance.evaluate(y)`` supplies a passive nonlinear SI potential
    and its work-conjugate derivatives. It is solved jointly by constrained
    Newton steps; it is never reinterpreted as an external physical load.
    """
    tolerance = _scalar(relative_tolerance, 'relative_tolerance')
    if tolerance >= 1.:
        raise ValueError('Relative tolerance must be below one.')
    for value in (maximum_newton_iterations, maximum_backtracks):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError('Nonlinear iteration/backtrack limits must be positive integers.')
    if resistance is not None and hasattr(resistance, 'validate_system'):
        resistance.validate_system(system)
    k = system['hessian_n_s_m']; force = system['load_n']; c = system['constraint_matrix']
    diagonal = np.diag(k)
    if np.any(diagonal <= 0.):
        raise ValueError('Joint energy scaling is not positive.')
    scale = 1./np.sqrt(diagonal)
    h = scale[:, None]*k*scale[None, :]
    rhs = scale*force
    scaled_constraints = c*scale[None, :]
    row_norm = np.linalg.norm(scaled_constraints, axis=1)
    if np.any(row_norm <= 0.) or not np.isfinite(row_norm).all():
        raise ValueError('A supplied constraint has no resolvable velocity direction.')
    normalized_u, singular, vt = np.linalg.svd(scaled_constraints/row_norm[:, None], full_matrices=True)
    rank_tolerance = 64*np.finfo(float).eps*max(c.shape)*singular[0]
    rank = int(np.count_nonzero(singular > rank_tolerance))
    null = vt[rank:].T
    reduced = null.T@h@null
    if len(reduced):
        chol = np.linalg.cholesky(.5*(reduced+reduced.T))
        answer = np.linalg.solve(chol.T, np.linalg.solve(chol, null.T@rhs))
        q = null@answer
    else:
        q = np.zeros(len(k))
    y = scale*q
    nonlinear = _resistance(resistance, y)
    iterations = 0; backtracks = 0; nonlinear_history = []
    if resistance is not None:
        # The linear minimizer is an initial guess, never a physical solution
        # with a fabricated load. Every nonlinear trial evaluates the SAME
        # supplied potential, and only final KKT/work checks authorize return.
        for iterations in range(maximum_newton_iterations+1):
            gradient = k@y-force+nonlinear['gradient_n']
            projected = null.T@(scale*gradient)
            projected_scale = max(float(np.linalg.norm(null.T@(scale*(k@y)))+
                np.linalg.norm(null.T@rhs)+np.linalg.norm(null.T@(scale*nonlinear['gradient_n']))), 1e-300)
            projected_relative = float(np.linalg.norm(projected)/projected_scale)
            if not np.isfinite(projected_relative) or not np.isfinite(projected_scale):
                raise RuntimeError('Nonlinear stationarity is outside finite arithmetic range.')
            if projected_relative <= .01*tolerance:
                break
            if iterations == maximum_newton_iterations:
                raise RuntimeError('Joint nonlinear stationarity did not converge within its iteration budget.')
            total_hessian = k+nonlinear['hessian_n_s_m']
            scaled_hessian = scale[:, None]*total_hessian*scale[None, :]
            local = null.T@scaled_hessian@null
            chol = np.linalg.cholesky(.5*(local+local.T))
            direction = -scale*(null@np.linalg.solve(chol.T, np.linalg.solve(chol, projected)))
            slope = float(gradient@direction)
            if not np.isfinite(direction).all() or not np.isfinite(slope) or slope >= 0.:
                raise RuntimeError('Joint Newton direction is not a finite descent direction.')
            accepted = False
            for line in range(maximum_backtracks):
                alpha = 2.**(-line); step = alpha*direction; candidate = y+step
                trial = _resistance(resistance, candidate)
                # Expand the quadratic difference directly. This avoids
                # subtracting two large total objectives near stationarity.
                linear_difference = float((k@y-force)@step+.5*step@k@step)
                difference = linear_difference+trial['potential_w']-nonlinear['potential_w']
                arithmetic = 64*np.finfo(float).eps*(abs(linear_difference)+
                    abs(trial['potential_w'])+abs(nonlinear['potential_w']))
                if (not np.isfinite(difference) or not np.isfinite(arithmetic)):
                    raise RuntimeError('Joint nonlinear line search is outside finite arithmetic range.')
                if difference <= 1e-4*alpha*slope+arithmetic:
                    accepted = True; y = candidate; nonlinear = trial; backtracks += line
                    nonlinear_history.append(dict(iteration=iterations, alpha=alpha,
                        potential_change_w=difference, directional_derivative_w=slope,
                        arithmetic_objective_allowance_w=arithmetic,
                        projected_relative_residual=projected_relative))
                    break
            if not accepted:
                raise RuntimeError('Joint nonlinear potential line search failed.')
        q = y/scale
    # L=Phi+lambda.Cy; these multipliers pull back through the SAME Jacobian.
    # Recover a dual in the row-normalized system, then minimize the ORIGINAL
    # weighted-row multiplier norm within its exact retained nullspace. Solving
    # directly with the unnormalized SVD loses precision for duplicate rows.
    normalized_dual = normalized_u[:, :rank]@((vt[:rank]@(rhs-h@q-scale*nonlinear['gradient_n']))/singular[:rank])
    multiplier = normalized_dual/row_norm
    dual_null = normalized_u[:, rank:]/row_norm[:, None]
    if dual_null.shape[1]:
        correction, _, dual_rank, _ = np.linalg.lstsq(dual_null, -multiplier,
            rcond=64*np.finfo(float).eps*max(dual_null.shape))
        if dual_rank != dual_null.shape[1]:
            raise RuntimeError('Minimum-norm constraint dual is numerically unresolved.')
        multiplier += dual_null@correction
    # Finish the Euclidean minimum exactly along KNOWN duplicate-contact null
    # directions. SVD null vectors can have tiny spurious gauge components whose
    # rescaling degrades this otherwise exact quadrature identity. This groups
    # only bit-identical unweighted rows, never nearly dependent physical rows.
    contact = system['contact_matrix']; nplate = system['nplate']
    groups = {}
    for i, row in enumerate(contact):
        groups.setdefault(tuple(row), []).append(i)
    for group in groups.values():
        if len(group) > 1:
            selected = np.asarray(group, int)
            weights = system['contact_weights'][selected]
            aggregate = weights@multiplier[nplate+selected]
            multiplier[nplate+selected] = weights*aggregate/(weights@weights)
    # Nullspace/SVD assembly can leave O(eps) mixing from a strongly forced
    # component into a symmetry-zero component. Its local backward error then
    # fails even though the normwise residual is tiny. Correct the ORIGINAL KKT
    # equations; do not excuse that component using an unrelated force scale.
    refinement_steps = 0
    independent = None
    for refinement in range(5):
        error = k@y-force+nonlinear['gradient_n']+c.T@multiplier
        local_scale = (np.abs(k)@np.abs(y)+np.abs(force)+
            np.abs(nonlinear['gradient_n'])+np.abs(c.T)@np.abs(multiplier))
        component_error = np.divide(np.abs(error), local_scale,
            out=np.zeros_like(error), where=local_scale > 0.)
        if np.any((local_scale == 0.) & (error != 0.)):
            raise RuntimeError('An unforced joint component has an unexplained residual.')
        if np.max(component_error) <= tolerance:
            break
        if refinement == 4:
            break  # The original gates below report failure, never success.
        if independent is None:
            # Select rank-many actual original rows with reorthogonalized
            # pivoted Gram-Schmidt. This changes only algebra, not constraints.
            original = scaled_constraints/row_norm[:, None]
            remainder = original.copy(); selected = []; vectors = []
            for _ in range(rank):
                norms = np.linalg.norm(remainder, axis=1)
                index = int(np.argmax(norms)); vector = original[index].copy()
                for _ in range(2):
                    for basis_vector in vectors:
                        vector -= (vector@basis_vector)*basis_vector
                length = np.linalg.norm(vector)
                if not np.isfinite(length) or length <= rank_tolerance:
                    raise RuntimeError('Original-row KKT refinement cannot resolve constraint rank.')
                vector /= length; selected.append(index); vectors.append(vector)
                remainder -= (remainder@vector)[:, None]*vector
                remainder[selected] = 0.
            independent = np.asarray(selected, int)
        actual_rows = scaled_constraints[independent]/row_norm[independent, None]
        tangent_hessian = scale[:, None]*(k+nonlinear['hessian_n_s_m'])*scale[None, :]
        kkt = np.block([[tangent_hessian, actual_rows.T],
            [actual_rows, np.zeros((rank, rank))]])
        correction = np.linalg.solve(kkt, -np.r_[scale*error, (c[independent]@y)/row_norm[independent]])
        y += scale*correction[:len(y)]
        multiplier[independent] += correction[len(y):]/row_norm[independent]
        # Restore the original weighted minimum dual convention after a solve
        # using an independent row subset, then finish exact duplicate modes.
        if dual_null.shape[1]:
            adjustment, _, projection_rank, _ = np.linalg.lstsq(dual_null, -multiplier,
                rcond=64*np.finfo(float).eps*max(dual_null.shape))
            if projection_rank != dual_null.shape[1]:
                raise RuntimeError('Refined minimum-norm dual is unresolved.')
            multiplier += dual_null@adjustment
        for group in groups.values():
            if len(group) > 1:
                selected = np.asarray(group, int); weights = system['contact_weights'][selected]
                aggregate = weights@multiplier[nplate+selected]
                multiplier[nplate+selected] = weights*aggregate/(weights@weights)
        nonlinear = _resistance(resistance, y)
        refinement_steps += 1
    reaction_generalized = -c.T@multiplier
    stationarity = k@y-force+nonlinear['gradient_n']-reaction_generalized
    residual_scale = max(float(np.linalg.norm(k@y)+np.linalg.norm(force)+np.linalg.norm(reaction_generalized)+
        np.linalg.norm(nonlinear['gradient_n'])), 1e-300)
    residual = float(np.linalg.norm(stationarity)/residual_scale)
    component_scale = np.abs(k)@np.abs(y)+np.abs(force)+np.abs(c.T)@np.abs(multiplier)+np.abs(nonlinear['gradient_n'])
    component_residual = np.divide(np.abs(stationarity), component_scale,
        out=np.zeros_like(stationarity), where=component_scale > 0.)
    if np.any((component_scale == 0.) & (stationarity != 0.)):
        raise RuntimeError('An unforced joint component has an unexplained residual.')
    velocity = (system['velocity_map']@y).reshape(-1, 3)
    residual_velocity = np.einsum('nia,na->ni', system['tangent_basis'], y[nplate:].reshape(-1, 2))
    omega = y[:nplate].reshape(-1, 3)/system['radius_m']
    slip = system['contact_matrix']@y
    gauge_error = system['gauge_matrix']@y
    trace_maps = system['contact_velocity_maps']
    plate_trace = np.einsum('qsij,j->qsi', trace_maps[:, :, :, :nplate], y[:nplate])
    residual_trace = np.einsum('qsij,j->qsi', trace_maps[:, :, :, nplate:], y[nplate:])
    contact_speed = np.sum(np.linalg.norm(plate_trace, axis=2)+np.linalg.norm(residual_trace, axis=2), axis=1)
    plate_speed = []
    for plate in range(system['plate_count']):
        take = system['vertex_plate'] == plate
        weight = system['nodal_area_m2'][take]
        plate_speed.append(np.linalg.norm(y[3*plate:3*plate+3])+
            np.average(np.linalg.norm(residual_velocity[take], axis=1), weights=weight))
    gauge_speed = np.repeat(plate_speed, 3)
    if (not all(np.isfinite(a).all() for a in
            (y, multiplier, residual, residual_scale, component_scale, component_residual,
             slip, gauge_error, contact_speed, gauge_speed))
            or residual > tolerance or np.max(component_residual) > tolerance
            or np.any(np.abs(slip) > tolerance*contact_speed)
            or np.any(np.abs(gauge_error) > tolerance*gauge_speed)):
        raise RuntimeError('Joint stationarity or an original contact/gauge constraint is unresolved.')
    resultant = system['contact_weights']*multiplier[nplate:]
    first_force = -resultant[:, None]*system['contact_normal']
    contact_reaction = -system['contact_matrix'].T@resultant
    gauge_reaction = -system['gauge_matrix'].T@multiplier[:nplate]
    relative_v = velocity-system['mantle_velocity_m_s']
    basal_dissipation = float(np.sum(system['basal_drag_weight_n_s_m'][:, None]*relative_v**2))
    viscous_dissipation = float(np.linalg.norm(system['strain_design']@y)**2)
    other_dissipation = float(np.linalg.norm(system['other_plate_drag_factor']@y[:nplate])**2)
    external_power = float((system['nodal_load_n']+system['torque_load_n'])@y)
    mantle_power = float(np.sum(-system['basal_drag_weight_n_s_m'][:, None]*relative_v*system['mantle_velocity_m_s']))
    passive = basal_dissipation+viscous_dissipation+other_dissipation+nonlinear['resisting_power_w']
    power_error = passive-external_power-mantle_power
    contact_power = float(contact_reaction@y); gauge_power = float(gauge_reaction@y)
    rhs_power = float(force@y); quadratic_power = float(y@k@y)
    objective = evaluate(system, y)[0]+nonlinear['potential_w']
    traction = resultant/system['contact_length_m']
    plate_torque = (system['radius_m']*contact_reaction[:nplate]).reshape(-1, 3)
    resistance_plate_torque = (-system['radius_m']*nonlinear['gradient_n'][:nplate]).reshape(-1, 3)
    algebraic_error = rhs_power-quadratic_power-nonlinear['resisting_power_w']
    work_scale = passive+abs(external_power)+abs(mantle_power)
    # A common rigid mantle/material motion has exactly zero physical power.
    # Relative accuracy against that zero cannot be required in floating point.
    # Propagate the already-gated KKT/constraint residual into work explicitly,
    # then add a gamma_n allowance for the absolute constituent products (W).
    # This is an arithmetic allowance, not a new constitutive power tolerance.
    constraint_work_envelope = float(np.abs(multiplier)@np.abs(c@y))
    solver_work_envelope = float(np.abs(y)@np.abs(stationarity))+constraint_work_envelope
    absolute_y = np.abs(y)
    factor_work = float(np.linalg.norm(np.abs(system['strain_design'])@absolute_y)**2 +
        np.linalg.norm(np.abs(system['other_plate_drag_factor'])@absolute_y[:nplate])**2)
    basal_work = float(np.sum(system['basal_drag_weight_n_s_m'][:, None]*
        (np.abs(velocity)+np.abs(system['mantle_velocity_m_s']))**2))
    constituent_work = (float(absolute_y@np.abs(k)@absolute_y + np.abs(force)@absolute_y +
        np.abs(multiplier)@np.abs(c)@absolute_y + np.abs(nonlinear['gradient_n'])@absolute_y) + factor_work + basal_work)
    # Diagnostic arithmetic estimate for the reported constituent products:
    # sixteen passes of these inner dimensions includes Gram/sum transforms.
    # This is not a certified bound through ill-conditioned geometry, rank,
    # eigenvalue or assembly operations; the separate residual gates still apply.
    operation_count = 16*(len(y)+system['velocity_map'].shape[0]+system['strain_design'].shape[0]+len(c)+16)
    epsilon_count = operation_count*np.finfo(float).eps
    arithmetic_work_allowance = epsilon_count/(1.-epsilon_count)*constituent_work
    power_allowance = tolerance*work_scale+solver_work_envelope+arithmetic_work_allowance
    reaction_work_allowance = tolerance*work_scale+constraint_work_envelope+arithmetic_work_allowance
    if (not all(np.isfinite(a).all() for a in (resultant, first_force, contact_reaction,
            gauge_reaction, traction, plate_torque, resistance_plate_torque, algebraic_error,
            omega, velocity, residual_velocity,
            basal_dissipation, viscous_dissipation, other_dissipation,
            external_power, mantle_power, power_error, contact_power, gauge_power,
            rhs_power, quadratic_power, objective, work_scale, constraint_work_envelope,
            solver_work_envelope, constituent_work, arithmetic_work_allowance, power_allowance,
            reaction_work_allowance))
            or abs(power_error) > power_allowance
            or abs(contact_power)+abs(gauge_power) > reaction_work_allowance):
        raise RuntimeError('Joint physical power or reaction is unresolved or nonfinite.')
    return dict(generalized_velocity_m_s=y, plate_omega_rad_s=omega,
        residual_velocity_m_s=residual_velocity, total_velocity_m_s=velocity,
        contact_resultant_n=resultant, contact_traction_n_m=traction,
        contact_first_force_n=first_force, contact_second_force_n=-first_force,
        contact_generalized_reaction_n=contact_reaction,
        contact_plate_torque_n_m=plate_torque,
        gauge_generalized_reaction_n=gauge_reaction, gauge_multiplier_n=multiplier[:nplate],
        resistance_generalized_reaction_n=-nonlinear['gradient_n'],
        resistance_plate_torque_n_m=resistance_plate_torque,
        diagnostics=dict(joint_relative_residual=residual, stationarity_n=stationarity,
            componentwise_stationarity_relative=component_residual,
            contact_slip_m_s=slip, gauge_error_m_s=gauge_error,
            contact_local_velocity_scale_m_s=contact_speed, gauge_plate_velocity_scale_m_s=gauge_speed,
            constraint_rows=len(c), constraint_rank=rank, redundant_rows=len(c)-rank,
            singular_values=singular, rank_tolerance=rank_tolerance,
            basal_dissipation_w=basal_dissipation, viscous_dissipation_w=viscous_dissipation,
            other_drag_dissipation_w=other_dissipation, external_power_w=external_power,
            resistance_potential_w=nonlinear['potential_w'], resistance_dissipation_w=nonlinear['resisting_power_w'],
            resistance_virtual_power_w=-nonlinear['resisting_power_w'],
            resistance_hessian_representation=nonlinear['hessian_representation'],
            resistance_diagnostics=nonlinear['diagnostics'], nonlinear_iterations=iterations,
            nonlinear_backtracks=backtracks, nonlinear_history=nonlinear_history,
            original_kkt_refinement_steps=refinement_steps,
            basal_reference_input_power_w=mantle_power, physical_power_residual_w=power_error,
            solver_work_error_envelope_w=solver_work_envelope,
            arithmetic_work_allowance_w=arithmetic_work_allowance,
            arithmetic_work_operation_count=operation_count,
            absolute_constituent_work_w=constituent_work,
            physical_power_acceptance_allowance_w=power_allowance,
            contact_virtual_power_w=contact_power, gauge_virtual_power_w=gauge_power,
            algebraic_rhs_power_w=rhs_power, algebraic_quadratic_power_w=quadratic_power,
            algebraic_power_residual_w=algebraic_error,
            objective_w=objective,
            scope='Frozen supplied bilateral normal constraints and/or passive resistance laws; no admission, timestep, native integration or calibration.'))
