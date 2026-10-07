"""Experimental, frozen-load SI viscous-sheet response and tensile damage.

This is a constitutive building block, NOT an enabled native evolution policy.
It does not move ocean material, solve the other plate/interface resistances,
choose tractions or physical coefficients, split owners, or manufacture intake.
Native rigid transport cannot consume its nonrigid velocity without a new
coupled transport/force transaction. Applying only its damage to a live rigid
world would therefore not constitute a work-consistent physical integration.

The sheet law is sigma = 2 eta [D + tr(D) P], with plane-stress incompressible
coupling. A positive basal drag makes the frozen linear problem coercive.
Internal velocity has zero area-weighted Euler mode on each connected owner
component. Actual nodal forces (N) are partitioned by that constraint; opposed
tractions can retain internal work when their total rigid torque is zero.

Damage is an explicitly uncalibrated reduced constitutive law. A caller-chosen
fraction of tensile overstress work spends a finite fracture budget, with the
remainder dissipated as heat. That budget is Gc * reference_volume / ell:
Gc is J/m2 of fracture surface, ell is a physical regularization length in m.
No additional energy is added, and this module never infers a connected crack
or changes viscosity from the resulting scalar damage.
"""
from __future__ import annotations

import hashlib

import numpy as np
import viscous_sheet

VERSION = 1


def _scalar(value, name, *, positive=True):
    if (np.asarray(value).dtype.kind not in 'iuf' or not np.isscalar(value)
            or not np.isfinite(value) or (value <= 0 if positive else value < 0)):
        raise ValueError(name + ' must be a finite ' + ('positive' if positive else 'nonnegative') + ' scalar.')
    return float(value)


def _fingerprint(items):
    """Identity/alignment check, not an authenticity or security mechanism."""
    digest = hashlib.sha256()
    for name, value in items:
        array = np.ascontiguousarray(value)
        digest.update(repr((name, array.dtype.str, array.shape)).encode('ascii'))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _context_fingerprint(context):
    items = [(name, context[name]) for name in ('version', 'owner_labels', 'component',
             'radius_m', 'viscosity_pa_s', 'thickness_m', 'face_area_m2', 'basal_drag_pa_s_m')]
    items.extend(('sheet.'+name, value) for name, value in sorted(context['sheet'].items()))
    items.extend(('projection.'+name, value) for name, value in sorted(vars(context['projection']).items()))
    return _fingerprint(items)


def _validate_context(context):
    if (type(context.get('version')) is not int or context['version'] != VERSION
            or context.get('context_fingerprint') != _context_fingerprint(context)):
        raise ValueError('The prepared SI sheet has changed; prepare a new context.')


def _response_fingerprint(response):
    items = [(name, response[name]) for name in ('internal_velocity_m_s', 'total_velocity_m_s',
             'strain_rate_per_s', 'stress_pa', 'membrane_power_by_face_w', 'tangent_nodal_force_n',
             'radial_reaction_n', 'balancing_constraint_force_n', 'unbalanced_rigid_torque_n_m')]
    items.extend(('diagnostics.'+name, value) for name, value in sorted(response['diagnostics'].items()))
    return _fingerprint(items)


def _check_saved(actual, expected, name):
    actual = np.asarray(actual, float)
    scale = float(np.max(np.abs(expected), initial=0.))
    if (actual.shape != expected.shape or not np.isfinite(actual).all()
            or not np.allclose(actual, expected, rtol=128.*np.finfo(float).eps,
                               atol=128.*np.finfo(float).eps*scale)):
        raise ValueError(name + ' does not match the solved constitutive response.')


def _field(value, count, name, *, positive=True):
    raw = np.asarray(value)
    if raw.dtype.kind not in 'iuf' or raw.dtype.kind == 'b':
        raise ValueError(name + ' must be real numeric values.')
    try:
        result = np.broadcast_to(np.asarray(value, float), (count,)).copy()
    except ValueError as error:
        raise ValueError(name + ' must be scalar or aligned with faces.') from error
    if not np.isfinite(result).all() or np.any(result <= 0 if positive else result < 0):
        raise ValueError(name + ' must be finite and ' + ('positive.' if positive else 'nonnegative.'))
    return result


def _vectors(value, count, name):
    array = np.asarray(value, float)
    if array.shape != (count, 3) or not np.isfinite(array).all():
        raise ValueError(name + ' must contain finite Nx3 vectors.')
    return array.copy()


def _connected_components(faces, count):
    parent = np.arange(count)

    def root(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for face in faces:
        first = root(int(face[0]))
        for node in face[1:]:
            other = root(int(node))
            if other != first:
                parent[other] = first
    return np.unique([root(node) for node in range(count)], return_inverse=True)[1]


class _InternalProjection:
    """Euclidean orthogonal projection in w=sqrt(lumped_area)*velocity."""

    def __init__(self, points, area, component):
        self.points, self.area, self.component = points, area, component
        self.root_area = np.sqrt(area)
        self.count = int(component.max()) + 1
        self.basis = np.empty((len(points), 3, 3))
        self.factor = np.empty((self.count, 3, 3))
        for item in range(self.count):
            take = component == item
            raw = np.stack([np.cross(axis, points[take]) for axis in np.eye(3)], axis=2)
            raw *= self.root_area[take, None, None]
            q, r = np.linalg.qr(raw.reshape(-1, 3), mode='reduced')
            singular = np.linalg.svd(r, compute_uv=False)
            if singular[-1] <= singular[0]*1e-6:
                raise ValueError('Every sheet component must resolve all three rigid Euler modes.')
            # Orthogonal QR projection avoids squaring the geometric condition
            # number through an Euler-inertia inverse during every CG iterate.
            self.basis[take] = q.reshape(-1, 3, 3)
            self.factor[item] = r

    def tangent(self, value):
        return value - self.points*np.einsum('ni,ni->n', self.points, value)[:, None]

    def coefficients(self, weighted_velocity):
        local = np.einsum('nij,ni->nj', self.basis, weighted_velocity)
        return np.column_stack([np.bincount(self.component, weights=local[:, j], minlength=self.count)
                                for j in range(3)])

    def fit(self, weighted_velocity):
        return np.linalg.solve(self.factor, self.coefficients(weighted_velocity)[..., None])[..., 0]

    def apply(self, value):
        tangent = self.tangent(value)
        rigid = np.einsum('nij,nj->ni', self.basis, self.coefficients(tangent)[self.component])
        return self.tangent(tangent-rigid)


def prepare(unit_xyz, triangle_faces, face_area_m2, *, radius_m, viscosity_pa_s,
            thickness_m, basal_drag_pa_s_m, owner_labels=None):
    """Prepare a disconnected collection of single-owner SI triangle sheets.

    Vertices at distinct owners' interfaces must be duplicated by the caller;
    a triangle cannot transmit internal membrane stress between two owners.
    No nearest-point attachment, boundary-force conversion or remeshing occurs.
    All physical coefficients are explicit; no production defaults are chosen.
    """
    points = np.asarray(unit_xyz, float)
    faces = np.asarray(triangle_faces)
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 3:
        raise ValueError('The sheet needs at least three unit-sphere vertices.')
    if (faces.ndim != 2 or faces.shape[1:] != (3,) or len(faces) == 0 or faces.dtype.kind not in 'iu'
            or np.any(faces < 0) or np.any(faces >= len(points))):
        raise ValueError('Triangle faces must contain valid integer vertex indices.')
    if np.any(np.diff(np.sort(faces, axis=1), axis=1) == 0):
        raise ValueError('Triangle corners must be distinct.')
    if len(np.unique(np.sort(faces, axis=1), axis=0)) != len(faces):
        raise ValueError('Duplicate triangles would count physical dissipation twice.')
    radius = _scalar(radius_m, 'radius_m')
    drag = _scalar(basal_drag_pa_s_m, 'basal_drag_pa_s_m')
    eta = _field(viscosity_pa_s, len(faces), 'viscosity_pa_s')
    thick = _field(thickness_m, len(faces), 'thickness_m')
    area = _field(face_area_m2, len(faces), 'face_area_m2')
    labels = np.zeros(len(points), np.int64) if owner_labels is None else np.asarray(owner_labels)
    if labels.shape != (len(points),) or labels.dtype.kind not in 'iu':
        raise ValueError('Owner labels must be aligned integer values.')
    if np.any(labels[faces] != labels[faces[:, :1]]):
        raise ValueError('A sheet triangle cannot couple different rigid owners.')
    eta_h = eta*thick
    if not np.isfinite(eta_h).all():
        raise ValueError('Viscosity times thickness overflows SI units.')
    reference = float(np.max(eta_h))
    length_m = np.sqrt(2.*reference/drag)
    # The existing P1 algebra is homogeneous in length, area and velocity.
    # Supply METRES/METRES² here, despite its historical *_km argument names.
    # Multiplication by drag converts its operator to N s/m; strain_rate then
    # maps m/s to 1/s. Never expose its normalized-unit diagnostic labels.
    sheet = viscous_sheet.prepare(points, faces, area, length_m, radius=radius,
                                  viscosity_weights=eta_h/reference)
    if np.any(sheet['data'] <= 0):
        raise ValueError('Every sheet vertex must belong to a positive-area triangle.')
    component = _connected_components(sheet['faces'], len(points))
    projection = _InternalProjection(sheet['points'], sheet['data'], component)
    context = dict(version=VERSION, sheet=sheet, projection=projection, owner_labels=labels.copy(),
                component=component, radius_m=radius, viscosity_pa_s=eta, thickness_m=thick,
                face_area_m2=area, basal_drag_pa_s_m=drag)
    context['context_fingerprint'] = _context_fingerprint(context)
    return context


def solve(context, nodal_force_n, *, rigid_velocity_m_s=None, iterations=800, tolerance=1e-9):
    """Return a frozen-load internal response; raise on failed true stationarity.

    Forces are integrated nodal forces in N, not line tractions or velocities.
    ``rigid_velocity_m_s`` is a supplied Euler field, not solved here. Rigid and
    internal source work sum to F.(v_rigid+v_internal) exactly; constraint forces
    do zero internal work. A production integrator must still couple interface
    resistance, actual material motion and both plates' reaction forces.
    """
    _validate_context(context)
    if isinstance(iterations, (bool, np.bool_)) or not isinstance(iterations, (int, np.integer)) or iterations < 1:
        raise ValueError('iterations must be a positive integer.')
    tol = _scalar(tolerance, 'tolerance')
    if tol >= 1:
        raise ValueError('tolerance must be below one.')
    sheet, projection = context['sheet'], context['projection']
    points, area = sheet['points'], sheet['data']
    root_area = projection.root_area[:, None]
    drag = context['basal_drag_pa_s_m']
    force = _vectors(nodal_force_n, len(points), 'nodal_force_n')
    tangent_force = projection.tangent(force)
    rigid = (np.zeros_like(force) if rigid_velocity_m_s is None
             else _vectors(rigid_velocity_m_s, len(points), 'rigid_velocity_m_s'))
    rigid_norm = float(np.linalg.norm(rigid*root_area))
    if np.linalg.norm(rigid-projection.tangent(rigid)) > 1e-10*max(float(np.linalg.norm(rigid)), 1e-300):
        raise ValueError('The supplied rigid velocity must be tangent.')
    if np.linalg.norm(projection.apply(rigid*root_area)) > 1e-10*max(rigid_norm, 1e-300):
        raise ValueError('The supplied background is not a rigid Euler field per component.')

    def physical_apply(velocity):
        return drag*viscous_sheet.apply(sheet, velocity)

    def apply(weighted):
        return projection.apply(physical_apply(projection.apply(weighted)/root_area)/root_area)

    full_rhs = (tangent_force-physical_apply(rigid))/root_area
    rhs = projection.apply(full_rhs)
    scale = float(np.linalg.norm(rhs))
    # Use a relative INTERNAL-force criterion. A large rigid force must not
    # make a resolvable smaller internal force pass without solving. Only a
    # machine-roundoff floor may scale with the full projected input.
    force_scale = max(scale, float(np.linalg.norm(full_rhs)))
    roundoff_floor = 128.*np.finfo(float).eps*force_scale
    threshold = tol*scale + roundoff_floor
    internal = np.zeros_like(rhs)
    residual = rhs.copy()
    used = 0
    precondition, _ = viscous_sheet._preconditioner(
        sheet, np.ones(len(points), bool), np.zeros(len(points), bool), np.zeros_like(points))

    def preconditioned(value):
        return projection.apply(root_area*precondition(root_area*value)/drag)

    if float(np.linalg.norm(residual)) > threshold:
        z = preconditioned(residual)
        direction = z.copy()
        rz = float(np.sum(residual*z))
        for used in range(1, int(iterations)+1):
            product = apply(direction)
            curvature = float(np.sum(direction*product))
            if not np.isfinite(curvature) or curvature <= 0 or not np.isfinite(rz) or rz <= 0:
                raise ValueError('The projected SI sheet lost positive definite curvature.')
            internal += direction*(rz/curvature)
            residual -= product*(rz/curvature)
            recompute = np.linalg.norm(residual) <= threshold or used == iterations
            if recompute:
                internal = projection.apply(internal)
                residual = rhs-apply(internal)
                if np.linalg.norm(residual) <= threshold:
                    break
            z = preconditioned(residual)
            next_rz = float(np.sum(residual*z))
            direction = z.copy() if recompute else z+direction*(next_rz/rz)
            rz = next_rz
    internal = projection.apply(internal)
    residual = rhs-apply(internal)
    residual_norm = float(np.linalg.norm(residual))
    if not np.isfinite(internal).all() or not np.isfinite(residual_norm) or residual_norm > threshold:
        raise ValueError('Frozen SI traction solve did not reach true projected stationarity.')
    velocity = internal/root_area
    fields = viscous_sheet.strain_rate(sheet, velocity)
    strain, divergence = fields['D'], fields['divergence']
    stress = 2.*context['viscosity_pa_s'][:, None, None]*(strain+divergence[:, None, None]*sheet['plane'])
    volume = context['face_area_m2']*context['thickness_m']
    face_power = volume*np.einsum('fij,fij->f', stress, strain)
    basal_power = drag*float(np.sum(area[:, None]*velocity**2))
    source_power = float(np.sum(tangent_force*velocity))
    membrane_power = float(np.sum(face_power))
    rigid_power = float(np.sum(tangent_force*rigid))
    total_power = float(np.sum(tangent_force*(rigid+velocity)))
    unbalanced = tangent_force-physical_apply(rigid+velocity)
    torque = context['radius_m']*np.cross(points, unbalanced)
    component_torque = np.column_stack([np.bincount(context['component'], weights=torque[:, j],
                                                   minlength=projection.count) for j in range(3)])
    if np.any(face_power < 0) or basal_power < 0:
        raise ValueError('A passive SI sheet emitted negative dissipation.')
    power_residual = source_power-membrane_power-basal_power
    constraint_power = -float(np.sum(unbalanced*velocity))
    arithmetic_work_scale = (float(np.sum(np.abs(tangent_force*velocity))) + membrane_power
                             + basal_power + float(np.sum(np.abs(physical_apply(rigid)*velocity))))
    # Cauchy-Schwarz converts the accepted true force residual to a work bound;
    # a separate roundoff term covers the scalar dot-product/assembly sums.
    power_bound = threshold*float(np.linalg.norm(internal)) + 512.*np.finfo(float).eps*arithmetic_work_scale
    if (not np.isfinite(power_residual) or not np.isfinite(power_bound)
            or max(abs(power_residual), abs(constraint_power)) > power_bound):
        raise ValueError('Frozen SI traction response does not close its accepted work budget.')
    response = dict(context_fingerprint=context['context_fingerprint'],
                internal_velocity_m_s=velocity, total_velocity_m_s=rigid+velocity,
                strain_rate_per_s=strain, stress_pa=stress, membrane_power_by_face_w=face_power,
                tangent_nodal_force_n=tangent_force, radial_reaction_n=tangent_force-force,
                balancing_constraint_force_n=-unbalanced,
                unbalanced_rigid_torque_n_m=component_torque,
                diagnostics=dict(version=VERSION, experimental=True, frozen_load_only=True,
                    native_transport_coupled=False, boundary_resistance_coupled=False,
                    converged=True, iterations=used, relative_residual=residual_norm/scale if scale else 0.,
                    projected_residual_n_m_inv=residual_norm,
                    projected_force_n_m_inv=scale, stationarity_threshold_n_m_inv=threshold,
                    projection_roundoff_floor_n_m_inv=roundoff_floor,
                    roundoff_limited=bool(roundoff_floor > tol*scale),
                    tolerance=tol, internal_source_work_w=source_power,
                    membrane_dissipation_w=membrane_power, basal_dissipation_w=basal_power,
                    internal_power_residual_w=power_residual, power_acceptance_bound_w=power_bound,
                    rigid_source_work_w=rigid_power, total_source_work_w=total_power,
                    source_partition_residual_w=total_power-rigid_power-source_power,
                    constraint_internal_work_w=constraint_power,
                    components=projection.count,
                    model='SI Newtonian P1 sheet with constrained internal Euler modes; no live source/transport coupling'))
    response['response_fingerprint'] = _response_fingerprint(response)
    return response


def initialize_failure(context, *, tensile_strength_pa, fracture_energy_j_m2,
                       regularization_length_m, damage_efficiency):
    """Explicit reduced damage parameters and material-reference energy budget.

    The extensive budget must follow material through any future remap; it may
    not be recomputed from a changed mesh to create new fracture-energy credit.
    This routine does not infer damage or prior work for a historical world.
    """
    _validate_context(context)
    count = len(context['face_area_m2'])
    strength = _field(tensile_strength_pa, count, 'tensile_strength_pa', positive=False)
    toughness = _field(fracture_energy_j_m2, count, 'fracture_energy_j_m2')
    length = _field(regularization_length_m, count, 'regularization_length_m')
    efficiency = _field(damage_efficiency, count, 'damage_efficiency', positive=False)
    if np.any(efficiency > 1):
        raise ValueError('damage_efficiency cannot exceed the available dissipative work.')
    budget = toughness*context['face_area_m2']*context['thickness_m']/length
    if not np.isfinite(budget).all() or np.any(budget <= 0):
        raise ValueError('The physical fracture-energy budget must be finite and positive.')
    return dict(version=VERSION, context_fingerprint=context['context_fingerprint'],
                tensile_strength_pa=strength, fracture_energy_j_m2=toughness,
                regularization_length_m=length, damage_efficiency=efficiency,
                capacity_energy_j=budget, spent_energy_j=np.zeros(count))


def advance_failure(context, response, history, *, dt_s):
    """Partition existing tensile dissipation into fracture work and heat.

    Newtonian principal stresses commute with strain rate. The positive
    overstress times positive extension is capped by actual face dissipation,
    then multiplied by the explicit efficiency. This reduced damage law does
    not supply a plastic stress cap, elastic storage, crack geometry, or a ridge.
    Full exhaustion is only a local eligibility signal for a future connected
    rupture test; failure is never inferred from an underfed trench.

    Response and history must retain the exact prepared geometry/constitutive
    identity. No remap is supported; changing mesh or material parameters needs
    a separately reviewed transfer of both stored energy and material identity.
    """
    _validate_context(context)
    dt = _scalar(dt_s, 'dt_s')
    if type(history.get('version')) is not int or history['version'] != VERSION:
        raise ValueError('Unsupported fracture-energy history version.')
    if (response.get('context_fingerprint') != context['context_fingerprint']
            or history.get('context_fingerprint') != context['context_fingerprint']):
        raise ValueError('Response/history belongs to a different geometry or constitutive context.')
    diagnostic = response.get('diagnostics', {})
    if diagnostic.get('converged') is not True or diagnostic.get('frozen_load_only') is not True:
        raise ValueError('Damage requires an explicitly converged frozen-load response.')
    if response.get('response_fingerprint') != _response_fingerprint(response):
        raise ValueError('The solved response has changed; solve again before advancing damage.')
    count = len(context['face_area_m2'])
    names = ('tensile_strength_pa', 'fracture_energy_j_m2', 'regularization_length_m',
             'damage_efficiency', 'capacity_energy_j', 'spent_energy_j')
    state = {name: _field(history[name], count, name, positive=name not in
                         ('tensile_strength_pa', 'damage_efficiency', 'spent_energy_j')) for name in names}
    if any(np.shape(history[name]) != (count,) for name in names):
        raise ValueError('Saved damage fields must retain their exact face alignment.')
    if np.any(state['damage_efficiency'] > 1) or np.any(state['spent_energy_j'] > state['capacity_energy_j']):
        raise ValueError('Invalid fracture-energy budget or efficiency.')
    expected_capacity = (state['fracture_energy_j_m2']*context['face_area_m2']
                         *context['thickness_m']/state['regularization_length_m'])
    _check_saved(state['capacity_energy_j'], expected_capacity, 'Fracture-energy capacity')
    fields = viscous_sheet.strain_rate(context['sheet'], response['internal_velocity_m_s'])
    strain = fields['D']
    stress = 2.*context['viscosity_pa_s'][:, None, None]*(strain
             + fields['divergence'][:, None, None]*context['sheet']['plane'])
    _check_saved(response['strain_rate_per_s'], strain, 'Saved strain rate')
    _check_saved(response['stress_pa'], stress, 'Saved stress')
    gradient = context['sheet']['gradient']
    first = gradient[:, 1]/np.linalg.norm(gradient[:, 1], axis=1)[:, None]
    normal = np.cross(gradient[:, 1], gradient[:, 2])
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    basis = np.stack((first, np.cross(normal, first)), axis=2)
    eigen = np.linalg.eigvalsh(basis.transpose(0, 2, 1)@strain@basis)
    trace = np.trace(strain, axis1=1, axis2=2)
    # Use only the physical two-dimensional strain eigenvalues. A rounded
    # radial zero eigenvalue must not classify compression as local opening.
    principal_stress = 2.*context['viscosity_pa_s'][:, None]*(eigen+trace[:, None])
    tensile = np.sum(np.maximum(principal_stress-state['tensile_strength_pa'][:, None], 0.)
                     *np.maximum(eigen, 0.), axis=1)
    volume = context['face_area_m2']*context['thickness_m']
    available = volume*np.einsum('fij,fij->f', stress, strain)
    _check_saved(response['membrane_power_by_face_w'], available, 'Saved membrane power')
    requested = np.minimum(tensile*volume, available)*state['damage_efficiency']*dt
    spent = np.minimum(requested, state['capacity_energy_j']-state['spent_energy_j'])
    total = available*dt
    if not np.isfinite(total).all() or not np.isfinite(spent).all():
        raise ValueError('Fracture/heat transaction overflows its explicit interval.')
    state['spent_energy_j'] += spent
    state['version'] = VERSION
    state['context_fingerprint'] = context['context_fingerprint']
    damage = state['spent_energy_j']/state['capacity_energy_j']
    return dict(history=state, damage=damage,
                currently_opening=np.max(eigen, axis=1) > 0.,
                locally_exhausted=state['spent_energy_j'] >= state['capacity_energy_j'],
                fracture_increment_j=spent, membrane_heat_j=total-spent,
                diagnostics=dict(experimental=True, geometry_changed=False,
                    damage_backreaction_applied=False, native_transport_coupled=False,
                    fracture_work_j=float(np.sum(spent)), membrane_heat_j=float(np.sum(total-spent)),
                    membrane_work_j=float(np.sum(total)),
                    allocation_residual_j=float(np.sum(total-(total-spent)-spent)),
                    constitutive_law='Caller-selected tensile-overstress dissipation fraction, capped by finite material fracture energy'))
