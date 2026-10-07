"""Isolated joint rigid/internal balance on a supplied frozen SI sheet.

This is NOT a simulation policy. One lumped nodal basal quadrature and one P1
membrane operator act on the complete tangent velocity. That quadrature differs
from production's exact spherical-cell rigid basal integral; this module does
not preserve or add the production basal matrix. No transport, failure history,
source inventory, geometry, or owner is changed.

Sparse boundary rows sample total velocity. Their transposes apply integrated
forces, so the same source/resistance supplies rigid and internal work once.
The solve is matrix-free Newton/PCG; only the existing local 2x2 preconditioner
blocks are inverted. Euler/internal decomposition happens after the joint solve.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

import ocean_traction
import plate_balance
import viscous_sheet

VERSION = 1


def _values(value, count, name, *, positive=False):
    return ocean_traction._field(value, count, name, positive=positive)


@dataclass(frozen=True)
class VelocityRows:
    """COO vector rows: sample[k] = sum vector[j].velocity[node[j]].

    Vectors are dimensionless; input and sampled velocities are m/s. Repeated
    nodes/rows add. ``adjoint`` maps forces in N back to nodal forces in N.
    """
    count: int
    node_count: int
    row: np.ndarray
    node: np.ndarray
    vector: np.ndarray

    def sample(self, velocity):
        value = ocean_traction._vectors(velocity, self.node_count, 'velocity_m_s')
        return np.bincount(self.row, weights=np.einsum('ni,ni->n', self.vector, value[self.node]),
                           minlength=self.count)

    def adjoint(self, force):
        value = np.asarray(force, float)
        if value.shape != (self.count,) or not np.isfinite(value).all():
            raise ValueError('Row forces must be finite and aligned.')
        result = np.zeros((self.node_count, 3))
        np.add.at(result, self.node, self.vector*value[self.row, None])
        return result


def velocity_rows(node_count, count, row, node, vector):
    """Create explicit sparse attachments; no nearest-point mapping is inferred."""
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) or x < 1
           for x in (node_count, count)):
        raise ValueError('Velocity row and node counts must be positive integers.')
    row, node, vector = np.asarray(row), np.asarray(node), np.asarray(vector, float)
    if (row.ndim != 1 or node.shape != row.shape or row.dtype.kind not in 'iu'
            or node.dtype.kind not in 'iu' or vector.shape != (len(row), 3)
            or not np.isfinite(vector).all() or np.any(row < 0) or np.any(row >= count)
            or np.any(node < 0) or np.any(node >= node_count)):
        raise ValueError('Sparse velocity attachments must be finite aligned rows and nodes.')
    arrays = [np.array(value, copy=True) for value in (row, node, vector)]
    for value in arrays:
        value.flags.writeable = False
    return VelocityRows(int(count), int(node_count), *arrays)


def combine_rows(*terms):
    """Linear combination of aligned row maps, with scalar or per-row factors."""
    if not terms:
        raise ValueError('At least one velocity row map is required.')
    first = terms[0][1]
    if not isinstance(first, VelocityRows):
        raise ValueError('Expected a VelocityRows map.')
    rows, nodes, vectors = [], [], []
    for coefficient, mapping in terms:
        if (not isinstance(mapping, VelocityRows) or mapping.count != first.count
                or mapping.node_count != first.node_count):
            raise ValueError('Combined velocity maps must have identical dimensions.')
        factor = np.broadcast_to(np.asarray(coefficient, float), (mapping.count,))
        if not np.isfinite(factor).all():
            raise ValueError('Velocity-map factors must be finite.')
        rows.append(mapping.row); nodes.append(mapping.node)
        vectors.append(mapping.vector*factor[mapping.row, None])
    return velocity_rows(first.node_count, first.count, np.concatenate(rows),
                         np.concatenate(nodes), np.concatenate(vectors))


def moving_hinge_terms(down_normal, over_normal, relative_tangent, *, length_m,
                       excess_mass_kg_per_m, dip_radians, gravity_m_s2,
                       retained_length_scale, slab_stokes_pa_s, slab_anchor_pa_s,
                       bending_pa_s, megathrust_n_per_m, strike_slip_n_per_m,
                       regularization_m_s, response_version, resistance_version):
    """One moving-hinge v1 source and its passive boundary resistances.

    Both normal maps point from incoming toward overriding lithosphere;
    q=down-over is positive intake. ``relative_tangent`` samples their signed
    relative tangential speed. Coefficients and attachments are explicit inputs,
    not production defaults. Bending is the caller's integrated viscosity law
    in Pa s; megathrust/strike-slip strengths are N per metre of front.
    """
    if (type(response_version) is not int or response_version != 1
            or type(resistance_version) is not int or resistance_version != 1):
        raise ValueError('Only moving-hinge v1 and passive resistance v1 are supported.')
    # Combining also checks attachment dimensions before any assembly.
    closing = combine_rows((1., down_normal), (-1., over_normal))
    opening = combine_rows((-1., closing))
    combine_rows((1., closing), (0., relative_tangent))
    n = closing.count
    length = _values(length_m, n, 'length_m', positive=True)
    mass = _values(excess_mass_kg_per_m, n, 'excess_mass_kg_per_m')
    dip = _values(dip_radians, n, 'dip_radians', positive=True)
    if np.any(dip >= np.pi/2):
        raise ValueError('Slab dips must lie strictly between zero and pi/2.')
    gravity = ocean_traction._scalar(gravity_m_s2, 'gravity_m_s2')
    retained = _values(retained_length_scale, n, 'retained_length_scale')
    if np.any(retained > 1):
        raise ValueError('Retained-length resistance scale cannot exceed one.')
    stokes = _values(slab_stokes_pa_s, n, 'slab_stokes_pa_s')*retained*length
    anchor = _values(slab_anchor_pa_s, n, 'slab_anchor_pa_s')*retained*length
    bending = _values(bending_pa_s, n, 'bending_pa_s')*length
    mega = _values(megathrust_n_per_m, n, 'megathrust_n_per_m')*length
    slip = _values(strike_slip_n_per_m, n, 'strike_slip_n_per_m')*length
    width = _values(regularization_m_s, n, 'regularization_m_s', positive=True)
    horizontal = combine_rows((np.cos(dip), down_normal), (1.-np.cos(dip), over_normal))
    vertical = combine_rows((np.sin(dip), closing))
    return dict(
        sources=[dict(name='slab_gravity', force_n=closing.adjoint(gravity*mass*length*np.sin(dip)))],
        quadratics=[dict(name='slab_stokes_horizontal', rows=horizontal, coefficient=stokes),
                    dict(name='slab_stokes_vertical', rows=vertical, coefficient=stokes),
                    dict(name='slab_anchor', rows=over_normal, coefficient=anchor)],
        passive=[dict(name='hinge', shape='closing_quadratic', rows=(opening,), coefficient=bending),
                 dict(name='megathrust', shape='cone', rows=(opening, relative_tangent),
                      coefficient=mega, width_m_s=width),
                 dict(name='strike_slip', shape='abs', rows=(relative_tangent,),
                      coefficient=slip, width_m_s=width)])


def _terms(context, sources, quadratics, passive):
    ocean_traction._validate_context(context)
    count = len(context['sheet']['points'])
    # This prototype represents one rigid carrier per connected owner. Giving
    # disconnected pieces independent Euler modes is a separate topology law.
    labels = [np.unique(context['owner_labels'][context['component'] == c])
              for c in range(context['projection'].count)]
    if any(len(label) != 1 for label in labels) or len(set(int(label[0]) for label in labels)) != len(labels):
        raise ValueError('The joint prototype requires one connected component per owner.')
    seen = set()

    def name(row):
        value = row.get('name')
        if not isinstance(value, str) or not value or value in seen:
            raise ValueError('Every source and resistance needs a unique nonempty name.')
        seen.add(value)
        return value

    source = [(name(row), ocean_traction._vectors(row['force_n'], count, 'source_force_n')) for row in sources]
    quadratic, nonlinear = [], []
    for row in quadratics:
        title, mapping = name(row), row['rows']
        if not isinstance(mapping, VelocityRows) or mapping.node_count != count:
            raise ValueError('Quadratic row map does not belong to this sheet.')
        quadratic.append((title, mapping, _values(row['coefficient'], mapping.count, title)))
    for row in passive:
        title, shape, maps = name(row), row['shape'], tuple(row['rows'])
        if shape not in ('abs', 'negative', 'cone', 'closing_quadratic'):
            raise ValueError('Unsupported passive boundary shape.')
        if (len(maps) != (2 if shape == 'cone' else 1)
                or any(not isinstance(m, VelocityRows) or m.node_count != count for m in maps)
                or any(m.count != maps[0].count for m in maps)):
            raise ValueError('Passive row maps must align with this sheet.')
        coefficient = _values(row['coefficient'], maps[0].count, title)
        width = (None if shape == 'closing_quadratic' else
                 _values(row['width_m_s'], maps[0].count, 'width_m_s', positive=True))
        nonlinear.append((title, shape, maps, coefficient, width))
    return source, quadratic, nonlinear


def _terms_fingerprint(source, quadratic, nonlinear):
    """Bind the actual frozen forcing/row laws, not a security signature.

    The validated arrays include every attachment, coefficient, regularization
    width and potential shape. This identifies the numerical forcing after
    unit conversion, independently of the constitutive/context fingerprint.
    """
    items = [('version', VERSION), ('term_counts', [len(source), len(quadratic), len(nonlinear)])]

    def mapping(prefix, rows):
        items.extend((prefix+'.'+key, value) for key, value in (
            ('count', rows.count), ('node_count', rows.node_count),
            ('row', rows.row), ('node', rows.node), ('vector', rows.vector)))

    for index, (name, force) in enumerate(source):
        prefix = 'source.'+str(index)
        items.extend(((prefix+'.name', name), (prefix+'.force_n', force)))
    for index, (name, rows, coefficient) in enumerate(quadratic):
        prefix = 'quadratic.'+str(index)
        items.extend(((prefix+'.name', name), (prefix+'.coefficient', coefficient)))
        mapping(prefix+'.rows', rows)
    for index, (name, shape, maps, coefficient, width) in enumerate(nonlinear):
        prefix = 'passive.'+str(index)
        items.extend(((prefix+'.name', name), (prefix+'.shape', shape),
                      (prefix+'.coefficient', coefficient), (prefix+'.row_count', len(maps)),
                      (prefix+'.width_m_s', np.empty(0) if width is None else width)))
        for row_index, rows in enumerate(maps):
            mapping(prefix+'.rows.'+str(row_index), rows)
    return ocean_traction._fingerprint(items)


def _passive_values(shape, rates, width):
    """Production passive-v1 potentials and derivatives, now sampled in SI."""
    if shape == 'cone':
        opening, tangent = rates
        closing, first, second = plate_balance._cone_closing_terms(opening, width, 1)
        magnitude = np.sqrt(closing*closing+tangent*tangent+width*width)
        potential = (closing*closing+tangent*tangent)/(magnitude+width)
        normal_first = closing*first/magnitude
        tangent_first = tangent/magnitude
        gradient = (normal_first, tangent_first)
        hessian = ((first*first+closing*second)/magnitude-normal_first**2/magnitude,
                   -normal_first*tangent_first/magnitude,
                   1./magnitude-tangent_first**2/magnitude)
        return potential, gradient, hessian
    (rate,) = rates
    if shape == 'closing_quadratic':
        closed = np.minimum(rate, 0.)
        return .5*closed*closed, (closed,), ((rate < 0.).astype(float),)
    law = plate_balance._abs_terms if shape == 'abs' else plate_balance._passive_negative_terms
    value, first, second = law(rate, width)
    return value, (first,), (second,)


def solve(context, *, sources, quadratics=(), passive=(), rigid_only=False,
          tolerance=1e-9, max_newton_iterations=60, max_cg_iterations=1600):
    """Solve joint tangent velocities without mutating any supplied state.

    Source forces are N. Quadratic coefficients are N s/m; passive abs,
    negative and cone coefficients are N and widths m/s. Closing-quadratic
    coefficients are N s/m. One scalar basal drag and the supplied eta/H fields
    come from ``ocean_traction.prepare``. No production coefficient is inferred.
    ``rigid_only`` restricts the SAME quadrature/operator to its Euler subspace.
    """
    if type(rigid_only) is not bool:
        raise ValueError('rigid_only must be an explicit boolean.')
    tol = ocean_traction._scalar(tolerance, 'tolerance')
    if tol >= 1:
        raise ValueError('tolerance must be below one.')
    for value in (max_newton_iterations, max_cg_iterations):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError('Iteration limits must be positive integers.')
    source, quadratic, nonlinear = _terms(context, sources, quadratics, passive)
    terms_fingerprint = _terms_fingerprint(source, quadratic, nonlinear)
    sheet, mode = context['sheet'], context['projection']
    points, root = sheet['points'], mode.root_area[:, None]
    drag = context['basal_drag_pa_s_m']
    force = sum((value for _, value in source), np.zeros_like(points))
    force = mode.tangent(force)

    def project(value):
        value = mode.tangent(value)
        return mode.tangent(value-mode.apply(value)) if rigid_only else value

    def base(velocity):
        return drag*viscous_sheet.apply(sheet, velocity)

    def evaluate(weighted):
        velocity = project(weighted)/root
        base_force = base(velocity)
        objective = .5*float(np.sum(velocity*base_force))-float(np.sum(force*velocity))
        gradient = base_force-force
        force_terms = [base_force, -force]
        quadratic_work, potential, resistance_work = {}, {}, {}
        curvature = []
        for title, mapping, coefficient in quadratic:
            rate = mapping.sample(velocity)
            work = float(coefficient@(rate*rate))
            quadratic_work[title] = work
            objective += .5*work
            term = mapping.adjoint(coefficient*rate)
            gradient += term; force_terms.append(term)
        for title, shape, maps, coefficient, width in nonlinear:
            rates = tuple(mapping.sample(velocity) for mapping in maps)
            value, first, second = _passive_values(shape, rates, width)
            potential[title] = float(coefficient@value)
            resistance_work[title] = float(sum(coefficient@(rate*derivative) for rate, derivative in zip(rates, first)))
            objective += potential[title]
            term = sum((mapping.adjoint(coefficient*derivative) for mapping, derivative in zip(maps, first)),
                       np.zeros_like(points))
            gradient += term; force_terms.append(term)
            curvature.append((maps, coefficient, second))
        projected = project(gradient/root)
        terms = [project(term/root) for term in force_terms]
        scale = sum(float(np.linalg.norm(term)) for term in terms)
        rounding = 256.*np.finfo(float).eps*scale
        limits = [(float(np.linalg.norm(projected)), tol*scale+rounding)]
        if not rigid_only:
            # Check each mode independently: a large rigid load must not hide
            # smaller resolvable internal forces (or vice versa).
            for select in (mode.apply, lambda v: project(v)-mode.apply(v)):
                mode_scale = sum(float(np.linalg.norm(select(term))) for term in terms)
                limits.append((float(np.linalg.norm(select(projected))), tol*mode_scale+rounding))
        relative = max((norm/bound if bound else (0. if norm == 0 else np.inf)) for norm, bound in limits)
        if not np.isfinite(objective) or not np.isfinite(relative) or not np.isfinite(projected).all():
            raise ValueError('Joint frozen balance produced non-finite objective or force.')

        def hessian(value):
            v = project(value)/root
            result = base(v)
            for _, mapping, coefficient in quadratic:
                result += mapping.adjoint(coefficient*mapping.sample(v))
            for maps, coefficient, second in curvature:
                rates = [mapping.sample(v) for mapping in maps]
                if len(maps) == 1:
                    result += maps[0].adjoint(coefficient*second[0]*rates[0])
                else:
                    nn, nt, tt = second
                    result += maps[0].adjoint(coefficient*(nn*rates[0]+nt*rates[1]))
                    result += maps[1].adjoint(coefficient*(nt*rates[0]+tt*rates[1]))
            return project(result/root)
        return dict(value=objective, gradient=projected, hessian=hessian,
                    gate_ratio=relative, force_scale=scale, limits=limits,
                    quadratic_work=quadratic_work, potential=potential, resistance_work=resistance_work)

    precondition, _ = viscous_sheet._preconditioner(
        sheet, np.ones(len(points), bool), np.zeros(len(points), bool), np.zeros_like(points))

    def inverse_diagonal(value):
        return project(root*precondition(root*project(value))/drag)

    def direction_for(record):
        rhs = -record['gradient']
        answer = np.zeros_like(rhs); residual = rhs.copy()
        norm = float(np.linalg.norm(rhs))
        threshold = max(tol*.05, 128.*np.finfo(float).eps)*norm
        z = inverse_diagonal(residual); direction = z.copy()
        rz = float(np.sum(residual*z))
        for count in range(1, int(max_cg_iterations)+1):
            product = record['hessian'](direction)
            denominator = float(np.sum(direction*product))
            if denominator <= 0 or rz <= 0 or not np.isfinite(denominator+rz):
                raise ValueError('Joint projected Hessian lost positive curvature.')
            answer += (rz/denominator)*direction
            residual -= (rz/denominator)*product
            check = np.linalg.norm(residual) <= threshold or count == max_cg_iterations
            if check:
                answer = project(answer)
                residual = rhs-record['hessian'](answer)
                if np.linalg.norm(residual) <= threshold:
                    return answer, count
            z = inverse_diagonal(residual); next_rz = float(np.sum(residual*z))
            direction = z.copy() if check else z+direction*(next_rz/rz)
            rz = next_rz
        raise ValueError('Joint frozen balance CG did not reach recomputed stationarity.')

    weighted = np.zeros_like(points)
    record = evaluate(weighted)
    newton = cg_count = backtracks = 0
    while record['gate_ratio'] > 1.:
        if newton >= max_newton_iterations:
            raise ValueError('Joint frozen balance did not reach recomputed stationarity.')
        direction, iterations = direction_for(record)
        cg_count += iterations
        slope = float(np.sum(record['gradient']*direction))
        if not np.isfinite(slope) or slope >= 0:
            raise ValueError('Joint Newton direction does not decrease the potential.')
        alpha = 1.
        while alpha > 1e-14:
            candidate = project(weighted+alpha*direction)
            trial = evaluate(candidate)
            roundoff = 32.*np.finfo(float).eps*max(abs(record['value']), abs(trial['value']), 1e-300)
            if (trial['value'] <= record['value']+1e-4*alpha*slope
                    or (-alpha*slope <= roundoff and trial['value'] <= record['value']+roundoff
                        and trial['gate_ratio'] < record['gate_ratio'])):
                weighted, record = candidate, trial
                break
            alpha *= .5; backtracks += 1
        else:
            raise ValueError('Joint frozen balance line search failed.')
        newton += 1

    velocity = project(weighted)/root
    internal = mode.apply(weighted)/root
    rigid = velocity-internal
    rotations = mode.fit(weighted)/context['radius_m']
    fields = viscous_sheet.strain_rate(sheet, velocity)
    strain, divergence = fields['D'], fields['divergence']
    stress = 2.*context['viscosity_pa_s'][:, None, None]*(strain+divergence[:, None, None]*sheet['plane'])
    membrane_faces = context['face_area_m2']*context['thickness_m']*np.einsum('fij,fij->f', stress, strain)
    membrane = float(membrane_faces.sum())
    basal = drag*float(np.sum(sheet['data'][:, None]*velocity**2))
    source_work = {name: float(np.sum(value*velocity)) for name, value in source}
    driving = sum(source_work.values())
    resisting = membrane+basal+sum(record['quadratic_work'].values())+sum(record['resistance_work'].values())
    residual_power = driving-resisting
    work_scale = float(np.sum(np.abs(force*velocity)))+abs(resisting)
    power_bound = record['limits'][0][1]*float(np.linalg.norm(weighted))+1024.*np.finfo(float).eps*work_scale
    if (any(not np.isfinite(value) or value < 0 for value in
            [membrane, basal, *record['quadratic_work'].values(), *record['resistance_work'].values()])
            or not np.isfinite(residual_power+power_bound) or abs(residual_power) > power_bound):
        raise ValueError('Joint frozen response fails passive work closure.')
    sampled = {name: mapping.sample(velocity) for name, mapping, _ in quadratic}
    sampled.update({name: [mapping.sample(velocity) for mapping in maps]
                    for name, _, maps, _, _ in nonlinear})
    return dict(context_fingerprint=context['context_fingerprint'],
        frozen_terms_fingerprint=terms_fingerprint,
        total_velocity_m_s=velocity, rigid_velocity_m_s=rigid,
        internal_velocity_m_s=internal, component_omega_rad_s=rotations,
        strain_rate_per_s=strain, stress_pa=stress, boundary_rates_m_s=sampled,
        diagnostics=dict(version=VERSION, experimental=True, frozen_geometry_only=True,
            native_transport_coupled=False, production_discretization_equivalent=False,
            basal_quadrature='single lumped nodal area; differs from production exact spherical-cell rigid metric',
            rigid_only=rigid_only, converged=True, stationarity_gate_ratio=record['gate_ratio'],
            force_mode_residuals=[dict(norm_n_m_inv=a, acceptance_n_m_inv=b) for a, b in record['limits']],
            newton_iterations=newton, cg_iterations=cg_count, backtracks=backtracks,
            source_work_w=source_work, total_source_work_w=driving,
            rigid_source_work_w=float(np.sum(force*rigid)), internal_source_work_w=float(np.sum(force*internal)),
            membrane_dissipation_w=membrane, basal_dissipation_w=basal,
            quadratic_dissipation_w=record['quadratic_work'], passive_resistance_work_w=record['resistance_work'],
            passive_potential_w=record['potential'], total_resisting_work_w=resisting,
            power_residual_w=residual_power, power_acceptance_bound_w=power_bound,
            objective_w=record['value'], state_changed=False))
