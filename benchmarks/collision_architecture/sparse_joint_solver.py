"""Constrained matrix-free joint SI research solve; no native call site.

Material operators remain local. A bounded dense constraint-row Gram is used
for projection/rank evidence; there is no material-sized matrix or nullspace.
"""
import hashlib
import numpy as np

from . import sparse_shared_contact as assembly


def _finite(*values):
    if not all(np.isfinite(value).all() for value in values):
        raise RuntimeError('Joint iteration is outside finite arithmetic range.')


def _norm(value):
    value = np.asarray(value)
    scale = np.max(np.abs(value), initial=0.)
    result = 0. if scale == 0. else float(scale*np.sqrt(np.sum((value/scale)**2)))
    _finite(result)
    return result


def _absolute_map(operator, value, *, transpose=False):
    """Coefficient-product bound, preserving local scatter cancellation costs."""
    if transpose:
        local = np.einsum('bij,bi->bj', abs(operator.blocks), value[operator.rows])
        result = np.bincount(operator.columns.ravel(), weights=local.ravel(), minlength=operator.shape[1])
    else:
        local = np.einsum('bij,bj->bi', abs(operator.blocks), value[operator.columns])
        result = np.bincount(operator.rows.ravel(), weights=local.ravel(), minlength=operator.shape[0])
    _finite(result)
    return result


def _gram_diagonal(operator, weights=None):
    blocks = operator.blocks**2
    if weights is not None:
        blocks = blocks*weights[operator.rows, None]
    result = np.bincount(operator.columns.ravel(), weights=blocks.sum(axis=1).ravel(), minlength=operator.shape[1])
    _finite(result)
    return result


def _diagonal(system):
    if 'hessian_diagonal_n_s_m' in system:
        result = np.asarray(system['hessian_diagonal_n_s_m'], float).copy()
    else:
        result = _gram_diagonal(system['velocity_map'], np.repeat(system['basal_drag_weight_n_s_m'], 3))
        result += _gram_diagonal(system['strain_design'])
        result[:system['nplate']] += np.diag(system['hessian_n_s_m'].other_plate_matrix)
    if result.shape != system['load_n'].shape or np.any(result < 0.):
        raise ValueError('The base Hessian diagonal must be aligned, finite and nonnegative.')
    _finite(result)
    return result


def _absolute_hessian(system, value):
    if 'hessian_absolute_action' in system:
        result = np.asarray(system['hessian_absolute_action'](value), float)
    else:
        b, f = system['velocity_map'], system['strain_design']
        weights = np.repeat(system['basal_drag_weight_n_s_m'], 3)
        result = _absolute_map(b, weights*_absolute_map(b, value), transpose=True)
        result += _absolute_map(f, _absolute_map(f, value), transpose=True)
        cut = system['nplate']
        result[:cut] += abs(system['hessian_n_s_m'].other_plate_matrix)@value[:cut]
    if result.shape != value.shape or np.any(result < 0.):
        raise ValueError('Absolute Hessian action must be an aligned nonnegative product bound.')
    _finite(result)
    return result


class _Constraints:
    def __init__(self, system, diagonal, maximum_rows):
        self.system = system
        self.diagonal = diagonal.copy()
        fixed = np.asarray(system.get('fixed_residual_dofs', np.zeros((len(system['points']), 2), bool)))
        if fixed.dtype.kind != 'b' or fixed.shape != (len(system['points']), 2):
            raise ValueError('Fixed residual coordinates require an aligned Boolean mask.')
        if ('fixed_residual_dofs_sha256' in system and hashlib.sha256(fixed.tobytes()).hexdigest()
                != system['fixed_residual_dofs_sha256']):
            raise ValueError('Fixed residual mask changed after its frame gauge was assembled.')
        self.free = np.r_[np.ones(system['nplate'], dtype=bool), ~fixed.ravel()]
        if np.any(diagonal[self.free] <= 0.) or np.any(diagonal < 0.):
            raise ValueError('Every free coordinate requires a positive combined Hessian diagonal at the current trial.')
        _finite(diagonal)
        self.scale = np.zeros_like(diagonal)
        self.scale[self.free] = 1/np.sqrt(diagonal[self.free])
        _finite(self.scale)
        self.count = system['nplate']+len(system['contact_weights'])
        # Only structurally zero restricted rows are removed. Their original
        # outputs are still retained and checked. No magnitude threshold is
        # used to discard a nonzero physical equation.
        row_support = _absolute_constraints(system, self.free.astype(float))
        self.active = np.flatnonzero(row_support > 0.)
        self.zero_rows = np.flatnonzero(row_support == 0.)
        active_count = len(self.active)
        if active_count > maximum_rows:
            raise ValueError('Constraint-row budget exceeded; no unbounded dense Schur allocation is permitted.')
        # Compute C D^-1 C^T only through original local actions. Storage O(m²+N).
        gram = np.empty((active_count, active_count))
        for j, original_row in enumerate(self.active):
            unit = np.zeros(self.count); unit[original_row] = 1.
            gram[:, j] = self.action(self.scale**2*self.transpose(unit))[self.active]
        _finite(gram)
        if not active_count:
            self.row_norm = np.empty(0); self.rank_tolerance = 0.
            self.rank = 0; self.eigenvalues = np.empty(0)
            self.inverse = np.empty((0, 0)); self.dual_null = np.empty((0, 0))
            return
        row_norm = np.sqrt(np.diag(gram))
        if np.any(row_norm <= 0.):
            raise ValueError('A nonzero restricted constraint row is numerically unresolved.')
        self.row_norm = row_norm
        gram = gram/row_norm[:, None]/row_norm[None, :]
        asymmetry = np.max(abs(gram-gram.T))
        if asymmetry > 512*np.finfo(float).eps*max(active_count, 1):
            raise RuntimeError('Constraint Gram action/transpose is not symmetric.')
        gram = .5*(gram+gram.T)
        eigen, vectors = np.linalg.eigh(gram)
        self.rank_tolerance = 64*np.finfo(float).eps*max(active_count, 1)*max(float(eigen[-1]), 1.)
        if eigen[0] < -self.rank_tolerance:
            raise RuntimeError('Constraint Gram is not positive semidefinite.')
        keep = eigen > self.rank_tolerance
        self.rank = int(keep.sum()); self.eigenvalues = eigen
        self.inverse = (vectors[:, keep]/eigen[keep])@vectors[:, keep].T
        # Redundant dual directions live in row space only. QR does not touch N.
        null = vectors[:, ~keep]/row_norm[:, None]
        self.dual_null = np.linalg.qr(null, mode='reduced')[0] if null.shape[1] else null

    def action(self, value):
        return assembly.constraint_action(self.system, value)

    def transpose(self, value):
        return assembly.constraint_transpose(self.system, value)

    def normalized_action(self, value):
        return self.action(self.scale*value)[self.active]/self.row_norm

    def normalized_transpose(self, value):
        expanded = np.zeros(self.count)
        expanded[self.active] = value/self.row_norm
        return self.scale*self.transpose(expanded)

    def project(self, value):
        # Twice removes O(eps) leakage through the finite row-Gram inverse.
        result = value*self.free
        for _ in range(2):
            result -= self.normalized_transpose(self.inverse@self.normalized_action(result))
        _finite(result)
        return result

    def multipliers(self, gradient):
        reduced = -self.inverse@self.normalized_action(self.scale*gradient)
        active_result = reduced/self.row_norm
        if self.dual_null.shape[1]:
            active_result -= self.dual_null@(self.dual_null.T@active_result)
        result = np.zeros(self.count)
        result[self.active] = active_result
        _finite(result)
        return result


def _pcg(action, right, constraints, *, tolerance, maximum_iterations):
    """CG in the projected, diagonal-congruence coordinates; exact actions refreshed."""
    rhs = constraints.project(right)
    rhs_norm = _norm(rhs)
    if rhs_norm == 0.:
        return np.zeros_like(rhs), 0
    x = np.zeros_like(rhs); residual = rhs.copy(); direction = residual.copy()
    square = float(residual@residual)
    for iteration in range(1, maximum_iterations+1):
        ad = constraints.project(action(constraints.project(direction)))
        curvature = float(direction@ad)
        _finite(curvature, square)
        if curvature <= 0.:
            raise RuntimeError('Nonpositive or unresolved curvature on the constraint nullspace.')
        alpha = square/curvature
        x += alpha*direction
        residual -= alpha*ad
        # Restarts use the true operator residual, not only a drifting recurrence.
        refresh = iteration % 40 == 0 or _norm(residual) <= tolerance*rhs_norm
        if refresh:
            x = constraints.project(x)
            residual = constraints.project(rhs-action(x))
            if _norm(residual) <= tolerance*rhs_norm:
                return x, iteration
            direction = residual.copy(); square = float(residual@residual)
        else:
            following = float(residual@residual)
            direction = residual+(following/square)*direction
            square = following
    raise RuntimeError('Projected CG exhausted its iteration budget without a true residual solution.')


def _resistance(operator, y):
    if operator is None:
        return dict(potential_w=0., gradient_n=np.zeros_like(y), gradient_absolute_scale_n=np.zeros_like(y),
            resisting_power_w=0., diagonal_n_s_m=np.zeros_like(y), hessian_n_s_m=None, diagnostics={})
    readonly = y.copy(); readonly.flags.writeable = False
    result = dict(operator.evaluate(readonly))
    for name in ('gradient_n', 'diagonal_n_s_m'):
        result[name] = np.asarray(result[name], float)
        if result[name].shape != y.shape:
            raise ValueError('Nonlinear resistance vectors must align with the joint system.')
    result['gradient_absolute_scale_n'] = np.asarray(result.get('gradient_absolute_scale_n', abs(result['gradient_n'])), float)
    result.setdefault('diagnostics', {})
    potential = float(result['potential_w']); power = float(result['resisting_power_w'])
    bound = result['gradient_absolute_scale_n']
    if (bound.shape != y.shape or np.any(bound < abs(result['gradient_n']))
            or np.any(result['diagonal_n_s_m'] < 0.) or potential < 0. or power < 0.):
        raise ValueError('Passive resistance must provide nonnegative potential, power and diagonal/product bounds.')
    _finite(potential, power, result['gradient_n'], bound, result['diagonal_n_s_m'])
    error = abs(float(y@result['gradient_n'])-power)
    work_scale = float(abs(y)@bound)+power
    if error > 256*np.finfo(float).eps*max(len(y), 1)*work_scale:
        raise ValueError('Resistance gradient and reported resisting power disagree.')
    result['potential_w'] = potential; result['resisting_power_w'] = power
    return result


def _absolute_constraints(system, y, *, transpose=False):
    cut = system['nplate']; g, c = system['gauge_matrix'], system['contact_matrix']
    if transpose:
        return _absolute_map(g, y[:cut], transpose=True)+_absolute_map(c, system['contact_weights']*y[cut:], transpose=True)
    return np.r_[_absolute_map(g, y), system['contact_weights']*_absolute_map(c, y)]


def _state(system, y, nonlinear, constraints):
    action = system['hessian_n_s_m']@y
    gradient = action-system['load_n']+nonlinear['gradient_n']
    multiplier = constraints.multipliers(gradient)
    unconstrained_stationarity = gradient+constraints.transpose(multiplier)
    fixed_reaction = unconstrained_stationarity*(~constraints.free)
    stationarity = unconstrained_stationarity-fixed_reaction
    component_scale = (_absolute_hessian(system, abs(y))+abs(system['load_n'])+
        nonlinear['gradient_absolute_scale_n']+_absolute_constraints(system, abs(multiplier), transpose=True)+abs(fixed_reaction))
    component = np.divide(abs(stationarity), component_scale, out=np.zeros_like(y), where=component_scale > 0.)
    component[(component_scale == 0.) & (stationarity != 0.)] = np.inf
    original = constraints.action(y)
    row_scale = _absolute_constraints(system, abs(y))
    row_error = np.divide(abs(original), row_scale, out=np.zeros_like(original), where=row_scale > 0.)
    row_error[(row_scale == 0.) & (original != 0.)] = np.inf
    free = constraints.free
    global_scale = (_norm(action[free])+_norm(system['load_n'][free])+_norm(nonlinear['gradient_n'][free])
        +_norm(constraints.transpose(multiplier)[free]))
    global_error = 0. if global_scale == 0. else _norm(stationarity)/global_scale
    _finite(action, gradient, multiplier, stationarity, component_scale, component, original, row_scale, row_error)
    return dict(action=action, gradient=gradient, multiplier=multiplier, stationarity=stationarity,
        fixed_reaction=fixed_reaction, pre_fixed_stationarity=unconstrained_stationarity,
        component_scale=component_scale, component=component, original=original,
        row_scale=row_scale, row_error=row_error, global_error=global_error)


def solve(system, *, resistance=None, relative_tolerance=1e-10,
          maximum_iterations=2000, maximum_newton_iterations=50,
          maximum_constraint_rows=512, maximum_backtracks=60):
    """Solve the supplied frozen functional; all original rows and work are gated.

    Constraints are homogeneous bilateral equalities. A supplied nonlinear law
    must be a pure, convex passive potential with consistent gradient/Hessian.
    No contact admission, topology update or coefficient calibration is inferred.
    """
    tolerance = float(relative_tolerance)
    if not np.isfinite(tolerance) or not 0. < tolerance <= 1e-4:
        raise ValueError('Relative tolerance must be finite in (0, 1e-4].')
    for value in (maximum_iterations, maximum_newton_iterations, maximum_constraint_rows, maximum_backtracks):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError('Solver resource budgets must be positive integers.')
    if 'validate_system' in system:
        system['validate_system'](system)
    if resistance is not None and hasattr(resistance, 'validate_system'):
        resistance.validate_system(system)
    y = np.zeros_like(system['load_n']); diagonal = _diagonal(system)
    nonlinear = _resistance(resistance, y)
    constraints = _Constraints(system, diagonal+nonlinear['diagonal_n_s_m'], maximum_constraint_rows)
    history = []; cg_total = 0; backtracks = 0
    inner_tolerance = min(tolerance*.01, 2e-13)
    state = _state(system, y, nonlinear, constraints)
    for iteration in range(maximum_newton_iterations+1):
        if (state['global_error'] <= tolerance and np.max(state['component'], initial=0.) <= tolerance
                and np.max(state['row_error'], initial=0.) <= tolerance):
            break
        if iteration == maximum_newton_iterations:
            raise RuntimeError('Joint Newton solve exhausted its budget before original force/constraint acceptance.')
        # Rebuild only the small projection Gram when nonlinear diagonal changes.
        current_diagonal = diagonal+nonlinear['diagonal_n_s_m']
        if not np.array_equal(current_diagonal, constraints.diagonal):
            constraints = _Constraints(system, current_diagonal, maximum_constraint_rows)
        scale = constraints.scale
        def hessian(value):
            result = system['hessian_n_s_m']@value
            if nonlinear['hessian_n_s_m'] is not None:
                result += nonlinear['hessian_n_s_m']@value
            _finite(result)
            return result
        step, cg_count = _pcg(lambda z: scale*hessian(scale*z), -scale*state['gradient'],
            constraints, tolerance=inner_tolerance, maximum_iterations=maximum_iterations)
        cg_total += cg_count
        direction = scale*step
        slope = float(state['gradient']@direction)
        if not np.isfinite(slope) or slope >= 0.:
            raise RuntimeError('Joint Newton direction is not a finite descent direction.')
        length = 1.; accepted = False
        for trial in range(maximum_backtracks):
            scaled_candidate = np.divide(y+length*direction, scale, out=np.zeros_like(y), where=constraints.free)
            candidate = scale*constraints.project(scaled_candidate)
            next_nonlinear = _resistance(resistance, candidate)
            displacement = candidate-y
            # Difference form avoids subtracting large quadratic objective values.
            change = float(displacement@(state['action']-system['load_n'])+
                .5*displacement@(system['hessian_n_s_m']@displacement))
            change += next_nonlinear['potential_w']-nonlinear['potential_w']
            products = (float(abs(displacement)@(abs(state['action'])+abs(system['load_n'])))+
                abs(next_nonlinear['potential_w'])+abs(nonlinear['potential_w']))
            arithmetic = 64*np.finfo(float).eps*max(len(y), 1)*products
            _finite(change, arithmetic)
            if change <= 1e-4*length*slope+arithmetic:
                accepted = True; break
            length *= .5; backtracks += 1
        if not accepted:
            raise RuntimeError('Total nonlinear potential line search exhausted its budget.')
        y = candidate; nonlinear = next_nonlinear
        state = _state(system, y, nonlinear, constraints)
        history.append(dict(cg_iterations=cg_count, step_fraction=length,
            maximum_component_residual=float(np.max(state['component'])),
            maximum_original_row_error=float(np.max(state['row_error'])), potential_change_w=change))
    return _result(system, y, nonlinear, state, constraints, tolerance,
        dict(nonlinear_iterations=len(history), cg_iterations=cg_total, nonlinear_backtracks=backtracks,
             iteration_history=history, maximum_constraint_rows=maximum_constraint_rows))


def _result(system, y, nonlinear, state, constraints, tolerance, iterations):
    cut = system['nplate']; multiplier = state['multiplier']
    work_function = system.get('physical_work', lambda value: assembly.work(system, value))
    powers = dict(work_function(y))
    if 'unsolved_power_defect_w' in powers:
        powers['linear_functional_power_defect_w'] = powers.pop('unsolved_power_defect_w')
    if 'generalized_gradient_work_w' in powers:
        powers['linear_gradient_work_w'] = powers.pop('generalized_gradient_work_w')
    passive = sum(powers[name] for name in ('basal_dissipation_w', 'viscous_dissipation_w', 'other_drag_dissipation_w'))
    passive += nonlinear['resisting_power_w']
    power_error = passive-powers['external_power_w']-powers['basal_reference_input_power_w']
    solver_envelope = float(abs(y)@abs(state['stationarity'])+abs(multiplier)@abs(state['original']))
    work_scale = passive+abs(powers['external_power_w'])+abs(powers['basal_reference_input_power_w'])
    velocity = (system['velocity_map']@y).reshape(-1, 3)
    if 'physical_work_absolute_scale_w' in system:
        basal_bound = float(system['physical_work_absolute_scale_w'](y))
    else:
        basal_bound = float(np.sum(system['basal_drag_weight_n_s_m'][:, None]*
            (abs(velocity)+abs(system['mantle_velocity_m_s']))**2))
    products = float(abs(y)@state['component_scale'])+basal_bound
    operations = 16*(len(y)+system['velocity_map'].shape[0]+system['strain_design'].shape[0]+constraints.count+16)
    eps = operations*np.finfo(float).eps
    if eps >= 1.: raise RuntimeError('Arithmetic estimate is unresolved at this problem size.')
    arithmetic = eps/(1.-eps)*products
    allowance = tolerance*work_scale+solver_envelope+arithmetic
    resultant = system['contact_weights']*multiplier[cut:]
    contact_reaction = -system['contact_matrix'].rmatvec(resultant)
    gauge_reaction = -system['gauge_matrix'].rmatvec(multiplier[:cut])
    contact_work = float(contact_reaction@y); gauge_work = float(gauge_reaction@y)
    first_force = -resultant[:, None]*system['contact_normal']
    radius = system['radius_m']
    omega = y[:cut].reshape(-1, 3)/radius
    residual_velocity = np.einsum('nia,na->ni', system['tangent_basis'], y[cut:].reshape(-1, 2))
    traction = resultant/system['contact_length_m']
    contact_torque = (radius*contact_reaction[:cut]).reshape(-1, 3)
    resistance_torque = (-radius*nonlinear['gradient_n'][:cut]).reshape(-1, 3)
    fixed_reaction = state['fixed_reaction']
    fixed_nodal_force = np.einsum('nia,na->ni', system['tangent_basis'], fixed_reaction[cut:].reshape(-1, 2))
    # z=0 is a material-to-plate-frame attachment. The nodal force therefore
    # has the opposite torque on that SAME Euler frame; their total virtual
    # work is exactly the eliminated-coordinate work, not external power.
    fixed_torque = np.zeros((system['plate_count'], 3))
    np.add.at(fixed_torque, system['vertex_plate'], -radius*np.cross(system['points'], fixed_nodal_force))
    fixed_work = float(fixed_reaction@y)
    fixed_material_work = float(np.sum(fixed_nodal_force*velocity))
    fixed_plate_work = float(np.sum(fixed_torque*omega))
    fixed_exchange_error = fixed_material_work+fixed_plate_work-fixed_work
    objective = powers['objective_w']+nonlinear['potential_w']
    _finite(*powers.values(), passive, power_error, solver_envelope, work_scale, products, arithmetic, allowance,
        resultant, contact_reaction, gauge_reaction, contact_work, gauge_work, first_force, omega,
        residual_velocity, traction, contact_torque, resistance_torque, velocity, objective,
        fixed_reaction, fixed_nodal_force, fixed_torque, fixed_work, fixed_material_work, fixed_plate_work, fixed_exchange_error)
    fixed_products = float(np.sum(abs(fixed_nodal_force*velocity))+np.sum(abs(fixed_torque*omega)))
    fixed_exchange_allowance = 64*np.finfo(float).eps*max(len(y), 1)*fixed_products
    _finite(fixed_products, fixed_exchange_allowance)
    if (abs(power_error) > allowance or abs(contact_work)+abs(gauge_work)+abs(fixed_work) > allowance
            or abs(fixed_exchange_error) > fixed_exchange_allowance
            or np.any(y[~constraints.free] != 0.)
            or state['global_error'] > tolerance
            or np.max(state['component'], initial=0.) > tolerance
            or np.max(state['row_error'], initial=0.) > tolerance):
        raise RuntimeError('Original joint force, constraint or physical power acceptance failed.')
    free = constraints.free
    residual_scale = (_norm(state['action'][free])+_norm(system['load_n'][free])+
        _norm(nonlinear['gradient_n'][free])+_norm(constraints.transpose(multiplier)[free]))
    relative = 0. if residual_scale == 0. else _norm(state['stationarity'])/residual_scale
    diagnostics = dict(iterations, **powers, joint_relative_residual=relative,
        stationarity_n=state['stationarity'], componentwise_stationarity_relative=state['component'],
        contact_slip_m_s=system['contact_matrix']@y, gauge_error_m_s=system['gauge_matrix']@y,
        original_row_relative_error=state['row_error'], original_row_absolute_product_scale_m_s=state['row_scale'],
        constraint_rows=constraints.count, constraint_rank=constraints.rank,
        active_constraint_rows=len(constraints.active), structurally_zero_constraint_rows=constraints.zero_rows,
        free_unknowns=int(constraints.free.sum()), fixed_residual_coordinates=int((~constraints.free).sum()),
        pre_fixed_stationarity_n=state['pre_fixed_stationarity'],
        redundant_rows=constraints.count-constraints.rank, constraint_gram_eigenvalues=constraints.eigenvalues,
        constraint_rank_tolerance=constraints.rank_tolerance,
        resistance_potential_w=nonlinear['potential_w'], resistance_dissipation_w=nonlinear['resisting_power_w'],
        resistance_diagnostics=nonlinear['diagnostics'], physical_power_residual_w=power_error,
        physical_power_acceptance_allowance_w=allowance, solver_work_error_envelope_w=solver_envelope,
        arithmetic_work_allowance_w=arithmetic, arithmetic_work_operation_count=operations,
        absolute_constituent_work_w=products, contact_virtual_power_w=contact_work, gauge_virtual_power_w=gauge_work,
        fixed_virtual_power_w=fixed_work, fixed_material_power_w=fixed_material_work,
        fixed_plate_power_w=fixed_plate_work, fixed_exchange_power_error_w=fixed_exchange_error,
        fixed_exchange_power_allowance_w=fixed_exchange_allowance,
        scope='Matrix-free frozen joint solve with bounded dense constraint-row Gram; no native integration, calibration or contact admission.')
    diagnostics['objective_w'] = objective
    return dict(generalized_velocity_m_s=y, plate_omega_rad_s=omega,
        residual_velocity_m_s=residual_velocity, total_velocity_m_s=velocity,
        contact_resultant_n=resultant, contact_traction_n_m=traction,
        contact_first_force_n=first_force, contact_second_force_n=-first_force,
        contact_generalized_reaction_n=contact_reaction, contact_plate_torque_n_m=contact_torque,
        gauge_generalized_reaction_n=gauge_reaction, gauge_multiplier_n=multiplier[:cut],
        fixed_generalized_reaction_n=fixed_reaction, fixed_nodal_force_n=fixed_nodal_force,
        fixed_plate_torque_n_m=fixed_torque,
        resistance_generalized_reaction_n=-nonlinear['gradient_n'], resistance_plate_torque_n_m=resistance_torque,
        diagnostics=diagnostics)
