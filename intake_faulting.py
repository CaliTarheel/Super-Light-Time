"""Explicit rate-independent fault release for the persistent-intake rule.

Feasible constrained motion advances without a fracture search. If rigid
motion is incompatible, viable yielded cuts are tried independently and ranked
by the remaining intake shortfall. Only the best improving cut is retained,
then motion is solved again. This idealization omits fracture surface energy
and gradual necking; it must be explicitly selected.

When rigid intake is geometrically incompatible, a nonnegative self-balanced
attachment stress certificate identifies a release direction. No infeasible
velocity is transported and no constraint is deleted to obtain a solution.
"""
from copy import deepcopy
import numpy as np

import balance_force_ledger as forces
import constrained_plate_motion as constrained
import force_rifting
import native_topology
import plate_balance as pb
import plate_limit_analysis as limit

MAX_FAULTS_PER_SOLVE = 16
YIELD_TOLERANCE = 1e-6


def incompatibility_stress(rows, minimum):
    """Farkas certificate: nonnegative lambda, C.T lambda=0, b.lambda>0."""
    linprog, _, _ = constrained._scipy()
    result = linprog(np.zeros(len(rows)), A_eq=np.vstack((rows.T, np.ones(len(rows)))),
        b_eq=np.r_[np.zeros(rows.shape[1]), 1.], bounds=(0., None), method='highs')
    if (not result.success or np.any(result.x < 0.)
            or np.linalg.norm(rows.T@result.x) > 1e-10
            or float(minimum@result.x) <= 1e-10):
        raise constrained.IncompatibleIntake('No qualified geometric incompatibility stress certificate was found.')
    return result.x


def intake_shortfall(intake):
    """Minimum length-weighted unmet intake, used only to rank trial splits.

    Slack is diagnostic: its velocities are never used to advance the world.
    Positive obligations are retained in the actual constrained force solve.
    """
    rows, minimum = intake['rows'], intake['minimum']
    if not len(rows):
        return 0.
    weight = np.asarray(intake.get('weights_km', np.ones(len(rows))), float)
    weight = weight/weight.sum()
    linprog, _, _ = constrained._scipy()
    result = linprog(np.r_[np.zeros(rows.shape[1]), weight],
        A_ub=np.column_stack((-rows, -np.eye(len(rows)))), b_ub=-minimum,
        bounds=[(None, None)]*rows.shape[1]+[(0., None)]*len(rows), method='highs')
    if not result.success:
        raise ValueError('Could not qualify the trial fracture intake shortfall: '+result.message)
    return float(weight@np.maximum(minimum-rows@result.x[:rows.shape[1]], 0.))


def retains_obligations(before, after, parent, daughter):
    """A zero-displacement split must retain all previous intake operators.

    Restrict both daughters to their old common motion. Every old row must
    still occur; losing a trench is not an acceptable way to improve intake.
    """
    reduced = np.zeros((len(after.intake_constraints['rows']), before.size))
    for index, plate in enumerate(after.plates):
        source = parent if plate == daughter else plate
        target = int(np.flatnonzero(before.plates == source)[0])
        reduced[:, 3*target:3*target+3] += after.intake_constraints['rows'][:, 3*index:3*index+3]
    old = before.intake_constraints
    new = after.intake_constraints
    for row, minimum in zip(old['rows'], old['minimum']):
        if not np.any((np.max(np.abs(reduced-row), axis=1) < 1e-10)
                      & (new['minimum'] >= minimum)):
            return False
    return True


def candidates(s, balance, ledger):
    """Strength onset only; no solved traction is used as a finite mode load."""
    found = []
    for p in balance.plates:
        if np.count_nonzero(s.plate == p) < 2*limit.MIN_PIECE_CELLS:
            continue
        setup = limit._setup(s, p, balance, None, None, ledger=ledger)
        cells = setup['cells']; edges = setup['args'][2]
        blocked = setup['craton'].copy()
        degree = np.bincount(edges.ravel(), minlength=len(cells))
        blocked |= (degree[edges[:, 0]] <= 1) | (degree[edges[:, 1]] <= 1)
        axes, _ = limit.bounded_candidate_axes(s, p, setup, max_axes=10)
        best = limit.search_prepared(setup['args'], blocked, limit._craton_glue(s, p, cells),
                                     coarse=False, axes=axes)
        if best['axis'] is not None and best['ratio'] > 1.+YIELD_TOLERANCE:
            found.append(dict(plate=p, setup=setup, **best))
    return sorted(found, key=lambda entry: (-entry['ratio'], int(s.plate_uid[entry['plate']])))


def commit(s, candidate, *, incompatible=False):
    """Recheck the actual whole-patch partition before any ownership change."""
    p, setup = candidate['plate'], candidate['setup']
    cells, axis = setup['cells'], candidate['axis']
    region = np.zeros(s.n, bool); region[cells[candidate['piece']]] = True
    parcel_side, region = force_rifting._material_sides(s, p, region)
    piece = region[cells]
    points, torque, edges, mid, length, strength = setup['args']
    a, b = edges.T; cut = piece[a] != piece[b]
    if (min(int(piece.sum()), int((~piece).sum())) < limit.MIN_PIECE_CELLS
            or not cut.any() or np.any(cut & setup['craton'])):
        return None
    normal = points[b]-points[a]
    normal -= mid*np.sum(normal*mid, axis=1)[:, None]
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-30)
    velocity = np.cross(axis, mid)
    sign = np.where(piece[a], 1., -1.)
    opening = -sign*np.sum(velocity*normal, axis=1)
    slip = np.abs(np.sum(velocity*np.cross(mid, normal), axis=1))
    tension, compression, shear = strength
    capacity = float(np.sum((length*pb.RADIUS_M*(tension*np.maximum(opening, 0.)
        +compression*np.maximum(-opening, 0.)+shear*slip))[cut]))
    power = float(np.sum(torque[piece]@axis))
    ratio = power/capacity if capacity > 0. else 0.
    if ratio <= 1.+YIELD_TOLERANCE:
        return None
    cut_length = float(length[cut].sum())
    continental = float(length[cut & setup['continental']].sum()/cut_length)
    center = native_topology._unit(np.sum(s.xyz[region]*s.cell_area[region, None], axis=0))
    loading = dict(cause='persistent_intake_yield', version=1, loading_ratio=ratio,
        cut_length_km=cut_length/1e3, cut_continental_fraction=continental,
        incompatible_rigid_intake=bool(incompatible), assigned_opening_km=0.,
        assigned_velocity_impulse=False,
        model='rate-independent strength onset followed by a new constrained motion solve',
        limitation='Zero-displacement fault release; fracture surface energy and gradual necking are not modeled.')
    traces = native_topology._trace_side(s, parcel_side, region)
    uid = int(s.next_plate_uid)
    if not native_topology._commit(s, p, region, parcel_side, traces, np.zeros((2, 3)),
            'continental' if continental > 0. else 'oceanic', loading, center):
        return None
    q = int(np.flatnonzero(s.active & (s.plate_uid == uid))[0])
    force_rifting._mark_rift(s, p, q, np.flatnonzero(setup['internal'])[cut], center, axis)
    return dict(time_myr=float(s.t), parent_plate_uid=int(s.plate_uid[p]), daughter_plate_uid=uid, **loading)


def rank_trials(s, dt, balance, options):
    """Try cuts on isolated copies; choose one that improves retained intake."""
    original = intake_shortfall(balance.intake_constraints)
    best = None
    for candidate in options:
        trial = deepcopy(s)
        event = commit(trial, candidate, incompatible=True)
        if event is None:
            continue
        trial._rasterize()
        trial._boundaries()
        child = int(np.flatnonzero(trial.active & (trial.plate_uid == event['daughter_plate_uid']))[0])
        motion = pb.Balance(trial, dt)
        if not retains_obligations(balance, motion, candidate['plate'], child):
            continue
        shortfall = intake_shortfall(motion.intake_constraints)
        if shortfall >= original-constrained.FEASIBILITY_TOLERANCE_CM_YR:
            continue
        power = np.inf
        try:
            motion.solve()
            power = float(motion.intake_reaction@(motion.intake_constraints['rows']@motion.x))
            shortfall = 0.
        except constrained.IncompatibleIntake:
            pass
        # Prefer complete resolution, then less remaining shortfall. Among
        # feasible cuts, prefer less supplied attachment power, then less cut.
        key = (shortfall, power, event['cut_length_km'], -event['loading_ratio'],
               event['parent_plate_uid'])
        event.update(intake_shortfall_before_cm_yr=original,
                     intake_shortfall_after_cm_yr=shortfall,
                     candidates_tested=len(options))
        if best is None or key < best[0]:
            best = key, trial, event
    return None if best is None else best[1:]


def solve(s, dt):
    """Transactional fault/motion solve; every accepted velocity is feasible."""
    before = deepcopy(vars(s))
    events = []
    try:
        for iteration in range(MAX_FAULTS_PER_SOLVE+1):
            balance = pb.Balance(s, dt)
            intake = balance.intake_constraints
            try:
                balance.solve()
            except constrained.IncompatibleIntake:
                certificate = incompatibility_stress(intake['rows'], intake['minimum'])
                # The intact equilibrium plus this self-balanced stress is
                # only an onset path. It is never returned as accepted motion.
                balance.intake_constraints = dict(rows=np.zeros((0, balance.size)))
                balance.solve()
                balance.intake_constraints = intake
                scale = max(float(np.linalg.norm(balance.torque)),
                            float(np.linalg.norm(balance.stiffness)), 1.)
                balance.intake_reaction = certificate*scale
            else:
                s.persistent_intake_faults = list(getattr(s, 'persistent_intake_faults', []))+events
                diagnostics = balance.diagnostics(balance.rotation())
                diagnostics['intake_fault_release'] = dict(version=1, events=deepcopy(events),
                    search_axes_per_plate=10, obligations_satisfied=True,
                    interpretation='Advance feasible constrained motion; fracture only to relieve '
                                   'incompatible intake, choosing one improving cut at a time.')
                return balance.rotation(), diagnostics
            if iteration == MAX_FAULTS_PER_SOLVE:
                raise ValueError('Persistent intake exceeded its explicit fault-release budget; step rolled back.')
            selected = None
            for attempt in range(8):
                options = candidates(s, balance, forces.export(balance, balance.x))
                selected = rank_trials(s, dt, balance, options)
                if selected is not None:
                    break
                # This certified self-stress is an onset path only, not an
                # accepted force or velocity. Each trial must improve intake.
                balance.intake_reaction *= 10.
            if selected is None:
                raise constrained.IncompatibleIntake('Positive ocean intake requires a split, but no yielded '
                    'cut improved intake while preserving its obligations and material guards; step rolled back.')
            trial, event = selected
            s.__dict__.clear(); s.__dict__.update(vars(trial))
            events.append(event)
        raise ValueError('Persistent intake fault solve exceeded its bounded work.')
    except BaseException:
        s.__dict__.clear(); s.__dict__.update(before)
        raise
