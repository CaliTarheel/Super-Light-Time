"""Read-only cut onset and reduced opening analysis under a plate's own forces.

The rigid force balance (``plate_balance``) sums slab pull and ridge push into
one Euler torque per plate. Opposing pulls can cancel there and leave the plate
stationary. Historical velocity-loaded rift paths could not expose that
internal tension to a breakup decision. This module asks the
missing question directly, in forces: *can the plate's own loads tear it along
some cut, and does that cut run through continent or only through ocean?*

Method (rigid-block plastic mechanism search). A mechanism is a piece
``A`` of the plate's control cells rotating rigidly relative to the rest with
unit angular velocity about an axis ``a``. The external loads do power

    P(A, a) = sum_{i in A} m_i . a

where ``m_i`` is the torque of loads applied to cell ``i`` at the current
intact equilibrium. Attached-slab worlds use the complete active motion
assembly and an owned-control spatial lift. The cut
dissipates

    D(A, a) = sum_{edges ij, i in A, j not in A} L_ij R (S_t <u.n>+ + S_c <u.n>- + S_s |u.t|)

with ``u = a x r`` the relative velocity per unit rotation. ``S_t``, ``S_c`` and
``S_s`` are depth-integrated tensile, compressive and strike-slip strengths in
N/m from yield-strength envelopes. The mechanism's loading ratio is
``rho = P / D``; ``rho > 1`` gives a descending onset direction in the declared
reduced model (equality is marginal). For fixed ``a``,
maximising ``P - D/rho`` over ``A`` is a minimum s-t cut, and the best ratio is
found by Dinkelbach iteration. Axes are scanned over the sphere.

Interpretation limits: onset does not give a finite speed for velocity-dependent
loads. The attached path reevaluates those operators in ``breakup_mode``;
the historical nonattached path retains its prior opening estimate. Spatial
lifting is a reduced approximation, not a resolved continuum stress field.
``rho < 1`` excludes only this searched family at the held state, not unresolved
curved cuts or distributed necking. Strength is a reference-strain-rate
envelope. Nothing here changes the simulation state. See SLAB-FORCE-LEDGER.md.
"""
from __future__ import annotations

from collections import deque
from functools import lru_cache
import math
import numpy as np

import plate_balance
import slab_memory

VERSION = 1
RADIUS_M = plate_balance.RADIUS_M
SECONDS_PER_MYR = plate_balance.SECONDS_PER_MYR
GAS_CONSTANT = 8.314
REFERENCE_STRAIN_RATE_S = 1e-15
# Ranalli (1995) brittle coefficients for the three Andersonian regimes with
# hydrostatic pore pressure (lambda = 0.36): sigma1 - sigma3 = beta rho g z (1 - lambda).
BYERLEE_BETA = dict(tension=.75, compression=3., shear=1.2)
PORE_FACTOR = .64
# Dislocation-creep laws, A in MPa^-n s^-1: wet quartzite (Ranalli 1995), Maryland
# diabase (Mackwell et al. 1998), dry olivine (Hirth & Kohlstedt 2003).
WET_QUARTZITE = (3.2e-4, 2.3, 154e3)
DIABASE = (8., 4.7, 485e3)
DRY_OLIVINE = (1.1e5, 3.5, 530e3)
# Goetze low-temperature plasticity caps olivine stress in cold lithosphere.
PEIERLS_STRESS_PA, PEIERLS_Q, PEIERLS_RATE = 8.5e9, 535e3, 5.7e11
MANTLE_POTENTIAL_C = 1350.
# Declared surface heat flow per continental class (mW/m^2): Phanerozoic
# continent, craton, juvenile arc. Kinds follow s.crust (1, 2, 3).
CONTINENTAL_HEAT_FLOW_MW_M2 = {1: 65., 2: 45., 3: 80.}
CONTINENTAL_CRUST_KM, UPPER_CRUST_KM = 35., 20.
DEFAULT_AXES = 40
# Boundaries opening faster than this carry no unresolved resistance.
OPENING_FLOOR_KM_MYR = .5


def _creep(temperature, law, rate):
    a, n, q = law
    return (rate/a)**(1./n)*np.exp(q/(n*GAS_CONSTANT*temperature))*1e6


def _peierls(temperature, rate):
    root = np.sqrt(np.maximum(GAS_CONSTANT*temperature/PEIERLS_Q*math.log(PEIERLS_RATE/rate), 0.))
    return np.maximum(PEIERLS_STRESS_PA*(1.-root), 0.)


def _integrate(depth, ductile, density, temperature, mode):
    brittle = BYERLEE_BETA[mode]*density*plate_balance.GRAVITY_M_S2*depth*PORE_FACTOR
    stress = np.minimum(brittle, ductile)
    stress[temperature >= MANTLE_POTENTIAL_C+273.15-1.] = 0.
    return float(np.sum(.5*(stress[1:]+stress[:-1])*np.diff(depth)))


_DEPTH_M = np.linspace(0., 200e3, 4001)


@lru_cache(maxsize=None)
def oceanic_strength_n_per_m(age_myr, mode='tension', rate=REFERENCE_STRAIN_RATE_S):
    """Integrated strength of half-space-cooled oceanic lithosphere, N/m."""
    age = max(float(age_myr), .1)*SECONDS_PER_MYR
    z = _DEPTH_M
    temperature = 273.15+MANTLE_POTENTIAL_C*np.vectorize(math.erf)(z/(2.*math.sqrt(1e-6*age)))
    temperature = np.maximum(temperature, 273.15)
    crust = z < 7e3
    ductile = np.where(crust, _creep(temperature, DIABASE, rate),
                       np.minimum(_creep(temperature, DRY_OLIVINE, rate), _peierls(temperature, rate)))
    return _integrate(z, ductile, np.where(crust, 2900., 3300.), temperature, mode)


@lru_cache(maxsize=None)
def continental_strength_n_per_m(heat_flow_mw_m2, mode='tension', rate=REFERENCE_STRAIN_RATE_S):
    """Integrated strength of a 35 km crust over dry olivine, steady geotherm, N/m.

    Radiogenic heat 1 uW/m^3 decays over 10 km; conductivity 2.5 W/m/K.
    """
    z = _DEPTH_M
    production, scale, conductivity = 1e-6, 10e3, 2.5
    q0 = float(heat_flow_mw_m2)*1e-3
    temperature = 273.15+((q0-production*scale)*z/conductivity
                          + production*scale**2/conductivity*(1.-np.exp(-z/scale)))
    temperature = np.minimum(temperature, MANTLE_POTENTIAL_C+273.15)
    upper, moho = z < UPPER_CRUST_KM*1e3, z < CONTINENTAL_CRUST_KM*1e3
    ductile = np.where(upper, _creep(temperature, WET_QUARTZITE, rate),
                       np.where(moho, _creep(temperature, DIABASE, rate),
                                np.minimum(_creep(temperature, DRY_OLIVINE, rate), _peierls(temperature, rate))))
    return _integrate(z, ductile, np.where(moho, 2800., 3300.), temperature, mode)


_AGE_TABLE = np.array([.5, 1, 2, 5, 10, 15, 20, 30, 40, 60, 80, 100, 130, 160, 200, 250, 300], float)


def heat_flows(heat_flow=None):
    """Declared heat flow per continental kind, with integer-keyed overrides."""
    flows = dict(CONTINENTAL_HEAT_FLOW_MW_M2)
    flows.update({int(kind): float(flow) for kind, flow in (heat_flow or {}).items()})
    if set(flows) != set(CONTINENTAL_HEAT_FLOW_MW_M2):
        raise ValueError('Continental heat flow applies only to crust kinds 1, 2 and 3.')
    return flows


def cell_strengths(crust, age, heat_flow=None):
    """Per-cell (tension, compression, shear) integrated strengths in N/m."""
    crust, age = np.asarray(crust, int), np.asarray(age, float)
    flows = heat_flows(heat_flow)
    result = []
    for mode in ('tension', 'compression', 'shear'):
        table = np.array([oceanic_strength_n_per_m(a, mode) for a in _AGE_TABLE])
        value = np.interp(np.clip(age, _AGE_TABLE[0], _AGE_TABLE[-1]), _AGE_TABLE, table)
        for kind, flow in flows.items():
            value = np.where(crust == kind, continental_strength_n_per_m(flow, mode), value)
        result.append(value)
    return tuple(result)


# Inherited weakness uses the native rift laws' dimensionless resistance
# 1/(1 + 1.2 suture) (progressive_rifting.relative_strength), normalised to
# intact crust of the same kind. A parcel on an inherited rift (rift_id > 0)
# counts as a weak belt, at the suture value those laws give rift and weak
# belts. It is a transfer of the project's existing weakness convention, not
# a new calibration.
INHERITED_SUTURE = .8
_BASELINE_SUTURE = {1: .2, 2: .02}


def inherited_weakness(s):
    """Per-cell strength factor (<= 1) from sutures and inherited rifts."""
    factor = np.ones(len(s.xyz))
    kind = np.asarray(s.kind, int)
    continental = np.isin(kind, tuple(_BASELINE_SUTURE))
    if not np.any(continental):
        return factor
    suture = np.asarray(s.suture, float)[continental]
    suture = np.where(np.asarray(s.rift_id)[continental] > 0, np.maximum(suture, INHERITED_SUTURE), suture)
    base = np.where(kind[continental] == 2, _BASELINE_SUTURE[2], _BASELINE_SUTURE[1])
    weak = np.minimum((1.+1.2*base)/(1.+1.2*suture), 1.)
    cell, mass = np.asarray(s.parcel_cell)[continental], np.asarray(s.mass, float)[continental]
    total = np.bincount(cell, weights=mass, minlength=len(factor))
    weighted = np.bincount(cell, weights=mass*weak, minlength=len(factor))
    held = (total > 0.) & (np.asarray(s.crust, int) > 0)
    factor[held] = weighted[held]/total[held]
    return factor


# ------------------------------------------------------------------ max flow
class _Flow:
    """Dinic maximum flow on float capacities; small, dependency-free."""

    def __init__(self, count):
        self.count = count
        self.head = [[] for _ in range(count)]
        self.to, self.cap = [], []

    def add(self, a, b, capacity, reverse=0.):
        self.head[a].append(len(self.to)); self.to.append(b); self.cap.append(float(capacity))
        self.head[b].append(len(self.to)); self.to.append(a); self.cap.append(float(reverse))

    def run(self, source, sink, tolerance):
        import flow_acceleration
        result = flow_acceleration.run(self.count, self.head, self.to, self.cap,
                                       source, sink, tolerance)
        if result is not None:
            total, reachable, capacities, _ = result
            self.cap[:] = capacities
            return total, reachable
        return self._run_python(source, sink, tolerance)

    def _run_python(self, source, sink, tolerance):
        to, cap, head = self.to, self.cap, self.head
        total = 0.
        while True:
            level = [-1]*self.count
            level[source] = 0
            queue = deque([source])
            while queue:
                node = queue.popleft()
                for e in head[node]:
                    if cap[e] > tolerance and level[to[e]] < 0:
                        level[to[e]] = level[node]+1
                        queue.append(to[e])
            if level[sink] < 0:
                break
            pointer = [0]*self.count
            while True:
                # Iterative blocking-flow DFS.
                path, node = [], source
                while node != sink:
                    edges = head[node]
                    while pointer[node] < len(edges):
                        e = edges[pointer[node]]
                        if cap[e] > tolerance and level[to[e]] == level[node]+1:
                            break
                        pointer[node] += 1
                    else:
                        if not path:
                            node = None
                            break
                        level[node] = -1
                        node = to[path.pop() ^ 1]
                        pointer[node] += 1
                        continue
                    path.append(e)
                    node = to[e]
                if node is None:
                    break
                push = min(cap[e] for e in path)
                for e in path:
                    cap[e] -= push
                    cap[e ^ 1] += push
                total += push
        reach = np.zeros(self.count, bool)
        reach[source] = True
        queue = deque([source])
        while queue:
            node = queue.popleft()
            for e in head[node]:
                if cap[e] > tolerance and not reach[to[e]]:
                    reach[to[e]] = True
                    queue.append(to[e])
        return total, reach


def _flow_arrays(weight, a, b, forward, backward):
    """The graph _best_piece adds to _Flow, as (offset, adjacency, to, cap) arrays.

    Same edge ids (terminal edges in node order, then internal edges, each
    followed by its reverse), same capacities, and the same per-node
    insertion order, which is ascending edge id: a stable sort of the tails.
    """
    n = len(weight)
    source, sink = n, n+1
    weight = np.asarray(weight, float)
    terminal = np.flatnonzero((weight > 0.) | (weight < 0.))
    positive = weight[terminal] > 0.
    t, m = len(terminal), len(terminal)+len(a)
    tail, to, cap = np.empty(2*m, np.int64), np.empty(2*m, np.int64), np.empty(2*m, float)
    tail[0:2*t:2] = to[1:2*t:2] = np.where(positive, source, terminal)
    to[0:2*t:2] = tail[1:2*t:2] = np.where(positive, terminal, sink)
    cap[0:2*t:2] = np.where(positive, weight[terminal], -weight[terminal])
    cap[1:2*t:2] = 0.
    tail[2*t::2] = to[2*t+1::2] = np.asarray(a, np.int64)
    to[2*t::2] = tail[2*t+1::2] = np.asarray(b, np.int64)
    cap[2*t::2], cap[2*t+1::2] = forward, backward
    offset = np.zeros(n+3, np.int64)
    np.cumsum(np.bincount(tail, minlength=n+2), out=offset[1:])
    return offset, np.argsort(tail, kind='stable'), to, cap


def _best_piece(weight, a, b, forward, backward):
    """argmax_A sum_{A} weight - sum cut costs (forward: a in A, b out)."""
    import flow_acceleration
    n = len(weight)
    source, sink = n, n+1
    # Tolerance from the loads alone: uncuttable (forbidden) edges carry
    # enormous capacities that must not inflate it past real edge costs.
    scale = float(np.abs(weight).sum()) or 1.
    if flow_acceleration.active():
        # The native loop needs only arrays; building them directly skips
        # the per-edge Python graph and its list conversions.
        cut, reach, _, _ = flow_acceleration.run_arrays(
            n+2, *_flow_arrays(weight, a, b, forward, backward), source, sink, 1e-13*scale)
        return float(weight[weight > 0.].sum()-cut), reach[:n]
    flow = _Flow(n+2)
    for i, w in enumerate(weight):
        if w > 0.:
            flow.add(source, i, w)
        elif w < 0.:
            flow.add(i, sink, -w)
    for i, j, f, r in zip(a, b, forward, backward):
        flow.add(int(i), int(j), f, r)
    cut, reach = flow.run(source, sink, 1e-13*scale)
    piece = reach[:n]
    return float(weight[weight > 0.].sum()-cut), piece


def fibonacci_axes(count):
    index = np.arange(count)+.5
    z = 1.-2.*index/count
    phi = math.pi*(1.+5**.5)*index
    rho = np.sqrt(1.-z*z)
    return np.column_stack((rho*np.cos(phi), rho*np.sin(phi), z))


def analyse(points, torques, edges, edge_mid, edge_length_m, strength, forbidden=None,
            axes=None, iterations=12, refine=2, seed_axes=(), coarse=True, refine_start_deg=12.):
    """Worst rigid-block mechanism of one plate.

    points (n,3) unit cell centres; torques (n,3) N m of all loads per cell
    (their sum should vanish for a plate in equilibrium); edges (E,2) internal
    cell pairs; edge_mid (E,3) unit midpoints; edge_length_m (E,);
    strength = (tension, compression, shear) per edge in N/m; forbidden (E,)
    marks edges no mechanism may cut (e.g. continental, for an ocean-only scan).
    Returns the best ratio rho = P/D, its axis, piece and cut edges.
    """
    points, torques = np.asarray(points, float), np.asarray(torques, float)
    edges = np.asarray(edges, int)
    a, b = edges.T
    mid = np.asarray(edge_mid, float)
    normal = points[b]-points[a]
    normal -= mid*np.sum(normal*mid, axis=1)[:, None]
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-30)
    tangent = np.cross(mid, normal)
    tension, compression, shear = (np.asarray(v, float) for v in strength)
    lever = np.asarray(edge_length_m, float)*RADIUS_M
    blocked = np.zeros(len(a), bool) if forbidden is None else np.asarray(forbidden, bool)
    infinite = 1e6*float((np.maximum(tension, compression).max(initial=1.)+shear.max(initial=1.))*lever.sum())
    def evaluate(axis, floor=0.):
        """Best ratio for one axis, or (0, None) if it cannot exceed floor."""
        axis = axis/np.linalg.norm(axis)
        weight = torques@axis
        if not np.any(weight > 0.):
            return 0., None
        velocity = np.cross(axis, mid)
        # A = side a, normal points a->b (out of A): A moving along the normal
        # closes the cut, so opening is minus the normal velocity.
        opening = -np.sum(velocity*normal, axis=1)
        slip = np.abs(np.sum(velocity*tangent, axis=1))
        # a in A, b outside: A moves by +velocity relative to b.
        forward = lever*(tension*np.maximum(opening, 0.)+compression*np.maximum(-opening, 0.)+shear*slip)
        backward = lever*(tension*np.maximum(-opening, 0.)+compression*np.maximum(opening, 0.)+shear*slip)
        forward[blocked] = backward[blocked] = infinite

        def scaled(values, ratio):
            # A forbidden edge stays uncuttable at every Dinkelbach ratio;
            # scaling its cost by a small trial ratio would make it cheap.
            return np.where(blocked, infinite, values*ratio)
        # Dinkelbach for max P/D: start from the cost-free piece (every cell
        # whose load does positive work), then solve max_A P - ratio*D until
        # no piece beats the current ratio.
        def measure(candidate):
            if not candidate.any() or candidate.all():
                return None
            dissipation = float(forward[candidate[a] & ~candidate[b]].sum()
                                + backward[candidate[b] & ~candidate[a]].sum())
            power = float(weight[candidate].sum())
            return power/dissipation if dissipation > 0. and power > 0. else None
        piece = weight > 0.
        ratio = measure(piece)
        if ratio is None:
            return 0., None
        if floor > ratio:
            # One cut decides whether this axis can beat the best found so far.
            value, candidate = _best_piece(weight, a, b, scaled(forward, floor), scaled(backward, floor))
            updated = measure(candidate) if value > 1e-12*float(np.abs(weight).sum()) else None
            if updated is None or updated <= floor:
                return 0., None
            piece, ratio = candidate, updated
        for _ in range(iterations):
            value, candidate = _best_piece(weight, a, b, scaled(forward, ratio), scaled(backward, ratio))
            updated = measure(candidate) if value > 1e-12*float(np.abs(weight).sum()) else None
            if updated is None or updated <= ratio*(1.+1e-9):
                break
            piece, ratio = candidate, updated
        return float(ratio), piece

    scored, floor = [], 0.
    candidates = list(seed_axes)+(list(np.asarray(axes, float)) if axes is not None
                                  else list(fibonacci_axes(DEFAULT_AXES)) if coarse else [])
    for axis in candidates:
        # Keep a margin so a few near-best axes survive for refinement.
        ratio, piece = evaluate(np.asarray(axis, float), .8*floor)
        floor = max(floor, ratio)
        scored.append((ratio, tuple(np.asarray(axis, float)/np.linalg.norm(axis)), piece))
    scored.sort(key=lambda row: -row[0])
    best = dict(ratio=0., axis=None, piece=np.zeros(len(points), bool), cut=np.zeros(len(a), bool))
    # Pattern search around the leading coarse axes (skipped for explicit axes).
    for ratio, axis, piece in scored[:(refine if axes is None else len(scored))]:
        axis = np.asarray(axis)
        step = math.radians(refine_start_deg) if axes is None else 0.
        while step > math.radians(1.5):
            frame = np.linalg.svd(axis[None, :])[2][1:]
            improved = False
            for direction in (frame[0], -frame[0], frame[1], -frame[1]):
                trial = axis*math.cos(step)+direction*math.sin(step)
                trial_ratio, trial_piece = evaluate(trial, ratio)
                if trial_ratio > ratio*(1.+1e-6):
                    ratio, axis, piece, improved = trial_ratio, trial/np.linalg.norm(trial), trial_piece, True
            if not improved:
                step *= .5
        if piece is not None and ratio > best['ratio']:
            best = dict(ratio=float(ratio), axis=axis.copy(), piece=piece.copy(), cut=piece[a] != piece[b])
    return best


def boundary_resistance(s, balance, omega):
    """Each boundary edge's resisting generalized force at the solved motion.

    Evaluates the balance's own edge-local resistance terms: slab Stokes and
    anchor drag, hinge bending, and the plastic strike-slip, megathrust and
    compression elements. Returns (edges, forces), where forces[k] is the
    gradient of that edge's dissipation in the balance's coordinates; the
    resisting torque on a plate is minus its block, times R/CM_YR_M_S.
    Collision-interface and weld terms have no single boundary edge and are
    left to the caller's residual.
    """
    x = np.zeros(balance.size)
    factor = plate_balance.CM_YR_M_S/RADIUS_M*SECONDS_PER_MYR
    for plate, index in balance.slot.items():
        x[3*index:3*index+3] = np.asarray(omega, float)[plate]/factor
    edges, forces = [], []
    trench = np.asarray(balance.trench_edges, int)
    if len(trench) and not getattr(balance, 'slab_tethers', False):
        for rows, coefficient in ((balance.slab_horizontal_rows, balance.slab_stokes_coefficient),
                                  (balance.slab_vertical_rows, balance.slab_stokes_coefficient),
                                  (balance.slab_anchor_rows, balance.slab_anchor_coefficient)):
            if len(rows):
                edges.append(trench)
                forces.append((coefficient*(rows@x))[:, None]*rows)
    if len(balance.hinge_coefficient):
        closing = np.minimum(balance.hinge@x, 0.)
        edges.append(trench)
        forces.append((balance.hinge_coefficient*closing)[:, None]*balance.hinge)
    delta = plate_balance.HUBER_CONTINUATION_KM_MYR[-1]*plate_balance.KM_MYR_CM_YR
    version = getattr(balance, 'resistance_version', 0)
    for element in balance.elements:
        if element.get('edges') is None or element['shape'] == 'arc_opening':
            continue
        width = delta*element['scale']
        coefficient = element['coefficient']
        if element['shape'] == 'cone':
            normal, tangent = element['rows']
            opening, tangential = normal@x, tangent@x
            closed, first, _ = plate_balance._cone_closing_terms(opening, width, version)
            magnitude = np.sqrt(closed*closed+tangential*tangential+width*width)
            direction = (closed[:, None]*(first[:, None]*normal)+tangential[:, None]*tangent)/magnitude[:, None]
            forces.append(coefficient[:, None]*direction)
        else:
            (operator,) = element['rows']
            rate = operator@x
            if element['shape'] == 'abs':
                _, first, _ = plate_balance._abs_terms(rate, width)
            else:
                law = plate_balance._passive_negative_terms if version == 1 else plate_balance._negative_terms
                _, first, _ = law(rate, width)
            forces.append((coefficient*first)[:, None]*operator)
        edges.append(np.asarray(element['edges'], int))
    if not edges:
        return np.zeros(0, int), np.zeros((0, balance.size))
    return np.concatenate(edges), np.vstack(forces)


# ------------------------------------------------------------ simulation use
def active_balance(s, dt):
    """Select the same attached-slab law as the native evolution policy.

    The mere presence of a trench or slab inventory does not select coupled
    neck evolution. Its saved shutdown policy is the native selector.
    """
    import trench_history
    import effective_subduction
    return plate_balance.Balance(s, dt, slab_tethers=(not effective_subduction.enabled(s)
                                and trench_history._rupture_governed(s)))


def plate_loads(s, plate, balance=None, omega=None, band_km=None, *, ledger=None):
    """Current per-cell torques, preserving the active law's rigid virtual work.

    Attached worlds use the explicit force ledger. The historical nonattached
    diagnostic remains available with its original interpretation; it is not
    substituted for a coupled neck law.
    """
    balance = balance or active_balance(s, s.config.get('dt_myr', 1.))
    if not getattr(balance, 'uses_force_ledger', getattr(balance, 'slab_tethers', False)):
        return _legacy_plate_loads(s, plate, balance, omega, band_km)
    import balance_force_ledger
    omega = np.asarray(s.omega if omega is None else omega, float)
    factor = plate_balance.CM_YR_M_S/RADIUS_M*SECONDS_PER_MYR
    x = np.concatenate([omega[p]/factor for p in balance.plates])
    report = ledger if ledger is not None else balance_force_ledger.export(balance, x, band_km=band_km)
    local = report['plates'][int(plate)]
    cells = np.asarray(local['cells'], int)
    torque = np.zeros((len(s.xyz), 3))
    torque[cells] = local['torques_n_m']
    components = {key: np.asarray(value).sum(axis=0).tolist()
                  for key, value in local['components_n_m'].items()}
    net = torque[cells].sum(axis=0)
    gross = float(local['diagnostics']['gross_component_torque_n_m'])
    return cells, torque, dict(components=components,
        force_accounting=('active effective-subduction work-conjugate ledger'
            if getattr(balance, 'effective_subduction', False) else 'active attached-slab work-conjugate ledger'),
        unresolved_boundary_torque_n_m=net.tolist(),
        unresolved_fraction_of_gross=float(np.linalg.norm(net)/max(gross, 1e-30)),
        artificial_reaction_added=False, ledger_diagnostics=report['diagnostics'],
        resistance_weight=np.zeros(len(cells)))


def _legacy_plate_loads(s, plate, balance=None, omega=None, band_km=None):
    """Per-cell load torques (N m) on one plate, and their bookkeeping."""
    balance = balance or plate_balance.Balance(s, s.config.get('dt_myr', 1.))
    band_km = float(s.config.get('deformation_width_km', 400.) if band_km is None else band_km)
    omega = np.asarray(s.omega if omega is None else omega, float)
    points = np.asarray(s.xyz, float)
    cells = np.flatnonzero(np.asarray(s.plate) == plate)
    torque = np.zeros((len(points), 3))
    components = {}
    dip = math.radians(slab_memory.SUBDUCTION_DIP_DEG)
    trench = balance.trench_edges
    slab = np.zeros((len(points), 3))
    if len(trench):
        owners = balance.slab_owner[trench]
        force = plate_balance.GRAVITY_M_S2*balance.slab_load[trench]*math.sin(dip)
        on_p = owners == balance.bp[trench]
        direction = np.where(on_p[:, None], balance.normal[trench], -balance.normal[trench])
        line = direction*(balance.length_m[trench]*force)[:, None]
        down = np.where(on_p, balance.bp[trench], balance.bq[trench])
        over = np.where(on_p, balance.bq[trench], balance.bp[trench])
        reaction = plate_balance.subduction_response_version(s) == 1
        area = np.asarray(s.cell_area, float)
        owner = np.asarray(s.plate)
        cos_band = math.cos(band_km/plate_balance.RADIUS_KM)
        # Slab pull enters the plate across its flexural (outer-rise) width,
        # not at one boundary cell: spread each edge's force over its plate's
        # cells within the band. Without this a staircase tooth receives the
        # pull of two trench edges and is held by one internal edge.
        for k, e in enumerate(trench):
            for target, sign in ((down[k], 1.), (over[k], -1.)) if reaction else ((down[k], 1.),):
                near = np.flatnonzero((owner == target) & (points@balance.radial[e] >= cos_band))
                if not len(near):
                    near = np.array([s.ba[e] if owner[s.ba[e]] == target else s.bb[e]])
                share = area[near]/area[near].sum()
                slab[near] += sign*RADIUS_M*np.cross(points[near], share[:, None]*line[k])
    components['slab'] = slab
    ridge = plate_balance.ridge_force_density(s)
    area_m2 = np.asarray(s.cell_area, float)*1e6
    components['ridge'] = (np.zeros_like(torque) if ridge is None
                           else RADIUS_M*np.cross(points, ridge)*area_m2[:, None])
    keel = np.asarray(plate_balance.KEEL, float)[np.clip(np.asarray(s.crust, int), 0, 3)]
    velocity = np.cross(omega[plate], points)*RADIUS_M/SECONDS_PER_MYR
    drag = -plate_balance.ASTHENOSPHERE_DRAG_PA_S_M*keel[:, None]*velocity*area_m2[:, None]
    components['basal'] = RADIUS_M*np.cross(points, drag)
    # Boundary resistances act where the balance computes them: each edge's
    # slab drag, bending and interface terms are applied at that edge, spread
    # over the same flexural band as slab pull. A young rift or ridge carries
    # only what its own terms give it, so it is not pinned against the pull.
    resisting = np.zeros_like(torque)
    edges, forces = boundary_resistance(s, balance, omega)
    owner = np.asarray(s.plate)
    mine = owner == plate
    cos_band = math.cos(band_km/plate_balance.RADIUS_KM)
    index = balance.slot.get(plate)
    if index is not None and len(edges):
        block = -forces[:, 3*index:3*index+3]*RADIUS_M/plate_balance.CM_YR_M_S
        for e, moment in zip(edges, block):
            if not np.any(moment):
                continue
            near = np.flatnonzero(mine & (points@balance.radial[e] >= cos_band))
            if not len(near):
                side = s.ba[e] if owner[s.ba[e]] == plate else s.bb[e]
                if owner[side] != plate:
                    continue
                near = np.array([side])
            # A torque M is carried by forces f_i = K^-1 M x r_i over these
            # cells, where K = sum w_i (I - r_i r_i^T) is their moment.
            w = area_m2[near]/area_m2[near].sum()
            local = points[near]
            k = np.eye(3)-np.einsum('c,ci,cj->ij', w, local, local)
            carrier = np.linalg.solve(k+1e-12*np.eye(3), moment)
            resisting[near] += np.cross(local, np.cross(carrier, local)*w[:, None])
    components['boundary'] = resisting
    for value in components.values():
        torque += value
    net = torque[cells].sum(axis=0)
    gross = float(np.linalg.norm(components['slab'][cells].sum(axis=0))
                  + sum(np.linalg.norm(v[cells], axis=1).sum() for v in components.values()))
    # What remains (collision interfaces, welds, basal-quadrature differences)
    # closes the plate's equilibrium on its non-opening boundaries.
    residual = -net
    edges = np.flatnonzero(((np.asarray(s.bp) == plate) | (np.asarray(s.bq) == plate))
                           & (np.asarray(s.bl) > 1e-10)
                           & (np.asarray(s.normal_speed) <= OPENING_FLOOR_KM_MYR))
    weight = np.zeros(len(points))
    for e in edges:
        near = np.flatnonzero(mine & (points@balance.radial[e] >= cos_band))
        if not len(near):
            near = np.array([s.ba[e] if owner[s.ba[e]] == plate else s.bb[e]])
        weight[near] += float(s.bl[e])*area_m2[near]/area_m2[near].sum()
    if weight[cells].sum() <= 0.:
        weight[cells] = area_m2[cells]*keel[cells]
    w = weight[cells]/weight[cells].sum()
    local = points[cells]
    moment = np.eye(3)*w.sum()-np.einsum('c,ci,cj->ij', w, local, local)
    carrier = np.linalg.solve(moment, residual)
    force = np.cross(carrier, local)*w[:, None]
    torque[cells] += np.cross(local, force)
    return cells, torque, dict(components={k: v[cells].sum(axis=0).tolist() for k, v in components.items()},
                               unresolved_boundary_torque_n_m=residual.tolist(),
                               unresolved_fraction_of_gross=float(np.linalg.norm(residual)/max(gross, 1e-30)),
                               resistance_weight=w)


def _opening_rate(s, cells, best, length, mid, torque, edges):
    """Opening speed if the excess load drives the mechanism against basal drag.

    The two pieces rotate apart about the mechanism axis; each is resisted by
    asthenospheric traction C*keel*v over its area. With reduced drag moment
    I = I_A I_B / (I_A + I_B), the relative rate is (P - D)/I. This is the
    rigid-plastic, viscously resisted estimate, not a solved rift evolution.
    """
    axis, piece = best['axis'], best['piece']
    points = np.asarray(s.xyz, float)[cells]
    area = np.asarray(s.cell_area, float)[cells]*1e6
    keel = np.asarray(plate_balance.KEEL, float)[np.clip(np.asarray(s.crust, int)[cells], 0, 3)]
    arm = np.linalg.norm(np.cross(axis, points), axis=1)**2*RADIUS_M**2
    moment = plate_balance.ASTHENOSPHERE_DRAG_PA_S_M*keel*area*arm
    first, second = float(moment[piece].sum()), float(moment[~piece].sum())
    power = float((torque[piece]@axis).sum())
    excess = power*(1.-1./best['ratio'])
    if excess <= 0. or first <= 0. or second <= 0.:
        return dict(predicted_opening_cm_yr_mean=0., predicted_opening_cm_yr_max=0., relative_rotation_rad_myr=0.,
                    piece_rotation_rad_myr=[0., 0., 0.], rest_rotation_rad_myr=[0., 0., 0.])
    rate = excess/(first*second/(first+second))                      # rad/s
    cut = best['cut']
    speed = np.linalg.norm(np.cross(axis, mid[cut]), axis=1)*RADIUS_M*rate/plate_balance.CM_YR_M_S
    per_myr = rate*SECONDS_PER_MYR
    # Drag-weighted split of the relative rotation: zero net drag torque change.
    return dict(predicted_opening_cm_yr_mean=float(np.average(speed, weights=length[cut])),
                predicted_opening_cm_yr_max=float(speed.max(initial=0.)),
                relative_rotation_rad_myr=float(per_myr),
                piece_rotation_rad_myr=(axis*per_myr*second/(first+second)).tolist(),
                rest_rotation_rad_myr=(-axis*per_myr*first/(first+second)).tolist())


def _setup(s, plate, balance, heat_flow, band_km, inherited=False, ledger=None):
    balance = balance or active_balance(s, s.config.get('dt_myr', 1.))
    if balance.uses_force_ledger and ledger is None:
        import balance_force_ledger
        balance.solve()
        ledger = balance_force_ledger.export(balance, balance.x, band_km=band_km)
    cells, torque, loads = plate_loads(s, plate, balance, band_km=band_km, ledger=ledger)
    local = -np.ones(len(s.xyz), int)
    local[cells] = np.arange(len(cells))
    ea, eb = np.asarray(s.edge_a), np.asarray(s.edge_b)
    internal = (local[ea] >= 0) & (local[eb] >= 0)
    ea, eb = ea[internal], eb[internal]
    crust = np.asarray(s.crust, int)
    tension, compression, shear = cell_strengths(crust, s.age, heat_flow)
    if inherited:
        weak = inherited_weakness(s)
        tension, compression, shear = (v*weak for v in (tension, compression, shear))
    strength = tuple(.5*(v[ea]+v[eb]) for v in (tension, compression, shear))
    length = np.asarray(s.edge_length, float)[internal]*1e3
    mid = np.asarray(s.edge_mid, float)[internal]
    return dict(cells=cells, torque=torque, loads=loads, balance=balance, ledger=ledger,
                internal=internal, length=length, mid=mid,
                continental=(crust[ea] > 0) | (crust[eb] > 0),
                # Cutting between two craton cells splits a craton; a cut along
                # a craton margin (one craton cell) is allowed, as rifts often are.
                craton=(crust[ea] == 2) & (crust[eb] == 2),
                args=(s.xyz[cells], torque[cells], np.column_stack((local[ea], local[eb])), mid, length, strength))


MIN_PIECE_CELLS = 10
MAX_GLUE_ROUNDS = 12


def _craton_glue(s, plate, cells):
    """Chain every plate cell holding each craton group into one unit.

    A craton group must stay on one side of any committed split. Its patches
    are assigned by centroid, so both the centroid cells and the cells of its
    parcels are chained with uncuttable links (local cell indices).
    """
    from material_geometry import patch_centres
    local = -np.ones(len(s.xyz), int)
    local[cells] = np.arange(len(cells))
    selected = (np.asarray(s.parcel_plate) == plate) & (np.asarray(s.parcel_craton) >= 0)
    links = []
    for group in np.unique(np.asarray(s.parcel_craton)[selected]):
        mine = selected & (np.asarray(s.parcel_craton) == group)
        _, centres, _ = patch_centres(s.pos[mine], s.mass[mine], s.parcel_patch[mine])
        held = np.unique(np.r_[local[s._indices(centres)], local[np.asarray(s.parcel_cell)[mine]]])
        held = held[held >= 0]
        links.extend([held[0], other] for other in held[1:])
    return links


def bounded_candidate_axes(s, plate, setup, *, seed_axes=(), max_axes=10, held_only=False):
    """Deterministic finite directions, not a certificate over all mechanisms.

    Spatial loads retain opposed forces even when their sum is zero. Search
    both signs because compression and tension cost different amounts. No
    random fracture, velocity kick, or lowered cut strength is introduced.
    """
    if isinstance(max_axes, bool) or int(max_axes) != max_axes or not 6 <= max_axes <= 16:
        raise ValueError('Bounded force-rift search requires 6 to 16 axes.')
    axes, kinds = [], []

    def add(value, kind):
        value = np.asarray(value, float)
        if value.shape != (3,) or not np.isfinite(value).all() or np.linalg.norm(value) <= 1e-30:
            raise ValueError('Force-rift candidate axis must be finite and nonzero.')
        value = value/np.linalg.norm(value)
        for axis in (value, -value):
            if len(axes) >= int(max_axes):
                break
            if not any(np.dot(axis, old) > 1.-1e-12 for old in axes):
                axes.append(axis.copy()); kinds.append(kind)

    for axis in seed_axes:
        add(axis, 'previous supported mode')
    if held_only and len(axes):
        return np.asarray(axes, float).reshape(-1, 3), kinds
    torque = np.asarray(setup['args'][1], float)
    if torque.ndim != 2 or torque.shape[1] != 3 or not np.isfinite(torque).all():
        raise ValueError('Bounded search requires finite spatial loads.')
    scale = float(np.max(np.abs(torque), initial=0.))
    if scale > 0.:
        values, vectors = np.linalg.eigh((torque/scale).T@(torque/scale))
        for i in np.argsort(-values, kind='stable'):
            if values[i] > max(float(values.max()), 1.)*1e-14:
                add(vectors[:, i], 'principal spatial load')
    # Stored fault strike is a tangent axis: rotation about it opens across
    # the fault. Average its axial second moment, not cancellable signs.
    if all(hasattr(s, name) for name in ('parcel_plate', 'rift_id', 'rift_tangent', 'mass')):
        selected = ((np.asarray(s.parcel_plate) == plate) & (np.asarray(s.rift_id) > 0)
                    & (np.asarray(s.mass) > 0.))
        direction = np.asarray(s.rift_tangent, float)[selected]
        weight = np.asarray(s.mass, float)[selected]
        valid = (np.isfinite(direction).all(axis=1) & np.isfinite(weight)
                 & (np.linalg.norm(direction, axis=1) > 1e-20))
        if np.any(valid):
            direction, weight = direction[valid], weight[valid]
            direction = direction/np.linalg.norm(direction, axis=1)[:, None]
            weight = weight/weight.max()
            values, vectors = np.linalg.eigh(np.einsum('i,ij,ik->jk', weight, direction, direction))
            if values[-1] > 0.:
                add(vectors[:, -1], 'inherited material rift')
    return np.asarray(axes, float).reshape(-1, 3), kinds


def worst_mechanism(s, plate, balance=None, *, seed_axes=(), coarse=True, forbid_cratons=False,
                    forbid_continent=False, heat_flow=None, band_km=None, setup=None, edge_strength_scale=None,
                    inherited=False, ledger=None, bounded_axes=None):
    """The single worst mechanism, warm-startable from previous axes.

    The legacy coarse=False search refines prior axes from 6 degrees. Bounded
    policy checks instead hold the prior axis pair between scheduled rescans;
    principal-load and inherited-rift axes are reconsidered on rescan.
    """
    setup = setup or _setup(s, plate, balance, heat_flow, band_km, inherited, ledger)
    cells = setup['cells']
    points, torque, edges, mid, length, strength = setup['args']
    if edge_strength_scale:
        # Already-stretched rift edges (global mesh edge -> factor <= 1).
        scale = np.ones(len(edges))
        index = {int(e): k for k, e in enumerate(np.flatnonzero(setup['internal']))}
        for edge, factor in edge_strength_scale.items():
            if int(edge) in index:
                scale[index[int(edge)]] = float(factor)
        strength = tuple(v*scale for v in strength)
    count = len(edges)
    blocked = setup['craton'].copy() if forbid_cratons else np.zeros(count, bool)
    if forbid_continent:
        blocked |= setup['continental']
    glue = _craton_glue(s, plate, cells) if forbid_cratons else []
    # A cell held to its plate by a single internal edge is a staircase tooth;
    # it cannot tear off as a resolved plate, so it moves with its neighbour.
    degree = np.bincount(edges.ravel(), minlength=len(cells))
    tooth = degree[edges[:, 0]] <= 1
    tooth |= degree[edges[:, 1]] <= 1
    blocked |= tooth
    candidate_kinds = None
    if bounded_axes is None:
        best = search_prepared((points, torque, edges, mid, length, strength),
                               blocked, glue, seed_axes, coarse)
    else:
        axes, candidate_kinds = bounded_candidate_axes(s, plate, setup,
            seed_axes=seed_axes, max_axes=bounded_axes, held_only=not coarse)
        best = search_prepared((points, torque, edges, mid, length, strength),
                               blocked, glue, (), False, axes=axes)
    cut = best['cut']
    result = dict(ratio=best['ratio'], axis=best['axis'], cells=cells, piece=best['piece'], cut=cut,
                  cut_edges=np.flatnonzero(setup['internal'])[cut],
                  cut_length_km=float(length[cut].sum()/1e3),
                  cut_continental_fraction=float(length[cut & setup['continental']].sum()/max(float(length[cut].sum()), 1e-30)),
                  unresolved_fraction_of_gross=setup['loads']['unresolved_fraction_of_gross'])
    if candidate_kinds is not None:
        result.update(search_policy='bounded_axes_v1', searched_axis_count=len(candidate_kinds),
            candidate_axis_sources=candidate_kinds,
            search_scope='scheduled candidate family' if coarse else 'held prior axis pair',
            search_interpretation='Complete native cuts and strengths along finite candidate axes; '
                                  'absence of failure is not a certificate over unsearched directions.')
    if best['axis'] is not None:
        if setup['balance'].uses_force_ledger:
            import breakup_mode
            result.update(breakup_mode.solve(s, plate, cells, best['piece'], best['axis'],
                dict(edges=edges, mid=mid, length_m=length, cut=cut), strength,
                setup['balance'], ledger=setup['ledger'], band_km=band_km))
        else:
            result.update(_opening_rate(s, cells, best, length, setup['mid'], setup['torque'][cells], None))
    return result


def search_prepared(args, blocked, glue, seed_axes=(), coarse=True, *, axes=None):
    """Pure ordered search on immutable per-plate numerical inputs.

    Each call owns its work arrays and glue list. Source loads, state history,
    source transactions and opening-rate bookkeeping stay in the parent.
    """
    points, torque, edges, mid, length, strength = args
    count = len(edges)
    glue = [list(link) for link in glue]
    seeds = [a for a in seed_axes if a is not None]
    resolved = False
    for round_index in range(MAX_GLUE_ROUNDS):
        extra = np.asarray(glue, int).reshape(-1, 2)
        # Virtual zero-length links that no mechanism may cut: they hold a
        # craton group, or a sub-resolution piece, to the rest of its side.
        all_edges = np.vstack((edges, extra))
        all_mid = np.vstack((mid, points[extra[:, 0]])) if len(extra) else mid
        all_length = np.r_[length, np.zeros(len(extra))]
        all_strength = tuple(np.r_[v, np.zeros(len(extra))] for v in strength)
        best = analyse(points, torque, all_edges, all_mid, all_length, all_strength,
                       forbidden=np.r_[blocked, np.ones(len(extra), bool)],
                       seed_axes=seeds if axes is None else (), axes=axes,
                       coarse=(coarse or not len(seeds)) and round_index == 0 if axes is None else False,
                       refine_start_deg=12. if (coarse or not len(seeds)) and round_index == 0 else 6.)
        if best['axis'] is None:
            break
        small = min(int(best['piece'].sum()), int((~best['piece']).sum()))
        if small >= MIN_PIECE_CELLS:
            resolved = True
            break
        # A mechanism thinner than a resolvable plate is a mesh artifact:
        # glue its cells to their neighbours and search again.
        side = best['piece'] if best['piece'].sum() <= (~best['piece']).sum() else ~best['piece']
        crossing = side[edges[:, 0]] != side[edges[:, 1]]
        glue.extend(edges[crossing].tolist())
        seeds = [best['axis']]
    if not resolved:
        # No mechanism at or above resolution was found: report none rather
        # than a sliver that no plate split could represent.
        best = dict(ratio=0., axis=None, piece=np.zeros(len(points), bool), cut=np.zeros(len(all_edges), bool))
    best['cut'] = best['cut'][:count]
    return best


def plate_mechanisms(s, plate, balance=None, axes=None, heat_flow=None, band_km=None, inherited=False):
    """Worst mechanism overall and worst ocean-only mechanism for one plate."""
    setup = _setup(s, plate, balance, heat_flow, band_km, inherited)
    cells, torque, loads, internal = setup['cells'], setup['torque'], setup['loads'], setup['internal']
    length, mid, continental, args = setup['length'], setup['mid'], setup['continental'], setup['args']
    report = dict(version=VERSION, plate_uid=int(s.plate_uid[plate]), plate_name=s.names[plate],
                  cells=int(len(cells)), loads={k: v for k, v in loads.items() if k != 'resistance_weight'},
                  slab_load_band_km=float(
                      s.config.get('deformation_width_km', 400.) if band_km is None else band_km),
                  reference_strain_rate_s=REFERENCE_STRAIN_RATE_S,
                  continental_heat_flow_mw_m2=heat_flows(heat_flow))
    # Both scans reject sub-resolution pieces the same way (see
    # worst_mechanism); each is seeded with the other's axis.
    found = {}
    if continental.any():
        found['ocean_only'] = worst_mechanism(s, plate, setup=setup, forbid_continent=True,
                                              coarse=axes is None)
    seeds = [v['axis'] for v in found.values() if v['axis'] is not None]
    found['any'] = worst_mechanism(s, plate, setup=setup, seed_axes=seeds, coarse=True)
    if 'ocean_only' not in found:
        found['ocean_only'] = found['any']
    elif found['ocean_only']['ratio'] > found['any']['ratio']:
        found['any'] = found['ocean_only']
    elif not np.any(found['any']['cut'] & continental) and found['any']['ratio'] > found['ocean_only']['ratio']:
        found['ocean_only'] = found['any']
    for name in ('any', 'ocean_only'):
        best = found[name]
        cut = best['cut']
        cut_continental = float(length[cut & continental].sum())
        report[name] = dict(loading_ratio=best['ratio'], fails=bool(best['ratio'] >= 1.),
            predicted_opening_cm_yr_mean=best.get('predicted_opening_cm_yr_mean', 0.),
            predicted_opening_cm_yr_max=best.get('predicted_opening_cm_yr_max', 0.),
            axis_xyz=None if best['axis'] is None else best['axis'].tolist(),
            cut_length_km=float(length[cut].sum()/1e3),
            cut_continental_length_km=cut_continental/1e3,
            cut_continental_fraction=cut_continental/max(float(length[cut].sum()), 1e-30),
            piece_area_km2=float(np.asarray(s.cell_area)[cells][best['piece']].sum()),
            piece_cells=cells[best['piece']].tolist(),
            cut_edges=best['cut_edges'].tolist())
        if setup['balance'].uses_force_ledger and best['axis'] is not None:
            report[name].update({key: value for key, value in best.items()
                if key.startswith(('mode_', 'opening_mode_', 'common_restriction_',
                                   'differential_virtual_work_', 'common_motion_'))})
    any_ratio, ocean_ratio = report['any']['loading_ratio'], report['ocean_only']['loading_ratio']
    report['continent_first'] = bool(report['any']['cut_continental_fraction'] > 0. and any_ratio > ocean_ratio)
    report['continent_to_ocean_ratio'] = any_ratio/ocean_ratio if ocean_ratio > 0. else math.inf
    report['interpretation'] = (
        'loading_ratio = held intact-state force power / cut plastic cost of the worst searched '
        'rigid-block mechanism. > 1 is onset within the declared spatial approximation, not a '
        'prescribed finite opening speed; equality is marginal. The ocean-only scan forbids '
        'cutting any edge touching continental crust. Attached worlds reevaluate active resistance '
        'in the reduced opening response; finite evolution requires separate qualification.')
    return report
