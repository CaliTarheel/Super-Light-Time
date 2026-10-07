"""Persistent underthrust contacts on conserved, separate material sheets.

This reduced contact model supplies dissipative convergence resistance and
local loading fronts to the existing bounded geometric deformation solver.
It never adds a second scalar shortening strain, combines columns, or deletes
overlapping crust. Covered-area erosion uses four spherical subtriangle
quadrature samples per contacted face; exact pair overlaps are separate.
"""
from __future__ import annotations

from copy import copy, deepcopy
import hashlib
import math
import numpy as np

import mesh_geometry
import mesh_coverage
import material_surface
import structure_engine
import collision_fronts

VERSION = 1
SUTURE_WELD_VERSION = 1
POLARITY_POLICY_VERSION = 2
POLARITY_TIE_TOLERANCE_M = 1e-7  # A tenth of a micrometre: clipping roundoff, not relief.
# The suture no longer damps omega here. It is a plastic element assembled by
# plate_balance on the overlap-area rate, so its capacity enters the same
# inertia-free force balance as ridge push and slab pull instead of relaxing an
# already-solved rotation. What survives in this module is the state that
# element reads (below) and the material consolidation it eventually earns.
#
# Yield capacity is PER METRE OF SUTURE TRACE and carries no width or thickness
# factor. Copley, Avouac & Royer (2010, JGR 115 B03410) fit the Indian plate's
# Euler poles with a Himalaya-Tibet resistance of ~5-6e12 N/m -- per metre of
# margin, roughly twice ridge push plus net subduction combined -- and that is a
# line force, not a force per unit underthrust area. The distinction matters
# here: the eclogite sink removes up to 60% of the buried column while leaving
# the overlap footprint untouched, so a thickness-scaled weld would evaporate as
# the root founders and a width-scaled one would stiffen without bound as the
# 2,100 km underthrust sheet grows.
#
# Cold, dry lower crust is strong; a young suture that is still hot is not.
# Continental strength envelopes for a 30-60 Myr-old orogen span ~1.5e12 N/m
# (hot, Moho >800 C) to ~1e13 N/m (cold cratonic; Burov & Watts 2006, GSA Today
# 16(1); Jackson 2002) and the conductive relaxation time of a doubled crust is
# 30-80 Myr, so the strengthening clock below is a decade-of-Myr closure with an
# honest factor-of-two band, not a measured rock property.
WELD_HOT_N_PER_M = 1.5e12
WELD_COLD_N_PER_M = 1e13
WELD_THERMAL_TAU_MYR = 50.
# Sutures are the weaknesses the next Wilson cycle reuses: the Iapetus and
# Caledonian sutures localized Mesozoic rifting three to four hundred Myr after
# they closed (Wilson 1966; Thomas 2006, GSA Today 16(2)). An idle suture
# therefore anneals DOWN toward an inherited-weakness floor rather than staying
# welded shut for ever, and never re-strengthens once weakened -- that ratchet is
# what makes it an inherited structure. The floor and the 150 Myr time constant
# are chosen to keep a finished collision shut for the whole orogenic lifetime
# (the user's standard: a finished collision must not un-collide) while leaving
# the old trace as the preferred break after a few hundred Myr of quiescence.
WELD_COHESION_FLOOR = .25
WELD_ANNEAL_TAU_MYR = 150.
# Consolidation thresholds. Buried means most of the face is hidden under the
# sheet above it; welded means its own persistent suture maturity is saturated.
CONSOLIDATION_EXPOSED_FRACTION = .5
CONSOLIDATION_SUTURE = .95
# A rim of the under sheet always stays with its own plate. The rim is where the
# pair's gravitational driver actually lives (a boundary integral over the
# shallow edge the sink never reaches), so consolidating it away would remove
# the driver instead of resisting it; the weld does the arresting, not this.
CONSOLIDATION_RIM_AREA_KM2 = 4e6
# Numerical rate limiter, not physics: welding is instantaneous once mature, but
# re-parenting several million km2 of material in one step would move a plate's
# carried area and the overlap ledger discontinuously under a balance that is
# solved from scratch each step. One Tibet (2.5e6 km2, this project's plateau
# yardstick) per contact per step drains the 8.1e6 km2 standing backlog at
# 396 Myr over four steps instead of one.
CONSOLIDATION_STEP_AREA_KM2 = 2.5e6
ARRAY_FIELDS = dict(material_collision_sheet=np.int64, material_exposed_fraction=np.float64,
                    material_burial_myr=np.float64, material_collision_suture=np.float64)
PARCEL_FIELDS = ('parcel_collision_sheet', 'parcel_exposed_fraction', 'parcel_burial_myr',
                 'parcel_collision_suture')
WELD_ROW_FIELDS = ('last_convergent_myr', 'weld_cohesion')


def _unit(value):
    value = np.asarray(value, float)
    return value/np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-30)


def _new_components(faces, owners):
    """Label newly born material by shared edges and plate ownership."""
    return _components(faces, owners)


def _shared_edges(faces, owners=None):
    """Adjacent face indices, optionally requiring the same plate owner.

    A corner touch does not connect material. A non-manifold edge links
    consecutive rows transitively. Consolidation works within one sheet, so
    it omits the owner key; newborn sheets must include it.
    """
    faces = np.asarray(faces)
    if not len(faces): return np.empty(0, np.int64), np.empty(0, np.int64)
    edge = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]], axis=2).reshape(-1, 2)
    if owners is not None:
        owners = np.asarray(owners)
        if owners.shape != (len(faces),):
            raise ValueError('Shared-edge ownership must align with material faces.')
        edge = np.column_stack((edge, np.repeat(owners, 3)))
    _, inverse = np.unique(edge, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind='stable')
    hit = np.flatnonzero(inverse[order[1:]] == inverse[order[:-1]])
    return order[hit]//3, order[hit+1]//3


def _components(faces, owners=None):
    """Connected-component label per face over eligible shared edges."""
    n = len(faces)
    parent = np.arange(max(n, 1))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for a, b in zip(*_shared_edges(faces, owners)):
        one, two = root(int(a)), root(int(b))
        if one != two: parent[max(one, two)] = min(one, two)
    return np.array([root(i) for i in range(n)], np.int64)


def _would_disconnect(faces, donor, moving):
    """True when re-parenting `moving` would cut or empty a body of `donor`.

    A material sheet is a connected body. A transfer that split one in two would
    manufacture a second plate fragment out of bookkeeping, and one that emptied
    a body would delete a sheet that still holds crust; both are forbidden. Face
    adjacency is the material surface's own shared-edge graph, so the test is
    the same connectivity the sheet IDs themselves are built on.
    """
    donor = np.unique(np.asarray(donor, np.int64)); moving = np.asarray(moving, np.int64)
    keep = np.setdiff1d(donor, moving)
    if not len(keep): return True
    before = _components(faces[donor])
    after = _components(faces[keep])
    body = before[np.searchsorted(donor, keep)]
    if len(np.unique(body)) != len(np.unique(before)): return True
    return len(np.unique(np.column_stack((body, after)), axis=0)) != len(np.unique(body))


def _whole_bodies(faces, donor, moving):
    """Donor faces belonging to a body `moving` would take entire.

    A body of the donor sheet that is welded and buried edge to edge has no rim
    of its own left; re-parenting it whole would delete a sheet rather than
    consolidate one, and a plate would gain a body it never made contact with
    around its whole perimeter. The whole body stays, not just one face of it.
    """
    donor = np.unique(np.asarray(donor, np.int64))
    bodies = _components(faces[donor])
    inside = np.isin(donor, np.asarray(moving, np.int64))
    whole = [int(body) for body in np.unique(bodies[inside]) if bool(np.all(inside[bodies == body]))]
    return donor[np.isin(bodies, whole)] if whole else np.empty(0, np.int64)


def _restore_connectivity(faces, donor, moving):
    """Release the fewest welded faces that keep every donor body one piece.

    Each retained fragment is reconnected to its body's largest retained piece
    along a shortest chain of selected faces, which is then released back to the
    donor. Releasing only shrinks the transfer, so the rim and per-step caps
    applied before this stay satisfied.
    """
    donor = np.asarray(donor, np.int64); moving = np.asarray(moving, np.int64)
    if not len(moving): return moving, 0
    slot = {int(face): index for index, face in enumerate(donor)}
    neighbours = [[] for _ in donor]
    for a, b in zip(*_shared_edges(faces[donor])):
        neighbours[int(a)].append(int(b)); neighbours[int(b)].append(int(a))
    selected = np.zeros(len(donor), bool); selected[[slot[int(f)] for f in moving]] = True
    bodies = _components(faces[donor])
    released = 0
    for body in np.unique(bodies):
        inside = np.flatnonzero(bodies == body)
        for _ in range(len(inside)+1):
            retained = inside[~selected[inside]]
            if len(retained) <= 1: break
            pieces = _components(faces[donor[retained]])
            if len(np.unique(pieces)) <= 1: break
            anchor = int(np.argmax(np.bincount(pieces)))
            # Breadth-first from the anchor piece, expanding only through
            # selected faces: the chain recorded behind any other retained piece
            # it reaches is a shortest bridge of welded faces back to the anchor.
            came = {}; frontier = [int(index) for index in retained[pieces == anchor]]
            seen = set(frontier); wanted = {int(index) for index in retained[pieces != anchor]}
            reached = []
            while frontier:
                following = []
                for at in frontier:
                    for step in neighbours[at]:
                        if step in seen: continue
                        seen.add(step); came[step] = at
                        if step in wanted: reached.append(step)
                        elif selected[step]: following.append(step)
                frontier = following
            if not reached: break
            for step in reached:
                back = came[step]
                while selected[back]:
                    selected[back] = False; released += 1; back = came[back]
    return donor[selected], released


def weld_enabled(s):
    """The weld's state, consolidation and anneal are all version gated."""
    return getattr(s, 'suture_weld_version', 0) == SUTURE_WELD_VERSION


def _weld_defaults(row):
    """Per-contact weld state, defaulted in place for new and migrated rows.

    A migrated row records how long it shortened for and when it began, but not
    WHEN those episodes fell, so the clock is reconstructed from what it does
    record. A row whose last measured front was still closing is converging now.
    Otherwise its shortening is placed as early as its own history allows --
    started_myr + convergent_myr, never later than it was last seen -- which is
    the reading that credits an idle suture with the most cooling its own record
    can support, and no more. A migration that can date the episodes from saved
    frames should write the field itself rather than rely on this.
    """
    if 'last_convergent_myr' not in row:
        seen = float(row.get('last_seen_myr', row.get('started_myr', 0.)))
        closing = float(row.get('normal_speed_km_myr', 0.)) < -.02
        row['last_convergent_myr'] = seen if closing else min(
            seen, float(row.get('started_myr', seen))+float(row.get('convergent_myr', 0.)))
    row.setdefault('weld_cohesion', 1.)


def weld_capacity_factor(row, t):
    """Dimensionless multiplier on a suture's per-metre plastic yield capacity.

    plate_balance owns the force itself (F_REF x the boundary's own strength
    sample); this is the part that depends on contact state and therefore lives
    with the state. Two factors multiply: conductive strengthening of the suture
    since it last took up shortening, from WELD_HOT_N_PER_M to WELD_COLD_N_PER_M
    on WELD_THERMAL_TAU_MYR, and the inherited-weakness cohesion that anneals
    down while the suture is idle. Both are decade-of-Myr closures; see the
    constant block for the literature band.
    """
    idle = max(float(t)-float(row.get('last_convergent_myr', t)), 0.)
    thermal = WELD_HOT_N_PER_M+(WELD_COLD_N_PER_M-WELD_HOT_N_PER_M)*(-np.expm1(-idle/WELD_THERMAL_TAU_MYR))
    return float(row.get('weld_cohesion', 1.))*float(thermal)/WELD_COLD_N_PER_M


def weld_rows(s):
    """(row, top owner slot, under owner slot) for every suture that can yield.

    plate_balance iterates this to assemble the plastic element on each pair's
    overlap-area rate. Only a contact with a live overlap between two DIFFERENT
    plates has that coordinate: an accreted pair is already one plate and a quiet
    pair has no overlap left to close or open. Slots are material plate indices
    (s.parcel_plate / s.omega rows), not the generation-unique plate UIDs.
    """
    if not weld_enabled(s): return []
    result = []
    for row in getattr(s, 'collision_contacts', ()):
        if row.get('state') != 'active' or not (row.get('overlap_area_km2', 0.) > 0.): continue
        top, under = int(row.get('top_owner', -1)), int(row.get('under_owner', -1))
        if top < 0 or under < 0 or top == under: continue
        _weld_defaults(row)
        result.append((row, top, under))
    return result


def _anneal_welds(s, dt):
    """Idle sutures lose cohesion toward the inherited-weakness floor.

    Monotone: cohesion never recovers. A suture that has been quiet long enough
    to weaken is the structure the next extensional episode reuses, and letting
    renewed shortening re-weld it to 1.0 would erase exactly that memory.
    """
    for row in s.collision_contacts:
        _weld_defaults(row)
        if float(row['last_convergent_myr']) >= float(s.t): continue
        cohesion = float(row['weld_cohesion'])
        row['weld_cohesion'] = float(WELD_COHESION_FLOOR
            +(cohesion-WELD_COHESION_FLOOR)*np.exp(-float(dt)/WELD_ANNEAL_TAU_MYR))


def ensure_fields(s):
    """Persistent per-face fields participate in the normal parcel remapping."""
    n = len(s.parcel_patch)
    old = getattr(s, 'parcel_collision_sheet', np.empty(0, np.int64))
    if len(old) > n:
        raise ValueError('Collision sheet fields were not transferred with material geometry.')
    for name, default, dtype in (('parcel_collision_sheet', 0, np.int64),
                                ('parcel_exposed_fraction', 1., float),
                                ('parcel_burial_myr', 0., float),
                                ('parcel_collision_suture', 0., float)):
        value = getattr(s, name, np.empty(0, dtype))
        if not hasattr(s, name) or len(value) < n:
            setattr(s, name, np.r_[value, np.full(n-len(value), default, dtype)])
    if not hasattr(s, 'collision_contacts'):
        s.collision_contacts = []
        s.next_collision_contact_id = 1
        s.next_collision_sheet_id = 1
    if weld_enabled(s):
        for row in s.collision_contacts: _weld_defaults(row)
    missing = np.flatnonzero(s.parcel_collision_sheet == 0)
    if len(missing):
        components = _new_components(s.material_surface['faces'][missing], s.parcel_plate[missing])
        for component in np.unique(components):
            s.parcel_collision_sheet[missing[components == component]] = s.next_collision_sheet_id
            s.next_collision_sheet_id += 1
    # An explicit fault/owner split starts a daughter sheet but keeps existing
    # over/under relationships. Accretion never welds independent sheet IDs.
    for sheet in np.unique(s.parcel_collision_sheet):
        use = s.parcel_collision_sheet == sheet
        owners = np.unique(s.parcel_plate[use])
        if len(owners) <= 1: continue
        for owner in owners[1:]:
            daughter = s.next_collision_sheet_id; s.next_collision_sheet_id += 1
            s.parcel_collision_sheet[use & (s.parcel_plate == owner)] = daughter
            for row in list(s.collision_contacts):
                if sheet not in (row['top_sheet'], row['under_sheet']): continue
                clone = deepcopy(row)
                clone['id'] = s.next_collision_contact_id; s.next_collision_contact_id += 1
                clone['top_sheet' if clone['top_sheet'] == sheet else 'under_sheet'] = int(daughter)
                clone['inherited_from_contact'] = row['id']
                clone['overlap_area_km2'] = 0.
                clone['state'] = 'quiet'
                s.collision_contacts.append(clone)


def validate_alignment(s):
    """Reject stale face state before a geometry transaction can omit it.

    Birth must append fields through the native synchronization lifecycle.
    Remapping cannot repair an old array by padding it after face rows changed.
    """
    n = len(s.parcel_patch)
    for name in PARCEL_FIELDS:
        value = getattr(s, name, None)
        if not isinstance(value, np.ndarray) or value.shape != (n,) or not np.isfinite(value).all():
            raise ValueError(f'Collision field {name} must align with every material face before adaptation.')
    if not np.issubdtype(s.parcel_collision_sheet.dtype, np.integer) or np.any(s.parcel_collision_sheet <= 0):
        raise ValueError('Every material face requires a persistent positive collision sheet before adaptation.')
    # The weld's state is per contact, not per face, so adaptation cannot pad or
    # remap it. It can only be missing, which means a row escaped ensure_fields.
    if weld_enabled(s):
        for row in s.collision_contacts:
            if not all(np.isfinite(row.get(field, np.nan)) for field in WELD_ROW_FIELDS):
                raise ValueError('Every collision contact requires finite suture weld state before adaptation.')


def eligible_hits(query, faces, sheets, contacts):
    """Hide a lower sheet only where its recorded upper sheet actually contains.

    No nearest-face extrapolation, blanket removal, or array-row ancestry is
    permitted. A separated sheet immediately becomes exposed again.
    """
    query, faces = np.asarray(query), np.asarray(faces)
    hit_sheets = np.asarray(sheets)[faces]
    valid = np.ones(len(query), bool)
    present = set(map(int, np.unique(hit_sheets)))
    for row in contacts:
        top, under = int(row['top_sheet']), int(row['under_sheet'])
        if top not in present or under not in present: continue
        covered_queries = query[hit_sheets == top]
        valid[(hit_sheets == under) & np.isin(query, covered_queries)] = False
    return valid


def _would_cycle(contacts, top, under):
    children = {}
    for row in contacts: children.setdefault(row['top_sheet'], set()).add(row['under_sheet'])
    pending = [under]; visited = set()
    while pending:
        at = pending.pop()
        if at == top: return True
        if at in visited: continue
        visited.add(at); pending.extend(children.get(at, ()))
    return False


def _exposure(s, overlap):
    fraction = np.ones(len(s.mass))
    selected = np.unique(np.r_[overlap['first'], overlap['second']])
    if not len(selected): return fraction
    triangle = s.material_surface['vertices'][s.material_surface['faces'][selected]]
    a, b, c = triangle.transpose(1, 0, 2)
    ab, bc, ca = _unit(a+b), _unit(b+c), _unit(c+a)
    sub = np.stack((np.stack((a, ab, ca), axis=1), np.stack((ab, b, bc), axis=1),
                    np.stack((ca, bc, c), axis=1), np.stack((ab, bc, ca), axis=1)), axis=1)
    points = _unit(sub.sum(axis=2)).reshape(-1, 3)
    q, f, _ = mesh_geometry.locate_points(points, overlap['locator'], all_hits=True)
    allowed = eligible_hits(q, f, s.parcel_collision_sheet, s.collision_contacts)
    # Each sample tests whether its own material sheet is exposed. Shared-edge
    # ties within one sheet are harmless; different sheets retain provenance.
    own = np.repeat(s.parcel_collision_sheet[selected], 4)
    exposed = np.zeros(len(points), bool)
    np.logical_or.at(exposed, q, allowed & (s.parcel_collision_sheet[f] == own[q]))
    flat = sub.reshape(-1, 3, 3)
    aa, bb, cc = flat.transpose(1, 0, 2)
    numerator = np.abs(np.einsum('ni,ni->n', aa, np.cross(bb, cc)))
    denominator = 1+np.einsum('ni,ni->n', aa, bb)+np.einsum('ni,ni->n', bb, cc)+np.einsum('ni,ni->n', cc, aa)
    weights = (2*np.arctan2(numerator, denominator)).reshape(-1, 4)
    fraction[selected] = np.sum(weights*exposed.reshape(-1, 4), axis=1)/weights.sum(axis=1)
    return np.clip(fraction, 0., 1.)


def admit_contact(s, one, two, aa, bb, weights, heights):
    """Append one first-contact record; callers may supply a detached ledger."""
    sheets = s.parcel_collision_sheet
    # Decide first-contact polarity from actual contacting buoyancy;
    # each intersection contributes its own area once to each sheet.
    # Whole-face weights count a coarse face repeatedly when the other
    # sheet is refined, changing polarity without changing geology.
    # Compensated sums and a physical near-zero tie tolerance prevent
    # clipping roundoff from overriding the stable sheet-ID tie break.
    # Existing contacts, including earlier policies, are never rescored.
    score = {}
    total = math.fsum(map(float, weights))
    for sheet in (int(one), int(two)):
        faces = np.where(sheets[aa] == sheet, aa, bb)
        score[sheet] = math.fsum(float(height)*float(weight)
            for height, weight in zip(heights[faces], weights))/total
    tied = abs(score[int(one)]-score[int(two)]) <= POLARITY_TIE_TOLERANCE_M
    top = int(min(one, two)) if tied else max(score, key=score.get)
    under = int(two) if top == one else int(one)
    import inherited_collision_polarity
    inherited = inherited_collision_polarity.choose(s, one, two, aa, bb)
    if inherited is not None:
        top, under = inherited['top_sheet'], inherited['under_sheet']
    cycle_override = _would_cycle(s.collision_contacts, top, under)
    if inherited is not None and cycle_override:
        raise ValueError('Inherited finite collision polarity conflicts with existing stack order; explicit handoff required.')
    if cycle_override: top, under = under, top
    row = dict(id=s.next_collision_contact_id, top_sheet=int(top), under_sheet=int(under),
        started_myr=float(s.t), last_seen_myr=float(s.t), cumulative_convergence_km=0.,
        convergent_myr=0., suture_strength=0., state='active', overlap_area_km2=0.,
        polarity_policy_version=POLARITY_POLICY_VERSION,
        polarity_height_scores_m={str(sheet):value for sheet, value in score.items()},
        polarity_tie_tolerance_m=POLARITY_TIE_TOLERANCE_M,
        polarity_cycle_override=bool(cycle_override))
    if inherited is not None:
        row.update(inherited)
    top_face=int(aa[0] if sheets[aa[0]]==top else bb[0])
    under_face=int(bb[0] if sheets[aa[0]]==top else aa[0])
    row.update(top_owner=int(s.parcel_plate[top_face]),under_owner=int(s.parcel_plate[under_face]))
    if weld_enabled(s): _weld_defaults(row)
    s.next_collision_contact_id += 1; s.collision_contacts.append(row)
    return row


def overlap_signature(s):
    """Fingerprint of every input that decides the exact overlap ledger."""
    surface = s.material_surface
    return hashlib.sha256(surface['vertices'].tobytes()+surface['faces'].tobytes()
        +s.parcel_collision_sheet.tobytes()+s.parcel_plate.tobytes()).hexdigest()


def current_overlap(s):
    """The exact overlap ledger of the latest refresh, or None if there is none.

    A ledger measured on other geometry, sheets or owners must not be read as
    today's contacts, so a stale one raises rather than being silently reused.
    """
    overlap = getattr(s, '_collision_overlap', None)
    if overlap is None:
        return None
    if overlap_signature(s) != getattr(s, '_collision_signature', None):
        raise ValueError('Accretion requires current persistent collision geometry.')
    return overlap


def same_owner_overlap_pairs(s, *, with_contact_area=False):
    """Face pairs (first, second) of distinct sheets welded inside one plate.

    A weld is a thrust or suture contact internal to one plate: the pairs
    ``refresh`` labels 'accreted' (a current positive-area overlap between two
    sheets with one motion owner), restricted to faces with mass and a buoyant
    kind on both sides, and to sheet pairs whose summed overlap exceeds
    ``column_density.UNORDERED_OVERLAP_FLOOR_KM2`` (1 km2). At or below that
    floor two sheets merely abut: arc emplacement leaves such roundoff overlaps
    (0.018 m2 stopped SEP21T at 85 Myr, PHYSICS_REPAIRS §17), and whether two
    abutting sheets overlap or leave a gap by that much is an accident of
    clipping, not geology. The floor is summed over the whole sheet pair, so a
    genuine thrust contact counts however finely its faces are cut. Without a
    ledger the result is empty.

    With ``with_contact_area`` it also returns, for each face pair, the summed
    overlap of its sheet pair in km2 (the weld's contact area).
    """
    import column_density
    overlap = current_overlap(s) if hasattr(s, 'collision_contacts') else None
    empty = np.empty((0, 2), np.int64)
    if overlap is None:
        return (empty, np.empty(0)) if with_contact_area else empty
    first = np.asarray(overlap['first'], np.int64)
    second = np.asarray(overlap['second'], np.int64)
    area = np.asarray(overlap['area_km2'], float)
    plate, mass, kind = np.asarray(s.parcel_plate), np.asarray(s.mass), np.asarray(s.kind)
    sheets = np.asarray(s.parcel_collision_sheet)
    keep = ((area > 0.) & (plate[first] == plate[second]) & (sheets[first] != sheets[second])
            & (mass[first] > 0.) & (mass[second] > 0.) & (kind[first] > 0) & (kind[second] > 0))
    first, second, area = first[keep], second[keep], area[keep]
    if not len(first):
        return (empty, np.empty(0)) if with_contact_area else empty
    pair = np.sort(np.column_stack((sheets[first], sheets[second])), axis=1)
    unique, inverse = np.unique(pair, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    contact = np.bincount(inverse, weights=area, minlength=len(unique))[inverse]
    weld = contact > column_density.UNORDERED_OVERLAP_FLOOR_KM2
    pairs = np.column_stack((first[weld], second[weld]))
    return (pairs, contact[weld]) if with_contact_area else pairs


def refresh(s, dt=0.):
    """Measure current contacts; advance burial/suture clocks at most once/step."""
    import entry_regions
    if not entry_regions.enabled(s):
        return _refresh(s, dt)
    # The explicitly selected entry experiment can encounter an unsupported
    # partial/conflicting order. Stage every field touched by this refresh so
    # refusal precedes publication of IDs, clocks, caches, exposure or sutures.
    # Ordinary worlds retain their existing path and copying cost.
    fields = PARCEL_FIELDS+('suture', 'collision_contacts', 'next_collision_contact_id',
        'next_collision_sheet_id', '_collision_signature', '_collision_overlap',
        '_weld_annealed_myr', '_burial_updated_myr', 'collision_diagnostics',
        'continental_entry_regions', 'parcel_entry_region')
    staged = copy(s)
    for name in fields:
        if hasattr(s, name):
            setattr(staged, name, deepcopy(getattr(s, name)))
    _refresh(staged, dt)
    for name in fields:
        if hasattr(staged, name):
            setattr(s, name, getattr(staged, name))


def _refresh(s, dt):
    ensure_fields(s)
    surface = s.material_surface
    signature = overlap_signature(s)
    if signature == getattr(s, '_collision_signature', None):
        overlap = s._collision_overlap
    else:
        overlap = mesh_coverage.material_overlaps(surface['vertices'], surface['faces'],
            s.parcel_collision_sheet, radius_km=surface['radius_km'])
        s._collision_signature, s._collision_overlap = signature, overlap
    first, second, area = overlap['first'], overlap['second'], overlap['area_km2']
    sheets = s.parcel_collision_sheet
    records = {tuple(sorted((row['top_sheet'], row['under_sheet']))): row for row in s.collision_contacts}
    for row in s.collision_contacts:
        row['overlap_area_km2'] = 0.; row['state'] = 'quiet'
        row['local_fronts'] = []
    if len(first) and '_front_geometry' not in overlap:
        overlap['_front_geometry'] = collision_fronts.prepare(surface, sheets, overlap)
    heights = structure_engine.material_height(s.kind, s.relief)
    pair = np.sort(np.column_stack((sheets[first], sheets[second])), axis=1)
    unique, inverse = np.unique(pair, axis=0, return_inverse=True)
    for index, (one, two) in enumerate(unique):
        take = np.flatnonzero(inverse == index)
        aa, bb, weights = first[take], second[take], area[take]
        owner_a = int(s.parcel_plate[aa[0]]); owner_b = int(s.parcel_plate[bb[0]])
        key = (int(one), int(two)); row = records.get(key)
        if row is None:
            row = admit_contact(s, one, two, aa, bb, weights, heights)
            records[key] = row
        top_faces = np.unique(np.r_[aa[sheets[aa] == row['top_sheet']], bb[sheets[bb] == row['top_sheet']]])
        under_faces = np.unique(np.r_[aa[sheets[aa] == row['under_sheet']], bb[sheets[bb] == row['under_sheet']]])
        top_owner, under_owner = int(s.parcel_plate[top_faces[0]]), int(s.parcel_plate[under_faces[0]])
        front_cache = overlap.setdefault('_fronts_by_pair', {})
        ordered_pair = (row['top_sheet'], row['under_sheet'])
        if ordered_pair not in front_cache:
            front_cache[ordered_pair] = collision_fronts.construct(surface, sheets, overlap, take,
                row['top_sheet'], overlap['_front_geometry'])
        fronts = deepcopy(front_cache[ordered_pair])
        state = 'accreted' if top_owner == under_owner else 'active'
        for front in fronts:
            front.update(parent_contact_id=row['id'], geometry_epoch_myr=float(s.t),
                top_owner=top_owner, under_owner=under_owner, state=state,
                normal_speed_km_myr=float(np.dot(np.cross(s.omega[under_owner]-s.omega[top_owner],
                    front['center']), front['normal'])*surface['radius_km']))
        # Parent geometry/signed speed are overview diagnostics only. All
        # loading paths expand the local fronts, so closing and opening
        # patches cannot cancel each other before a constitutive response.
        total = math.fsum(front['overlap_area_km2'] for front in fronts)
        centre = _unit(np.sum([np.asarray(f['center'])*f['overlap_area_km2'] for f in fronts], axis=0))
        normal = _unit(np.sum([np.asarray(f['normal'])*f['length_km'] for f in fronts], axis=0))
        speed = math.fsum(f['normal_speed_km_myr']*f['overlap_area_km2'] for f in fronts)/total
        row.update(last_seen_myr=float(s.t), overlap_area_km2=float(weights.sum()),
            center=centre.tolist(), normal=normal.tolist(), top_owner=top_owner, under_owner=under_owner,
            top_plate_uid=int(s.plate_uid[top_owner]), under_plate_uid=int(s.plate_uid[under_owner]),
            normal_speed_km_myr=speed, state=state, local_fronts=fronts, local_fronts_version=1,
            convergence_aggregation='overlap-area-weighted positive local closing speed',
            front_identity='front_index is current geometry only; parent contact ID retains geological history')
        if dt > 0. and row.get('updated_myr') != float(s.t):
            convergence = math.fsum(max(-f['normal_speed_km_myr'], 0.)*f['overlap_area_km2'] for f in fronts)/total
            row['cumulative_convergence_km'] += convergence*dt
            row['convergent_myr'] += dt if convergence > .02 else 0.
            row['suture_strength'] = float(-np.expm1(-row['cumulative_convergence_km']/100.))
            # The weld's thermal clock is the time since the suture last took up
            # shortening, on the same threshold that accumulates convergent_myr:
            # a still-deforming orogen keeps a hot, weak core, and only an idle
            # one cools into the strong welded plate interior.
            if weld_enabled(s) and convergence > .02: row['last_convergent_myr'] = float(s.t)
            row['updated_myr'] = float(s.t)
            # Persistent per-face maturity advances only where a current
            # closing front intersects that face. Sub-face participation is
            # area averaged; a distant opening-only face gets no new suture.
            exposure_rate = np.zeros(len(s.mass))
            for front in fronts:
                closing = max(-front['normal_speed_km_myr'], 0.)
                if not closing: continue
                indices = np.asarray(front['overlap_indices'], int)
                for face in (first[indices], second[indices]):
                    np.add.at(exposure_rate, face, closing*area[indices])
            exposure_rate /= surface['area_km2']
            touched = exposure_rate > 0.
            old_suture = s.parcel_collision_suture[touched]
            s.parcel_collision_suture[touched] = old_suture + (1.-old_suture)*(-np.expm1(-exposure_rate[touched]*dt/100.))
            s.suture[touched] = np.maximum(s.suture[touched], s.parcel_collision_suture[touched])
    # Every row anneals, including the quiet ones the pair loop never visits --
    # a suture with no overlap left is the most idle suture there is. Stamped
    # once per epoch so a second refresh in the same step (the dt = 0 one that
    # follows consolidation) cannot advance the clock twice.
    if weld_enabled(s) and dt > 0. and getattr(s, '_weld_annealed_myr', None) != float(s.t):
        _anneal_welds(s, dt); s._weld_annealed_myr = float(s.t)
    s.parcel_exposed_fraction = _exposure(s, overlap)
    if dt > 0. and getattr(s, '_burial_updated_myr', None) != float(s.t):
        s.parcel_burial_myr += (1.-s.parcel_exposed_fraction)*dt
        s._burial_updated_myr = float(s.t)
    stack = 0.
    if len(first):
        stack = float(np.max(s.structure['thickness_km'][first]+s.structure['thickness_km'][second]))
    s.collision_diagnostics = dict(model='persistent underthrust contacts and bounded geometric shortening',
        active_contacts=sum(row['state'] == 'active' for row in s.collision_contacts),
        accreted_contacts=sum(row['state'] == 'accreted' for row in s.collision_contacts),
        retained_contacts=len(s.collision_contacts),
        active_local_fronts=sum(len(row.get('local_fronts', ())) for row in s.collision_contacts if row['state'] == 'active'),
        local_front_geometry='positive-length connected spherical clipped patches; normals from free sheet edges',
        sutured_contacts=sum(row['suture_strength'] > .01 for row in s.collision_contacts),
        buried_faces=int(np.count_nonzero(s.parcel_exposed_fraction < 1.-1e-12)),
        contact_pairs=len(first), pair_overlap_area_km2=float(area.sum()),
        buried_area_quadrature_km2=float(surface['area_km2']@(1.-s.parcel_exposed_fraction)),
        exposure_quadrature_points_per_face=4, maximum_pair_stack_thickness_km=stack,
        reference_area_km2=float(s.mass.sum()), physical_column_volume_km3=float(surface['area_km2']@s.structure['thickness_km']),
        pair_overlap_is_not_union=True, contact_policy='retain every independent material face and column',
        geometric_shortening_volume_residual_km3=getattr(s, 'material_geometry_volume_residual_km3', 0.),
        column_budget=deepcopy(getattr(s, 'material_column_budget', {})))
    if weld_enabled(s):
        welds = weld_rows(s)
        s.collision_diagnostics.update(suture_weld_version=SUTURE_WELD_VERSION, yielding_sutures=len(welds),
            minimum_weld_cohesion=min((float(row['weld_cohesion']) for row, _, _ in welds), default=1.),
            maximum_weld_capacity_factor=max((weld_capacity_factor(row, s.t) for row, _, _ in welds), default=0.),
            weld_scope='per metre of suture trace; independent of underthrust width and column thickness')


def resist_motion(s, dt):
    """Measure the contact fronts a suture presents. Never touch omega.

    THE WELD NO LONGER LIVES HERE. It is a plastic element that plate_balance
    assembles on each pair's overlap-area rate, inside the same inertia-free
    force balance that solves omega from scratch every step; weld_rows and
    weld_capacity_factor above are the state it reads. Relaxing an
    already-solved rotation afterwards, as this function used to, was a
    velocity law wearing a force law's clothes: it could only ever damp
    convergence (the penalty was one-sided on min(n.x, 0)), so a suture that
    had finished closing offered no resistance at all to being pulled apart
    again, and its 20%-per-step ceiling and 1,400 km arrest width were free
    parameters carrying no force.

    What remains is the front census the same solve used to build, kept as a
    diagnostic so the recorded quantities and their validation are unchanged
    for readers of collision_resistance_diagnostics. The proxy fields it can no
    longer measure report the no-op honestly: zero removed, zero residual.
    """
    if not np.isscalar(dt) or not np.isfinite(dt) or dt <= 0:
        raise ValueError('Collision resistance timestep must be finite and positive.')
    ensure_fields(s)
    area=np.bincount(s.parcel_plate,weights=s.material_surface['area_km2'],minlength=len(s.omega))
    omega=np.asarray(s.omega,float)
    if not np.isfinite(area).all() or np.any(area<0) or not np.isfinite(omega).all():
        raise ValueError('Collision motion needs finite nonnegative material areas and finite angular velocities.')
    groups=set();input_contacts=0
    for row in iter_motion_fronts(s.collision_contacts):
        if row.get('state')!='active':continue
        if not np.isfinite(row.get('overlap_area_km2',0.)):
            raise ValueError('Collision motion requires finite overlap areas.')
        if row.get('overlap_area_km2',0.)<=0.:continue
        p,q=int(row['top_owner']),int(row['under_owner'])
        if p==q or min(area[p],area[q])<=0.:continue
        center=np.asarray(row['center'],float);normal=np.asarray(row['normal'],float)
        direction=np.cross(center,normal)
        if direction.shape!=(3,) or not np.isfinite(direction).all():
            raise ValueError('Collision motion front requires a finite three-dimensional direction.')
        if np.linalg.norm(direction)<1e-12:continue
        if p>q:p,q=q,p;normal=-normal
        # Exact geometric identity only. Distinct fronts are not rounded or
        # coalesced by angle; copied subdivision records normalize identically.
        groups.add((p,q,*map(float,center),*map(float,normal)))
        input_contacts+=1
    momentum=(area@omega).tolist()
    energy=float(np.sum(area[:,None]*omega**2))
    s.collision_resistance_diagnostics=dict(contacts=0,
        input_active_contacts=input_contacts,distinct_motion_fronts=len(groups),
        rotational_energy_proxy_before=energy,rotational_energy_proxy_after=energy,
        rotational_energy_proxy_removed=0.,
        angular_motion_proxy_before=momentum,angular_motion_proxy_after=momentum,
        angular_motion_proxy_residual_norm=0.,
        angular_motion_proxy_contribution_scale=float(np.sum(area*np.linalg.norm(omega,axis=1))),
        angular_motion_proxy_tolerance=0.,stationarity_residual_norm=0.,
        stationarity_tolerance=0.,newton_iterations=0,line_search_backtracks=0,
        maximum_damping_per_contact=0.,maximum_isolated_front_fraction=0.,
        relaxation_myr=0.,multiple_contacts_apply_sequentially=False,
        area_weighted_angular_momentum_conserved=True,
        model='diagnostics only; the suture is a plastic element in plate_balance on the overlap-area rate',
        proxy_limit='Material-area-weighted angular motion and omega-squared are numerical proxies, not physical momentum or thermodynamic energy.',
        cap_scope='no cap and no relaxation: this function no longer changes omega.')


def iter_motion_fronts(rows):
    """Expand current fronts, retaining a legacy-row fallback.

    A quiet parent or an explicitly empty current list never reactivates stale
    geometry. Local indices are epoch diagnostics, not persistent episodes.
    """
    for row in rows:
        if 'local_fronts' not in row:
            yield row
        elif row.get('state') == 'active':
            for front in row['local_fronts']:
                yield dict(row, **front)


def deformation_boundaries(s, boundaries):
    """Resolve sub-control-cell contact fronts through the existing solver."""
    active = []
    for row in iter_motion_fronts(getattr(s, 'collision_contacts', ())):
        if row.get('state') != 'active' or np.linalg.norm(row.get('normal', [])) <= .5:
            continue
        # Forces and coupled resistance can change omega after the recorded
        # contact epoch. Admission uses the actual impending material motion;
        # saved historical diagnostics retain their original measured speed.
        speed = float(np.dot(np.cross(s.omega[row['under_owner']]-s.omega[row['top_owner']],
            row['center']), row['normal'])*s.material_surface['radius_km'])
        if speed < -.02:
            active.append(row)
    if not active: return boundaries
    result = {key: np.asarray(value).copy() for key, value in boundaries.items()}
    existing = len(result['bl'])
    extra = dict(bmid=np.array([r['center'] for r in active]), bn=np.array([r['normal'] for r in active]),
        bl=np.array([max(1., min(r.get('length_km', 2.*np.sqrt(r['overlap_area_km2'])), 2000.)) for r in active]),
        bp=np.array([r['top_owner'] for r in active], int), bq=np.array([r['under_owner'] for r in active], int))
    for key in extra: result[key] = np.concatenate((result[key], extra[key]))
    result['valid'] = np.r_[result.get('valid', np.ones(existing, bool)), np.ones(len(active), bool)]
    return result


def _empty_consolidation(enabled=True):
    return dict(version=SUTURE_WELD_VERSION, enabled=bool(enabled), moved=False,
        faces=0, area_km2=0., qualifying_faces=0, qualifying_area_km2=0.,
        whole_component_faces=0, rim_withheld_area_km2=0., cap_withheld_area_km2=0.,
        connectivity_released_faces=0, contacts=[],
        exposed_fraction_threshold=CONSOLIDATION_EXPOSED_FRACTION, suture_threshold=CONSOLIDATION_SUTURE,
        rim_area_km2=CONSOLIDATION_RIM_AREA_KM2, step_area_cap_km2=CONSOLIDATION_STEP_AREA_KM2,
        model='welded, buried under-sheet material joins the plate above it; control ownership untouched',
        control_ownership_changed=False)


def _plan_consolidation(s):
    """Choose material transfers and inventory them without changing ownership.

    A suture that has finished welding is one plate mechanically. The buried
    lower sheet has no boundary of its own left to be driven from -- it is
    roofed by the sheet above and holds no ridge, trench or transform -- so
    leaving it on the donor plate lets the force balance drag two rigid bodies
    apart through crust that is already metamorphically joined. Consolidation
    is the material half of the weld: the plastic element in plate_balance
    resists separating them, this stops them being counted as separable at all.

    It moves MATERIAL ownership only. s.plate, the control-cell ownership, is
    taken by native_engine._rasterize from the exposed winner of each cell, and
    a face that qualifies here is by definition not exposed, so no control area
    changes hands and no plate is created or retired. Nothing here touches
    thickness, area, relief or the overlap ledger, so the column volume and the
    gravitational cross term are bitwise unchanged across the transfer; the
    daughter-sheet split that follows in ensure_fields clones the contact rows
    and keeps the by-sheet and by-contact provenance of every ledger that keys
    on them.

    Guards, applied in this order and all reported in the returned inventory:
      1. one contact may claim a face; rows are visited in ascending contact ID
      2. no connected body of the donor sheet may move whole
      3. a rim of at least CONSOLIDATION_RIM_AREA_KM2 of the donor sheet stays
      4. at most CONSOLIDATION_STEP_AREA_KM2 moves per contact per step
      5. the retained donor stays exactly as many connected bodies as before
    Trimming for 3 and 4 keeps the most deeply buried faces first: burial is
    what earns consolidation, and the shallow rim is where the pair's driver is.
    """
    report = _empty_consolidation()
    overlap = getattr(s, '_collision_overlap', None)
    if overlap is None or not len(overlap['first']):
        return np.empty(0, int), np.empty(0, int), report
    faces = s.material_surface['faces']
    face_area = np.asarray(s.material_surface['area_km2'], float)
    sheets = s.parcel_collision_sheet
    first, second = overlap['first'], overlap['second']
    active = getattr(s, 'active', None)
    claimed = np.zeros(len(sheets), bool)
    moved_faces, moved_owner = [], []
    for row in sorted(s.collision_contacts, key=lambda record: int(record['id'])):
        if row.get('state') != 'active': continue
        top_owner, under_owner = int(row.get('top_owner', -1)), int(row.get('under_owner', -1))
        if top_owner < 0 or under_owner < 0 or top_owner == under_owner: continue
        if active is not None and not (bool(active[top_owner]) and bool(active[under_owner])): continue
        top, under = int(row['top_sheet']), int(row['under_sheet'])
        take = np.flatnonzero(((sheets[first] == top) & (sheets[second] == under))
                              | ((sheets[first] == under) & (sheets[second] == top)))
        if not len(take): continue
        contacted = np.unique(np.r_[first[take][sheets[first[take]] == under],
                                    second[take][sheets[second[take]] == under]])
        candidate = contacted[(s.parcel_exposed_fraction[contacted] < CONSOLIDATION_EXPOSED_FRACTION)
                              & (s.parcel_collision_suture[contacted] >= CONSOLIDATION_SUTURE)
                              & ~claimed[contacted]]
        record = dict(contact_id=int(row['id']), top_sheet=top, under_sheet=under,
            top_owner=top_owner, under_owner=under_owner, qualifying_faces=int(len(candidate)),
            qualifying_area_km2=float(face_area[candidate].sum()), faces=0, area_km2=0.,
            whole_component_faces=0, rim_withheld_area_km2=0., cap_withheld_area_km2=0.,
            connectivity_released_faces=0)
        report['qualifying_faces'] += record['qualifying_faces']
        report['qualifying_area_km2'] += record['qualifying_area_km2']
        if not len(candidate):
            report['contacts'].append(record); continue
        donor = np.flatnonzero((sheets == under) & (s.parcel_plate == under_owner))
        blocked = _whole_bodies(faces, donor, candidate)
        if len(blocked):
            record['whole_component_faces'] = int(np.count_nonzero(np.isin(candidate, blocked)))
            candidate = candidate[~np.isin(candidate, blocked)]
        if len(candidate):
            budget = min(CONSOLIDATION_STEP_AREA_KM2,
                         max(0., float(face_area[donor].sum())-CONSOLIDATION_RIM_AREA_KM2))
            order = np.lexsort((s.parcel_patch[candidate], -s.parcel_burial_myr[candidate],
                                s.parcel_exposed_fraction[candidate]))
            ranked = candidate[order]
            fits = np.cumsum(face_area[ranked]) <= budget
            withheld = float(face_area[ranked[~fits]].sum())
            if budget < CONSOLIDATION_STEP_AREA_KM2: record['rim_withheld_area_km2'] = withheld
            else: record['cap_withheld_area_km2'] = withheld
            candidate = np.sort(ranked[fits])
        if len(candidate):
            candidate, released = _restore_connectivity(faces, donor, candidate)
            record['connectivity_released_faces'] = int(released)
        if len(candidate) and _would_disconnect(faces, donor, candidate):
            raise RuntimeError('Consolidation would disconnect the donor material sheet.')
        if len(candidate):
            claimed[candidate] = True
            moved_faces.append(candidate); moved_owner.append(np.full(len(candidate), top_owner))
            record.update(faces=int(len(candidate)), area_km2=float(face_area[candidate].sum()))
        for key in ('faces', 'area_km2', 'whole_component_faces', 'rim_withheld_area_km2',
                    'cap_withheld_area_km2', 'connectivity_released_faces'):
            report[key] += record[key]
        report['contacts'].append(record)
    if not moved_faces:
        return np.empty(0, int), np.empty(0, int), report
    return np.concatenate(moved_faces), np.concatenate(moved_owner), report


def _commit_consolidation(s, moving, owners, report):
    """Prepare all aligned owners before publishing the material transfer."""
    # Entered material carries a moving hinge and buoyancy load against its old
    # owner. Relabeling without an energy handoff would silently discard work.
    import entry_regions
    if entry_regions.enabled(s):
        entry_regions.ensure_fields(s)
        membership = np.asarray(getattr(s, entry_regions.FIELD))
        if np.any(membership[moving] > 0):
            raise ValueError('Consolidation of entry-bearing material requires an explicit energy and owner handoff.')

    next_owners = s.parcel_plate.copy()
    next_owners[moving] = owners.astype(next_owners.dtype)
    # reassign_owners prepares replacement arrays and updates only its mapping.
    # Stage in a separate mapping so an error leaves the live surface untouched.
    next_surface = s.material_surface.copy()
    surface_report = material_surface.reassign_owners(next_surface, next_owners)
    next_trace = None
    if hasattr(s, 'trace_plate') and len(getattr(s, 'trace_patch', ())):
        next_trace = s.trace_plate.copy()
        order = np.argsort(s.parcel_patch[moving]); patches = s.parcel_patch[moving][order]
        at = np.searchsorted(patches, s.trace_patch)
        follows = (at < len(patches)) & (patches[np.minimum(at, len(patches)-1)] == s.trace_patch)
        if np.any(follows):
            next_trace[follows] = owners[order[at[follows]]].astype(next_trace.dtype)

    # Keep the array and surface mapping identities: other simulation components
    # may already hold references to them. All derived arrays are ready here.
    s.material_surface.update(next_surface)
    s.parcel_plate[:] = next_owners
    if next_trace is not None:
        s.trace_plate[:] = next_trace
    report['surface'] = surface_report
    report['moved'] = True


def consolidate(s):
    """Re-parent welded, buried under-sheet crust onto the plate above it."""
    if not weld_enabled(s):
        s.collision_consolidation_diagnostics = _empty_consolidation(enabled=False)
        return s.collision_consolidation_diagnostics
    ensure_fields(s)
    moving, owners, report = _plan_consolidation(s)
    if len(moving):
        _commit_consolidation(s, moving, owners, report)
    s.collision_consolidation_diagnostics = report
    return report


def erosion_fraction(s, *, trace=False):
    """Current exposure gates rock denudation while retaining buried heat/magma."""
    if not hasattr(s, 'parcel_exposed_fraction'): return None
    if not trace: return s.parcel_exposed_fraction
    hits = material_surface.sample_surface(s.material_surface, s.trace_xyz)
    q, f = hits['query_index'], hits['face_index']
    allowed = eligible_hits(q, f, s.parcel_collision_sheet, s.collision_contacts)
    order = np.argsort(s.parcel_patch)
    at = np.searchsorted(s.parcel_patch[order], s.trace_patch)
    own = s.parcel_collision_sheet[order[at]]
    result = np.zeros(len(s.trace_patch))
    np.maximum.at(result, q, (allowed & (s.parcel_collision_sheet[f] == own[q])).astype(float))
    return result


def snapshot_fields(s):
    ensure_fields(s)
    result = dict(collision_contact_version=VERSION, material_collision_sheet=s.parcel_collision_sheet.copy(),
        material_exposed_fraction=s.parcel_exposed_fraction.copy(), material_burial_myr=s.parcel_burial_myr.copy(),
        material_collision_suture=s.parcel_collision_suture.copy(), collision_contacts=deepcopy(s.collision_contacts),
        collision_diagnostics=deepcopy(getattr(s, 'collision_diagnostics', {})),
        collision_resistance_diagnostics=deepcopy(getattr(s, 'collision_resistance_diagnostics', {})))
    # The weld's own state is per contact, so it rides inside the contact rows
    # that are already saved; only the flag that makes them required is new.
    if weld_enabled(s):
        result.update(suture_weld_version=SUTURE_WELD_VERSION,
            collision_consolidation_diagnostics=deepcopy(getattr(s, 'collision_consolidation_diagnostics',
                                                                 _empty_consolidation())))
        if getattr(s, 'suture_weld_coordinate_version', 0):
            result['suture_weld_coordinate_version'] = s.suture_weld_coordinate_version
            result['suture_weld_coordinate_migration'] = deepcopy(getattr(s, 'suture_weld_coordinate_migration', {}))
    import collision_interface
    result.update(collision_interface.snapshot(s))
    return result


def _validate_weld_frame(frame, contacts):
    """Per-contact weld state, on the same require-the-version rule as the rest.

    A version-zero frame predates the weld and must carry none of its state; a
    version-one frame must carry all of it, finite, with cohesion inside the
    band the anneal can reach and the convergence epoch not in the future.
    """
    version = frame.get('suture_weld_version', 0)
    coordinate_version = frame.get('suture_weld_coordinate_version', 0)
    if (isinstance(coordinate_version, (bool, np.bool_)) or coordinate_version not in (0, 1)
            or (coordinate_version and not version)):
        raise ValueError('Unsupported or unversioned local weld coordinates.')
    if version not in (0, SUTURE_WELD_VERSION): raise ValueError('Unsupported suture weld version.')
    carried = any(field in row for row in contacts for field in WELD_ROW_FIELDS)
    if not version:
        if carried or 'collision_consolidation_diagnostics' in frame:
            raise ValueError('Suture weld contact state requires its version.')
        return
    epoch = float(frame.get('time_myr', 0.))
    for row in contacts:
        for field in WELD_ROW_FIELDS:
            value = row.get(field)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
                raise ValueError(f'Invalid suture weld contact field {field}.')
        if not np.isfinite(row['last_convergent_myr']) or row['last_convergent_myr'] > epoch+1e-9:
            raise ValueError('A suture cannot last have converged after the saved epoch.')
        if not WELD_COHESION_FLOOR-1e-12 <= row['weld_cohesion'] <= 1.+1e-12:
            raise ValueError('Suture weld cohesion must lie between its inherited-weakness floor and one.')


def validate_frame(frame):
    import collision_interface
    collision_interface.validate_frame(frame)
    version = frame.get('collision_contact_version', 0)
    if version not in (0, VERSION): raise ValueError('Unsupported collision contact version.')
    if not version:
        if set(ARRAY_FIELDS).intersection(frame) or 'collision_contacts' in frame:
            raise ValueError('Collision contact fields require their version.')
        if frame.get('suture_weld_version', 0) or frame.get('suture_weld_coordinate_version', 0):
            raise ValueError('The suture weld requires persistent contact identity.')
        return
    count = len(frame['material_faces'])
    if frame.get('surface_reconstruction_version') != 1 or frame.get('arc_surface_version') != 2:
        raise ValueError('Persistent contacts require continuous version-two native surfaces.')
    for name, dtype in ARRAY_FIELDS.items():
        value = np.asarray(frame.get(name, []))
        if value.shape != (count,) or not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError(f'Invalid collision history field {name}.')
        if dtype == np.int64 and (not np.issubdtype(value.dtype, np.integer) or np.any(value == 0)):
            raise ValueError('Collision sheet IDs must be positive integers.')
        if name in ('material_exposed_fraction', 'material_collision_suture') and np.any(value > 1.):
            raise ValueError(f'{name} must be a fraction.')
    contacts = frame.get('collision_contacts', [])
    if not isinstance(contacts, list):
        raise ValueError('Collision contacts must be a list of records.')
    seen = set(); rows = []
    for row in contacts:
        if not isinstance(row, dict):
            raise ValueError('Collision contacts must be mapping records.')
        top, under = row.get('top_sheet'), row.get('under_sheet')
        if (not isinstance(top, int) or not isinstance(under, int) or min(top, under) <= 0
                or top == under or tuple(sorted((top, under))) in seen or _would_cycle(rows, top, under)):
            raise ValueError('Collision relationships must be unique, directed and acyclic.')
        seen.add(tuple(sorted((top, under)))); rows.append(row)
    _validate_weld_frame(frame, contacts)

    # Legacy contact rows had no local geometry schema. Quiet legacy records
    # can also receive an empty current list during refresh; neither case
    # retroactively requires fields that were never saved in those histories.
    versioned = []
    for row in contacts:
        local_version = row.get('local_fronts_version')
        if 'local_fronts_version' not in row:
            if 'local_fronts' in row and row['local_fronts'] != []:
                raise ValueError('Nonempty collision local fronts require their version.')
            continue
        if isinstance(local_version, (bool, np.bool_)) or not isinstance(local_version, (int, np.integer)) or local_version != 1:
            raise ValueError('Unsupported collision local-front version.')
        if not isinstance(row.get('local_fronts'), list):
            raise ValueError('Collision local fronts must be a list of mapping records.')
        versioned.append(row)
    if not versioned:
        return

    def integer(value, name, minimum=0):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f'Invalid collision local-front integer {name}.')
        return int(value)

    def scalar(value, name, minimum=None):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value) or (minimum is not None and value < minimum)):
            raise ValueError(f'Invalid collision local-front scalar {name}.')
        return float(value)

    def vector(value, name):
        try:
            result = np.asarray(value, float)
        except (ValueError, TypeError) as exc:
            raise ValueError(f'Invalid collision local-front vector {name}.') from exc
        if result.shape != (3,) or not np.isfinite(result).all():
            raise ValueError(f'Invalid collision local-front vector {name}.')
        return result

    epoch = scalar(frame.get('time_myr'), 'frame time_myr', 0.)
    epoch_tolerance = max(1e-9, abs(epoch)*1e-12)
    owners = np.asarray(frame.get('material_owner', []))
    if owners.shape != (count,) or not np.issubdtype(owners.dtype, np.integer) or np.any(owners < 0):
        raise ValueError('Collision local fronts require aligned material owner IDs.')
    sheets = np.asarray(frame['material_collision_sheet'])
    owner_pairs = np.unique(np.column_stack((sheets, owners)), axis=0)
    owner_by_sheet = {}
    for sheet, owner in owner_pairs:
        owner_by_sheet.setdefault(int(sheet), set()).add(int(owner))
    diagnostics = frame.get('collision_diagnostics', {})
    if not isinstance(diagnostics, dict):
        raise ValueError('Collision diagnostics must be a mapping.')
    pair_count = diagnostics.get('contact_pairs')
    if pair_count is not None:
        pair_count = integer(pair_count, 'contact_pairs')
        if pair_count > count*(count-1)//2:
            raise ValueError('Collision overlap count exceeds the possible material face pairs.')
    all_indices = set(); contact_ids = set()
    for row in versioned:
        contact_id = integer(row.get('id'), 'parent contact ID', 1)
        if contact_id in contact_ids:
            raise ValueError('Versioned collision parent contact IDs must be unique.')
        contact_ids.add(contact_id)
        state = row.get('state')
        if state not in ('active', 'accreted', 'quiet'):
            raise ValueError('Invalid collision local-front parent state.')
        top_owner = integer(row.get('top_owner'), 'top owner')
        under_owner = integer(row.get('under_owner'), 'under owner')
        parent_area = scalar(row.get('overlap_area_km2'), 'parent overlap area', 0.)
        local = row['local_fronts']
        if state == 'quiet':
            if local or parent_area != 0.:
                raise ValueError('Quiet collision contacts cannot retain current local fronts or overlap area.')
            continue
        if (not local or (state == 'accreted') != (top_owner == under_owner)
                or owner_by_sheet.get(row['top_sheet']) != {top_owner}
                or owner_by_sheet.get(row['under_sheet']) != {under_owner}):
            raise ValueError('Current collision fronts must agree with their parent and material owners.')
        if abs(scalar(row.get('last_seen_myr'), 'parent last_seen_myr', 0.)-epoch) > epoch_tolerance:
            raise ValueError('Current collision parent geometry must match the saved epoch.')
        local_ids = set(); local_areas = []
        for front in local:
            if not isinstance(front, dict):
                raise ValueError('Collision local fronts must be mapping records.')
            if (integer(front.get('parent_contact_id'), 'parent_contact_id', 1) != contact_id
                    or integer(front.get('top_owner'), 'front top owner') != top_owner
                    or integer(front.get('under_owner'), 'front under owner') != under_owner
                    or front.get('state') != state):
                raise ValueError('Collision local-front IDs, owners and state must agree with the parent.')
            for key in ('top_sheet', 'under_sheet', 'top_plate_uid', 'under_plate_uid'):
                if key in front and front[key] != row.get(key):
                    raise ValueError('Collision local-front inherited identity must agree with the parent.')
            index = integer(front.get('front_index'), 'front_index')
            if index >= len(local) or index in local_ids:
                raise ValueError('Collision local-front indices must uniquely address their current list.')
            local_ids.add(index)
            if abs(scalar(front.get('geometry_epoch_myr'), 'geometry_epoch_myr', 0.)-epoch) > epoch_tolerance:
                raise ValueError('Collision local-front geometry must match the saved epoch.')
            area = scalar(front.get('overlap_area_km2'), 'overlap area', 0.)
            if area == 0.:
                raise ValueError('A current collision local front requires positive overlap area.')
            local_areas.append(area)
            length = scalar(front.get('length_km'), 'length_km', 0.)
            scalar(front.get('footprint_radius_km'), 'footprint_radius_km', 0.)
            for key in ('upper_free_edge_length_km', 'lower_free_edge_length_km'):
                if key in front: scalar(front[key], key, 0.)
            speed = scalar(front.get('normal_speed_km_myr'), 'normal speed')
            center, normal = vector(front.get('center'), 'center'), vector(front.get('normal'), 'normal')
            if abs(np.linalg.norm(center)-1.) > 1e-7:
                raise ValueError('Collision local-front center must be a unit spherical vector.')
            resolved = front.get('direction_resolved')
            if not isinstance(resolved, (bool, np.bool_)):
                raise ValueError('Collision local-front direction_resolved must be Boolean.')
            if resolved:
                if abs(np.linalg.norm(normal)-1.) > 1e-7 or abs(float(center@normal)) > 1e-7 or length <= 0.:
                    raise ValueError('Resolved collision local-front normal must be a unit tangent with positive length.')
            elif np.linalg.norm(normal) > 1e-12 or length != 0. or speed != 0.:
                raise ValueError('Unresolved collision local fronts require zero normal, length and speed.')
            indices = front.get('overlap_indices')
            if not isinstance(indices, list) or not indices:
                raise ValueError('Collision local fronts require a nonempty list of overlap indices.')
            for value in indices:
                value = integer(value, 'overlap index')
                bound = pair_count if pair_count is not None else count*(count-1)//2
                if value >= bound or value in all_indices:
                    raise ValueError('Collision local-front overlap indices must be in bounds and disjoint.')
                all_indices.add(value)
        if abs(math.fsum(local_areas)-parent_area) > max(1e-6, parent_area*1e-10):
            raise ValueError('Collision local-front areas must partition their parent overlap area.')
    # Complete index coverage is knowable only when every current parent uses
    # this schema. Legacy active rows can retain their older representation.
    versioned_identity = {id(row) for row in versioned}
    if (pair_count is not None and all(row.get('state') == 'quiet' or id(row) in versioned_identity for row in contacts)
            and len(all_indices) != pair_count):
        raise ValueError('Current collision local fronts must partition all recorded overlap pairs.')
