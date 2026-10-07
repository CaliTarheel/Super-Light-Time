"""Opt-in research transactions around real native accretion and topology.

No production callsite uses this module. An authored intact witness is an input
assumption, not an attachment admission or release law. See the adjacent contract.
"""
from contextlib import contextmanager
from copy import deepcopy

import numpy as np

import collision_contacts
from convex_partition import Edge
import local_accretion
import material_surface
import native_topology
from . import attachment_transaction as finite


STORE_FIELD = '_research_finite_attachment_store'


@contextmanager
def _atomic(s):
    # All native mutations act on a detached candidate, preserving external
    # aliases too on failure. deepcopy rebinds instance-bound fixture methods
    # to the candidate; native Simulation methods resolve on its class.
    candidate = deepcopy(s)
    yield candidate
    # Build the entire publishable state before changing the original object.
    # Rebind instance-bound helpers to s, not the temporary candidate.
    accepted = deepcopy(vars(candidate), {id(candidate): s})
    vars(s).clear()
    vars(s).update(accepted)


def leaves(s):
    """Read existing material addresses; never initialize or repair ancestry."""
    lineage = s.material_lineage
    ids = np.asarray(s.parcel_patch)
    if not np.array_equal(ids, lineage['face_ids']):
        raise ValueError('Material lineage is not aligned with current face IDs.')
    if not np.array_equal(ids, s.material_surface['face_id']):
        raise ValueError('Material surface is not aligned with current face IDs.')
    return dict(face_id=ids.copy(), root_id=np.asarray(lineage['root_id']).copy(),
                reference_corners=np.asarray(lineage['reference_corners']).copy(),
                owner_uid=np.asarray(s.plate_uid)[s.parcel_plate].copy())


def _store(s, current):
    # Absence does not silently create an empty ledger for a saved world.
    if not hasattr(s, STORE_FIELD):
        raise ValueError('An explicit research attachment store is required.')
    store = deepcopy(getattr(s, STORE_FIELD))
    finite.prepare_transaction(store, current, current)
    return store


def _payload(s):
    """Quantities an instantaneous owner transaction must preserve exactly."""
    mesh = s.material_surface
    return dict(mass=s.mass.copy(), pos=s.pos.copy(),
                material_lineage=deepcopy(s.material_lineage),
                triangles=mesh['vertices'][mesh['faces']].copy(),
                reference_area=mesh['reference_area_km2'].copy(),
                structure=deepcopy(s.structure), trace_structure=deepcopy(s.trace_structure),
                trace_xyz=s.trace_xyz.copy(), trace_patch=s.trace_patch.copy())


def _validate_payload(s, before):
    if finite._canonical(_payload(s)) != finite._canonical(before):
        raise ValueError('Native owner transaction changed conserved material/column payload.')


def _witness(s, plan, authored):
    expected = {'bond_id', 'source_face_id', 'receiver_face_id', 'world_corners', 'asserted_intact'}
    if not isinstance(authored, dict) or set(authored) != expected or authored['asserted_intact'] is not True:
        raise ValueError('An explicitly authored intact finite receiving witness is required.')
    for key in ('bond_id', 'source_face_id', 'receiver_face_id'):
        if isinstance(authored[key], (bool, np.bool_)) or not isinstance(authored[key], (int, np.integer)) or authored[key] < 0:
            raise ValueError('Witness IDs must be nonnegative integers.')
    ids = np.asarray(s.parcel_patch)
    indices = []
    for key in ('source_face_id', 'receiver_face_id'):
        found = np.flatnonzero(ids == authored[key])
        if len(found) != 1:
            raise ValueError('Witness must name one actual current material face per endpoint.')
        indices.append(int(found[0]))
    a, b = indices
    if a not in plan['parcel_indices'] or ids[a] not in plan.get('contact_patch_ids', []):
        raise ValueError('Source witness is not an engaged transferring face.')
    if int(s.parcel_plate[b]) != int(plan['target']):
        raise ValueError('Receiving witness does not belong to the accepted target.')
    overlap = s._collision_overlap
    pair = (((overlap['first'] == a) & (overlap['second'] == b)) |
            ((overlap['first'] == b) & (overlap['second'] == a)))
    if not np.any(pair & (overlap['area_km2'] > 0.)):
        raise ValueError('Witness endpoints do not share an actual positive contact overlap.')
    contact = next((r for r in s.collision_contacts if r['id'] == plan.get('collision_contact_id')), None)
    sheets = s.parcel_collision_sheet
    if contact is None or {int(sheets[a]), int(sheets[b])} != {contact['top_sheet'], contact['under_sheet']}:
        raise ValueError('Witness does not belong to the accepted persistent contact.')
    points = np.asarray(authored['world_corners'])
    # Check dtype before conversion; never discard imaginary coordinates or
    # serialize pointer-bearing object arrays as if they were physical data.
    if (points.dtype.kind not in 'iuf' or (points.dtype.kind == 'f' and points.dtype.itemsize > 8)
            or points.shape != (3, 3) or not np.isfinite(points).all()):
        raise ValueError('World witness must contain real finite binary64-compatible 3D coordinates.')
    points = points.astype(float)
    if not np.allclose(np.linalg.norm(points, axis=1), 1., rtol=0., atol=64*np.finfo(float).eps):
        raise ValueError('World witness corners must be unit directions.')
    if Edge(points[0], points[1]).distances(points[2:])[0] <= 0.:
        raise ValueError('World witness must have strictly positive oriented area.')
    mesh = s.material_surface
    triangles = mesh['vertices'][mesh['faces']]
    endpoints = []
    for index in indices:
        triangle = triangles[index]
        # Original-edge filtered exact signs decide strict geometric inclusion.
        # A poorly conditioned floating solve must never authorize an outside
        # point or erase a represented zero-width edge incidence.
        if (Edge(triangle[0], triangle[1]).distances(triangle[2:])[0] <= 0.
                or any(np.any(Edge(a, b).distances(points) <= 0.)
                       for a, b in zip(triangle, np.roll(triangle, -1, axis=0)))):
            raise ValueError('Witness must be a positive oriented triangle strictly inside both material faces.')
        local = np.linalg.solve(triangle.T, points.T).T
        reconstructed = local @ triangle
        arithmetic_scale = np.abs(local) @ np.abs(triangle) + np.abs(points)
        if (not np.isfinite(local).all() or not np.all(local > 0.)
                or not np.isfinite(reconstructed).all()
                or np.any(np.abs(reconstructed-points) > 64*np.finfo(float).eps*arithmetic_scale)):
            raise ValueError('Witness homogeneous solve is unresolved or fails reconstruction.')
        chart = local @ s.material_lineage['reference_corners'][index]
        endpoints.append(dict(root_id=int(s.material_lineage['root_id'][index]), corners=chart))
    return dict(bond_id=int(authored['bond_id']), creation_event_id=0, created_myr=float(s.t),
                source=endpoints[0], receiver=endpoints[1])


def accrete(s, dt, authored_witness):
    """Plan and apply at the caller's current epoch, atomically with one bond.

    dt advances native contact/loading histories, not simulation time. The
    caller explicitly asserts the new bond; this function only validates its
    finite material identity against the actual native plan/contact.
    """
    if (isinstance(dt, (bool, np.bool_)) or not np.isscalar(dt)
            or not isinstance(dt, (int, float, np.integer, np.floating))
            or not np.isfinite(dt) or dt <= 0.):
        raise ValueError('Positive finite contact-history interval required.')
    with _atomic(s) as s:
        current = leaves(s)
        store = _store(s, current)
        preserved = _payload(s)
        collision_contacts.refresh(s, dt)
        plans = local_accretion.plan_accretions(s, dt)
        if not plans:
            return dict(accepted=False, reason='native planner produced no plan')
        if len(plans) != 1:
            raise ValueError('Research transaction requires one unambiguous native plan.')
        plan = plans[0]
        proposed = deepcopy(current)
        proposed['owner_uid'][plan['parcel_indices']] = plan['target_uid']
        # The existing intact ledger is checked BEFORE witness construction
        # and BEFORE apply_accretion's first omega/owner mutation.
        staged = finite.prepare_transaction(store, current, proposed)
        bond = _witness(s, plan, authored_witness)
        authored_store = deepcopy(store)
        authored_store['bonds'].append(bond)
        finite.prepare_transaction(authored_store, proposed, proposed)
        finite.validate_prepared(staged, store, current, proposed)
        old_events = len(s.events)
        if not local_accretion.apply_accretion(s, plan):
            raise ValueError('Native accretion refused its freshly preflighted plan.')
        material_surface.reassign_owners(s.material_surface, s.parcel_plate)
        _validate_payload(s, preserved)
        if finite._canonical(leaves(s)) != finite._canonical(proposed):
            raise ValueError('Actual native owners differ from the preflighted proposal.')
        events = [e for e in s.events[old_events:] if e['type'] == 'accretion'
                  and e.get('details', {}).get('contact_id') == plan['contact_id']]
        if len(events) != 1 or not isinstance(events[0].get('id'), int):
            raise ValueError('Accepted docking must emit one identified native accretion event.')
        # Rehosting may emit events first. Bind the actual accepted event ID,
        # never a guessed counter; provisional zero is never committed.
        bond['creation_event_id'] = events[0]['id']
        completed = finite.prepare_transaction(authored_store, proposed, leaves(s))
        setattr(s, STORE_FIELD, completed['store'])
        return dict(accepted=True, event_id=events[0]['id'], contact_id=int(plan['contact_id']),
                    source_uid=int(plan['source_uid']), target_uid=int(plan['target_uid']),
                    bond_id=bond['bond_id'], incidence=completed['proposed_incidence'])


def split_continent(s, parent_slot, chosen, loading=None):
    """Guard a real native split before identity allocation; no release law.

    The caller supplies the ordinary native cut proposal. Existing paired
    material may move together, but separating any intact portion is unsupported.
    """
    with _atomic(s) as s:
        current = leaves(s)
        store = _store(s, current)
        preserved = _payload(s)
        if chosen is None:
            return dict(accepted=False, reason='no native split proposal')
        side = np.asarray(chosen['parcel_side'])
        if side.dtype.kind != 'b' or side.shape != s.parcel_plate.shape:
            raise ValueError('Split requires a complete boolean material partition.')
        predicted_uid = int(s.next_plate_uid)
        if predicted_uid < 0 or predicted_uid in np.asarray(s.plate_uid):
            raise ValueError('Next native identity is not fresh.')
        proposed = deepcopy(current)
        proposed['owner_uid'][(s.parcel_plate == parent_slot) & side] = predicted_uid
        staged = finite.prepare_transaction(store, current, proposed)
        finite.validate_prepared(staged, store, current, proposed)
        if not native_topology.split_continent(s, parent_slot, chosen, loading):
            # Native rejection has no intended accepted changes. Restoring even
            # a future partially mutating rejection keeps this boundary atomic.
            raise ValueError('Native topology refused the preflighted split.')
        _validate_payload(s, preserved)
        if finite._canonical(leaves(s)) != finite._canonical(proposed):
            raise ValueError('Actual native split differs from the preflighted material partition.')
        completed = finite.prepare_transaction(store, current, leaves(s))
        setattr(s, STORE_FIELD, completed['store'])
        return dict(accepted=True, new_plate_uid=predicted_uid, incidence=completed['proposed_incidence'])
