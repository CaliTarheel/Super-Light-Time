"""Real native transaction experiments, not admission/rupture physics tests."""
from copy import deepcopy
import hashlib
from pathlib import Path
import pickle
import tempfile
from types import MethodType, FunctionType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import collision_contacts
import checkpoint
import crust_inventory
import dense_crust
import local_accretion
import material_surface
import mesh_coverage
import native_topology
import raster_engine
import structure_engine
from benchmarks.collision_architecture import attachment_transaction as finite
from benchmarks.collision_architecture import native_attachment_transaction as native
from tests.test_terrane_attachment import three_body_fixture
from tests.test_native_topology import world


def _freeze(value):
    """All reachable state, with arrays/RNG and opaque spatial caches included."""
    if isinstance(value, np.ndarray):
        return ('array', value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, np.random.Generator):
        return ('rng', _freeze(value.bit_generator.state))
    if isinstance(value, dict):
        return ('dict', tuple(sorted((repr(k), _freeze(v)) for k, v in value.items())))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(map(_freeze, value)))
    if isinstance(value, set):
        return ('set', tuple(sorted(map(repr, value))))
    if isinstance(value, MethodType):
        return ('bound', value.__func__.__module__, value.__func__.__qualname__)
    if isinstance(value, FunctionType):
        return ('function', value.__module__, value.__qualname__)
    if hasattr(value, '__dict__'):
        return (type(value).__name__, _freeze(vars(value)))
    return (type(value).__name__, pickle.dumps(value, protocol=5))


def _empty_store():
    return dict(version=finite.VERSION, scope=finite.SCOPE, bonds=[])


def _columns(s):
    """Populate actual native composition, retained phase, heat and return ledgers."""
    for column in (s.structure, s.trace_structure):
        if not crust_inventory.present(column):
            crust_inventory.initialize(column)
        if not dense_crust.present(column):
            dense_crust.initialize(column, 700.)
        dense_crust.advance(column, .5, 800., 10., 5., detachment_fraction=1.)
        dense_crust.validate(column)


def docking_world():
    s = three_body_fixture()
    n = len(s.mass)
    s.material_lineage = dict(face_ids=s.parcel_patch.copy(), root_id=s.parcel_patch.copy(),
        parent_id=np.full(n, -1, np.int64), reference_corners=np.tile(np.eye(3), (n, 1, 1)),
        level=np.zeros(n, np.int32))
    structure_engine.initialize_parcels(s)
    structure_engine.initialize_traces(s, np.arange(n))
    _columns(s)
    # These are actual real native event serialization and real RNG state.
    s.event_keys = set()
    s._record = MethodType(raster_engine.Simulation._record, s)
    s.rng = np.random.default_rng(918)
    s.material_adaptivity = dict(registry=[dict(parent_id=91, children=np.array([92, 93]))])
    s.next_patch_uid = 24
    setattr(s, native.STORE_FIELD, _empty_store())
    s.t = 102.
    return s


def witness(s, source=0, receiver=8):
    """Authored finite input, strictly interior to a real positive overlap."""
    triangles = s.material_surface['vertices'][s.material_surface['faces']]
    polygon = mesh_coverage.clip_triangle(triangles[source], triangles[receiver])
    center = polygon.sum(axis=0); center /= np.linalg.norm(center)
    points = .9*center+.1*polygon[:3]
    points /= np.linalg.norm(points, axis=1)[:, None]
    return dict(bond_id=1, source_face_id=int(s.parcel_patch[source]),
        receiver_face_id=int(s.parcel_patch[receiver]), world_corners=points, asserted_intact=True)


def split_world():
    s = world(kind=1)
    s.t = 125.; s.born[0] = 7.
    s.parcel_plate[:] = 0; s.trace_plate[:] = 0; s.plate[:] = 0
    s.active[:] = False; s.active[0] = True
    s.support[:] = 0.; s.support[0] = 1.
    s._sync_material(); s._boundaries()
    _columns(s)
    crack = SimpleNamespace(center=np.array([1., 0., 0.]), normal=np.array([0., 1., 0.]),
                            signed_distance=lambda points: points[:, 1])
    chosen = dict(crack=crack, parcel_side=s.pos[:, 1] > 0, trace_side=s.trace_xyz[:, 1] > 0,
                  grid_distance=s.xyz[:, 1], rotations=np.zeros((2, 3)))
    setattr(s, native.STORE_FIELD, _empty_store())
    return s, chosen


def authored_split_bond(s, a, b):
    # Explicit pre-existing paired material addresses, not inferred from world
    # proximity. This tests transaction consistency, not a prior docking event.
    corners = np.array([[.8, .1, .1], [.1, .8, .1], [.1, .1, .8]])
    charts = s.material_lineage['reference_corners']
    record = dict(bond_id=1, creation_event_id=1, created_myr=100.,
        source=dict(root_id=int(s.material_lineage['root_id'][a]), corners=corners@charts[a]),
        receiver=dict(root_id=int(s.material_lineage['root_id'][b]), corners=corners@charts[b]))
    setattr(s, native.STORE_FIELD, dict(version=1, scope=finite.SCOPE, bonds=[record]))


class NativeAttachmentTransactions(unittest.TestCase):
    def assertStateEqual(self, s, before):
        self.assertEqual(_freeze(vars(s)), before)

    def test_real_docking_matches_native_operators_and_preserves_phase_heat(self):
        s = docking_world(); control = deepcopy(s)
        before_payload = native._payload(s)
        authored = witness(s)
        collision_contacts.refresh(control, 2.)
        plan, = local_accretion.plan_accretions(control, 2.)
        self.assertTrue(local_accretion.apply_accretion(control, plan))
        material_surface.reassign_owners(control.material_surface, control.parcel_plate)
        result = native.accrete(s, 2., authored)
        self.assertTrue(result['accepted'])
        self.assertEqual(result['event_id'], 1)
        for name in vars(control):
            if name != native.STORE_FIELD:
                self.assertEqual(_freeze(getattr(s, name)), _freeze(getattr(control, name)), name)
        native._validate_payload(s, before_payload)
        for column in (s.structure, s.trace_structure):
            dense_crust.validate(column)
            self.assertGreater(float(column[dense_crust.DENSE].sum()), 0.)
            self.assertGreater(float(column[dense_crust.HEAT_RETURNED].sum()), 0.)
        store = getattr(s, native.STORE_FIELD)
        self.assertEqual(store['bonds'][0]['creation_event_id'], s.events[-1]['id'])
        self.assertIs(s._record.__self__, s)
        self.assertEqual(s.process_totals['accreted_km2'], s.mass[:8].sum())

    def test_real_reverse_plans_reject_before_apply_and_restore_planning_histories(self):
        s = docking_world(); native.accrete(s, 2., witness(s))
        for epoch in (104., 106., 108.):
            s.t = epoch
            before = _freeze(vars(s))
            aliases = (s.parcel_plate, s.omega, s.structure[dense_crust.HEAT], s.accretion_welding_state['loading'])
            alias_before = [x.copy() for x in aliases]
            with patch.object(local_accretion, 'apply_accretion', wraps=local_accretion.apply_accretion) as apply:
                with self.assertRaisesRegex(ValueError, 'owner cut'):
                    native.accrete(s, 2., None)
                apply.assert_not_called()
            self.assertStateEqual(s, before)
            for a, old in zip(aliases, alias_before):
                np.testing.assert_array_equal(a, old)
        self.assertEqual(len(s.events), 1)

    def test_exception_after_real_apply_leaves_original_and_external_aliases_untouched(self):
        s = docking_world(); before = _freeze(vars(s))
        aliases = (s.omega, s.parcel_plate, s.structure[dense_crust.HEAT], s.events)
        frozen_aliases = _freeze(aliases)
        actual = local_accretion.apply_accretion
        def apply_then_fail(candidate, plan):
            self.assertIsNot(candidate, s)
            self.assertIs(candidate._record.__self__, candidate)
            self.assertTrue(actual(candidate, plan))
            candidate.rng.random(3)
            candidate.material_adaptivity['registry'].append(dict(parent_id=999))
            candidate.next_patch_uid += 100
            candidate.structure[dense_crust.HEAT][0] += 1.
            raise RuntimeError('injected failure after actual native application')
        with patch.object(local_accretion, 'apply_accretion', side_effect=apply_then_fail):
            with self.assertRaisesRegex(RuntimeError, 'after actual'):
                native.accrete(s, 2., witness(s))
        self.assertStateEqual(s, before)
        self.assertEqual(_freeze(aliases), frozen_aliases)

    def test_postcondition_failure_after_real_apply_cannot_publish_corrupted_heat(self):
        s = docking_world(); before = _freeze(vars(s))
        actual = local_accretion.apply_accretion
        def corrupt(candidate, plan):
            result = actual(candidate, plan)
            candidate.structure[dense_crust.HEAT][0] += 1.
            return result
        with patch.object(local_accretion, 'apply_accretion', side_effect=corrupt):
            with self.assertRaisesRegex(ValueError, 'conserved material'):
                native.accrete(s, 2., witness(s))
        self.assertStateEqual(s, before)

    def test_invalid_or_missing_witness_never_reaches_owner_mutation(self):
        for defect in ('missing', 'remote', 'outside', 'line', 'object', 'complex', 'not_asserted'):
            with self.subTest(defect=defect):
                s = docking_world(); authored = witness(s)
                if defect == 'missing': authored = None
                elif defect == 'remote': authored['receiver_face_id'] = 16
                elif defect == 'outside': authored['world_corners'] *= -1.
                elif defect == 'line': authored['world_corners'][2] = authored['world_corners'][1]
                elif defect == 'object': authored['world_corners'] = authored['world_corners'].astype(object)
                elif defect == 'complex': authored['world_corners'] = authored['world_corners'].astype(complex)
                else: authored['asserted_intact'] = False
                before = _freeze(vars(s))
                with patch.object(local_accretion, 'apply_accretion') as apply:
                    with self.assertRaises(ValueError): native.accrete(s, 2., authored)
                    apply.assert_not_called()
                self.assertStateEqual(s, before)

    def test_missing_store_or_lineage_is_not_a_silent_historical_migration(self):
        for field in (native.STORE_FIELD, 'material_lineage'):
            s = docking_world(); delattr(s, field)
            before = _freeze(vars(s))
            with self.assertRaises((ValueError, AttributeError)):
                native.accrete(s, 2., witness(s))
            self.assertStateEqual(s, before)

    def test_no_plan_publishes_native_history_advance_without_claiming_docking(self):
        s = docking_world()
        s.accretion_welding_state['loading'][:] = 0.
        control = deepcopy(s); before = _freeze(vars(s)); payload = native._payload(s)
        collision_contacts.refresh(control, 2.)
        self.assertEqual(local_accretion.plan_accretions(control, 2.), [])
        result = native.accrete(s, 2., None)
        self.assertFalse(result['accepted'])
        self.assertEqual(_freeze(vars(s)), _freeze(vars(control)))
        self.assertNotEqual(_freeze(vars(s)), before)
        native._validate_payload(s, payload)
        self.assertEqual(s.events, [])
        self.assertEqual(getattr(s, native.STORE_FIELD)['bonds'], [])

    def test_interval_requires_positive_real_scalar_not_boolean_or_array(self):
        s = docking_world(); before = _freeze(vars(s))
        for dt in (True, np.bool_(True), np.array(2.), [2.], '2', 2j, 0., -2., np.nan, np.inf):
            with self.subTest(dt=repr(dt)):
                with self.assertRaises(ValueError): native.accrete(s, dt, witness(s))
                self.assertStateEqual(s, before)

    def test_original_edge_predicate_rejects_boundary_before_a_misleading_solve(self):
        s = docking_world(); collision_contacts.refresh(s, 2.)
        plan, = local_accretion.plan_accretions(s, 2.)
        authored = witness(s)
        authored['world_corners'] = s.material_surface['vertices'][s.material_surface['faces'][0]].copy()
        before = _freeze(vars(s))
        with patch.object(np.linalg, 'solve', return_value=np.ones((3, 3))) as solve:
            with self.assertRaisesRegex(ValueError, 'strictly inside'):
                native._witness(s, plan, authored)
            solve.assert_not_called()
        self.assertStateEqual(s, before)

    def test_unresolved_homogeneous_solve_fails_before_record_construction(self):
        s = docking_world(); collision_contacts.refresh(s, 2.)
        plan, = local_accretion.plan_accretions(s, 2.)
        authored = witness(s); before = _freeze(vars(s))
        for answer in (np.full((3, 3), np.nan), np.full((3, 3), .25)):
            with patch.object(np.linalg, 'solve', return_value=answer):
                with self.assertRaisesRegex(ValueError, 'unresolved'):
                    native._witness(s, plan, authored)
            self.assertStateEqual(s, before)

    def test_real_topology_cut_rejected_before_any_native_identity_allocation(self):
        s, chosen = split_world()
        a = int(np.flatnonzero(chosen['parcel_side'])[0])
        b = int(np.flatnonzero(~chosen['parcel_side'])[0])
        authored_split_bond(s, a, b)
        before = _freeze(vars(s)); aliases = (s.plate_uid, s.parcel_plate, s.omega)
        alias_before = _freeze(aliases)
        with patch.object(native_topology, '_allocate', wraps=native_topology._allocate) as allocate:
            with self.assertRaisesRegex(ValueError, 'owner cut'):
                native.split_continent(s, 0, chosen)
            allocate.assert_not_called()
        self.assertStateEqual(s, before)
        self.assertEqual(_freeze(aliases), alias_before)

    def test_real_topology_accepts_paired_material_together_without_releasing_bond(self):
        s, chosen = split_world()
        a, b = map(int, np.flatnonzero(chosen['parcel_side'])[:2])
        authored_split_bond(s, a, b)
        before_store = deepcopy(getattr(s, native.STORE_FIELD))
        payload = native._payload(s); uid = s.next_plate_uid
        result = native.split_continent(s, 0, chosen)
        self.assertTrue(result['accepted']); self.assertEqual(result['new_plate_uid'], uid)
        self.assertEqual(s.next_plate_uid, uid+1)
        self.assertEqual(finite._canonical(getattr(s, native.STORE_FIELD)), finite._canonical(before_store))
        np.testing.assert_array_equal(s.plate_uid[s.parcel_plate[[a, b]]], [uid, uid])
        native._validate_payload(s, payload)
        for column in (s.structure, s.trace_structure): dense_crust.validate(column)

    def test_whole_native_checkpoint_restores_ledger_and_same_transaction_result(self):
        s, chosen = split_world()
        a, b = map(int, np.flatnonzero(chosen['parcel_side'])[:2])
        authored_split_bond(s, a, b)
        # Real typed disk checkpoint of the entire native Simulation. This
        # fixture identity covers these inspected sources; it is not a full
        # deployment source manifest or a subsequent evolved-world trajectory.
        paths = [Path(native.__file__), Path(native_topology.__file__)]
        compatibility = dict(engine_sha256=hashlib.sha256(paths[0].read_bytes()).hexdigest(),
            auxiliary_sources_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths[1:]},
            numpy_version=np.__version__)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'native-research.npz'
            checkpoint.write_checkpoint(target, s, dict(config=deepcopy(s.config)), compatibility)
            restored, _ = checkpoint.read_checkpoint(target, compatibility, type(s))
        self.assertStateEqual(restored, _freeze(vars(s)))
        first = native.split_continent(s, 0, chosen)
        second = native.split_continent(restored, 0, deepcopy(chosen))
        self.assertEqual(finite._canonical(first), finite._canonical(second))
        self.assertStateEqual(restored, _freeze(vars(s)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
