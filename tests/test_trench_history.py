"""Persistent local trench identities and lifecycle, independent of plate forces."""
from copy import deepcopy
from types import SimpleNamespace
import pickle
import unittest

import numpy as np

import trench_history as history
import backarc
from trench_dynamics import extend_forearcs
from ridge_geometry import rotate


def fixture(fronts=((58, 63, 63),), width=128):
    height = width//2
    lon, lat = np.meshgrid((np.arange(width)+.5)*2*np.pi/width-np.pi,
                          np.pi/2-(np.arange(height)+.5)*np.pi/height)
    xyz = np.column_stack((np.cos(lat).ravel()*np.cos(lon).ravel(),
                           np.cos(lat).ravel()*np.sin(lon).ravel(), np.sin(lat).ravel()))
    aa = np.array([r*width+c for start, stop, c in fronts for r in range(start, stop)], int)
    bb = (aa//width)*width+(aa % width+1) % width
    mid = xyz[aa]+xyz[bb]
    mid /= np.linalg.norm(mid, axis=1)[:, None]
    normal = xyz[bb]-xyz[aa]
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    n, count = width*height, len(aa)
    plate = np.zeros(n, int)
    plate[bb] = 1
    s = SimpleNamespace(w=width, h=height, n=n, t=0., active=np.array([True, True]),
        plate_uid=np.array([11, 22]), plate=plate, capacity=2, xyz=xyz,
        omega=np.zeros((2, 3)), age=np.full(n, 80.), crust=np.zeros(n, np.uint8),
        ba=aa, bb=bb, bp=np.zeros(count, int), bq=np.ones(count, int),
        bmid=mid, bn=normal, bl=np.full(count, np.pi*6371/height),
        bcode=np.full(count, 2, np.uint8), down=np.zeros(count, int),
        normal_speed=np.full(count, -20.), polarity=np.array([[-1, 0], [0, -1]]),
        events=[])
    s.age[bb] = 40.
    def record(typ, description, key, **fields):
        s.events.append(dict(type=typ, key=key, **fields))
    s._record = record
    return s


def step(s, dt=2.):
    s.t += dt
    history.update(s, dt)


class TrenchHistoryTests(unittest.TestCase):
    def test_prepare_and_repeated_update_cannot_create_age_or_duplicate_events(self):
        s = fixture()
        history.initialize(s)
        self.assertEqual(len(s.trench_systems), 1)
        self.assertEqual(s.trench_systems[0]['maturity'], 0)
        before = deepcopy(s.trench_systems)
        for _ in range(12):
            history.prepare(s)
            history.update(s, 0.)
        self.assertEqual(s.trench_systems, before)
        self.assertEqual(len(s.events), 1)
        step(s)
        before = deepcopy(s.trench_systems)
        history.update(s, 2.)
        self.assertEqual(s.trench_systems, before)

    def test_maturation_requires_elapsed_convergence_and_not_just_age(self):
        s = fixture()
        history.initialize(s)
        for _ in range(5):
            step(s)
        row = s.trench_systems[0]
        self.assertEqual(row['phase'], 'mature')
        self.assertEqual(row['shortening_km'], 200.)
        np.testing.assert_array_equal(history.weights(s), 1.)
        slow = fixture()
        slow.normal_speed[:] = -3.
        history.initialize(slow)
        for _ in range(5):
            step(slow)
        self.assertAlmostEqual(slow.trench_systems[0]['maturity'], .3)

    def test_remote_systems_between_same_pair_are_not_merged(self):
        s = fixture(((28, 35, 31), (28, 35, 95)))
        history.initialize(s)
        self.assertEqual(len(s.trench_systems), 2)
        self.assertNotEqual(s.trench_id[0], s.trench_id[-1])
        for _ in range(12):
            step(s)
        self.assertEqual(len(s.trench_systems), 2)
        self.assertEqual(sum(e['type'] == 'trench_initiating' for e in s.events), 2)

    def test_advected_geometry_keeps_identity_across_seam_and_poles(self):
        for axis in ([0., 0., .06], [0., .06, 0.], [.04, .02, -.03]):
            with self.subTest(axis=axis):
                s = fixture(((27, 35, 63),))
                s.omega[1] = axis
                history.initialize(s)
                for _ in range(35):
                    expected = rotate(s.bmid, s.omega[1]*2.)
                    history.advect(s, 2.)
                    history.advect(s, 2.)  # Duplicate orchestration cannot double-rotate.
                    np.testing.assert_allclose(s.trench_systems[0]['geometry_xyz'], expected, atol=2e-14)
                    s.bmid = expected
                    s.bn = rotate(s.bn, s.omega[1]*2.)
                    step(s)
                    self.assertTrue(np.all(s.trench_id == 1))
                self.assertEqual(len(s.trench_systems), 1)

    def test_local_polarity_survives_ocean_age_reversal(self):
        s = fixture(((24, 35, 63),))
        history.initialize(s)
        step(s)
        s.age[s.ba], s.age[s.bb] = 1., 160.
        s.down[:] = 1  # Pair-wide fallback would now point the other way.
        history.prepare(s)
        np.testing.assert_array_equal(s.down, 0)
        step(s)
        self.assertEqual(len(s.trench_systems), 1)
        self.assertEqual(s.trench_systems[0]['downgoing_plate_uid'], 11)

    def test_brief_quiet_interval_preserves_polarity_and_episode(self):
        s = fixture()
        history.initialize(s)
        for _ in range(5):
            step(s)
        s.bcode[:], s.normal_speed[:] = 3, 0.
        step(s)
        self.assertEqual(s.trench_systems[0]['phase'], 'quiet')
        np.testing.assert_array_equal(history.weights(s), 0.)
        s.bcode[:], s.normal_speed[:] = 2, -20.
        step(s)
        self.assertEqual(s.trench_systems[0]['phase'], 'mature')
        self.assertEqual(s.trench_systems[0]['episode'], 1)
        self.assertEqual(len(s.trench_systems), 1)
        self.assertEqual(sum(e['type'] == 'trench_resumed' for e in s.events), 1)

    def test_collision_extension_and_absence_have_bounded_shutdown(self):
        for mode, count, reason in [('collision', 3, 'buoyant_collision'),
                                    ('extension', 5, 'sustained_extension'),
                                    ('quiet', 10, 'convergence_ceased')]:
            with self.subTest(mode=mode):
                s = fixture()
                history.initialize(s)
                for _ in range(5):
                    step(s)
                s.bcode[:], s.normal_speed[:] = {'collision': (4, -20.),
                                                'extension': (1, 20.), 'quiet': (3, 0.)}[mode]
                if mode == 'collision':
                    s.crust[s.ba], s.crust[s.bb] = 1, 1
                for _ in range(count):
                    step(s)
                self.assertEqual(s.trench_systems[0]['phase'], 'shutdown')
                self.assertEqual(s.trench_systems[0]['episodes'][0]['shutdown_reason'], reason)
                self.assertEqual(sum(e['type'] == 'trench_shutdown' for e in s.events), 1)
                np.testing.assert_array_equal(history.weights(s), 0.)

    def test_reactivation_is_new_episode_not_duplicate_system(self):
        s = fixture()
        history.initialize(s)
        step(s)
        s.bcode[:], s.normal_speed[:] = 1, 20.
        for _ in range(5):
            step(s)
        s.bcode[:], s.normal_speed[:] = 2, -20.
        step(s)
        self.assertEqual(len(s.trench_systems), 1)
        row = s.trench_systems[0]
        self.assertEqual(row['episode'], 2)
        self.assertEqual(len(row['episodes']), 2)
        self.assertAlmostEqual(row['maturity'], .2)
        self.assertEqual(sum(e['type'] == 'trench_reactivated' for e in s.events), 1)

    def test_reused_plate_slot_does_not_inherit_old_trench(self):
        s = fixture()
        history.initialize(s)
        for _ in range(5):
            step(s)
        s.plate_uid[0] = 33
        step(s)
        self.assertEqual(len(s.trench_systems), 2)
        self.assertEqual(s.trench_systems[0]['phase'], 'shutdown')
        self.assertEqual(s.trench_systems[0]['episodes'][0]['shutdown_reason'], 'plate_owner_lost')
        self.assertEqual(s.trench_systems[1]['plate_uids'], [22, 33])
        self.assertTrue(np.all(s.trench_id == 2))

    def test_collision_blocked_polarity_cannot_force_buoyant_crust_down(self):
        s = fixture()
        history.initialize(s)
        for _ in range(5):
            step(s)
        s.crust[s.ba] = 1
        s.down[:] = 1
        history.prepare(s)
        np.testing.assert_array_equal(history.weights(s), 0.)
        np.testing.assert_array_equal(s.down, 1)
        step(s)
        self.assertEqual(len(s.trench_systems), 2)
        self.assertEqual(s.trench_systems[1]['downgoing_plate_uid'], 22)
        self.assertEqual(s.trench_systems[1]['predecessor_trench_id'], 1)
        self.assertAlmostEqual(s.trench_maturity[0], .2)

    def test_local_overlap_weight_does_not_consume_remote_pair_or_continent(self):
        s = fixture(((28, 35, 63),))
        history.initialize(s)
        for _ in range(5):
            step(s)
        result = history.downgoing_weight(s, 0)
        self.assertTrue(np.all(result[s.ba] > 0))
        self.assertEqual(result[32*s.w+10], 0.)
        s.crust[s.ba[0]] = 1
        self.assertEqual(history.downgoing_weight(s, 0)[s.ba[0]], 0.)
        s.bcode[:] = 1
        history.prepare(s)
        np.testing.assert_array_equal(history.downgoing_weight(s, 0), 0.)

    def test_separation_inherits_maturity_without_false_reinitiation(self):
        s = fixture(((20, 42, 63),))
        history.initialize(s)
        for _ in range(5):
            step(s)
        # A finite extending reach splits one previously continuous trench.
        s.bcode[8:14], s.normal_speed[8:14] = 1, 20.
        step(s)
        self.assertEqual(len(s.trench_systems), 1, 'One changed frame is not a persistent trench separation')
        for _ in range(7):
            step(s)
        self.assertEqual(len(s.trench_systems), 2)
        self.assertEqual(s.trench_systems[1]['parent_trench_id'], 1)
        self.assertEqual(s.trench_systems[1]['maturity'], 1.)
        self.assertEqual(sum(e['type'] == 'trench_separated' for e in s.events), 1)
        for _ in range(12):
            step(s)
        self.assertEqual(len(s.trench_systems), 2)
        self.assertEqual(sum(e['type'] == 'trench_initiating' for e in s.events), 1)

    def test_rejoining_chains_has_a_successor_not_a_false_quiet_shutdown(self):
        s = fixture(((20, 42, 63),))
        s.bcode[8:14], s.normal_speed[8:14] = 1, 20.
        history.initialize(s)
        self.assertEqual(len(s.trench_systems), 2)
        for _ in range(5):
            step(s)
        s.bcode[:], s.normal_speed[:] = 2, -20.
        step(s)
        retired = [row for row in s.trench_systems if row['phase'] == 'joined']
        self.assertEqual(len(retired), 1)
        self.assertEqual(retired[0]['history'][-1]['reason'], 'continuous_trench_geometry_rejoined')
        self.assertIsNone(retired[0]['episodes'][-1]['end_myr'])
        self.assertIn(retired[0]['successor_trench_id'], [1, 2])
        self.assertFalse(any(e['type'] == 'trench_shutdown' for e in s.events))
        for _ in range(5):
            step(s)
        self.assertEqual(len(s.trench_systems), 2)
        self.assertEqual(sum(r['phase'] == 'mature' for r in s.trench_systems), 1)

    def test_repeated_short_separation_does_not_create_ids_events_or_drive_gap(self):
        s = fixture(((20, 42, 63),))
        history.initialize(s)
        for _ in range(5):
            step(s)
        for iteration in range(60):
            split = iteration % 3 != 2
            s.bcode[8:14] = 1 if split else 2
            s.normal_speed[8:14] = 20. if split else -20.
            step(s)
            self.assertEqual(len(s.trench_systems), 1)
            self.assertLessEqual(len(s.trench_systems[0].get('pending_branches',[])), 1)
            if split:
                np.testing.assert_array_equal(history.weights(s)[8:14], 0.)
                np.testing.assert_array_equal(history.weights(s)[:8], 1.)
                np.testing.assert_array_equal(history.weights(s)[14:], 1.)
        self.assertEqual([e['type'] for e in s.events], ['trench_initiating','trench_mature'])

    def test_joined_identity_redirects_backarc_and_never_reactivates_itself(self):
        s = fixture(((20, 42, 63),))
        s.bcode[8:14], s.normal_speed[8:14] = 1, 20.
        history.initialize(s)
        for _ in range(5):
            step(s)
        s.backarc_basins = [dict(trench_id=r['id'], phase='loading', loading_km=42.) for r in s.trench_systems]
        s.bcode[:], s.normal_speed[:] = 2, -20.
        step(s)
        joined = next(r for r in s.trench_systems if r['phase'] == 'joined')
        successor = joined['successor_trench_id']
        self.assertTrue(all(r['trench_id'] == successor for r in s.backarc_basins))
        self.assertTrue(all(r['loading_km'] == 42. for r in s.backarc_basins))
        for _ in range(60):
            s.bcode[:], s.normal_speed[:] = 1, 20.
            step(s)
        s.bcode[:], s.normal_speed[:] = 2, -20.
        step(s)
        self.assertEqual(joined['phase'], 'joined')
        self.assertNotIn(joined['id'], s.trench_id)

    def test_pending_separation_survives_checkpoint_exactly(self):
        s = fixture(((20, 42, 63),))
        history.initialize(s)
        for _ in range(5):
            step(s)
        s.bcode[8:14], s.normal_speed[8:14] = 1, 20.
        for _ in range(3):
            step(s)
        state = {k:v for k,v in vars(s).items() if k != '_record'}
        restored = SimpleNamespace(**pickle.loads(pickle.dumps(state)))
        restored._record = lambda *args, **kw: None
        for _ in range(6):
            step(s)
            step(restored)
        self.assertEqual(s.trench_systems,restored.trench_systems)
        self.assertEqual(len(s.trench_systems),2)

    def test_contact_disappearance_shuts_down_and_prepare_cannot_birth_new_faces(self):
        s = fixture()
        history.initialize(s)
        # New far-away geometry is outside the old system's spatial memory.
        s.bmid = rotate(s.bmid, np.array([1.4, 0., 0.]))
        history.prepare(s)
        self.assertEqual(len(s.trench_systems), 1)
        np.testing.assert_array_equal(s.trench_id, 0)
        for _ in range(10):
            step(s)
        self.assertEqual(len(s.trench_systems), 2)
        self.assertEqual(s.trench_systems[0]['phase'], 'shutdown')
        self.assertEqual(s.trench_systems[0]['episodes'][-1]['shutdown_reason'], 'contact_disappeared')

    def test_checkpoint_restart_is_exact_and_uninitialized_fixture_falls_back(self):
        s = fixture()
        np.testing.assert_array_equal(history.weights(s), 1.)
        history.initialize(s)
        for _ in range(4):
            step(s)
        # The manager serializes state, not the test's local recording closure.
        state = {k: v for k, v in vars(s).items() if k != '_record'}
        restored = SimpleNamespace(**pickle.loads(pickle.dumps(state)))
        restored._record = lambda *args, **kw: None
        for _ in range(8):
            step(s)
            step(restored)
        self.assertEqual(s.trench_systems, restored.trench_systems)
        np.testing.assert_array_equal(s.trench_id, restored.trench_id)
        np.testing.assert_array_equal(s.trench_maturity, restored.trench_maturity)

    def test_overriding_transfer_is_local_and_inherits_the_existing_slab(self):
        s = fixture(((22, 40, 63),))
        s.active = np.array([True, True, True])
        s.plate_uid = np.array([11, 22, 33])
        s.omega = np.zeros((3, 3))
        history.initialize(s)
        for _ in range(5):
            step(s)
        changed = np.arange(len(s.bb)) >= len(s.bb)//2
        s.plate[s.bb[changed]] = 2
        mapping = history.transfer_overriding(s, 1, 2)
        self.assertEqual(mapping, {1: 2})
        self.assertEqual(len(s.trench_systems), 2)
        self.assertEqual(s.trench_systems[0]['overriding_plate_uid'], 22)
        self.assertEqual(s.trench_systems[1]['overriding_plate_uid'], 33)
        self.assertEqual(s.trench_systems[1]['maturity'], 1.)
        s.bq[changed] = 2
        history.prepare(s)
        np.testing.assert_array_equal(s.trench_id[~changed], 1)
        np.testing.assert_array_equal(s.trench_id[changed], 2)
        np.testing.assert_array_equal(history.weights(s), 1.)
        self.assertEqual(history.transfer_overriding(s, 1, 2), {})
        # A complete transfer retains identity rather than creating another slab.
        s.plate[s.bb[~changed]] = 2
        self.assertEqual(history.transfer_overriding(s, 1, 2), {1: 1})
        self.assertEqual(len(s.trench_systems), 2)
        self.assertEqual(s.trench_systems[0]['maturity'], 1.)

    def _three_plate_mature(self):
        s = fixture(((22, 40, 63),))
        s.active = np.array([True, True, True])
        s.plate_uid = np.array([11, 22, 33])
        s.omega = np.zeros((3, 3))
        history.initialize(s)
        for _ in range(5):
            step(s)
        row = s.trench_systems[0]
        self.assertEqual((row['downgoing_plate_uid'], row['maturity']), (11, 1.))
        return s

    def test_downgoing_split_keeps_the_mature_trench_on_the_daughter(self):
        # A rift through the subducting plate: trenches are matched by plate
        # pair, so without a transfer the mature record lost contact and the
        # same margin restarted from zero maturity.
        s = self._three_plate_mature()
        changed = np.arange(len(s.ba)) >= len(s.ba)//2
        s.plate[s.ba[changed]] = 2
        mapping = history.transfer_split(s, 0, 2)
        self.assertEqual(mapping, {1: 2})
        self.assertEqual(len(s.trench_systems), 2)
        parent, child = s.trench_systems
        self.assertEqual((parent['downgoing_plate_uid'], child['downgoing_plate_uid']), (11, 33))
        self.assertEqual(child['plate_uids'], [22, 33])
        self.assertEqual((child['maturity'], child['parent_trench_id']), (1., 1))
        self.assertEqual(child['history'][-1]['reason'], 'local_downgoing_plate_transfer')
        # Once the boundary is rebuilt the daughter's faces match the child, so
        # the next update continues it instead of initiating a new trench.
        s.bp[changed] = 2
        s.down[changed] = 2
        history.prepare(s)
        np.testing.assert_array_equal(s.trench_id[~changed], 1)
        np.testing.assert_array_equal(s.trench_id[changed], 2)
        step(s)
        self.assertEqual(len(s.trench_systems), 2)
        self.assertEqual(s.trench_systems[1]['phase'], 'mature')
        # A complete split keeps the original identity.
        s2 = self._three_plate_mature()
        s2.plate[s2.ba] = 2
        self.assertEqual(history.transfer_downgoing(s2, 0, 2), {1: 1})
        self.assertEqual(len(s2.trench_systems), 1)
        self.assertEqual((s2.trench_systems[0]['downgoing_plate_uid'], s2.trench_systems[0]['maturity']), (33, 1.))

    def test_absorbed_subducting_remnant_hands_its_slab_to_the_receiver(self):
        # Whole-plate absorption of the subducting remnant into a third plate:
        # the slab still hangs from the same edge, now owned by the receiver.
        s = self._three_plate_mature()
        s.plate[s.plate == 0] = 2
        self.assertEqual(history.transfer_split(s, 0, 2), {1: 1})
        row = s.trench_systems[0]
        self.assertEqual((row['downgoing_plate_uid'], row['overriding_plate_uid']), (33, 22))
        self.assertEqual((row['phase'], row['maturity']), ('mature', 1.))
        s.active[0] = False
        s.bp[:] = 2
        s.down[:] = 2
        history.prepare(s)
        np.testing.assert_array_equal(s.trench_id, 1)
        step(s)
        self.assertEqual(len(s.trench_systems), 1)
        self.assertEqual(s.trench_systems[0]['phase'], 'mature')
        self.assertFalse(any(e['type'] == 'trench_shutdown' for e in s.events))

    def test_absorption_into_the_trench_partner_ends_the_trench_without_self_subduction(self):
        s = self._three_plate_mature()
        s.plate[s.plate == 0] = 1  # The remnant joins the plate it subducted beneath.
        self.assertEqual(history.transfer_split(s, 0, 1), {})
        row = s.trench_systems[0]
        self.assertEqual(row['phase'], 'shutdown')
        self.assertEqual(row['episodes'][-1]['shutdown_reason'], 'boundary_absorbed')
        self.assertNotEqual(row['downgoing_plate_uid'], row['overriding_plate_uid'])
        # A partial move onto the partner keeps the trench on the faces that
        # still separate the two plates, and never writes a same-plate row.
        s2 = self._three_plate_mature()
        changed = np.arange(len(s2.ba)) >= len(s2.ba)//2
        s2.plate[s2.ba[changed]] = 1
        self.assertEqual(history.transfer_downgoing(s2, 0, 1), {})
        self.assertEqual(len(s2.trench_systems), 1)
        self.assertEqual(s2.trench_systems[0]['phase'], 'mature')
        self.assertEqual(s2.trench_systems[0]['downgoing_plate_uid'], 11)

    def test_real_forearc_retreat_is_gated_and_confined_to_mature_local_system(self):
        from tests.test_trench_dynamics import globe
        s = globe(128, smooth=True)
        local = fixture(((0, 64, 63),))
        local.bcode[:] = 3
        local.bcode[24:40] = 2
        for field in ('t', 'plate_uid', 'active', 'ba', 'bb', 'bp', 'bq', 'down',
                      'bmid', 'bn', 'bl', 'bcode', 'normal_speed'):
            setattr(s, field, deepcopy(getattr(local, field)))
        history.initialize(s)
        before = s.support.copy()
        self.assertEqual(extend_forearcs(s, 2.), {})
        np.testing.assert_array_equal(s.support, before)
        for _ in range(5):
            step(s)
        # A genuinely new remote contact of the same pair has no developed
        # slab yet and cannot borrow this middle trench's established supply.
        s.bcode[:] = 2
        history.prepare(s)
        untracked = deepcopy(s)
        del untracked.trench_systems
        full = extend_forearcs(untracked, 2.)
        limited = extend_forearcs(s, 2.)
        self.assertTrue(limited)
        self.assertLess(limited[0, 1]['area_km2'], full[0, 1]['area_km2'])
        remote = np.abs(s.xyz[:, 2]) > np.sin(np.radians(35.))
        np.testing.assert_array_equal(s.support[:, remote], before[:, remote])
        self.assertGreater(np.max(np.abs(untracked.support[:, remote]-before[:, remote])), 0.)

    def test_backarc_motion_and_loading_need_local_maturity(self):
        from tests.test_backarc import fixture as arc_fixture, relative_opening
        legacy = arc_fixture()
        backarc.apply_motion(legacy, 2.)
        reference = relative_opening(legacy)
        for maturity in (0., .25, 1.):
            with self.subTest(maturity=maturity):
                s = arc_fixture()
                s.trench_systems = []
                s.trench_id = np.ones(1, int)
                s.trench_maturity = np.full(1, maturity)
                mantle = s.mantle.copy()
                backarc.apply_motion(s, 2.)
                self.assertAlmostEqual(relative_opening(s), reference*maturity)
                np.testing.assert_array_equal(s.mantle, mantle)
                s.backarc_basins = []
                backarc.update(s, 2.)
                self.assertEqual(bool(s.backarc_basins), maturity > 0)
                if maturity > 0:
                    self.assertEqual(s.backarc_basins[0]['trench_id'], 1)

    def test_nearby_separate_trenches_do_not_share_backarc_loading_or_motion(self):
        from tests.test_backarc import fixture as arc_fixture
        s = arc_fixture()
        for field in ('ba', 'bb', 'bp', 'bq', 'down', 'bcode', 'bl',
                      'normal_speed', 'trench_retreat_speed'):
            setattr(s, field, np.repeat(getattr(s, field), 2))
        s.bmid = np.vstack((s.bmid, rotate(s.bmid, [0., 0., .08])))
        s.bn = np.vstack((s.bn, rotate(s.bn, [0., 0., .08])))
        s.trench_systems = []
        s.trench_id = np.array([10, 20])
        s.trench_maturity = np.array([0., 1.])
        s.backarc_basins[0]['trench_id'] = 10
        before = s.omega.copy()
        backarc.apply_motion(s, 2.)
        np.testing.assert_array_equal(s.omega, before)
        s.backarc_basins = []
        backarc.update(s, 2.)
        self.assertEqual([r['trench_id'] for r in s.backarc_basins], [20])
        s.trench_maturity[:] = 1.
        s.t += 2.
        backarc.update(s, 2.)
        self.assertEqual(sorted(r['trench_id'] for r in s.backarc_basins), [10, 20])


if __name__ == '__main__':
    unittest.main()
