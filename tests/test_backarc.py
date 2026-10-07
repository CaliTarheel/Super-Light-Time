"""Small kinematic namespaces isolate back-arc motion gates and accounting."""
from copy import deepcopy
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import backarc
import plate_balance
import slab_memory
from trench_dynamics import retreat_speed


def fixture(convergence=50., cached_convergence=None, phase='rifting'):
    radius = backarc.RADIUS_KM
    row = dict(id=1, phase=phase, parent_plate_uid=10, arc_plate_uid=20,
               downgoing_plate_uid=30, center=[1., 0., 0.], started_myr=-10.,
               last_active_myr=0., loading_km=60., opening_km=0., closing_km=0.,
               extension_rate_km_myr=0.)
    s = SimpleNamespace(
        capacity=3, plate_uid=np.array([10, 20, 30]), active=np.ones(3, bool),
        plate=np.array([0, 0, 1, 2]), cell_area=np.array([5., 5., 2., 4.]),
        omega=np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., convergence/radius]]),
        mantle=np.arange(9, dtype=float).reshape(3, 3)/1000.,
        crust=np.array([1, 1, 3, 0]), age=np.full(4, 80.),
        ba=np.array([3]), bb=np.array([2]), bp=np.array([2]), bq=np.array([1]),
        down=np.array([2]), bcode=np.array([2]), bl=np.array([1000.]),
        bmid=np.array([[1., 0., 0.]]), bn=np.array([[0., 1., 0.]]),
        normal_speed=np.array([-convergence if cached_convergence is None else -cached_convergence]),
        trench_retreat_speed=np.array([8.]),
        mass=np.empty(0), kind=np.empty(0, np.uint8), parcel_plate=np.empty(0, int),
        backarc_basins=[row], config={'rift_strength': 1., 'seed': 12},
        names=['Parent', 'Arc', 'Down'], next_backarc_id=2, t=2.)
    s.recorded=[]
    s._record=lambda *args, **kwargs: s.recorded.append((args, kwargs))
    return s


def relative_opening(s):
    # Here +Y is inland and opening moves the arc toward -Y.
    return -np.cross(s.omega[1]-s.omega[0], s.bmid[0])[1]*backarc.RADIUS_KM


class BackarcKinematicTests(unittest.TestCase):
    def test_common_rigid_rotation_cannot_trigger_loading_or_motion(self):
        s = fixture()
        s.omega[:] = [.001, -.002, .003]
        before = s.omega.copy()
        backarc.apply_motion(s, 2.)
        np.testing.assert_array_equal(s.omega, before)
        s.backarc_basins=[]
        self.assertIsNone(backarc.update(s, 2.))
        self.assertEqual(s.backarc_basins, [])
        self.assertEqual(s.recorded, [])

    def test_actual_divergence_beats_stale_subduction_and_retreat_labels(self):
        s = fixture(-10., cached_convergence=80.)
        before = s.omega.copy()
        backarc.apply_motion(s, 2.)
        np.testing.assert_array_equal(s.omega, before)
        s.backarc_basins=[]
        self.assertIsNone(backarc.update(s, 2.))
        self.assertEqual(s.backarc_basins, [])

    def test_drive_preserves_area_weighted_angular_change_and_mantle(self):
        s = fixture()
        original, mantle = s.omega.copy(), s.mantle.copy()
        backarc.apply_motion(s, 2.)
        change = s.omega-original
        np.testing.assert_allclose(10.*change[0]+2.*change[1], 0., atol=1e-18)
        np.testing.assert_array_equal(s.omega[2], original[2])
        np.testing.assert_array_equal(s.mantle, mantle)
        self.assertGreater(relative_opening(s), 0.)
        self.assertLessEqual(relative_opening(s), 8.)

    def test_current_convergence_sets_retreat_magnitude_after_forces_change(self):
        s = fixture(3., cached_convergence=80.)
        backarc.apply_motion(s, 2.)
        current_limit = float(retreat_speed(3., 80., 0.))
        self.assertLessEqual(relative_opening(s), current_limit+1e-12)

    def test_relaxation_cannot_overshoot_local_retreat_at_allowed_largest_step(self):
        s = fixture(100.)
        s.config['rift_strength']=3.
        backarc.apply_motion(s, 5.)
        self.assertLessEqual(relative_opening(s), 8.+1e-12)

    def test_no_drive_after_trench_loss_or_for_unrelated_remote_margin(self):
        for change in ('lost', 'wrong_plate', 'remote', 'closed', 'disabled'):
            with self.subTest(change=change):
                s = fixture()
                if change == 'lost': s.bcode[:]=3
                elif change == 'wrong_plate': s.backarc_basins[0]['downgoing_plate_uid']=999
                elif change == 'remote': s.backarc_basins[0]['center']=[-1., 0., 0.]
                elif change == 'closed': s.backarc_basins[0]['phase']='closed'
                else: s.config['rift_strength']=0.
                before=s.omega.copy()
                backarc.apply_motion(s, 2.)
                np.testing.assert_array_equal(s.omega, before)

    def test_faster_existing_opening_is_not_piled_on_with_extra_retreat(self):
        s=fixture()
        s.omega[1,2]=-12./backarc.RADIUS_KM
        before=s.omega.copy()
        backarc.apply_motion(s, 2.)
        np.testing.assert_array_equal(s.omega, before)

    def test_loading_without_measured_opening_and_quiet_transition_are_distinct(self):
        s=fixture()
        s.backarc_basins=[]
        self.assertIsNone(backarc.update(s, 2.))
        self.assertEqual(len(s.backarc_basins), 1)
        row=s.backarc_basins[0]
        self.assertEqual(row['phase'], 'loading')
        self.assertEqual(row['opening_km'], 0.)
        self.assertEqual(row['extension_rate_km_myr'], 0.)
        self.assertGreater(row['loading_rate_km_myr'], 0.)
        self.assertGreater(row['loading_km'], 0.)
        s.bcode[:]=3; s.t=14.
        backarc.update(s, 2.)
        self.assertEqual(row['phase'], 'quiet')
        self.assertEqual(row['opening_km'], 0.)
        self.assertEqual(row['extension_rate_km_myr'], 0.)
        self.assertEqual(row['loading_rate_km_myr'], 0.)
        self.assertEqual([args[0] for args, _ in s.recorded], ['backarc_loading', 'backarc_quiet'])

    def test_driven_pairs_exclude_duplicate_support_retreat_only_for_existing_arc(self):
        s=fixture()
        self.assertEqual(backarc.driven_pairs(s), {(2, 1)})
        for change in ('closed', 'no_arc', 'lost_arc'):
            other=deepcopy(s)
            if change=='closed': other.backarc_basins[0]['phase']='closed'
            elif change=='no_arc': other.backarc_basins[0]['arc_plate_uid']=None
            else: other.active[1]=False
            self.assertEqual(backarc.driven_pairs(other), set())

    def test_measured_opening_quiet_reactivation_and_consumption_drive_phase_events(self):
        s=fixture()
        s.ba[:]=0; s.bb[:]=2; s.bp[:]=0; s.bq[:]=1; s.down[:]=-1
        s.bcode[:]=1; s.normal_speed[:]=5.
        s.omega[1, 2]=5./backarc.RADIUS_KM
        row=s.backarc_basins[0]
        backarc.update(s, 2.)
        self.assertEqual(row['phase'], 'spreading')
        self.assertEqual(row['opening_km'], 10.)
        s.t=4.; backarc.update(s, 2.)
        self.assertEqual(row['opening_km'], 20.)
        s.t=14.; s.bcode[:]=3; s.normal_speed[:]=0.; backarc.update(s, 2.)
        self.assertEqual(row['phase'], 'quiet')
        self.assertEqual(row['opening_km'], 20.)
        s.t=16.; s.bcode[:]=1; s.normal_speed[:]=5.; backarc.update(s, 2.)
        self.assertEqual(row['phase'], 'spreading')
        self.assertEqual(row['opening_km'], 30.)
        s.t=18.; s.bcode[:]=4; s.normal_speed[:]=-15.; backarc.update(s, 2.)
        self.assertEqual(row['phase'], 'closed')
        self.assertEqual(row['closing_km'], 30.)
        self.assertEqual([args[0] for args, _ in s.recorded],
                         ['backarc_spreading', 'backarc_quiet', 'backarc_spreading', 'backarc_closed'])
        s.t=20.; s.bcode[:]=1; s.normal_speed[:]=5.; backarc.update(s, 2.)
        self.assertEqual(row['phase'], 'closed')
        self.assertEqual(row['opening_km'], 30.)

    def test_age_without_compression_does_not_claim_basin_consumption(self):
        s=fixture()
        row=s.backarc_basins[0]
        row['opening_km']=100.
        s.bcode[:]=3; s.t=200.
        backarc.update(s, 2.)
        self.assertEqual(row['phase'], 'quiet')
        self.assertEqual(row['closing_km'], 0.)
        self.assertEqual(row['opening_km'], 100.)


# Anchoring threshold for a continental/arc upper plate: slab depth
# length*sin(dip) must exceed the 100 km lithosphere cap (130.5 km of slab).
ANCHOR_KM = plate_balance.PLATE_THICKNESS_CAP_KM/math.sin(math.radians(slab_memory.SUBDUCTION_DIP_DEG))


def sea_anchor_fixture(upper_km_myr=0., slab_km=400., convergence=50.):
    """Driver 1: the arc-margin upper plate (slot 1) over the downgoing slot 2.

    +Y at the trench is inland (bn points from the downgoing plate into the
    overriding plate). The swept retreat is deliberately left at 8 km/Myr to
    show that driver 1 ignores it.
    """
    s = fixture(convergence)
    s.backarc_basins = []
    s.subduction_response_version = 1
    s.backarc_driver_version = 1
    s.omega[1] = [0., 0., upper_km_myr/backarc.RADIUS_KM]
    s.slab_km = slab_km
    return s


def attached(s):
    """Slab attachment as slab_memory.line_load reports it: downgoing slot 2."""
    return (np.where(s.down >= 0, s.down, -1), np.full(len(s.ba), 1e6),
            np.full(len(s.ba), float(s.slab_km)))


class SeaAnchorDriverTests(unittest.TestCase):
    def update(self, s, dt=2.):
        with patch.object(backarc.slab_memory, 'line_load', attached):
            return backarc.update(s, dt)

    def speed(self, s):
        with patch.object(backarc.slab_memory, 'line_load', attached):
            return backarc.loading_speed(s)

    def test_stationary_upper_plate_does_not_load_despite_swept_retreat(self):
        s = sea_anchor_fixture(0.)
        self.assertIsNone(self.update(s))
        self.assertEqual(s.backarc_basins, [])
        self.assertEqual(s.backarc_driver_diagnostics['anchored_edges'], 1)
        self.assertEqual(s.backarc_driver_diagnostics['max_speed_km_myr'], 0.)
        legacy = fixture()
        legacy.backarc_basins = []
        backarc.update(legacy, 2.)
        self.assertEqual(legacy.backarc_basins[0]['loading_rate_km_myr'], 8.)

    def test_upper_plate_moving_inland_loads_at_its_mantle_frame_speed(self):
        s = sea_anchor_fixture(3.)
        self.assertIsNone(self.update(s))
        self.assertEqual(len(s.backarc_basins), 1)
        row = s.backarc_basins[0]
        self.assertEqual(row['phase'], 'loading')
        self.assertAlmostEqual(row['loading_rate_km_myr'], 3., places=12)
        memory = backarc.LOADING_MEMORY_MYR
        expected = 3.*memory*(1-math.exp(-2./memory))
        self.assertAlmostEqual(row['loading_km'], expected, places=12)
        s.t = 4.
        self.update(s)
        expected = expected*math.exp(-2./memory)+3.*memory*(1-math.exp(-2./memory))
        self.assertAlmostEqual(row['loading_km'], expected, places=12)
        self.assertIn('anchored slab', s.recorded[0][0][1])
        self.assertEqual(s.recorded[0][1]['details']['mechanism'], backarc._MECHANISM[1])
        report = s.backarc_driver_diagnostics
        self.assertEqual(report['loading_length_km'], 1000.)
        self.assertAlmostEqual(report['mean_speed_km_myr'], 3., places=12)
        self.assertEqual((report['frame'], report['law']), (backarc.DRIVER_FRAME, backarc.DRIVER_LAW))
        np.testing.assert_array_equal(s.trench_retreat_speed, [8.])  # never written

    def test_advancing_and_trench_parallel_upper_plates_do_not_load(self):
        for name, omega in (('advancing', [0., 0., -3./backarc.RADIUS_KM]),
                            ('parallel', [0., -3./backarc.RADIUS_KM, 0.])):
            with self.subTest(name=name):
                s = sea_anchor_fixture()
                s.omega[1] = omega
                self.assertEqual(self.speed(s)[0], 0.)
                self.assertIsNone(self.update(s))
                self.assertEqual(s.backarc_basins, [])

    def test_slab_must_reach_below_the_overriding_lithosphere(self):
        for slab, loads in ((ANCHOR_KM*(1-1e-6), False), (ANCHOR_KM*(1+1e-6), True)):
            with self.subTest(slab=slab):
                s = sea_anchor_fixture(3., slab_km=slab)
                self.update(s)
                self.assertEqual(bool(s.backarc_basins), loads)
                self.assertEqual(s.backarc_driver_diagnostics['anchored_edges'], int(loads))
        # A young oceanic upper plate is thinner, so a shorter slab anchors it.
        thin = float(plate_balance.plate_thickness_m(25., plate_balance.PLATE_THICKNESS_CAP_KM))/1e3
        self.assertLess(thin, 70.)
        threshold = thin/math.sin(math.radians(slab_memory.SUBDUCTION_DIP_DEG))
        for slab, expected in ((threshold*(1-1e-6), 0.), (threshold*(1+1e-6), 3.)):
            s = sea_anchor_fixture(3., slab_km=slab)
            s.crust[2] = 0
            s.age[2] = 25.
            self.assertAlmostEqual(self.speed(s)[0], expected, places=12)

    def test_reversed_edge_orientation_reads_the_other_side(self):
        # Overriding plate on the bp side: inland is -bn and the overriding
        # control cell is ba. A sign error or swapped ba/bb would fail here.
        def flipped(upper_km_myr, slab_km=400., crust_above=3, age_above=80.):
            s = sea_anchor_fixture(upper_km_myr, slab_km=slab_km)
            s.bp, s.bq = np.array([1]), np.array([2])
            s.ba, s.bb = np.array([2]), np.array([3])
            s.bn = np.array([[0., -1., 0.]])
            dv = np.cross(s.omega[2]-s.omega[1], s.bmid)*backarc.RADIUS_KM
            s.normal_speed = np.sum(dv*s.bn, axis=1)
            s.crust[2], s.age[2] = crust_above, age_above
            return s
        with self.subTest(case='away'):
            s = flipped(3.)
            self.assertAlmostEqual(self.speed(s)[0], 3., places=12)
            self.update(s)
            self.assertAlmostEqual(s.backarc_basins[0]['loading_rate_km_myr'], 3., places=12)
        with self.subTest(case='advancing'):
            s = flipped(-3.)
            self.assertEqual(self.speed(s)[0], 0.)
            self.update(s)
            self.assertEqual(s.backarc_basins, [])
        with self.subTest(case='young oceanic upper plate read from ba'):
            thin = float(plate_balance.plate_thickness_m(25., plate_balance.PLATE_THICKNESS_CAP_KM))/1e3
            threshold = thin/math.sin(math.radians(slab_memory.SUBDUCTION_DIP_DEG))
            for slab, expected in ((threshold*(1-1e-6), 0.), (threshold*(1+1e-6), 3.)):
                s = flipped(3., slab_km=slab, crust_above=0, age_above=25.)
                self.assertAlmostEqual(self.speed(s)[0], expected, places=12)

    def test_non_finite_oceanic_overriding_age_is_rejected(self):
        s = sea_anchor_fixture(3.)
        s.crust[2], s.age[2] = 0, float('nan')
        with self.assertRaisesRegex(ValueError, 'finite ages'):
            self.speed(s)
        # A continental overriding cell does not read its age.
        s.crust[2] = 1
        self.assertAlmostEqual(self.speed(s)[0], 3., places=12)

    def test_unattached_or_non_subducting_edges_do_not_load(self):
        s = sea_anchor_fixture(3.)
        with patch.object(backarc.slab_memory, 'line_load',
                          lambda s: (np.full(1, -1), np.zeros(1), np.zeros(1))):
            self.assertEqual(backarc.loading_speed(s)[0], 0.)
        s.bcode[:] = 3
        self.assertEqual(self.speed(s)[0], 0.)

    def test_common_rotation_that_moves_the_upper_plate_away_loads(self):
        # Driver 0's common-rotation invariance (first test class) is
        # kinematic. The sea anchor acts in the mantle frame, so a shared
        # rotation that moves the upper plate inland against its anchored slab
        # does load.
        s = sea_anchor_fixture(0.)
        s.omega[:] += [0., 0., 3./backarc.RADIUS_KM]
        self.update(s)
        self.assertAlmostEqual(s.backarc_basins[0]['loading_rate_km_myr'], 3., places=12)

    def test_version_selection_is_strict(self):
        s = sea_anchor_fixture(3.)
        s.subduction_response_version = 0
        with self.assertRaisesRegex(ValueError, 'moving-hinge'):
            self.update(s)
        for value in (True, 1., 2, '1', None):
            with self.subTest(value=value):
                s = sea_anchor_fixture(3.)
                s.backarc_driver_version = value
                with self.assertRaises(ValueError):
                    self.update(s)
        legacy = fixture()
        self.assertEqual(backarc.driver_version(legacy), 0)
        self.assertEqual(backarc.snapshot_fields(legacy), {})
        self.assertEqual(backarc.driver_version(SimpleNamespace(backarc_driver_version=np.int64(1))), 1)

    def test_rupture_follows_the_existing_gate(self):
        s = sea_anchor_fixture(3.)
        pending = None
        while pending is None and s.t < 400.:
            pending = self.update(s)
            s.t += 2.
        self.assertIsNotNone(pending)
        row = s.backarc_basins[0]
        self.assertGreaterEqual(row['loading_km'], backarc.RUPTURE_LOADING_KM)
        # 3 km/Myr relaxes toward 240 km of hinge lag; 60 km needs about 23 Myr
        # of loading, i.e. 12 two-Myr updates. The first update is the row's
        # start, so the rupture step is 11 steps after started_myr.
        crossing = -backarc.LOADING_MEMORY_MYR*math.log(
            1-backarc.RUPTURE_LOADING_KM/(3.*backarc.LOADING_MEMORY_MYR))
        self.assertAlmostEqual(s.t-2.-row['started_myr'], 2.*math.ceil(crossing/2.)-2.)


class SeaAnchorFrameTests(unittest.TestCase):
    def frame(self):
        s = sea_anchor_fixture(3.)
        with patch.object(backarc.slab_memory, 'line_load', attached):
            fields = backarc.snapshot_fields(s)
        return dict(fields, subduction_response_version=1, time_myr=2.)

    def test_valid_frame_and_versioned_rejections(self):
        frame = self.frame()
        self.assertEqual(frame['backarc_driver_version'], 1)
        backarc.validate_frame(frame)
        backarc.validate_frame({})
        bad = []
        for change in ('response', 'missing', 'law', 'frame', 'negative', 'nan', 'bool', 'stray',
                       'version', 'inconsistent', 'float_count'):
            other = deepcopy(frame)
            row = other.get('backarc_driver_diagnostics')
            if change == 'response': other['subduction_response_version'] = 0
            elif change == 'missing': del other['backarc_driver_diagnostics']
            elif change == 'law': row['law'] = 'retreat'
            elif change == 'frame': row['frame'] = 'no-net-rotation'
            elif change == 'negative': row['max_speed_km_myr'] = -1.
            elif change == 'nan': row['mean_speed_km_myr'] = float('nan')
            elif change == 'bool': other['backarc_driver_version'] = True
            elif change == 'stray': other = {'backarc_driver_diagnostics': row}
            elif change == 'version': row['version'] = 0
            elif change == 'inconsistent': row['loading_length_km'] = row['anchored_length_km']+10.
            else: row['anchored_edges'] = 1.
            bad.append((change, other))
        for change, other in bad:
            with self.subTest(change=change), self.assertRaises(ValueError):
                backarc.validate_frame(other)

    def test_snapshot_diagnostics_are_detached_copies(self):
        s = sea_anchor_fixture(3.)
        with patch.object(backarc.slab_memory, 'line_load', attached):
            fields = backarc.snapshot_fields(s)
        fields['backarc_driver_diagnostics']['max_speed_km_myr'] = 99.
        self.assertEqual(s.backarc_driver_diagnostics['max_speed_km_myr'], 3.)


if __name__ == '__main__':
    unittest.main()
