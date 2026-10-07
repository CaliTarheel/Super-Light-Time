"""Continuous hinge retreat preserves membership and respects physical limits."""
from copy import deepcopy
from collections import defaultdict
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from raster_engine import Simulation
from trench_dynamics import RADIUS_KM, extend_forearcs, retreat_speed


def globe(width=128, dtype=np.float64, smooth=False):
    """Lightweight prescribed two-plate globe; no physics integration or parcels."""
    height = width//2
    lon, lat = np.meshgrid((np.arange(width)+.5)*2*np.pi/width-np.pi,
                          np.pi/2-(np.arange(height)+.5)*np.pi/height)
    xyz = np.column_stack((np.cos(lat).ravel()*np.cos(lon).ravel(),
                           np.cos(lat).ravel()*np.sin(lon).ravel(), np.sin(lat).ravel()))
    edges = np.pi/2-np.arange(height+1)*np.pi/height
    area = np.repeat(np.diff(-np.sin(edges))*2*np.pi/width*RADIUS_KM**2, width)
    q = np.clip(.5+np.sin(lon.ravel())/.2, 0, 1) if smooth else (np.sin(lon.ravel()) > 0).astype(float)
    support = np.stack((1-q, q)).astype(dtype)
    s = SimpleNamespace(w=width, h=height, n=width*height, capacity=2, xyz=xyz, cell_area=area,
        support=support, plate=np.argmax(support, axis=0).astype(np.int16),
        omega=np.array([[0., 0., .006], [0., 0., 0.]]), age=np.full(width*height, 80.),
        crust=np.zeros(width*height, np.uint8), density=np.zeros(width*height),
        mass=np.empty(0), kind=np.empty(0, np.uint8), parcel_plate=np.empty(0, np.int16),
        bcode=np.array([2], np.uint8), down=np.array([0]), bp=np.array([0]), bq=np.array([1]))
    s._sample_coordinates = MethodType(Simulation._sample_coordinates, s)
    s._sample = Simulation._sample
    return s


def front_longitude(s):
    """Interpolate the 50% front on the equatorial row around longitude zero."""
    row = s.support[1].reshape(s.h, s.w)[s.h//2]
    longitude = (np.arange(s.w)+.5)*2*np.pi/s.w-np.pi
    crossings = np.flatnonzero((row[:-1] <= .5) & (row[1:] > .5))
    i = int(crossings[np.argmin(np.abs(longitude[crossings]))])
    return longitude[i]+(.5-row[i])/(row[i+1]-row[i])*(longitude[i+1]-longitude[i])


def segmented_trench():
    """Three oceanic segments beside a remotely cratonic overriding plate."""
    s = globe(64, smooth=True)
    s.ba = np.array([s.h//2-6, s.h//2, s.h//2+6])*s.w+s.w//2-1
    s.bb = s.ba+1
    s.bp, s.bq, s.down = np.zeros(3, int), np.ones(3, int), np.zeros(3, int)
    s.bcode = np.full(3, 2, np.uint8)
    s.bl = np.array([100., 150., 200.])
    s.normal_speed = np.array([-30., -80., -200.])
    s.age[s.ba] = [20., 80., 120.]
    s.mass = np.array([.5*s.cell_area[s.plate == 1].sum()])
    s.kind, s.parcel_plate = np.array([2], np.uint8), np.array([1], np.int16)
    s.process_totals = defaultdict(float)
    s.names, s.plate_uid, s.t = ['Downgoing', 'Overriding'], np.array([11, 22]), 20.
    # Allocation receives already measured transport and reconstructed edges;
    # their geometric transport is covered independently above.
    s._boundaries = lambda: None
    s._record = lambda *args, **kwargs: None
    return s


def prescribed_transfer(s, cells):
    cells = np.asarray(cells, int)
    gain = np.full(len(cells), .02)
    return {(0, 1): dict(cells=cells, gain=gain,
                         area_km2=float(np.sum(gain*s.cell_area[cells])))}


class TrenchDynamicsTests(unittest.TestCase):
    def test_retreat_speed_cap_age_and_cratonic_resistance(self):
        np.testing.assert_allclose(retreat_speed(np.array([-2., 0., 20., 10000.]), 80., 0.), [0., 0., 3., 8.])
        self.assertLess(retreat_speed(20., 20., 0.), retreat_speed(20., 80., 0.))
        self.assertLess(retreat_speed(20., 80., .5), retreat_speed(20., 80., 0.))
        self.assertTrue(np.isfinite(retreat_speed(20., -1., 0.)))

    def test_native_reconstruction_has_zero_retreat_at_zero_displacement(self):
        # The native sampler smooths sharp finite-volume ownership. Comparing
        # that smoothed value to a raw cell made a five-year step transfer
        # millions of km2 despite the 8 km/Myr retreat-speed cap.
        import plate_balance
        import trench_history
        from tests.test_slab_tether_native import world
        s=world()
        model=plate_balance.Balance(s,1.,slab_tethers=True)
        model.solve();s.omega=model.rotation()
        s._advect(5e-6);s.t+=5e-6
        s._rasterize();s._boundaries();trench_history.prepare(s)
        smoothed=s._sample(s.support[1],s._sample_coordinates(s.xyz))
        self.assertGreater(np.max(np.abs(smoothed-s.support[1])),.1)
        before=s.support.copy();owners=s.plate.copy()
        self.assertEqual(extend_forearcs(s,0.),{})
        np.testing.assert_array_equal(s.support,before)
        np.testing.assert_array_equal(s.plate,owners)
        rates=[]
        for interval in (5e-6,1e-4,.01,1.):
            probe=deepcopy(s);prior=probe.support.copy()
            result=extend_forearcs(probe,interval)
            swept=sum(row['area_km2'] for row in result.values())
            measured=float(np.sum((probe.support[1]-prior[1])*probe.cell_area))
            self.assertLessEqual(swept,8.*interval*float(probe.bl.sum()))
            self.assertAlmostEqual(swept,measured,delta=max(1e-11,swept*1e-8))
            np.testing.assert_allclose(probe.support.sum(axis=0),1.,atol=1e-12)
            rates.append(swept/interval)
        self.assertGreater(rates[-1],0.)
        np.testing.assert_allclose(rates,rates[-1],rtol=3e-5)

    def test_front_speed_matches_prescribed_retreat_across_steps_and_grids(self):
        measured = []
        duration = 240.
        expected = -.15*.006*duration
        for width, dt in ((128, 2.), (128, 5.), (256, 5.)):
            with self.subTest(width=width, dt=dt):
                s = globe(width)
                origin = front_longitude(s)
                before = s.support[1].copy()
                transferred = 0.
                for _ in range(round(duration/dt)):
                    result = extend_forearcs(s, dt)
                    transferred += sum(row['area_km2'] for row in result.values())
                actual = front_longitude(s)-origin
                measured.append(actual)
                self.assertAlmostEqual(actual, expected, delta=2*np.pi/width*.6,
                                       msg=f'Wrong retreat speed: {actual/expected:.3f} times prescribed')
                self.assertAlmostEqual(transferred, float(np.sum((s.support[1]-before)*s.cell_area)), delta=.001)
                np.testing.assert_allclose(s.support.sum(axis=0), 1., atol=1e-14)
        self.assertLess(max(measured)-min(measured), 2*np.pi/128*.4)

    def test_transfer_balances_membership_and_preserves_material_and_surface_fields(self):
        s = globe(128, dtype=np.float32, smooth=True)
        s.mass = np.array([234.])
        s.kind = np.array([1], np.uint8)
        s.parcel_plate = np.array([0], np.int16)
        s.pos = np.array([[1., 0., 0.]])
        frozen = {key: getattr(s, key).copy() for key in ('mass', 'kind', 'parcel_plate', 'pos', 'age', 'crust', 'density', 'omega')}
        before = s.support.copy()
        result = extend_forearcs(s, 5.)
        self.assertTrue(result)
        self.assertTrue(np.all(s.support >= 0))
        np.testing.assert_allclose(s.support.sum(axis=0), 1., atol=2e-7)
        np.testing.assert_allclose(s.support[1]-before[1], before[0]-s.support[0], atol=1.2e-7)
        measured = sum(row['area_km2'] for row in result.values())
        self.assertAlmostEqual(measured, float(np.sum((s.support[1]-before[1])*s.cell_area)), delta=3.)
        for key, value in frozen.items(): np.testing.assert_array_equal(getattr(s, key), value, err_msg=key)

    def test_hidden_buoyant_footprints_and_continental_cells_block_retreat(self):
        for field, value in (('crust', 1), ('crust', 2), ('density', .001)):
            with self.subTest(field=field, value=value):
                s = globe(smooth=True)
                getattr(s, field)[:] = value
                before = s.support.copy()
                self.assertEqual(extend_forearcs(s, 5.), {})
                np.testing.assert_array_equal(s.support, before)
        s = globe(smooth=True)
        s.density[:s.n//2] = .001
        before = s.support.copy()
        self.assertTrue(extend_forearcs(s, 5.))
        np.testing.assert_array_equal(s.support[:, :s.n//2], before[:, :s.n//2])

    def test_absent_subduction_or_nonconvergent_motion_is_a_noop(self):
        for mode in ('no-subduction', 'co-rotation', 'divergence'):
            with self.subTest(mode=mode):
                s = globe()
                if mode == 'no-subduction': s.bcode[:] = 3
                elif mode == 'co-rotation': s.omega[1] = s.omega[0]
                else:
                    # Restrict eligibility to the zero-longitude boundary;
                    # reversing rotation converges at the antipodal boundary.
                    s.density[np.abs(s.xyz[:, 1]) < .2] = np.where(s.xyz[np.abs(s.xyz[:, 1]) < .2, 0] < 0, 1., 0.)
                    s.omega *= -1
                before = s.support.copy()
                self.assertEqual(extend_forearcs(s, 5.), {})
                np.testing.assert_array_equal(s.support, before)

    def test_polar_sampling_stays_finite_and_chunk_order_does_not_change_results(self):
        first = globe(256, dtype=np.float32, smooth=True)
        first.omega[0] = [.008, 0., 0.]
        second = deepcopy(first)
        # Deepcopy retains a bound sampler for the copied object.
        a = extend_forearcs(first, 5., chunk_size=79)
        b = extend_forearcs(second, 5., chunk_size=16384)
        np.testing.assert_array_equal(first.support, second.support)
        np.testing.assert_array_equal(first.plate, second.plate)
        self.assertEqual(a.keys(), b.keys())
        for pair in a:
            np.testing.assert_array_equal(a[pair]['cells'], b[pair]['cells'])
            np.testing.assert_array_equal(a[pair]['gain'], b[pair]['gain'])
            self.assertEqual(a[pair]['area_km2'], b[pair]['area_km2'])
        moved = np.concatenate([item['cells'] for item in a.values()])
        self.assertTrue(np.any(np.abs(first.xyz[moved, 2]) > np.sin(np.deg2rad(80))))
        self.assertTrue(np.isfinite(first.support).all())
        np.testing.assert_allclose(first.support.sum(axis=0), 1., atol=2e-7)

    def test_boundary_segment_count_and_orientation_do_not_multiply_retreat(self):
        single = globe(smooth=True)
        segmented = deepcopy(single)
        # The same established hinge can have many raster boundary segments,
        # with either plate appearing on the first side of each segment.
        segmented.bcode = np.full(12, 2, np.uint8)
        segmented.down = np.zeros(12, np.int32)
        segmented.bp = np.tile([0, 1], 6)
        segmented.bq = 1-segmented.bp
        expected = extend_forearcs(single, 5.)
        actual = extend_forearcs(segmented, 5.)
        np.testing.assert_array_equal(segmented.support, single.support)
        self.assertEqual(actual.keys(), expected.keys())
        self.assertEqual(actual[0, 1]['area_km2'], expected[0, 1]['area_km2'])
        np.testing.assert_array_equal(actual[0, 1]['gain'], expected[0, 1]['gain'])

    def test_third_plate_junction_is_excluded_but_roundoff_tail_is_allowed(self):
        for third, expected in ((.2, False), (1e-5, True)):
            s = globe(smooth=True)
            s.support = np.vstack((s.support*(1-third), np.full(s.n, third)))
            s.capacity = 3
            s.omega = np.vstack((s.omega, np.zeros(3)))
            before = s.support[2].copy()
            result = extend_forearcs(s, 5.)
            self.assertEqual(bool(result), expected)
            np.testing.assert_allclose(s.support[2], before, atol=1e-14)

    def test_large_pair_transfer_cannot_concentrate_unbounded_local_arc_flux(self):
        s = segmented_trench()
        transfer = prescribed_transfer(s, np.flatnonzero(s.plate == 0))
        actual_area = transfer[0, 1]['area_km2']
        with patch('raster_engine.extend_forearcs', return_value=transfer):
            Simulation._migrate_trenches(s, 2.)
        # Known physical inputs give age/drag-limited 1 and 16/3 km/Myr,
        # then the explicit 8 km/Myr ceiling, despite a much larger transfer.
        expected = np.array([1., 16./3., 8.])
        np.testing.assert_allclose(s.trench_retreat_speed, expected, atol=1e-12)
        self.assertTrue(np.all(s.trench_retreat_speed <= 8.))
        self.assertEqual(s.process_totals['trench_swept_km2'], actual_area)
        eligible = float(np.sum(expected*s.bl)*2.)
        self.assertAlmostEqual(s.process_totals['trench_arc_sweep_km2'], eligible)
        self.assertGreater(actual_area, eligible*10,
                           'Fixture must leave real swept area unconverted to local arc supply')

    def test_arc_flux_requires_local_unblocked_realized_movement(self):
        for mode in ('hidden-buoyant', 'distant-only', 'no-transfer'):
            with self.subTest(mode=mode):
                s = segmented_trench()
                s.trench_retreat_speed = np.full(3, 8.)
                if mode == 'hidden-buoyant':
                    s.density[s.ba[0]] = .001
                    transfer = prescribed_transfer(s, np.flatnonzero(s.plate == 0))
                    expected = [0., 16./3., 8.]
                elif mode == 'distant-only':
                    transfer = prescribed_transfer(s, [0])
                    expected = [0., 0., 0.]
                else:
                    transfer, expected = {}, [0., 0., 0.]
                with patch('raster_engine.extend_forearcs', return_value=transfer):
                    Simulation._migrate_trenches(s, 2.)
                np.testing.assert_allclose(s.trench_retreat_speed, expected, atol=1e-12)
                self.assertAlmostEqual(s.process_totals['trench_arc_sweep_km2'],
                                       float(np.sum(np.asarray(expected)*s.bl)*2.))
                measured = sum(item['area_km2'] for item in transfer.values())
                self.assertEqual(s.process_totals['trench_swept_km2'], measured)


if __name__ == '__main__':
    unittest.main()
