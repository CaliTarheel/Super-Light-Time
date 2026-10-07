from copy import deepcopy
import tempfile
from pathlib import Path
import unittest

import numpy as np

from orientation import (inverse_cells, normalize_orientation, orient_frame,
                         orient_initial, orient_metadata, reproject_raster,
                         rotation_matrix)


def sphere(width=128):
    height = width//2
    lon, lat = np.meshgrid((np.arange(width)+.5)*2*np.pi/width-np.pi,
                          np.pi/2-(np.arange(height)+.5)*np.pi/height)
    return np.stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)), axis=-1)


class OrientationTests(unittest.TestCase):
    def test_normalization_and_rejection_do_not_silently_change_bad_requests(self):
        self.assertEqual(normalize_orientation(), dict(yaw=0., pitch=0., roll=0.))
        self.assertEqual(normalize_orientation(dict(yaw=540, pitch=270, roll=-720)),
                         dict(yaw=-180., pitch=-90., roll=0.))
        for bad in ([], '90', 2, dict(heading=3), dict(yaw=True), dict(pitch='30'),
                    dict(roll=None), dict(yaw=float('inf')), dict(roll=float('nan'))):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                normalize_orientation(bad)

    def test_true_rotation_preserves_lengths_angles_handedness_and_inverse(self):
        np.testing.assert_array_equal(np.array([1, 0, 0]) @ rotation_matrix(dict(yaw=90)), [0, 1, 0])
        np.testing.assert_array_equal(np.array([1, 0, 0]) @ rotation_matrix(dict(pitch=90)), [0, 0, -1])
        np.testing.assert_array_equal(np.array([0, 1, 0]) @ rotation_matrix(dict(roll=90)), [0, 0, 1])
        matrix = rotation_matrix(dict(yaw=71, pitch=-116, roll=32))
        np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=5e-16)
        self.assertAlmostEqual(np.linalg.det(matrix), 1., places=14)
        points = np.random.default_rng(8).normal(size=(50, 3))
        moved = points @ matrix
        np.testing.assert_allclose(moved @ matrix.T, points, atol=2e-15)
        np.testing.assert_allclose(moved @ moved.T, points @ points.T, atol=5e-15)
        np.testing.assert_allclose(np.cross(moved[0], moved[1]), np.cross(points[0], points[1]) @ matrix, atol=2e-15)

    def test_identity_is_byte_exact_and_does_not_alias_source(self):
        data = np.arange(32, dtype=np.float32)
        data[0] = -0.
        frame = dict(width=8, height=4, elevation=data, crust=np.arange(32, dtype=np.uint8)%4,
                     events=[dict(lon=12, lat=25, details={'x': [1, 2]})], time_myr=312.)
        result = orient_frame(frame, dict(yaw=360, pitch=-720))
        self.assertEqual(result['elevation'].tobytes(), data.tobytes())
        self.assertEqual(set(result), set(frame))
        result['elevation'][2] = 91
        result['events'][0]['details']['x'][0] = 9
        self.assertEqual(frame['elevation'][2], 2.)
        self.assertEqual(frame['events'][0]['details']['x'][0], 1)

    def test_quarter_yaw_is_exact_in_both_directions_and_preserves_shared_categories(self):
        width, height = 64, 32
        cells = np.arange(width*height, dtype=np.int32)
        frame = dict(width=width, height=height, plate=cells%17, domain=cells+300,
                     crust=(cells%4).astype(np.uint8), elevation=cells.astype(np.float32),
                     age=(cells/7).astype(np.float32).reshape(height, width))
        rotated = orient_frame(frame, dict(yaw=90))
        for key in ('elevation', 'plate', 'domain', 'crust', 'age'):
            expected = np.roll(frame[key].reshape(height, width), width//4, axis=1)
            np.testing.assert_array_equal(rotated[key].reshape(height, width), expected)
            np.testing.assert_array_equal(orient_frame(rotated, dict(yaw=-90))[key], frame[key])
        sampled = inverse_cells(cells, width, height, dict(yaw=90))
        np.testing.assert_array_equal(rotated['domain'], frame['domain'][sampled])

    def test_pole_crossing_does_not_punch_holes_in_intact_cap_interiors(self):
        width = 256
        points = sphere(width)
        radius = np.radians(33)
        cap = (points[..., 0] > np.cos(radius)).astype(np.uint8)*2
        frame = dict(width=width, height=width//2, crust=cap, domain=np.where(cap, 401, 19),
                     plate=np.where(cap, 7, 0))
        for orientation in (dict(pitch=-90), dict(pitch=90), dict(yaw=177, pitch=119, roll=62)):
            moved = orient_frame(frame, orientation)
            inverse = points @ rotation_matrix(orientation).T
            distance = np.arccos(np.clip(inverse[..., 0], -1, 1))
            interior = distance < radius-2*np.pi/width
            exterior = distance > radius+2*np.pi/width
            self.assertTrue(np.all(moved['crust'][interior] == 2))
            self.assertTrue(np.all(moved['crust'][exterior] == 0))
            self.assertTrue(np.all(moved['domain'][moved['crust'] == 2] == 401))
            self.assertTrue(np.all(moved['plate'][moved['crust'] == 2] == 7))
            self.assertEqual(set(np.unique(moved['crust'])), {0, 2})
        # A source polar cap, transported across the seam, has the same guarantee.
        frame['crust'] = (points[..., 2] > np.cos(radius)).astype(np.uint8)
        orientation = dict(pitch=90, yaw=180)
        moved = orient_initial(frame, orientation)
        inverse = points @ rotation_matrix(orientation).T
        self.assertTrue(np.all(moved['crust'][inverse[..., 2] > np.cos(radius-2*np.pi/width)] == 1))

    def test_scalar_bilinear_is_spherical_and_continuous_at_seam_and_poles(self):
        points = sphere(256)
        direction = np.array([1300., 2500., -500.])
        source = (points @ direction).astype(np.float32)
        orientation = dict(yaw=172, pitch=90, roll=31)
        moved = orient_frame(dict(width=256, height=128, elevation=source), orientation)['elevation']
        expected = points @ rotation_matrix(orientation).T @ direction
        self.assertLess(float(np.abs(moved-expected).max()), 1.)
        self.assertLess(float(np.abs((moved[:, 0]-moved[:, -1])-(expected[:, 0]-expected[:, -1])).max()), 1.)
        self.assertTrue(np.isfinite(moved).all())
        # With odd height, a destination pixel can hit an exact source pole.
        # Its scalar value has one unique limit, regardless of atan2 roundoff.
        width, height = 18, 9
        source = np.ones((height, width), np.float64)
        source[0] = np.arange(width)
        destination = np.empty_like(source)
        reproject_raster(source, destination, dict(pitch=90, yaw=10))
        self.assertTrue(np.isfinite(destination).all())
        self.assertGreaterEqual(float(destination.min()), float(source.min()))
        self.assertLessEqual(float(destination.max()), float(source.max()))

    def test_all_history_geometry_uses_one_rotation_and_keeps_material_identity(self):
        xyz = np.array([[1., 0, 0], [0, 1., 0], [0, 0, 1.]])
        velocity = np.array([.001, -.004, .007])
        orientation = dict(yaw=78, pitch=-51, roll=13)
        matrix = rotation_matrix(orientation)
        original = dict(width=8, height=4, crust=np.ones(32, np.uint8), trace_xyz=xyz,
                        trace_id=np.array([400, 800, 900]), trace_plate_uid=np.array([100, 100, 200]),
                        trace_relief_m=np.array([100., 200., 300.]),
                        plates=[dict(uid=100, angular_velocity=velocity.tolist())],
                        domains=[dict(uid=12, name='Fragment 012', area_km2=12.5, created_myr=40.)],
                        events=[dict(id=3, lon=0., lat=0., time_myr=20., details=dict(fractures=[
                            dict(center=[1., 0, 0], normal=[0, 0, 1.], offsets_rad=[.1, .2])]))])
        frozen = deepcopy(original)
        for epoch in (0., 200., 1000.):
            original['time_myr'] = epoch
            moved = orient_frame(original, orientation)
            np.testing.assert_allclose(moved['trace_xyz'], xyz @ matrix, atol=1e-15)
            omega = moved['plates'][0]['angular_velocity']
            np.testing.assert_allclose(np.cross(omega, moved['trace_xyz']), np.cross(velocity, xyz) @ matrix, atol=1e-17)
            event = moved['events'][0]
            lon, lat = np.radians([event['lon'], event['lat']])
            event_xyz = [np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)]
            np.testing.assert_allclose(event_xyz, xyz[0] @ matrix, atol=1e-15)
            np.testing.assert_allclose(event['details']['fractures'][0]['normal'], xyz[2] @ matrix, atol=1e-15)
            self.assertEqual(moved['domains'], original['domains'])
            self.assertEqual(moved['time_myr'], epoch)
            for key in ('trace_id', 'trace_plate_uid', 'trace_relief_m'):
                np.testing.assert_array_equal(moved[key], original[key])
        np.testing.assert_array_equal(original['trace_xyz'], frozen['trace_xyz'])
        self.assertEqual(original['events'], frozen['events'])
        self.assertEqual(original['plates'], frozen['plates'])

    def test_public_metadata_initial_and_chunked_memmap_apis_agree(self):
        width = 512  # More than one bounded chunk.
        points = sphere(width)
        source = (points[..., 0]*500+points[..., 2]*800).astype(np.float32)
        orientation = dict(yaw=-141, pitch=67, roll=9)
        frame = dict(width=width, height=width//2, elevation=source,
                     crust=(source > 0).astype(np.uint8), craton=(source > 400).astype(np.int16))
        expected = orient_frame(frame, orientation)
        with tempfile.TemporaryDirectory() as directory:
            destination = np.lib.format.open_memmap(Path(directory)/'rotated.npy', mode='w+', dtype=np.float32, shape=source.shape)
            progress = []
            try:
                reproject_raster(source, destination, orientation, progress=progress.append)
                np.testing.assert_array_equal(destination, expected['elevation'])
                self.assertEqual(progress[-1], 1.)
                self.assertEqual(progress, sorted(progress))
                self.assertGreater(len(progress), 1)
            finally:
                destination._mmap.close()
        cells = np.arange(source.size).reshape(source.shape)
        indices = inverse_cells(cells, width, width//2, orientation)
        np.testing.assert_array_equal(expected['crust'], frame['crust'].reshape(-1)[indices])
        initial = orient_initial({key: value for key, value in frame.items() if key != 'elevation'}, orientation)
        np.testing.assert_array_equal(initial['craton'], expected['craton'])
        point = orient_metadata({'selection': {'lon': 30., 'lat': 20.}}, dict(yaw=90))
        self.assertAlmostEqual(point['selection']['lon'], 120.)
        self.assertAlmostEqual(point['selection']['lat'], 20.)

    def test_slab_window_footprint_and_evidence_rotate_with_the_world(self):
        episode = dict(id=7, center=[1., 0., 0.], overriding_plate_uid=5,
                       incoming_plate_uids=[2, 3], started_myr=2., end_myr=34., radius_km=450.,
                       evidence=dict(xyz=[0., 1., 0.], normal=[1., 0., 0.]))
        frame = dict(width=8, height=4, crust=np.ones(32, np.uint8), ridge_episodes=[episode],
                     trace_ridge_uplift_m=np.array([12.]), trace_ridge_thermal_m=np.array([230.]))
        rotated = orient_frame(frame, dict(yaw=90))
        np.testing.assert_allclose(rotated['ridge_episodes'][0]['center'], [0., 1., 0.], atol=1e-15)
        np.testing.assert_allclose(rotated['ridge_episodes'][0]['evidence']['xyz'], [-1., 0., 0.], atol=1e-15)
        np.testing.assert_allclose(rotated['ridge_episodes'][0]['evidence']['normal'], [0., 1., 0.], atol=1e-15)
        self.assertEqual(rotated['ridge_episodes'][0]['incoming_plate_uids'], [2, 3])
        self.assertEqual(rotated['ridge_episodes'][0]['radius_km'], 450.)
        np.testing.assert_array_equal(rotated['trace_ridge_thermal_m'], frame['trace_ridge_thermal_m'])
        self.assertEqual(frame['ridge_episodes'][0]['center'], [1., 0., 0.])

    def test_backarc_centers_rotate_across_history_without_altering_phase_or_opening(self):
        basin = dict(id=12, started_myr=40., phase='rifting', parent_plate_uid=7,
                     arc_plate_uid=None, downgoing_plate_uid=9, center=[1., 0., 0.],
                     opening_km=24., extension_rate_km_myr=2.5, loading_km=11.,
                     last_active_myr=52., rupture_myr=None)
        for elapsed, center in ((52., [1., 0., 0.]), (54., [0., 1., 0.])):
            original = dict(basin, center=center)
            frame = dict(width=8, height=4, crust=np.ones(32, np.uint8),
                         time_myr=elapsed, backarc_basins=[original],
                         events=[dict(type='backarc_rifting', backarc_id=12, lon=0., lat=0.)])
            rotated = orient_frame(frame, dict(yaw=90))
            np.testing.assert_allclose(rotated['backarc_basins'][0]['center'],
                                       np.asarray(center) @ rotation_matrix(dict(yaw=90)), atol=1e-15)
            for key in original.keys() - {'center'}:
                self.assertEqual(rotated['backarc_basins'][0][key], original[key])
            self.assertEqual(rotated['time_myr'], elapsed)
            self.assertAlmostEqual(rotated['events'][0]['lon'], 90.)
            self.assertEqual(frame['backarc_basins'][0]['center'], center)
            identity = orient_frame(frame)
            self.assertEqual(identity['backarc_basins'], frame['backarc_basins'])
            self.assertIsNot(identity['backarc_basins'], frame['backarc_basins'])
        legacy = orient_frame(dict(width=8, height=4, crust=np.ones(32, np.uint8)), dict(pitch=90))
        self.assertNotIn('backarc_basins', legacy)

    def test_cancellation_and_invalid_buffers_are_explicit(self):
        source = np.ones((32, 64), np.float32)
        with self.assertRaises(InterruptedError):
            reproject_raster(source, np.empty_like(source), dict(pitch=20), cancel=lambda: True)
        with self.assertRaises(ValueError):
            reproject_raster(source, source, dict(pitch=20))
        with self.assertRaises(ValueError):
            reproject_raster(source, np.empty(source.shape, np.int16), dict(pitch=20))
        for frame in (dict(width=7, height=4), dict(width=True, height=1),
                      dict(width=8, height=4, crust=np.ones(31)),
                      dict(width=8, height=4, elevation=np.full(32, np.nan))):
            with self.assertRaises(ValueError):
                orient_frame(frame, dict(yaw=90))
        with self.assertRaises(ValueError):
            inverse_cells([-1], 64, 32)


if __name__ == '__main__':
    unittest.main()
