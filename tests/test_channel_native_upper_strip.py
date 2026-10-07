"""Native material identity and metric for the read-only upper flexure strip."""

import unittest

import numpy as np

from channel_native_upper_strip import (
    bind_trench_normal_upper_strip, bind_upper_strip, trench_normal_points)
from channel_upper_basal_reference import install_unloaded_upper_bases
from ridge_geometry import rotate
from tests.test_channel_region_native import uniform_world


def unit(vector):
    return vector / np.linalg.norm(vector)


def great_circle_points(a, b, count=5):
    angle = np.arccos(np.clip(a @ b, -1., 1.))
    fractions = np.linspace(0., 1., count)
    return np.array([(np.sin((1. - t) * angle) * a +
                      np.sin(t * angle) * b) / np.sin(angle)
                     for t in fractions])


def fixture():
    s = uniform_world()
    s.parcel_collision_sheet = np.arange(1, len(s.mass) + 1)
    face_id = int(s.channel_region_store['records'][0]['face_id'])
    triangle = s.channel_region_store['records'][0]['face_triangle']
    a, b, c = triangle
    points = great_circle_points(unit(.6 * a + .2 * b + .2 * c),
                                 unit(.2 * a + .6 * b + .2 * c))
    face_ids = np.full(len(points), face_id, dtype=np.int64)
    return s, face_ids, points


class NativeUpperStripTests(unittest.TestCase):
    def test_signed_hinge_normal_recovers_native_strip_and_rotates_equivariantly(self):
        s, face_ids, points = fixture()
        install_unloaded_upper_bases(s, [int(face_ids[0])])
        hinge_normal = unit(points[-1] - (points[-1] @ points[0]) * points[0])
        angle = np.arccos(np.clip(points[0] @ points[-1], -1., 1.))
        offsets = np.linspace(0., angle * 6371000., len(points))
        geometry = trench_normal_points(hinge_normal, points[2], offsets, 6371.)
        np.testing.assert_allclose(geometry['xyz'], points, atol=2e-14)
        np.testing.assert_allclose(geometry['hinge_xyz'], points[0], atol=2e-14)
        result = bind_trench_normal_upper_strip(
            s, face_ids, hinge_normal, points[2], offsets)
        np.testing.assert_allclose(result['xyz'], points, atol=2e-14)
        np.testing.assert_allclose(result['contact_offset_m'], offsets[2], atol=1e-6)
        rotation = np.array([.2, -.1, .3])
        rotated = trench_normal_points(
            rotate(hinge_normal[None], rotation)[0],
            rotate(points[2:3], rotation)[0], offsets, 6371.)
        np.testing.assert_allclose(rotated['xyz'], rotate(points, rotation), atol=2e-14)

    def test_trench_normal_rejects_wrong_side_or_unbracketed_contact(self):
        _, _, points = fixture()
        hinge_normal = unit(points[-1] - (points[-1] @ points[0]) * points[0])
        angle = np.arccos(np.clip(points[0] @ points[-1], -1., 1.))
        offsets = np.linspace(0., angle * 6371000., len(points))
        with self.assertRaisesRegex(ValueError, 'positive entry side'):
            trench_normal_points(-hinge_normal, points[2], offsets, 6371.)
        with self.assertRaisesRegex(ValueError, 'bracket'):
            trench_normal_points(hinge_normal, points[2], offsets * .2, 6371.)
        with self.assertRaisesRegex(ValueError, 'bracket'):
            trench_normal_points(hinge_normal, points[2], offsets + 2. * offsets[-1], 6371.)
        nonuniform = offsets.copy()
        nonuniform[2] += 100.
        with self.assertRaisesRegex(ValueError, 'ordered metre offsets'):
            trench_normal_points(hinge_normal, points[2], nonuniform, 6371.)

    def test_saved_regional_base_and_native_distance(self):
        s, face_ids, points = fixture()
        install_unloaded_upper_bases(s, [int(face_ids[0])])
        before = s.channel_region_store
        result = bind_upper_strip(s, face_ids, points)
        self.assertIs(s.channel_region_store, before)
        base = before['records'][0]['history']['regions'][0][
            'upper_basal_reference']['depth_m']
        np.testing.assert_array_equal(result['unloaded_upper_base_depth_m'], base)
        self.assertEqual(result['reference_epoch_myr'], s.t)
        self.assertEqual(result['collision_sheet'], s.parcel_collision_sheet[0])
        self.assertEqual(result['plate_owner'], s.parcel_plate[0])
        self.assertEqual(result['region_id'], ('uniform',) * len(points))
        angle = np.arccos(np.clip(points[0] @ points[-1], -1., 1.))
        np.testing.assert_allclose(result['spacing_m'] * (len(points) - 1),
                                   angle * 6371000., rtol=1e-12)
        np.testing.assert_allclose(np.diff(result['distance_m']),
                                   result['spacing_m'], rtol=1e-9)
        result['unloaded_upper_base_depth_m'][0] = 0.
        self.assertEqual(before['records'][0]['history']['regions'][0][
            'upper_basal_reference']['depth_m'], base)

    def test_missing_or_future_reference_and_stale_store_fail_closed(self):
        s, face_ids, points = fixture()
        with self.assertRaisesRegex(ValueError, 'unloaded base'):
            bind_upper_strip(s, face_ids, points)
        install_unloaded_upper_bases(s, [int(face_ids[0])])
        datum = s.channel_region_store['records'][0]['history']['regions'][0][
            'upper_basal_reference']
        datum['epoch_myr'] = s.t + 1.
        with self.assertRaisesRegex(ValueError, 'past pre-contact'):
            bind_upper_strip(s, face_ids, points)
        datum['epoch_myr'] = s.t
        s.material_surface['geometry_revision'] += 1
        with self.assertRaisesRegex(ValueError, 'current material geometry'):
            bind_upper_strip(s, face_ids, points)

    def test_irregular_curved_or_reversed_sampling_is_rejected(self):
        s, face_ids, points = fixture()
        install_unloaded_upper_bases(s, [int(face_ids[0])])
        curved = points.copy()
        curved[2] = unit(curved[2] + .01 *
                         s.channel_region_store['records'][0]['face_triangle'][2])
        with self.assertRaisesRegex(ValueError, 'great circle'):
            bind_upper_strip(s, face_ids, curved)
        irregular = points.copy()
        irregular[2] = unit(points[1] + .2 * points[2])
        with self.assertRaisesRegex(ValueError, 'great circle'):
            bind_upper_strip(s, face_ids, irregular)
        reversed_middle = points.copy()
        reversed_middle[1], reversed_middle[2] = points[2], points[1]
        with self.assertRaisesRegex(ValueError, 'great circle'):
            bind_upper_strip(s, face_ids, reversed_middle)

    def test_sheet_and_material_geometry_must_remain_identified(self):
        s, face_ids, points = fixture()
        install_unloaded_upper_bases(s, [int(face_ids[0])])
        different = face_ids.copy()
        different[2] = s.material_surface['face_id'][1]
        with self.assertRaisesRegex(ValueError, 'one identified sheet and plate owner'):
            bind_upper_strip(s, different, points)
        s.parcel_collision_sheet[1] = s.parcel_collision_sheet[0]
        s.parcel_plate[1] = s.parcel_plate[0] + 1
        s.material_surface['face_owner'][1] = s.parcel_plate[1]
        with self.assertRaisesRegex(ValueError, 'one identified sheet and plate owner'):
            bind_upper_strip(s, different, points)
        s.material_surface['face_owner'][1] -= 1
        with self.assertRaisesRegex(ValueError, 'aligned native sheet and plate owners'):
            bind_upper_strip(s, face_ids, points)
        s.material_surface['face_owner'][1] = s.parcel_plate[1]
        s.channel_region_store['records'][0]['face_triangle'] = (
            s.channel_region_store['records'][0]['face_triangle'].copy())
        s.channel_region_store['records'][0]['face_triangle'][0, 0] += 1e-5
        with self.assertRaisesRegex(ValueError, 'stale'):
            bind_upper_strip(s, face_ids, points)


if __name__ == '__main__':
    unittest.main()
