"""Physical and numerical contracts for connected material deformation."""
import copy
import unittest

import numpy as np

from deforming_regions import deform
from material_surface import initialize_surface, spherical_face_areas
from ridge_geometry import rotate


RADIUS = 6371.


def unit(value):
    return value / np.linalg.norm(value, axis=-1, keepdims=True)


def fixture(speed=20., craton=False):
    u, v = np.meshgrid(np.linspace(-900., 0., 10), np.linspace(-300., 300., 7))
    xyz = unit(np.column_stack((np.ones(u.size), u.ravel()/RADIUS, v.ravel()/RADIUS)))
    faces = []
    for row in range(6):
        for col in range(9):
            a = row*10+col
            faces.extend(((a, a+1, a+11), (a, a+11, a+10)))
    faces = np.asarray(faces)
    kind = np.ones(len(faces), np.uint8)
    if craton:
        centre = xyz[faces].mean(axis=1)
        kind[(centre[:, 1]*RADIUS > -260) & (centre[:, 1]*RADIUS < -130)
             & (np.abs(centre[:, 2]*RADIUS) < 80)] = 2
    surface = initialize_surface(xyz, faces, np.zeros(len(faces), np.int32), kind)
    boundary = dict(bmid=np.array([[1., 0., 0.]]), bn=np.array([[0., 1., 0.]]),
                    bl=np.array([1800.]), bp=np.array([0]), bq=np.array([1]))
    omega = np.array([[0., 0., -speed/RADIUS], [0., 0., speed/RADIUS]])
    return surface, omega, boundary


class DeformingRegionsTests(unittest.TestCase):
    def test_common_euler_motion_is_exactly_rigid_and_does_not_mutate_inputs(self):
        surface, omega, boundary = fixture()
        omega[:] = [.004, -.008, .012]
        before = copy.deepcopy(surface)
        result = deform(surface, omega, boundary, 2.)
        expected = unit(rotate(surface['vertices'], omega[surface['vertex_owner']]*2))
        np.testing.assert_array_equal(result['vertices'], expected)
        np.testing.assert_array_equal(result['areal_strain'], 0.)
        np.testing.assert_array_equal(result['residual_velocity_km_myr'], 0.)
        for key in before:
            if isinstance(before[key], np.ndarray):
                np.testing.assert_array_equal(surface[key], before[key])

    def test_extension_and_compression_change_actual_area_and_close_column_volume(self):
        for speed, sign in ((20., 1), (-20., -1)):
            surface, omega, boundary = fixture(speed)
            result = deform(surface, omega, boundary, 2.)
            old = spherical_face_areas(surface['vertices'], surface['faces'], RADIUS)
            actual = spherical_face_areas(result['vertices'], surface['faces'], RADIUS)
            self.assertGreater(sign*(actual.sum()/old.sum()-1), .025)
            np.testing.assert_allclose(result['area_ratio'], actual/old, atol=2e-14)
            np.testing.assert_allclose(result['areal_strain'], np.log(actual/old), atol=2e-14)
            # The caller's reciprocal thickness update preserves each column.
            thickness = 35./result['area_ratio']
            np.testing.assert_allclose(actual*thickness, old*35., rtol=3e-14)
            self.assertTrue(result['diagnostics']['solver']['converged'])
            self.assertTrue(np.all(actual > 0))
            self.assertGreater(result['diagnostics']['deforming_vertices'], 0)

    def test_every_craton_incident_vertex_stays_exactly_rigid(self):
        surface, omega, boundary = fixture(craton=True)
        result = deform(surface, omega, boundary, 2.)
        protected = np.unique(surface['faces'][surface['face_kind'] == 2])
        self.assertGreater(len(protected), 0)
        np.testing.assert_array_equal(result['vertices'][protected], result['rigid_vertices'][protected])
        np.testing.assert_array_equal(result['residual_velocity_km_myr'][protected], 0.)
        np.testing.assert_array_equal(result['areal_strain'][surface['face_kind'] == 2], 0.)
        self.assertGreater(np.max(np.linalg.norm(result['vertices']-result['rigid_vertices'], axis=1)), .001)

    def test_unused_vertices_are_finite_and_exactly_rigid(self):
        surface, omega, boundary = fixture()
        surface['vertices'] = np.vstack((surface['vertices'], [1., 0., 0.]))
        surface['vertex_owner'] = np.r_[surface['vertex_owner'], 0]
        result = deform(surface, omega, boundary, 2.)
        self.assertTrue(result['rigid_mask'][-1])
        np.testing.assert_array_equal(result['vertices'][-1], result['rigid_vertices'][-1])
        self.assertTrue(np.isfinite(result['vertices']).all())

    def test_globe_rotation_equivariance_including_poles(self):
        surface, omega, boundary = fixture(craton=True)
        original = deform(surface, omega, boundary, 2.)
        for axis in (np.array([0., -np.pi/2, 0.]), np.array([1.1, -.71, .82])):
            turned = copy.deepcopy(surface)
            turned['vertices'] = rotate(surface['vertices'], axis)
            rotated_boundary = copy.deepcopy(boundary)
            for key in ('bmid', 'bn'):
                rotated_boundary[key] = rotate(boundary[key], axis)
            result = deform(turned, rotate(omega, axis), rotated_boundary, 2.)
            np.testing.assert_allclose(result['vertices'], rotate(original['vertices'], axis), atol=2e-13)
            np.testing.assert_allclose(result['area_ratio'], original['area_ratio'], atol=2e-11)
            np.testing.assert_array_equal(result['craton_mask'], original['craton_mask'])

    def test_column_bounds_limit_geometry_and_do_not_clip_reported_strain(self):
        for speed, lower, upper in ((90., .99, 1.02), (-90., .98, 1.01)):
            surface, omega, boundary = fixture(speed)
            count = len(surface['faces'])
            result = deform(surface, omega, boundary, 2.,
                            face_min_area_ratio=np.full(count, lower),
                            face_max_area_ratio=np.full(count, upper))
            ratio = result['face_area_km2']/spherical_face_areas(surface['vertices'], surface['faces'], RADIUS)
            self.assertTrue(np.all(ratio >= lower-2e-12))
            self.assertTrue(np.all(ratio <= upper+2e-12))
            np.testing.assert_allclose(result['areal_strain'], np.log(ratio), atol=2e-14)
            self.assertGreater(result['diagnostics']['backtracks'], 0)
            self.assertGreater(result['diagnostics']['limited_faces'], 0)
            self.assertLess(result['diagnostics']['accepted_residual_fraction'], .5)

    def test_one_constrained_component_does_not_freeze_a_disconnected_belt(self):
        surface, omega, boundary = fixture()
        points, faces = surface['vertices'], surface['faces']
        doubled = initialize_surface(np.vstack((points, points)), np.vstack((faces, faces+len(points))),
                                     np.zeros(2*len(faces), np.int32), np.ones(2*len(faces), np.uint8))
        maximum = np.r_[np.ones(len(faces)), np.full(len(faces), 4.)]
        result = deform(doubled, omega, boundary, 2., face_max_area_ratio=maximum)
        self.assertEqual(len(result['diagnostics']['regions']), 2)
        np.testing.assert_allclose(result['area_ratio'][:len(faces)], 1., atol=2e-12)
        self.assertGreater(result['area_ratio'][len(faces):].max(), 1.05)

    def test_failed_solve_cannot_supply_deformation(self):
        surface, omega, boundary = fixture()
        result = deform(surface, omega, boundary, 2., iterations=1, tolerance=1e-13)
        self.assertFalse(result['diagnostics']['solver']['converged'])
        self.assertEqual(result['diagnostics']['solver']['iterations'], 1)
        np.testing.assert_allclose(result['vertices'], result['rigid_vertices'], atol=0.)
        np.testing.assert_array_equal(result['residual_velocity_km_myr'], 0.)
        np.testing.assert_array_equal(result['areal_strain'], 0.)

    def test_legitimate_initial_sliver_does_not_require_a_regular_triangle(self):
        surface, omega, boundary = fixture()
        points, faces = surface['vertices'], surface['faces']
        # An interior point near one existing corner creates two legitimate
        # slivers without a T-junction or changing the patch outline.
        at = 52
        a, b, c = faces[at]
        vertex = len(points)
        points = np.vstack((points, unit((1-2e-5)*points[a]+1e-5*(points[b]+points[c]))))
        faces = np.vstack((np.delete(faces, at, axis=0), [a, b, vertex], [b, c, vertex], [c, a, vertex]))
        surface = initialize_surface(points, faces, np.zeros(len(faces), np.int32), np.ones(len(faces), np.uint8))
        result = deform(surface, omega, boundary, 2.)
        self.assertTrue(result['diagnostics']['solver']['converged'])
        self.assertGreater(np.max(result['area_ratio']), 1.02)
        self.assertTrue(np.all(result['face_area_km2'] > 0))
        self.assertLessEqual(result['diagnostics']['substeps'], 64)

    def test_owner_and_admission_masks_do_not_invent_boundary_loading(self):
        for mode in ('unrelated_owner', 'invalid', 'not_admitted'):
            surface, omega, boundary = fixture()
            if mode == 'unrelated_owner':
                surface['face_owner'][:] = 2
                surface['vertex_owner'][:] = 2
                omega = np.vstack((omega, [.002, -.003, .001]))
            elif mode == 'invalid':
                boundary['valid'] = np.array([False])
            else:
                boundary['admit_normal'] = np.array([False])
                boundary['admit_shear'] = np.array([False])
            result = deform(surface, omega, boundary, 2.)
            np.testing.assert_array_equal(result['residual_velocity_km_myr'], 0.)
            np.testing.assert_array_equal(result['areal_strain'], 0.)

    def test_zero_timestep_preserves_geometry_and_no_realized_velocity(self):
        surface, omega, boundary = fixture()
        result = deform(surface, omega, boundary, 0.)
        np.testing.assert_allclose(result['vertices'], surface['vertices'], atol=2e-16)
        np.testing.assert_array_equal(result['actual_vertex_velocity_km_myr'], 0.)
        np.testing.assert_array_equal(result['accepted_residual_fraction'], 0.)


if __name__ == '__main__':
    unittest.main()
