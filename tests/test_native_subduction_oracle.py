"""Finite-trench removal checked with independent scalar spherical areas.

These fixtures supply ample transported support to isolate the geometry from
the finite-volume advection stencil. The real transport fixture is separate.
"""
import unittest
import numpy as np

import native_subduction
from tests.test_native_spreading_pairing import scene
from tests._native_spreading_oracle import polygon_area, material_rectangles


def z_rotation(angle):
    return np.array([[np.cos(angle), -np.sin(angle), 0.],
                     [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])


def fixture(*, split=1, level=3, maturity=1.):
    s, axis = scene(split=split, level=level, extent=600.)
    s.omega *= -1.
    s.bcode = np.array([2], np.int8)
    s.down = np.array([0], np.int16)
    s.normal_speed = np.array([-12.742])
    s.crust = np.zeros(s.n, np.int8)
    s.trench_systems = []
    s.trench_maturity = np.array([maturity])
    s.native_subduction_version = 1
    s.material_surface['face_owner'] = np.empty(0, np.int16)
    s.material_surface['face_kind'] = np.empty(0, np.int8)
    transported = np.zeros_like(s.support)
    transported[:2] = .75
    return s, axis, transported


def strip_area(axis, dt, *, half=False):
    left = axis @ z_rotation(-.001*dt).T
    right = axis @ z_rotation(.001*dt).T
    area = polygon_area(np.array([left[0], right[0], right[1], left[1]]))
    # Reflection y -> -y bisects this symmetric finite polygon exactly. Its
    # geodesic end caps cross y=0 at unit(left+right), not the original axis
    # endpoints; using the latter invents slightly different end triangles.
    return area/2. if half else area


class NativeSubductionOracleTests(unittest.TestCase):
    def measured(self, s, transported, dt=2.):
        original = transported.copy()
        material = {key: value.copy() for key, value in s.material_surface.items()
                    if isinstance(value, np.ndarray)}
        removed = native_subduction.removal(s, transported, dt)
        self.assertEqual(removed.shape, transported.shape)
        self.assertTrue(np.isfinite(removed).all())
        self.assertTrue(np.all(removed >= 0.))
        self.assertTrue(np.all(removed <= transported+1e-12))
        self.assertTrue(np.all(removed.sum(axis=0) <=
                               np.maximum(transported.sum(axis=0)-1., 0.)+1e-12))
        np.testing.assert_array_equal(transported, original)
        for key, value in material.items():
            np.testing.assert_array_equal(s.material_surface[key], value)
        return removed, float(removed.sum(axis=0) @ s.cell_area)

    def test_finite_strip_area_across_control_levels_and_timesteps(self):
        for level in (3, 4):
            for dt in (1., 2.):
                with self.subTest(level=level, dt=dt):
                    s, axis, transported = fixture(level=level)
                    removed, area = self.measured(s, transported, dt)
                    self.assertAlmostEqual(area, strip_area(axis, dt), delta=1e-5)
                    np.testing.assert_array_equal(removed[1:], 0.)

    def test_subdividing_the_same_front_does_not_change_removed_area(self):
        values = []
        for split in (1, 2, 4):
            s, axis, transported = fixture(split=split)
            _, area = self.measured(s, transported)
            self.assertAlmostEqual(area, strip_area(axis, 2.), delta=1e-5)
            values.append(area)
        self.assertLess(max(values)-min(values), 1e-5)

    def test_maturity_scales_the_same_geometric_request(self):
        for maturity in (0., .5, 1.):
            s, axis, transported = fixture(maturity=maturity)
            _, area = self.measured(s, transported)
            self.assertAlmostEqual(area, maturity*strip_area(axis, 2.), delta=1e-5)

    def test_opening_front_removes_no_support(self):
        s, _, transported = fixture()
        s.omega *= -1.
        s.normal_speed *= -1.
        removed, area = self.measured(s, transported)
        self.assertEqual(area, 0.)
        np.testing.assert_array_equal(removed, 0.)

    def test_only_downgoing_material_excludes_partial_water(self):
        for owner in (0, 1):
            s, axis, transported = fixture(level=4)
            end_vertices, faces = material_rectangles([(0., 500., -900., 900.)])
            # The supplied continent reaches the same fixed right-half footprint
            # at the end of the step, regardless of which plate owns it.
            angle = s.omega[owner, 2]*2.
            s.material_surface = dict(vertices=end_vertices @ z_rotation(-angle).T,
                faces=faces, vertex_owner=np.full(len(end_vertices), owner, np.int16),
                face_owner=np.full(len(faces), owner, np.int16),
                face_kind=np.ones(len(faces), np.int8))
            # Deliberately identical categorical labels: actual owned triangles
            # decide whether the incoming material is water or continent.
            s.crust[:] = 1
            _, area = self.measured(s, transported)
            self.assertAlmostEqual(area, strip_area(axis, 2., half=owner == 0), delta=1e-5)

    def test_removal_is_bounded_by_actual_available_overlap(self):
        s, _, transported = fixture()
        transported[0] = 1e-7
        transported[1] = 1.
        removed, area = self.measured(s, transported)
        self.assertGreater(area, 0.)
        self.assertLessEqual(float(removed.max()), 1e-7+1e-14)


if __name__ == '__main__':
    unittest.main()
