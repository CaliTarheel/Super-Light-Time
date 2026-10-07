"""Lateral magma growth keeps existing deformation instead of regularizing arcs."""
import unittest
import numpy as np
import mesh_geometry
from native_processes import _grow_arc_triangles, _arc_triangles
from gospl_export import rotation


def unit(points):
    return points/np.linalg.norm(points, axis=-1, keepdims=True)


def area(triangles):
    return mesh_geometry.spherical_area(triangles.reshape(-1, 3), np.arange(triangles.size//3).reshape(-1, 3))


def bearings_and_angles(triangles, centres):
    dot = np.einsum('nvi,ni->nv', triangles, centres)
    tangent = triangles-centres[:, None, :]*dot[:, :, None]
    length = np.linalg.norm(tangent, axis=2)
    return tangent/length[:, :, None], np.arctan2(length, dot)


class ArcShapeGrowthTests(unittest.TestCase):
    def footprint(self):
        return unit(np.array([[[1., -.03, -.015], [1., .022, -.009], [1., .007, .03]]]))

    def test_growth_preserves_each_deformed_corner_bearing_and_relative_radius(self):
        before = self.footprint(); centres = unit(before.sum(axis=1)); target = area(before)*2.7
        after = _grow_arc_triangles(before, target)
        np.testing.assert_allclose(area(after), target, rtol=3e-13)
        old_bearings, old_angles = bearings_and_angles(before, centres)
        new_bearings, new_angles = bearings_and_angles(after, centres)
        np.testing.assert_allclose(new_bearings, old_bearings, atol=3e-14)
        scale = new_angles/old_angles
        np.testing.assert_allclose(scale, np.broadcast_to(scale[:, :1], scale.shape), atol=2e-14)
        self.assertGreater(float(np.ptp(new_angles)), .001)

    def test_growth_is_rotation_equivariant_across_pole_and_seam(self):
        before = self.footprint(); target = area(before)*1.2
        matrix = rotation([.1, -.7, .3], 2.2)
        actual = _grow_arc_triangles(before@matrix, target)
        np.testing.assert_allclose(actual, _grow_arc_triangles(before, target)@matrix, atol=2e-14)

    def test_unchanged_area_preserves_vertices_and_regular_births_remain_regular(self):
        before = self.footprint()
        np.testing.assert_array_equal(_grow_arc_triangles(before, area(before)), before)
        regular = _arc_triangles(np.array([[0., 0., 1.]]), np.array([[1., 0., 0.]]), np.array([10.]))
        after = _grow_arc_triangles(regular, [1000.])
        np.testing.assert_allclose(area(after), [1000.], rtol=2e-11)
        _, angles = bearings_and_angles(after, np.array([[0., 0., 1.]]))
        np.testing.assert_allclose(angles, np.broadcast_to(angles[:, :1], angles.shape), atol=2e-14)

    def test_growth_rejects_shrinkage_inverted_and_unbounded_targets(self):
        before = self.footprint()
        for triangle, target in ((before, area(before)*.5), (before[:, ::-1], area(before)),
                                 (before, [1e15]), (before, [np.nan])):
            with self.subTest(target=target), self.assertRaises(ValueError):
                _grow_arc_triangles(triangle, target)
        self.assertEqual(_grow_arc_triangles(np.empty((0, 3, 3)), np.empty(0)).shape, (0, 3, 3))


if __name__ == '__main__': unittest.main()
