"""Independent spherical union/partition checks for native seafloor strips."""
import unittest
import numpy as np

import mesh_geometry
import native_spreading
from orientation import rotation_matrix
from tests._native_spreading_oracle import (
    rectangle, polygon_area, uncovered_rectangle_area, material_rectangles, RADIUS)


class NativeSpreadingGeometryTests(unittest.TestCase):
    target = (-150., 150., -100., 100.)

    def check_union(self, covered, *, refine=False, rotation=None):
        vertices, faces = material_rectangles(covered, refine=refine, rotation=rotation)
        preserved = vertices.copy(), faces.copy()
        context = native_spreading.prepare_material(dict(vertices=vertices, faces=faces))
        target = rectangle(*self.target)
        if rotation is not None: target = target @ rotation
        pieces = native_spreading.uncovered_polygons(target, context)
        expected = uncovered_rectangle_area(self.target, covered)
        self.assertAlmostEqual(sum(polygon_area(p) for p in pieces), expected, delta=1e-6)
        np.testing.assert_array_equal(vertices, preserved[0])
        np.testing.assert_array_equal(faces, preserved[1])
        # Independent pointwise coverage checks reject duplicated output pieces
        # even if another missing patch accidentally balances the area total.
        chart = []
        for polygon in pieces:
            if rotation is not None: polygon = polygon @ rotation.T
            chart.append(polygon[:, 1:]/polygon[:, :1]*RADIUS)
        for x in np.linspace(-143.71, 146.23, 17):
            for y in np.linspace(-93.87, 94.29, 13):
                hits = 0
                point = np.array([x, y])
                for polygon in chart:
                    edge = np.roll(polygon, -1, axis=0)-polygon
                    offset = point-polygon
                    cross = edge[:, 0]*offset[:, 1]-edge[:, 1]*offset[:, 0]
                    hits += bool(np.all(cross >= -1e-7) or np.all(cross <= 1e-7))
                wet = not any(a < x < b and c < y < d for a, b, c, d in covered)
                self.assertEqual(hits, int(wet), (x, y, covered))
        return context

    def test_empty_material_leaves_the_entire_strip(self):
        self.check_union([])

    def test_duplicate_coincident_sheets_subtract_union_once(self):
        self.check_union([(-170., 20., -120., 120.)]*3)

    def test_overlapping_sheets_and_triangle_refinement_preserve_union(self):
        covered = [(-170., 30., -120., 50.), (-50., 170., -40., 120.)]
        self.check_union(covered)
        self.check_union(covered, refine=True)

    def test_fully_covered_strip_is_empty_even_with_duplicates(self):
        self.check_union([(-200., 200., -150., 150.)]*2)

    def test_union_is_equivariant_through_poles_and_dateline(self):
        covered = [(-170., 30., -120., 50.), (-50., 170., -40., 120.)]
        for pose in (dict(yaw=179., pitch=0., roll=0.),
                     dict(yaw=0., pitch=89., roll=0.),
                     dict(yaw=37., pitch=-71., roll=23.)):
            self.check_union(covered, refine=True, rotation=rotation_matrix(pose))

    def test_cell_partition_and_capacity_preserve_exact_wet_area(self):
        covered = [(-170., -20., -120., 120.), (20., 170., -120., 120.)]
        vertices, faces = material_rectangles(covered)
        context = native_spreading.prepare_material(dict(vertices=vertices, faces=faces))
        expected = uncovered_rectangle_area(self.target, covered)
        for level in (2, 3, 4):
            mesh = mesh_geometry.icosphere(level)
            result = native_spreading.integrate_polygon(mesh, rectangle(*self.target), context)
            cells, area, capacity = (np.asarray(result[name]) for name in
                                     ('cells', 'area_km2', 'ocean_capacity_km2'))
            self.assertEqual(len(np.unique(cells)), len(cells))
            self.assertTrue(np.all(area >= 0.))
            self.assertTrue(np.all(area <= capacity+1e-6))
            self.assertTrue(np.all(capacity <= mesh['area_km2'][cells]+1e-6))
            self.assertAlmostEqual(float(area.sum()), expected, delta=1e-6)

    def test_foreign_cell_mask_excludes_only_the_forbidden_receivers(self):
        mesh = mesh_geometry.icosphere(3)
        vertices, faces = material_rectangles([])
        context = native_spreading.prepare_material(dict(vertices=vertices, faces=faces))
        target = rectangle(*self.target)
        base = native_spreading.integrate_polygon(mesh, target, context)
        self.assertAlmostEqual(float(np.sum(base['area_km2'])), polygon_area(target), delta=1e-6)
        allowed = np.ones(len(mesh['faces']), bool)
        forbidden = int(base['cells'][0])
        allowed[forbidden] = False
        result = native_spreading.integrate_polygon(mesh, target, context, allowed_cells=allowed)
        self.assertNotIn(forbidden, result['cells'])
        expected = float(np.sum(np.asarray(base['area_km2'])[np.asarray(base['cells']) != forbidden]))
        self.assertAlmostEqual(float(np.sum(result['area_km2'])), expected, delta=1e-6)
        empty = native_spreading.integrate_polygon(mesh, target, context,
                                                  allowed_cells=np.zeros(len(mesh['faces']), bool))
        self.assertEqual(len(empty['cells']), 0)
        self.assertEqual(float(np.sum(empty['area_km2'])), 0.)

    def test_ocean_intervals_use_material_union_and_preserve_a_real_gap(self):
        for half_gap in (20., .0005):
            covered = [(-170., -half_gap, -120., 120.),
                       (half_gap, 170., -120., 120.)]*2
            vertices, faces = material_rectangles(covered, refine=True)
            context = native_spreading.prepare_material(dict(vertices=vertices, faces=faces))
            ends = np.array([[1., -150./RADIUS, 0.], [1., 150./RADIUS, 0.]])
            ends /= np.linalg.norm(ends, axis=1)[:, None]
            intervals = native_spreading.ocean_intervals(*ends, context)
            self.assertEqual(len(intervals), 1)
            actual = []
            for value in intervals[0]:
                point = (1.-value)*ends[0]+value*ends[1]
                actual.append(float(point[1]/point[0]*RADIUS))
            np.testing.assert_allclose(actual, [-half_gap, half_gap], atol=1e-8, rtol=0.)

    def test_ocean_intervals_keep_separate_water_pieces(self):
        covered = [(-80., 30., -120., 120.), (-30., 80., -120., 120.)]
        vertices, faces = material_rectangles(covered)
        context = native_spreading.prepare_material(dict(vertices=vertices, faces=faces))
        ends = np.array([[1., -150./RADIUS, 0.], [1., 150./RADIUS, 0.]])
        ends /= np.linalg.norm(ends, axis=1)[:, None]
        intervals = native_spreading.ocean_intervals(*ends, context)
        self.assertEqual(len(intervals), 2)
        actual = []
        for limits in intervals:
            row = []
            for value in limits:
                point = (1.-value)*ends[0]+value*ends[1]
                row.append(float(point[1]/point[0]*RADIUS))
            actual.append(row)
        np.testing.assert_allclose(actual, [[-150., -80.], [80., 150.]], atol=1e-8, rtol=0.)


if __name__ == '__main__': unittest.main()
