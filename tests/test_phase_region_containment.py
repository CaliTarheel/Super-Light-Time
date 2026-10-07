"""Stable local phase-region containment without moving or reassigning markers."""
from copy import deepcopy
from decimal import Decimal, localcontext
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

import phase_evolution
import mesh_coverage


def unit(value):
    value = np.asarray(value, float)
    return value/np.linalg.norm(value)


def short_edge_triangle():
    a = unit([.3, .5, .8])
    tangent = unit(np.cross(a, [0., 0., 1.]))
    other = np.cross(a, tangent)
    return np.array([a, unit(a+1.e-6*tangent), unit(a+.01*other)]), tangent, other


def exact_signed_distances(polygon, point):
    """80-digit determinant oracle on the represented floats, not the new formula."""
    with localcontext() as context:
        context.prec = 80
        decimal = lambda values: [Decimal.from_float(float(value)) for value in values]
        query = decimal(point)
        result = []
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
            a, b = decimal(a), decimal(b)
            normal = [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
            length = sum(value*value for value in normal).sqrt()
            result.append(sum(x*y for x, y in zip(normal, query))/length)
        return result


def row(polygon, upper=(), *, face=0, weight=1.):
    return dict(face=face, weight=weight, polygon=polygon.copy(),
                pairs=np.arange(len(upper)), upper=np.array(upper, int))


def marker(points):
    points = np.atleast_2d(points)
    return SimpleNamespace(parcel_patch=np.array([17]), trace_patch=np.full(len(points), 17),
                           trace_xyz=points.copy())


class PhaseRegionContainmentTests(unittest.TestCase):
    def test_recorded_462_myr_collapsed_edge_uses_existing_region_footprint(self):
        fixture = Path(__file__).with_name('fixtures')/'phase-marker-462myr.json'
        capture = json.loads(fixture.read_text(encoding='utf-8'))
        failure, = capture['failures']
        point = np.asarray(failure['point'])
        self.assertEqual(failure['marker'], 3736)
        self.assertGreater(min(exact_signed_distances(failure['triangle'], point)), Decimal('.0028'))
        regions = [row(np.asarray(record['polygon']), record['upper'], weight=record['weight'])
                   for record in failure['regions']]
        # High precision confirms the represented tiny edge is a false
        # separating plane: merely improving the determinant cannot fix this.
        self.assertTrue(all(min(exact_signed_distances(r['polygon'], point)) < Decimal('-1e-11')
                            for r in regions))
        expected = regions[4]
        polygon = expected['polygon']
        self.assertLess(np.linalg.norm(polygon[1]-polygon[0])*6371., 8e-13)
        # Independent oracle: the large quadrilateral is interior by >2km;
        # its polygon area agrees to <3e-11km2. This reduction is test-only.
        quad = polygon[1:]
        self.assertGreater(min(exact_signed_distances(quad, point)), Decimal('.00037'))
        self.assertAlmostEqual(mesh_coverage._polygon_area(polygon, 6371.),
                               mesh_coverage._polygon_area(quad, 6371.), delta=3e-11)
        source = marker(point)
        before = deepcopy((source.trace_xyz, source.trace_patch, regions))
        selected, _, _ = phase_evolution._selected_regions(source, dict(regions=regions), True)
        self.assertIs(selected[0], expected)
        self.assertEqual([i for i,r in enumerate(regions)
                          if phase_evolution._region_contains(r['polygon'], point)], [4])
        np.testing.assert_array_equal(source.trace_xyz, before[0])
        np.testing.assert_array_equal(source.trace_patch, before[1])
        for region, old in zip(regions, before[2]):
            for key in region: np.testing.assert_array_equal(region[key], old[key])
        # Corner order changes fan diagonals, not the actual interior.
        for shift in range(len(polygon)):
            self.assertTrue(phase_evolution._region_contains(np.roll(polygon, shift, axis=0), point))

    def test_nearly_collapsed_edge_does_not_admit_outside_points(self):
        fixture = Path(__file__).with_name('fixtures')/'phase-marker-462myr.json'
        failed, = json.loads(fixture.read_text())['failures']
        polygon = np.asarray(failed['regions'][4]['polygon'])
        interior = np.asarray(failed['point'])
        # Across each long side, independently confirmed outside the reduced
        # quadrilateral; neither triangle tolerance nor short edge may fill it.
        quad = polygon[1:]
        for a,b in zip(quad, np.roll(quad,-1,axis=0)):
            normal = unit(np.cross(a,b-a))
            point = unit(unit(a+b)-1e-5*normal)
            self.assertLess(min(exact_signed_distances(quad, point)), Decimal('-9e-6'))
            self.assertFalse(phase_evolution._region_contains(polygon, point))
        self.assertFalse(phase_evolution._region_contains(polygon, -interior))

    def test_real_subnanometre_edge_is_not_welded_or_erased(self):
        # Near a coordinate axis this edge is representable and has a stable
        # direction despite being shorter than the captured collapsed edge.
        polygon = np.array([[0.,0.,1.], unit([1e-16,0.,1.]), unit([0.,.01,1.])])
        original = polygon.copy()
        area = mesh_coverage._polygon_area(polygon,6371.)
        self.assertGreater(area,0.)
        self.assertTrue(phase_evolution._region_contains(polygon,unit(polygon.sum(axis=0))))
        self.assertFalse(phase_evolution._region_contains(polygon,unit([-1e-6,.003,1.])))
        np.testing.assert_array_equal(polygon,original)
        self.assertEqual(mesh_coverage._polygon_area(polygon,6371.),area)

    def test_zero_area_fan_pieces_cannot_fill_an_outside_point(self):
        polygon = np.array([[0.,0.,1.],[0.,0.,1.],unit([.01,0.,1.]),unit([0.,.01,1.])])
        self.assertFalse(phase_evolution._region_contains(polygon,unit([-.01,-.01,1.])))
        self.assertTrue(phase_evolution._region_contains(polygon,unit([.002,.002,1.])))

    def test_recorded_14_myr_marker_matches_all_28_high_precision_region_tests(self):
        fixture = Path(__file__).with_name('fixtures')/'phase-marker-14myr.json'
        capture = json.loads(fixture.read_text(encoding='utf-8'))
        self.assertEqual(capture['time_myr'], 14.)
        failed, = capture['failed']
        self.assertEqual(failed['trace_id'], 1664)
        point = np.asarray(failed['point'])
        triangle = np.asarray(failed['triangle'])
        self.assertGreater(min(exact_signed_distances(triangle, point)), Decimal('.0033'))
        regions = [dict(record, face=0, polygon=np.asarray(record['polygon']),
                        upper=np.asarray(record['upper'], int), pairs=np.asarray(record['pairs'], int))
                   for record in failed['regions']]
        self.assertEqual(len(regions), 28)
        oracle = [index for index, region in enumerate(regions)
                  if min(exact_signed_distances(region['polygon'], point)) >= Decimal('-1e-11')]
        self.assertEqual(oracle, [0])
        self.assertGreater(min(exact_signed_distances(regions[0]['polygon'], point)), Decimal('.0014'))
        legacy = []
        for index, region in enumerate(regions):
            polygon = region['polygon']
            planes = np.cross(polygon, np.roll(polygon, -1, axis=0))
            planes /= np.maximum(np.linalg.norm(planes, axis=1)[:, None], 1.e-300)
            if np.all(planes@point >= -1.e-11):
                legacy.append(index)
        self.assertEqual(legacy, [])
        source = marker(point)
        selected, _, _ = phase_evolution._selected_regions(source, dict(regions=regions), True)
        self.assertIs(selected[0], regions[0])
        # Check each represented region individually as well as the combined
        # selection: no additional region was admitted by the arithmetic fix.
        for index, region in enumerate(regions):
            with self.subTest(region=index):
                if index in oracle:
                    choice, _, _ = phase_evolution._selected_regions(source, dict(regions=[region]), True)
                    self.assertIs(choice[0], region)
                else:
                    with self.assertRaisesRegex(ValueError, 'left its local stack region'):
                        phase_evolution._selected_regions(source, dict(regions=[region]), True)

    def test_valid_short_edge_vertex_matches_high_precision_orientation(self):
        polygon, _, _ = short_edge_triangle()
        point = polygon[0]
        exact = exact_signed_distances(polygon, point)
        self.assertLess(abs(exact[0]), Decimal('1e-60'))
        self.assertGreater(exact[1], Decimal('9e-7'))
        self.assertLess(abs(exact[2]), Decimal('1e-60'))
        # The prior arithmetic fails despite this being an exact polygon vertex.
        old = np.cross(polygon, np.roll(polygon, -1, axis=0))
        old /= np.linalg.norm(old, axis=1)[:, None]
        self.assertLess(float((old@point).min()), -1.e-11)
        source = marker(point)
        region = row(polygon, [9])
        prepared = dict(regions=[region])
        before = deepcopy((source.trace_xyz, source.trace_patch, prepared))
        selected, index, weight = phase_evolution._selected_regions(source, prepared, True)
        self.assertIs(selected[0], region)
        np.testing.assert_array_equal(index, [0])
        np.testing.assert_array_equal(weight, [1.])
        np.testing.assert_array_equal(source.trace_xyz, before[0])
        np.testing.assert_array_equal(source.trace_patch, before[1])
        for key in region:
            np.testing.assert_array_equal(region[key], before[2]['regions'][0][key])

    def test_genuinely_outside_marker_still_rejects_without_relocation(self):
        polygon, tangent, other = short_edge_triangle()
        point = unit(polygon[0]-1.e-7*tangent+1.e-4*other)
        self.assertLess(min(exact_signed_distances(polygon, point)), Decimal('-1e-8'))
        source = marker(point)
        before = deepcopy(source)
        with self.assertRaisesRegex(ValueError, 'left its local stack region'):
            phase_evolution._selected_regions(source, dict(regions=[row(polygon)]), True)
        np.testing.assert_array_equal(source.trace_xyz, before.trace_xyz)
        np.testing.assert_array_equal(source.trace_patch, before.trace_patch)

    def test_shared_edge_tie_keeps_lexicographic_upper_choice_and_local_regions(self):
        polygon, tangent, _ = short_edge_triangle()
        a, b, c = polygon
        d = unit(a-1.e-6*tangent)
        first = row(polygon, [7], weight=.5)
        second = row(np.array([a, c, d]), [2], weight=.5)
        point = a  # Exact shared endpoint; no interpolation error in the oracle.
        for candidate in (first, second):
            self.assertGreater(min(exact_signed_distances(candidate['polygon'], point)), Decimal('-1e-60'))
        source = marker([point, unit(a+b+c), unit(a+c+d)])
        for regions in ([first, second], [second, first]):
            selected, index, weight = phase_evolution._selected_regions(source, dict(regions=regions), True)
            self.assertIs(selected[0], second)
            self.assertIs(selected[1], first)
            self.assertIs(selected[2], second)
            np.testing.assert_array_equal(index, [0, 1, 2])
            np.testing.assert_array_equal(weight, [1., 1., 1.])

    def test_parcel_quadrature_is_unchanged(self):
        polygon, _, _ = short_edge_triangle()
        regions = [row(polygon, [9], face=2, weight=.3), row(polygon, face=2, weight=.7)]
        selected, index, weight = phase_evolution._selected_regions(None, dict(regions=regions), False)
        self.assertIs(selected, regions)
        np.testing.assert_array_equal(index, [2, 2])
        np.testing.assert_array_equal(weight, [.3, .7])


if __name__ == '__main__':
    unittest.main()
