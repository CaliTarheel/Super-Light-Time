"""A point touch must never manufacture a positive-length weld segment."""
import unittest

import numpy as np

import collision_fronts
import weld_geometry
from benchmarks.collision_architecture import runner
from tests.test_collision_architecture import overlapping_fixture


class DegenerateWeldArcTests(unittest.TestCase):
    def test_point_clip_edge_cannot_cover_a_whole_material_edge(self):
        a = np.array([1., 0., 0.])
        c = np.array([1., -.1, 0.]); c /= np.linalg.norm(c)
        d = np.array([1., .1, 0.]); d /= np.linalg.norm(d)
        tol = collision_fronts.CONNECT_TOLERANCE_KM/6371.
        self.assertIsNone(weld_geometry._intersection_arc(a, a, c, d, tol))
        self.assertIsNone(weld_geometry._intersection_arc(c, d, a, a, tol))
        self.assertEqual(collision_fronts.shared_arc(a, a, c, d, tol), 0.)

    def test_resolved_short_arc_is_retained(self):
        tol = collision_fronts.CONNECT_TOLERANCE_KM/6371.
        def point(theta):
            return np.array([np.cos(theta), np.sin(theta), 0.])
        result = weld_geometry._intersection_arc(point(-4*tol), point(4*tol),
                                                 point(-2*tol), point(2*tol), tol)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(collision_fronts._angle(*result)/(4*tol), 1., places=13)

    def test_short_resolved_edge_does_not_fail_its_plane_test_by_cancellation(self):
        a = np.array([.6167699478529308, -.6869660200853932, .38428182194000293])
        b = np.array([.6167677853546376, -.686966898320878, .3842837227363762])
        c = np.array([.6594335646413076, -.6678945260767503, .3450569168429632])
        d = np.array([.6142876788942576, -.6879685465977935, .38645818202080456])
        tol = collision_fronts.CONNECT_TOLERANCE_KM/6371.
        # Independent 70-digit evaluation on these stored floats gives plane
        # residuals -2.974584795e-12 and 1.681987122e-13, both within tol.
        # The subtracting-large-products cross gave -1.802486e-11 and falsely
        # rejected this real 19.177m clipped edge.
        result = weld_geometry._intersection_arc(a, b, c, d, tol)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(collision_fronts._angle(*result)*6371., .019177369617269, places=10)

    def test_rebuilt_rotation_preserves_weld_length_and_virtual_work(self):
        # Independently rebuilding this orientation introduces a duplicate
        # clipped vertex. Without the guard, a 5.5km weld grows to205km.
        cases = (([-.4069093894237079, .027385761972595773, -.913057921952384], -1.4091887705076653),
                 ([.6538951036495669, -.008688478973357859, .7565353288222906], -1.2754330012807031))
        for axis, angle in cases:
            with self.subTest(axis=axis):
                self.check_rotated_weld(np.array(axis), angle)

    def check_rotated_weld(self, axis, angle):
        # The second case loses 9.6m of valid weld through cancellation in a
        # short clipped edge's cross product, despite having no zero edge.
        x, y, z = axis
        skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
        q = np.eye(3)*np.cos(angle)+(1.-np.cos(angle))*np.outer(axis, axis)+np.sin(angle)*skew
        first = runner.assemble(overlapping_fixture())[0]
        other = runner.assemble(overlapping_fixture(rotation=q))[0]
        for side in (0, 1):
            lengths = sorted(r['trace_length_km'] for r in first.weld_rows if r['side'] == side)
            changed = sorted(r['trace_length_km'] for r in other.weld_rows if r['side'] == side)
            np.testing.assert_allclose(changed, lengths, rtol=1e-10, atol=1e-8)
        def response(model, coordinate):
            value, gradient, hessian = 0., np.zeros(6), np.zeros((6, 6))
            for element in model.elements:
                for operator, coefficient, (start, end) in zip(element['rows'][0], element['coefficient'], element['bounds']):
                    v, g, h, _ = weld_geometry.integrate(operator, coordinate, start, end, .01)
                    value += coefficient*v; gradient += coefficient*g; hessian += coefficient*h
            return value, gradient, hessian
        coordinate = np.array([.2, -.4, .7, -.3, .5, -.1])
        transform = np.kron(np.eye(2), q)
        value, gradient, hessian = response(first, coordinate)
        v2, g2, h2 = response(other, transform@coordinate)
        self.assertGreater(value, 0.)
        np.testing.assert_allclose(v2, value, rtol=1e-10)
        np.testing.assert_allclose(g2, transform@gradient, rtol=1e-9, atol=1e-2)
        np.testing.assert_allclose(h2, transform@hessian@transform.T, rtol=1e-8, atol=1e-2)


if __name__ == '__main__':
    unittest.main()
