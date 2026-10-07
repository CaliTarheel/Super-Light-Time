"""Oblique opening must agree across labels, production and review geometry."""
import unittest
from unittest.mock import patch

import numpy as np

from native_engine import Simulation
import native_spreading
import ridge_spreading
from tests.test_native_spreading_pairing import scene, RADIUS


def boundary_scene(opening, sliding):
    """One prescribed meridional ocean front, speeds in km/Myr at its centre."""
    s, axis = scene()
    s.omega[0] = [0., sliding/(2*RADIUS), -opening/(2*RADIUS)]
    s.omega[1] = -s.omega[0]
    s.edge_a, s.edge_b = s.ba.copy(), s.bb.copy()
    s.crust = np.zeros(s.n, np.uint8)
    s.age = np.full(s.n, 80.)
    s.polarity = np.full((3, 3), -1, np.int16)
    s.native_spreading_version = 1
    s.native_boundary_geometry.update(
        normals=np.array([[0., 1., 0.]]), midpoints=np.array([[1., 0., 0.]]),
        lengths=np.array([np.arccos(np.clip(axis[0]@axis[1], -1., 1.))*RADIUS]),
        diagnostics={})
    # Prescribe geometry, then exercise the real engine classification and
    # real material clipping/production. Trench history is outside this fixture.
    with patch('native_boundary_geometry.reconstruct', return_value=s.native_boundary_geometry), \
            patch('trench_history.prepare'):
        Simulation._boundaries(s)
    return s


class RidgeMotionPolicyTests(unittest.TestCase):
    def test_oblique_opening_creates_paired_crust_and_stays_a_ridge(self):
        # 8/40=.2 is between the former inconsistent .15 and .35 thresholds.
        s = boundary_scene(8., 40.)
        np.testing.assert_array_equal(s.bcode, [1])
        rows = native_spreading.boundary_segments(s)
        self.assertEqual([row['code'] for row in rows], [1])
        transported = s.support.copy()
        birth = native_spreading.advance(s, transported, 2.)
        area = float(birth@s.cell_area)
        self.assertGreater(area, 0.)
        report = s.spreading_diagnostics
        self.assertAlmostEqual(report['side_p_area_km2'], report['side_q_area_km2'], delta=1e-6)
        self.assertAlmostEqual(area, report['side_p_area_km2']+report['side_q_area_km2'], delta=1e-6)
        self.assertAlmostEqual(area, s.process_totals['ocean_created_km2'], delta=1e-6)
        np.testing.assert_allclose(transported.sum(axis=0), s.support.sum(axis=0), atol=1e-12)

    def test_sliding_stopped_closing_and_threshold_contacts_create_no_crust(self):
        for opening, sliding, code in ((0., 40., 3), (0., 0., 3), (-8., 40., 2),
                                       (6., 40., 3), (2., 0., 3), (1., 0., 3)):
            with self.subTest(opening=opening, sliding=sliding):
                s = boundary_scene(opening, sliding)
                np.testing.assert_array_equal(s.bcode, [code])
                self.assertEqual([r['code'] for r in native_spreading.boundary_segments(s)], [code])
                transported = s.support.copy()
                birth = native_spreading.advance(s, transported, 2.)
                np.testing.assert_array_equal(birth, 0.)
                np.testing.assert_array_equal(transported, s.support)
                self.assertEqual(s.process_totals['ocean_created_km2'], 0.)

    def test_native_front_cache_uses_native_policy_without_changing_raster_policy(self):
        s = boundary_scene(8., 40.)
        np.testing.assert_array_equal(ridge_spreading._active(s), [0])
        # The retained raster edition still uses its historical .35 policy.
        del s.native_mesh
        self.assertEqual(len(ridge_spreading._active(s)), 0)

    def test_motion_change_expires_spreading_without_reusing_old_labels(self):
        s = boundary_scene(8., 40.)
        native_spreading.prepare(s)
        s.omega[:, 2] = 0.  # Keep sliding, remove opening; bcode is deliberately stale.
        self.assertEqual([r['code'] for r in native_spreading.boundary_segments(s)], [3])
        birth = native_spreading.advance(s, s.support.copy(), 2.)
        np.testing.assert_array_equal(birth, 0.)


if __name__ == '__main__':
    unittest.main()
