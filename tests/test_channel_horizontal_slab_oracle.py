"""Horizontal intake and along-slab speed obey one power balance."""

import math
import unittest

import numpy as np

from channel_horizontal_slab_oracle import horizontal_intake_attachment


class HorizontalSlabOracleTests(unittest.TestCase):
    def test_stiff_neck_projects_horizontal_intake_and_depth(self):
        angle = math.radians(50.)
        result = horizontal_intake_attachment(1., 0., 50., 1., 1e12, 0.)
        self.assertAlmostEqual(result['along_slab_speed_m_s'],
                               1. / math.cos(angle), places=10)
        self.assertAlmostEqual(result['mantle_horizontal_speed_m_s'], 1., places=10)
        self.assertAlmostEqual(result['vertical_speed_m_s'],
                               math.tan(angle), places=10)
        # The old convention u=q would advance horizontally only cos(theta).
        self.assertGreater(1. - math.cos(angle), .35)

    def test_plate_forces_are_derivatives_and_power_is_conserved(self):
        for d, h, weight in np.random.default_rng(725).normal(size=(40, 3)):
            result = horizontal_intake_attachment(
                d, h, 50., 2.5, 6., weight)
            scale = max(abs(result['gravity_power_w']),
                        abs(result['plate_power_w']),
                        result['mantle_dissipation_w'],
                        result['neck_dissipation_w'], 1.)
            self.assertLess(abs(result['power_residual_w']) / scale, 2e-14)
            self.assertGreaterEqual(result['mantle_dissipation_w'], 0.)
            self.assertGreaterEqual(result['neck_dissipation_w'], 0.)
            delta = 1e-6
            def potential(new_d, new_h):
                return horizontal_intake_attachment(
                    new_d, new_h, 50., 2.5, 6., weight)['minimized_potential_w']
            down_derivative = -(potential(d + delta, h)
                                - potential(d - delta, h)) / (2. * delta)
            over_derivative = -(potential(d, h + delta)
                                - potential(d, h - delta)) / (2. * delta)
            np.testing.assert_allclose(down_derivative,
                                       result['down_plate_force_n'], rtol=1e-7,
                                       atol=1e-8)
            np.testing.assert_allclose(over_derivative,
                                       result['over_plate_force_n'], rtol=1e-7,
                                       atol=1e-8)

    def test_invalid_dip_and_drag_fail_closed(self):
        for dip, cs, cn in ((0., 1., 1.), (90., 1., 1.),
                            (50., 0., 1.), (50., 1., -1.)):
            with self.assertRaises(ValueError):
                horizontal_intake_attachment(1., 0., dip, cs, cn, 1.)


if __name__ == '__main__':
    unittest.main()
