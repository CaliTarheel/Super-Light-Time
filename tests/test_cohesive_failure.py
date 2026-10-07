"""Independent work/derivative checks for the retained force-rift necking law."""

import math
import unittest

import numpy as np

import cohesive_failure as cf


class CohesiveFailureTests(unittest.TestCase):
    def test_mixed_mode_uses_signed_normal_and_unsigned_shear(self):
        strengths = (2e12, 7e12, 3e12)
        self.assertEqual(cf.mixed_capacity_n(1000., strengths, .5, -.2),
                         1000. * (2e12 * .5 + 3e12 * .2))
        self.assertEqual(cf.mixed_capacity_n(1000., strengths, -.5, .2),
                         1000. * (7e12 * .5 + 3e12 * .2))
        self.assertEqual(cf.mixed_capacity_n(0., strengths, -.5, .2), 0.)
        # Subdividing the represented edge cannot create an extra cut cost.
        split = math.fsum(cf.mixed_capacity_n(length, strengths, .5, -.2)
                          for length in (17., 283., 700.))
        self.assertEqual(split, cf.mixed_capacity_n(1000., strengths, .5, -.2))

    def test_current_strength_matches_existing_beta_law(self):
        width, capacity = 150000., 4e18
        for opening in (0., 1., 30000., 300000., 2e7):
            expected = 1. / (1. + opening / width)
            self.assertAlmostEqual(cf.necking_multiplier(opening, width), expected, places=15)
            self.assertAlmostEqual(cf.necking_force_n(capacity, opening, width) /
                                   (capacity * expected), 1., places=15)

    def test_work_matches_independent_gauss_quadrature(self):
        roots, weights = np.polynomial.legendre.leggauss(64)
        capacity, width = 7e18, 150000.
        for opening, increment in ((0., 30000.), (50000., 450000.),
                                   (5e6, 30.), (2e5, 1e-4)):
            distance = increment * (roots + 1.) / 2.
            force = capacity / (1. + (opening + distance) / width)
            independent = float(increment / 2. * np.dot(weights, force))
            measured = cf.necking_work_j(capacity, opening, increment, width)
            self.assertAlmostEqual(measured / independent, 1., delta=4e-14)

    def test_work_derivative_is_current_mixed_resistance(self):
        capacity, width, opening = 9e18, 80000., 30000.
        step = 1.
        plus = cf.necking_work_j(capacity, 0., opening + step, width)
        minus = cf.necking_work_j(capacity, 0., opening - step, width)
        derivative = (plus - minus) / (2. * step)
        independent = capacity / (1. + opening / width)
        self.assertAlmostEqual(derivative / independent, 1., delta=7e-11)

    def test_paid_work_does_not_depend_on_source_step_subdivision(self):
        capacity, width, start, total = 1.3e19, 150000., 4200., 380000.
        exact = cf.necking_work_j(capacity, start, total, width)
        for count in (2, 7, 101, 1000):
            increment = total / count
            pieces = [cf.necking_work_j(capacity, start + i * increment,
                                       increment, width) for i in range(count)]
            self.assertAlmostEqual(math.fsum(pieces) / exact, 1., delta=6e-15)

    def test_budget_inverse_is_bounded_and_cannot_overdraw(self):
        capacity, width, opening, limit = 8e18, 150000., 60000., 500000.
        for target in (1e-8, 1., 13000., 300000., limit):
            # Independent antiderivative, avoiding this module's work helper.
            budget = capacity * width * math.log1p(target / (width + opening))
            accepted = cf.increment_for_work_m(capacity, opening, width, budget, limit)
            self.assertGreaterEqual(accepted, 0.)
            self.assertLessEqual(accepted, limit)
            spent = cf.necking_work_j(capacity, opening, accepted, width)
            self.assertLessEqual(spent, budget)
            self.assertAlmostEqual(accepted / target, 1., delta=4e-15)
            if accepted < limit:
                # No representable affordable increment remains above this one.
                larger = math.nextafter(accepted, math.inf)
                self.assertGreater(cf.necking_work_j(capacity, opening, larger, width), budget)

    def test_zero_work_and_finite_caps_have_explicit_semantics(self):
        self.assertEqual(cf.necking_work_j(5e18, 30., 0., 150000.), 0.)
        self.assertEqual(cf.necking_work_j(0., 30., 50., 150000.), 0.)
        self.assertEqual(cf.increment_for_work_m(5e18, 30., 150000., 0., 50.), 0.)
        self.assertEqual(cf.increment_for_work_m(0., 30., 150000., 0., 50.), 50.)
        self.assertEqual(cf.increment_for_work_m(5e18, 30., 150000., 5e30, 50.), 50.)
        self.assertEqual(cf.increment_for_work_m(5e18, 30., 150000., 5e30, 0.), 0.)

    def test_short_increment_retains_work_without_primitive_cancellation(self):
        # Subtracting two large primitive values would lose this increment.
        capacity, opening, increment, width = 1e18, 1e8, 1e-8, 150000.
        measured = cf.necking_work_j(capacity, opening, increment, width)
        midpoint = capacity / (1. + (opening + increment / 2.) / width) * increment
        self.assertGreater(measured, 0.)
        self.assertAlmostEqual(measured / midpoint, 1., delta=3e-15)

    def test_boolean_nonfinite_negative_or_wrong_shape_inputs_are_rejected(self):
        for bad in (True, False, None, '1', float('nan'), float('inf'), -1.):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    cf.necking_work_j(bad, 0., 1., 150000.)
                with self.assertRaises(ValueError):
                    cf.necking_multiplier(bad, 150000.)
                with self.assertRaises(ValueError):
                    cf.increment_for_work_m(1., 0., 150000., bad, 1.)
        for strengths in (None, (1., 2.), (1., 2., 3., 4.), (1., -2., 3.),
                          (1., True, 3.), (1., float('inf'), 3.)):
            with self.assertRaises(ValueError):
                cf.mixed_capacity_n(1., strengths, 1., 0.)
        with self.assertRaises(ValueError):
            cf.necking_multiplier(0., 0.)
        with self.assertRaises(ValueError):
            cf.necking_work_j(1., 0., -1., 150000.)
        with self.assertRaises(ValueError):
            cf.mixed_capacity_n(1., (1., 2., 3.), np.bool_(True), 0.)

    def test_nonfinite_arithmetic_is_refused_instead_of_funding_motion(self):
        with self.assertRaises(ValueError):
            cf.mixed_capacity_n(1e308, (1e308, 1e308, 1e308), 1., 1.)
        with self.assertRaises(ValueError):
            cf.necking_work_j(1e308, 0., 1e308, 1.)
        with self.assertRaises(ValueError):
            cf.necking_multiplier(1e308, 1e308)


if __name__ == '__main__':
    unittest.main()
