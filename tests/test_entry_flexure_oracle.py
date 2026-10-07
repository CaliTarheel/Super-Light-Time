"""Physical and discrete work checks for the read-only flexure benchmark."""

import unittest

import numpy as np

from experiments.entry_flexure_oracle import (
    anchored_channel_contact, equivalent_mode_stiffness,
    flexural_rigidity, periodic_bending_work)


class EntryFlexureOracleTests(unittest.TestCase):
    def anchored_case(self, count=65, *, base=None, tolerance=1e-8,
                      max_iterations=80):
        preferred = np.linspace(0., 80000., count)
        if base is None:
            base = np.minimum(preferred, 35000.)
        rigidity = flexural_rigidity(70e9, 30000., .25)
        result = anchored_channel_contact(
            preferred, base, 9.81 * 500. * 35000., rigidity,
            3300. * 9.81, 300000. / (count - 1),
            force_tolerance=tolerance, max_iterations=max_iterations)
        return preferred, base, result

    def test_anchored_nonlocal_contact_balances_force_and_virtual_work(self):
        preferred, base, result = self.anchored_case()
        depth = result['depth_m']
        uplift = result['upper_uplift_m']
        reaction = result['reaction_pa']
        np.testing.assert_array_equal(depth[[0, -1]], preferred[[0, -1]])
        self.assertGreater(np.count_nonzero(reaction > 0.), 0)
        self.assertGreater(np.count_nonzero(reaction == 0.), 0)
        self.assertTrue(np.all(depth + uplift >= base - 1e-9))
        self.assertTrue(np.all(reaction >= 0.))
        np.testing.assert_allclose(reaction, 3300. * 9.81 * uplift,
                                   rtol=2e-15)
        np.testing.assert_allclose(
            reaction * (depth + uplift - base), 0., atol=1.)
        np.testing.assert_allclose(
            result['energy_j_per_m'],
            result['bending_j_per_m'] + result['buoyancy_j_per_m']
            + result['upper_j_per_m'], rtol=1e-14)
        self.assertLess(result['free_force_residual_n_per_m'], 1e3)
        spacing = 300000. / (len(base) - 1)
        weights = np.full(len(base), spacing)
        weights[[0, -1]] *= .5
        body_force = weights @ (9.81 * 500. * 35000. - reaction)
        self.assertLess(
            abs(result['anchor_force_n_per_m'].sum() + body_force),
            1e5)
        for index in (16, 32):
            plus, minus = base.copy(), base.copy()
            plus[index] += 1.
            minus[index] -= 1.
            higher = self.anchored_case(base=plus)[2]['energy_j_per_m']
            lower = self.anchored_case(base=minus)[2]['energy_j_per_m']
            observed = (higher - lower) / 2.
            np.testing.assert_allclose(
                observed, spacing * reaction[index], rtol=2e-5)

    def test_flexure_couples_neighbors_and_refines_in_physical_space(self):
        preferred, base, coarse = self.anchored_case(65)
        _, _, fine = self.anchored_case(129)
        np.testing.assert_allclose(
            coarse['depth_m'][::16], fine['depth_m'][::32],
            rtol=2e-3, atol=50.)
        np.testing.assert_allclose(
            coarse['energy_j_per_m'], fine['energy_j_per_m'], rtol=2e-3)
        altered = base.copy()
        altered[24] += 1000.
        moved = self.anchored_case(base=altered)[2]
        self.assertGreater(
            abs(moved['depth_m'][20] - coarse['depth_m'][20]), 1.)

    def test_unloaded_preferred_path_and_unbalanced_rejection(self):
        preferred = np.linspace(0., 80000., 65)
        result = anchored_channel_contact(
            preferred, np.zeros(65), 0.,
            flexural_rigidity(70e9, 30000., .25),
            3300. * 9.81, 300000. / 64.)
        np.testing.assert_array_equal(result['depth_m'], preferred)
        np.testing.assert_array_equal(result['reaction_pa'], 0.)
        self.assertEqual(result['energy_j_per_m'], 0.)
        with self.assertRaisesRegex(ValueError, 'did not balance force'):
            self.anchored_case(tolerance=1e-30, max_iterations=1)

    def test_uniform_offset_has_no_bending_energy(self):
        preferred = np.zeros(64)
        depth = np.full(64, 1000.)
        energy, gradient = periodic_bending_work(
            depth, preferred, 5000., 6.22e21)
        self.assertEqual(energy, 0.)
        np.testing.assert_array_equal(gradient, 0.)
        self.assertGreater(.5 * 1000. * 5000. * float(depth @ depth), 0.)

    def test_sinusoidal_mode_recovers_wavelength_dependent_stiffness(self):
        young, nu, te = 70e9, .25, 10000.
        rigidity = flexural_rigidity(young, te, nu)
        wavelength = 300000.
        count = 128
        spacing = wavelength / count
        depth = 1000. * np.cos(2. * np.pi * np.arange(count) / count)
        energy, gradient = periodic_bending_work(
            depth, np.zeros(count), spacing, rigidity)
        spring_denominator = .5 * spacing * float(depth @ depth)
        measured = energy / spring_denominator
        continuum = equivalent_mode_stiffness(rigidity, wavelength)
        np.testing.assert_allclose(measured, continuum, rtol=5e-4)
        self.assertGreater(energy, 0.)
        np.testing.assert_allclose(
            equivalent_mode_stiffness(rigidity, wavelength / 3.),
            continuum * 81., rtol=1e-14)
        for index in (0, 13, 64):
            step = .01
            plus, minus = depth.copy(), depth.copy()
            plus[index] += step
            minus[index] -= step
            observed = (periodic_bending_work(
                plus, np.zeros(count), spacing, rigidity)[0]
                - periodic_bending_work(
                    minus, np.zeros(count), spacing, rigidity)[0]
            ) / (2. * step)
            np.testing.assert_allclose(gradient[index], observed, rtol=1e-7)

    def test_invalid_parameters_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Poisson ratio'):
            flexural_rigidity(70e9, 10000., .5)
        with self.assertRaisesRegex(ValueError, 'wavelength'):
            equivalent_mode_stiffness(1e22, 0.)
        with self.assertRaisesRegex(ValueError, 'aligned finite'):
            periodic_bending_work(np.zeros(4), np.zeros(4), 1000., 1e22)


if __name__ == '__main__':
    unittest.main()
