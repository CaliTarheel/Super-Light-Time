"""Nonlocal two-plate benchmark balances contact and reciprocal work."""

import unittest

import numpy as np

from experiments.entry_two_plate_flexure import anchored_two_plate_contact
from channel_two_plate_flexure import anchored_two_plate_contact as archived_solver


def contact_case(count=33):
    x = np.linspace(-160000., 160000., count)
    preferred = np.full(count, 20000.)
    base = 10000. + 25000. * np.exp(-(x / 65000.)**2)
    spacing = float(x[1] - x[0])
    return preferred, base, spacing


def solve(preferred, base, spacing, *, upper_rigidity=1e23):
    return anchored_two_plate_contact(
        preferred, base, 0., 1e23, upper_rigidity, 20000., spacing)


class TwoPlateFlexureTests(unittest.TestCase):
    def test_explicit_contact_mask_ignores_inactive_basal_reference(self):
        offsets = np.linspace(0., 320000., 33)
        preferred = np.full(len(offsets), 20000.)
        base = 10000. + 25000. * np.exp(-((offsets - 130000.) / 60000.)**2)
        mask = (offsets >= 80000.) & (offsets <= 180000.)
        def solve_mask(reference):
            return anchored_two_plate_contact(
                preferred, reference, 0., 1e23, 1e23, 20000.,
                float(offsets[1] - offsets[0]), contact_mask=mask)
        original = solve_mask(base)
        outside = base.copy()
        outside[~mask] = 200000.
        ignored = solve_mask(outside)
        np.testing.assert_allclose(ignored['energy_j_per_m'],
                                   original['energy_j_per_m'], rtol=2e-13)
        np.testing.assert_array_equal(ignored['reaction_pa'][~mask], 0.)
        np.testing.assert_allclose(ignored['depth_m'], original['depth_m'],
                                   rtol=2e-13, atol=1e-7)
        self.assertLess(np.min(ignored['depth_m'][~mask]
                                   + ignored['upper_uplift_m'][~mask]
                                   - outside[~mask]), 0.)
        delta = 1.
        raised = base.copy()
        raised[13] += delta
        derivative = (solve_mask(raised)['energy_j_per_m']
                      - original['energy_j_per_m']) / delta
        np.testing.assert_allclose(derivative,
                                   original['basal_reference_derivative_n_per_m'][13],
                                   rtol=2e-4)
        with self.assertRaisesRegex(ValueError, 'boolean physical contact mask'):
            anchored_two_plate_contact(preferred, base, 0., 1e23, 1e23,
                                       20000., float(offsets[1] - offsets[0]),
                                       contact_mask=mask.astype(int))
        free = anchored_two_plate_contact(
            preferred, outside, 0., 1e23, 1e23, 20000.,
            float(offsets[1] - offsets[0]), free_upper_hinge=True,
            contact_mask=mask)
        self.assertEqual(free['hinge_line_reaction_n_per_m'], 0.)
        self.assertEqual(free['reaction_pa'][0], 0.)

    def test_free_upper_hinge_lifts_real_basal_edge_with_finite_line_force(self):
        results=[]
        for count in (33,65):
            offsets=np.linspace(0.,160000.,count)
            spacing=float(offsets[1]-offsets[0])
            preferred=offsets*np.tan(np.deg2rad(50.))
            base=np.full(count,35000.)
            with self.assertRaisesRegex(ValueError,'infeasible at an anchored end'):
                anchored_two_plate_contact(preferred,base,0.,1e23,1e23,20000.,spacing)
            result=anchored_two_plate_contact(preferred,base,0.,1e23,1e23,
                                               20000.,spacing,free_upper_hinge=True)
            self.assertAlmostEqual(result['upper_uplift_m'][0],base[0],places=6)
            self.assertGreater(result['hinge_line_reaction_n_per_m'],0.)
            self.assertGreaterEqual(result['minimum_gap_m'],-1e-6)
            np.testing.assert_allclose(result['hinge_line_reaction_n_per_m'],
                                       result['reaction_pa'][0]*spacing/2.,rtol=2e-13)
            change=1.
            plus,minus=base.copy(),base.copy()
            plus[0]+=change;minus[0]-=change
            def energy(path):
                return anchored_two_plate_contact(preferred,path,0.,1e23,1e23,
                    20000.,spacing,free_upper_hinge=True)['energy_j_per_m']
            derivative=(energy(plus)-energy(minus))/(2.*change)
            np.testing.assert_allclose(derivative,
                                       result['hinge_line_reaction_n_per_m'],rtol=2e-8)
            results.append(result)
        np.testing.assert_allclose(results[0]['energy_j_per_m'],
                                   results[1]['energy_j_per_m'],rtol=.005)
        np.testing.assert_allclose(results[0]['hinge_line_reaction_n_per_m'],
                                   results[1]['hinge_line_reaction_n_per_m'],rtol=.002)

    def test_experimental_import_is_the_archived_native_solver(self):
        self.assertIs(anchored_two_plate_contact, archived_solver)

    def test_separated_plates_have_zero_contact_work(self):
        preferred = np.full(33, 80000.)
        base = np.full(33, 35000.)
        result = solve(preferred, base, 10000.)
        np.testing.assert_allclose(result['depth_m'], preferred, atol=1e-7)
        np.testing.assert_allclose(result['upper_uplift_m'], 0., atol=1e-7)
        np.testing.assert_array_equal(result['reaction_pa'], 0.)
        self.assertGreater(result['minimum_gap_m'], 0.)

    def test_contact_balances_both_bending_plates_and_nonpenetration(self):
        preferred, base, spacing = contact_case()
        result = solve(preferred, base, spacing)
        self.assertGreater(result['reaction_pa'].max(), 0.)
        self.assertGreater(result['lower_bending_j_per_m'], 0.)
        self.assertGreater(result['upper_bending_j_per_m'], 0.)
        self.assertGreater(result['upper_foundation_j_per_m'], 0.)
        self.assertGreaterEqual(result['minimum_gap_m'], -1e-6)
        self.assertLess(result['free_force_residual_n_per_m'], 10.)
        self.assertTrue(np.all(result['depth_m'] >= -1e-8))
        self.assertTrue(np.all(result['reaction_pa'] >= -1e-5))
        gap = result['depth_m'] + result['upper_uplift_m'] - base
        self.assertLess(float(np.max(np.abs(gap[result['contact']]))), 1e-6)
        self.assertEqual(result['upper_uplift_m'][0], 0.)
        self.assertEqual(result['upper_uplift_m'][-1], 0.)
        # Bending lets a neighboring, noncontact upper column respond.
        self.assertGreater(abs(result['upper_uplift_m'][4]), 0.)

    def test_unloaded_upper_base_has_reciprocal_virtual_work(self):
        preferred, base, spacing = contact_case()
        centre = len(base) // 2
        original = solve(preferred, base, spacing)
        delta = .01
        plus = base.copy()
        minus = base.copy()
        plus[centre] += delta
        minus[centre] -= delta
        derivative = (solve(preferred, plus, spacing)['energy_j_per_m']
                      - solve(preferred, minus, spacing)['energy_j_per_m']) / (2. * delta)
        np.testing.assert_allclose(derivative,
            original['reaction_pa'][centre] * spacing, rtol=2e-5)

    def test_physical_span_refinement_and_infeasible_anchors(self):
        coarse = contact_case(33)
        fine = contact_case(65)
        a = solve(*coarse)
        b = solve(*fine)
        np.testing.assert_allclose(a['depth_m'][16], b['depth_m'][32], rtol=.03)
        np.testing.assert_allclose(a['upper_uplift_m'][16],
                                   b['upper_uplift_m'][32], rtol=.03)
        bad = coarse[1].copy()
        bad[0] = coarse[0][0] + 1.
        with self.assertRaisesRegex(ValueError, 'infeasible'):
            solve(coarse[0], bad, coarse[2])
        with self.assertRaisesRegex(ValueError, 'did not converge'):
            anchored_two_plate_contact(coarse[0], coarse[1], 0.,
                1e23, 1e23, 20000., coarse[2], max_iterations=1)


if __name__ == '__main__':
    unittest.main()
