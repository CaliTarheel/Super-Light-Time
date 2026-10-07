"""Spreading births must not reassign pre-existing ocean support.

In this transaction, a receiving cell's older crust can only be diluted by
the new crust sharing it, never reassigned to the partner. Explicit ridge
jumps, capture, and other ownership handoffs are separate processes.
"""
import unittest
from unittest.mock import patch
import numpy as np
import native_spreading
from tests.test_native_spreading_pairing import scene


class SpreadingOwnershipTests(unittest.TestCase):
    def run_step(self, skew=None, dt=2.):
        s, _ = scene()
        if skew is not None: s.support[0] *= skew
        transported = s.support.copy()
        before = transported.copy()
        birth = native_spreading.advance(s, transported, dt)
        receiving = np.flatnonzero(birth > 0.)
        self.assertTrue(len(receiving), 'the fixture must actually accrete')
        return s, before, transported, birth, receiving

    def assert_no_handoff(self, before, after, birth, receiving):
        retained = before[:, receiving]*(1.-birth[receiving])
        np.testing.assert_array_less(retained-after[:, receiving], 1e-9)

    def test_existing_ocean_crust_is_never_handed_across_the_axis(self):
        _, before, after, birth, receiving = self.run_step()
        self.assert_no_handoff(before, after, birth, receiving)

    def test_unequal_support_across_the_axis_does_not_move_older_crust(self):
        # A continent and its thin skirt facing a vast ocean plate skews the
        # support contrast. Splitting a whole receiving cell by that contrast
        # transferred established crust across the axis on every step.
        _, before, after, birth, receiving = self.run_step(skew=.6)
        self.assert_no_handoff(before, after, birth, receiving)

    def test_receiving_cells_conserve_total_support(self):
        _, before, after, birth, receiving = self.run_step(skew=.6)
        np.testing.assert_allclose(after[:, receiving].sum(axis=0),
                                   before[:, receiving].sum(axis=0), rtol=0., atol=1e-9)

    def test_new_crust_is_credited_to_the_spreading_pair_alone(self):
        _, before, after, birth, receiving = self.run_step()
        gained = after[:, receiving]-before[:, receiving]*(1.-birth[receiving])
        np.testing.assert_allclose(gained[2], 0., atol=1e-12)
        self.assertGreater(gained[:2].sum(), 0.)

    def test_each_shore_receives_its_own_new_crust(self):
        # The fixture's axis is y=0: plate 0 rotates toward negative y, plate 1
        # toward positive y. These receiving cells do not straddle that axis,
        # so their geometric side is an independent ownership oracle.
        for skew in (None, .6):
            with self.subTest(skew=skew):
                s, before, after, birth, receiving = self.run_step(skew=skew)
                y = s.native_mesh['vertices'][s.native_mesh['faces'][receiving], 1]
                negative = np.all(y <= 0., axis=1) & np.any(y < 0., axis=1)
                positive = np.all(y >= 0., axis=1) & np.any(y > 0., axis=1)
                self.assertTrue(np.all(negative | positive))
                self.assertTrue(np.any(negative) and np.any(positive))
                gained = after[:, receiving]-before[:, receiving]*(1.-birth[receiving])
                expected = np.zeros_like(gained)
                produced = before[:, receiving].sum(axis=0)*birth[receiving]
                expected[0, negative] = produced[negative]
                expected[1, positive] = produced[positive]
                np.testing.assert_allclose(gained, expected, rtol=1e-12, atol=1e-14)

    def test_reported_side_areas_are_independent_sums(self):
        # Both sides were once the same variable printed twice, so the equality
        # held by construction and could not witness anything.
        s, _, _, _, _ = self.run_step()
        diagnostic = s.spreading_diagnostics
        self.assertAlmostEqual(diagnostic['side_p_area_km2'],
                               diagnostic['side_q_area_km2'], delta=1e-6)
        self.assertAlmostEqual(diagnostic['generated_area_km2'],
                               diagnostic['side_p_area_km2']+diagnostic['side_q_area_km2'],
                               delta=1e-6)

    def test_reported_q_area_retains_its_independent_measurement(self):
        # A tiny q-side measurement perturbation remains below the unchanged
        # congruence gate. It must appear in diagnostics, while shared congruent
        # deposition and ownership retain the exact original physical output.
        s, _, after, birth, _ = self.run_step()
        original = native_spreading._area
        scale = 1.+5e-10

        def measured_area(polygon, context):
            area = original(polygon, context)
            return area*scale if len(polygon) and np.mean(polygon[:, 1]) > 0. else area

        with patch.object(native_spreading, '_area', side_effect=measured_area):
            perturbed, _, changed_after, changed_birth, _ = self.run_step()
        np.testing.assert_array_equal(changed_birth, birth)
        np.testing.assert_array_equal(changed_after, after)
        before = s.spreading_diagnostics
        diagnostic = perturbed.spreading_diagnostics
        self.assertEqual(diagnostic['generated_area_km2'], before['generated_area_km2'])
        self.assertEqual(diagnostic['side_p_area_km2'], before['side_p_area_km2'])
        self.assertAlmostEqual(diagnostic['side_q_area_km2'],
                               before['side_q_area_km2']*scale, delta=1e-9)
        self.assertGreater(diagnostic['side_q_area_km2']-diagnostic['side_p_area_km2'], 1e-7)


if __name__ == '__main__':
    unittest.main()
