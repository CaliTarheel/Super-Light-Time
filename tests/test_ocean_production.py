"""A ridge fills the hole it opened, and the closure stops picking winners.

Rigidly rotating each plate leaves a cell short wherever two flanks pull
apart. Under policy 0 the ridge repaints that cell without changing its total,
so the hole is closed by `s.support = moved/total`, which scales every plate up
in proportion to what it already holds - the opening is awarded by incumbency,
and on the live world that handed the largest plate 569,322 km2 a step while a
small plate with its own ridge went backwards at -0.82x its entitlement.

Policy 1 spends one budget in the right order: the pair's opening fills the gap
first and only the remainder repaints, and the closure fits plates and cells at
once so it can restore cover without deciding ownership.
"""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tests'), str(ROOT)]
import native_processes as processes
import native_spreading
import trench_history
from test_native_processes import ocean_fixture, reset_trenches


def world(policy):
    s = ocean_fixture(level=3, two=True)
    s.omega[0] = [0, 0, -.005]; s.omega[1] = [0, 0, .005]
    reset_trenches(s)
    if policy:
        s.ocean_production_version = policy
    return s


class OceanProductionTests(unittest.TestCase):
    def test_policy_zero_is_the_behaviour_it_always_had(self):
        # The default path must be untouched, because every frame already on
        # disk was produced by it.
        plain, gated = world(0), world(0)
        gated.ocean_production_version = 0
        processes.advect_ocean(plain, 2.)
        processes.advect_ocean(gated, 2.)
        np.testing.assert_array_equal(plain.support, gated.support)

    def test_policy_zero_leaves_the_gap_for_the_closure(self):
        s = world(0)
        before = s.support.copy()
        processes.advect_ocean(s, 2.)
        # Spreading conserved each receiving cell, so cover was restored by the
        # proportional closure rather than by the ridge.
        self.assertEqual(s.spreading_diagnostics['generated_area_km2'] > 0., True)
        np.testing.assert_allclose(s.support.sum(axis=0), 1., rtol=0., atol=1e-12)

    def test_policy_one_fills_the_opening_with_new_crust(self):
        s = world(1)
        processes.advect_ocean(s, 2.)
        np.testing.assert_allclose(s.support.sum(axis=0), 1., rtol=0., atol=1e-9)
        self.assertGreater(s.spreading_diagnostics['generated_area_km2'], 0.)

    def test_the_opening_is_never_overspent(self):
        # The gap bounds the fill, so a cell can be brought to full cover and
        # no further however much a ridge is producing.
        s = world(1)
        processes.advect_ocean(s, 2.)
        self.assertLessEqual(float(s.support.sum(axis=0).max()), 1.+1e-9)
        self.assertGreaterEqual(float(s.support.min()), -1e-12)

    def test_the_closure_keeps_the_area_the_operators_set(self):
        s = world(1)
        processes.advect_ocean(s, 2.)
        report = s.partition_closure
        self.assertTrue(report['converged'], report)
        # A thousand square metres against plate areas of order 1e8 km2.
        self.assertLessEqual(report['row_residual_km2'], 1e-3)

    def test_both_flanks_are_paid_from_the_same_opening(self):
        s = world(1)
        processes.advect_ocean(s, 2.)
        d = s.spreading_diagnostics
        self.assertAlmostEqual(d['side_p_area_km2'], d['side_q_area_km2'], places=6)

    def test_an_unsupported_policy_is_refused(self):
        for bad in (True, 2, 'yes', -1):
            s = world(0)
            s.ocean_production_version = bad
            with self.assertRaises(ValueError):
                native_spreading.production_version(s)
        # An absent attribute is policy 0, which is how a restored checkpoint
        # that predates the policy must read.
        s = world(0)
        self.assertFalse(hasattr(s, 'ocean_production_version'))
        self.assertEqual(native_spreading.production_version(s), 0)


if __name__ == '__main__':
    unittest.main()
