"""Bounded source-free oracle for local breakoff before full underthrusting."""
import unittest

from experiments.reduced_collision_trajectory import run


class ReducedCollisionTrajectoryTests(unittest.TestCase):
    def test_local_slab_breakoff_halts_this_prescribed_continent(self):
        result = run(1.)
        first, last = result['ruptures'][0], result['ruptures'][-1]
        final = result['observations'][-1]
        self.assertEqual(len(result['ruptures']), 12)
        self.assertLess(first['time_myr'], 30.)
        self.assertLess(last['time_myr'], 40.)
        self.assertEqual(final['failed_patches'], 12)
        self.assertEqual(final['retained_slab_excess_mass_kg'], 0.)
        self.assertLess(final['relative_omega_rad_myr'], 1e-4)
        self.assertLess(final['relative_travel_km'], 800.)
        self.assertLess(final['entered_area_km2'], 600000.)
        self.assertGreater(final['entered_area_km2'], 400000.)


if __name__ == '__main__':
    unittest.main()
