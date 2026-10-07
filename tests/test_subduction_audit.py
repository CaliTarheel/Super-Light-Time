"""Read-only subduction audit on a complex-plate world."""
import unittest

import numpy as np

import native_engine
import slab_memory
import subduction_audit

CONFIG = dict(width=48, height=24, mesh_level=4, coast_geometry_level=2, plate_count=4,
              mechanics_nodes=128, seed=37, physics_profile='reviewed_v1',
              primordial_subduction={'enabled': True},
              primordial_ocean={'enabled': True, 'continental_attachments':
                                [dict(region_index=1, continental_plate_uid=3)]})


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = native_engine.Simulation(CONFIG)
        cls.report = subduction_audit.audit(cls.world)

    def test_shares_partition_the_loaded_trench_by_length_and_by_force(self):
        summary = self.report['summary']
        for prefix in ('loaded_', 'pull_'):
            total = sum(summary[prefix+kind+'_fraction'] for kind in ('converging', 'stationary', 'opening'))
            self.assertAlmostEqual(total, 1.)
        owners, load, _ = slab_memory.line_load(self.world)
        loaded = (owners >= 0) & (load > 0.)
        self.assertAlmostEqual(summary['loaded_length_km'], float(np.asarray(self.world.bl)[loaded].sum()))
        self.assertGreater(summary['total_slab_pull_n'], 0.)
        # Most inherited pull starts on converging trench, with no polarity error.
        self.assertGreater(summary['pull_converging_fraction'], .8)
        self.assertEqual(summary['polarity_mismatch_km'], 0.)

    def test_plate_records_and_inflation(self):
        plates = self.report['plates']
        self.assertTrue(plates)
        for plate in plates:
            self.assertGreaterEqual(plate['cancellation'], 0.)
            self.assertLessEqual(plate['cancellation'], 1.)
            self.assertGreater(plate['smooth_length_km'], 0.)
            # A resolved trace is at least as long as its smooth length and,
            # at this resolution, not grossly longer.
            self.assertGreater(plate['trace_inflation'], .95)
            self.assertLess(plate['trace_inflation'], 1.5)
        self.assertEqual(self.report['summary']['live_systems'], len(self.world.trench_systems))

    def test_audit_never_changes_state(self):
        before = [dict(r) for r in self.world.trench_systems]
        subduction_audit.audit(self.world)
        self.assertEqual([dict(r) for r in self.world.trench_systems], before)


if __name__ == '__main__':
    unittest.main()
