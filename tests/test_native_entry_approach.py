"""End-to-end audit fields for the finite-front native entry probe."""
import unittest

from experiments.native_entry_approach import run


class NativeEntryApproachTests(unittest.TestCase):
    def test_short_source_reports_conservation_and_inherited_trench(self):
        report=run(.01,.01,automatic_finite_front=True)
        self.assertEqual(report['accepted_source_intervals'],1)
        self.assertEqual(report['trench_system_count'],1)
        self.assertEqual(report['inherited_trench_phase'],'mature')
        self.assertEqual(report['trench_shutdown_version'],1)
        self.assertEqual(report['automatic_admitted_faces'],3)
        self.assertEqual(report['active_contact_count'],0)
        self.assertLess(abs(report['slab_excess_mass_relative_residual']),1e-12)
        self.assertLess(abs(report['last_source_column_volume_relative_residual']),1e-10)
        self.assertIsNone(report['last_source_phase_mass_relative_residual'])


if __name__=='__main__':unittest.main()
