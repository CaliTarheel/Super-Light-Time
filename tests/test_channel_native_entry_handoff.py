"""Read-only old-entry to projected-contact work and force accounting."""

import unittest

import numpy as np

import channel_region_native
import dense_crust
import mesh_coverage
from channel_native_entry_handoff import audit_uniform_entry_handoff
from channel_native_projected_ray import prepare_native_projected_pair
from tests.admissible_native_contact import admissible_native_contact_world


class NativeEntryHandoffTests(unittest.TestCase):
    def setUp(self):
        self.s, lower, upper = admissible_native_contact_world()
        self.prepared = prepare_native_projected_pair(self.s, lower, upper)
        self.lower = lower

    def audit(self, nodes=65, **kwargs):
        return audit_uniform_entry_handoff(
            self.s, self.prepared, np.linspace(0., 160000., nodes), **kwargs)

    def test_uniform_native_handoff_has_explicit_stable_budgets(self):
        before = self.s.channel_region_store
        ledgers = [self.audit(nodes) for nodes in (33, 65, 129)]
        self.assertIs(self.s.channel_region_store, before)
        for ledger in ledgers:
            self.assertGreater(ledger['old_entry_work_to_remove_j'], 0.)
            self.assertGreater(ledger['projected_contact_force_n'], 0.)
            self.assertAlmostEqual(
                ledger['work_difference_j'],
                ledger['represented_preferred_path_work_j']
                - ledger['old_entry_work_to_remove_j'], delta=1e8)
            self.assertLess(abs(ledger['relative_work_difference']), 1e-4)
            self.assertGreater(abs(ledger['relative_work_difference']), 1e-6)
            self.assertLess(abs(ledger['relative_force_difference']), 1e-4)
            self.assertGreater(abs(ledger['relative_force_difference']), 1e-7)
        np.testing.assert_allclose(
            [row['represented_preferred_path_work_j'] for row in ledgers],
            ledgers[0]['represented_preferred_path_work_j'], rtol=2e-12)

    def test_tighter_work_or_force_budget_rejects_handoff(self):
        with self.assertRaisesRegex(ValueError, 'work exceed their error budget'):
            self.audit(work_tolerance=1e-6)
        with self.assertRaisesRegex(ValueError, 'forces exceed their error budget'):
            self.audit(force_tolerance=1e-7)

    def test_stale_geometry_hinge_contact_and_material_fail_closed(self):
        self.s.material_surface['geometry_revision'] += 1
        with self.assertRaisesRegex(ValueError, 'current native regional geometry'):
            self.audit()
        self.s.material_surface['geometry_revision'] -= 1
        self.s.collision_contacts[0]['last_seen_myr'] -= 1.
        with self.assertRaisesRegex(ValueError, 'not current'):
            self.audit()
        self.s.collision_contacts[0]['last_seen_myr'] += 1.
        region = self.s.channel_region_store['records'][0]['history']['regions'][0]
        region['column']['thickness_km'][0] += 1.
        self.s.structure['thickness_km'][0] += 1.
        with self.assertRaisesRegex(ValueError, 'material loads are stale'):
            self.audit()

    def test_nonuniform_lower_history_needs_regional_entry_integral(self):
        record = self.s.channel_region_store['records'][0]
        a, b, c = record['face_triangle']
        middle = (a + b) / np.linalg.norm(a + b)
        polygons = [np.array([a, middle, c]), np.array([middle, b, c])]
        area = self.s.material_surface['area_km2'][0]
        source = record['history']['regions'][0]['column']
        columns = [{key: value.copy() for key, value in source.items()}
                   for _ in polygons]
        columns[1][dense_crust.DENSE][:] = 1.5
        columns[1][dense_crust.CONVERTED][:] = 1.5
        record['polygons'] = polygons
        record['history']['regions'] = [
            dict(region_id=str(index),
                 fraction=mesh_coverage._polygon_area(polygon, 6371.) / area,
                 column=column)
            for index, (polygon, column) in enumerate(zip(polygons, columns))]
        projected = channel_region_native._project(record, set(source))
        for key in self.s.structure:
            self.s.structure[key][0] = projected[key][0]
        with self.assertRaisesRegex(ValueError, 'one uniform regional phase'):
            self.audit()


if __name__ == '__main__':
    unittest.main()
