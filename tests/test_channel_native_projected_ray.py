"""One projected hinge ray bound to native upper and lower material."""

import unittest

import numpy as np

from channel_native_projected_ray import (
    prepare_native_projected_pair, solve_native_projected_ray)
from tests.admissible_native_contact import admissible_native_contact_world


class NativeProjectedRayTests(unittest.TestCase):
    def test_current_native_material_supplies_a_reciprocal_free_hinge_ray(self):
        s, lower_id, upper_id = admissible_native_contact_world()
        store = s.channel_region_store
        prepared = prepare_native_projected_pair(s, lower_id, upper_id)
        self.assertGreater(prepared['native_unprojected_contact_area_km2'],
                           prepared['projected_pair']['physical_contact_area_km2'])
        count = len(prepared['line_union']['line_weight_m'])
        ray = 3 * count // 4
        offsets = np.linspace(0., 160000., 65)
        def solve():
            return solve_native_projected_ray(
                s, prepared, ray, offsets, 1e23, 1e23, 20000.)
        result = solve()
        self.assertIs(s.channel_region_store, store)
        self.assertEqual(result['upper_material']['plate_owner'], 1)
        self.assertEqual(result['upper_material']['face_id'][0], upper_id)
        np.testing.assert_array_equal(
            result['upper_material']['unloaded_upper_base_depth_m'], 35000.)
        self.assertTrue(result['solution']['contact_mask'][0, 0])
        self.assertGreater(result['solution']['hinge_line_reaction_n_per_m'][0], 0.)
        self.assertGreaterEqual(result['solution']['minimum_gap_m'], -1e-6)
        self.assertGreater(result['source_load_n'], 0.)
        datum = store['records'][1]['history']['regions'][0][
            'upper_basal_reference']
        datum['depth_m'] += 1.
        plus = solve()['solution']['energy_j']
        datum['depth_m'] -= 2.
        minus = solve()['solution']['energy_j']
        datum['depth_m'] += 1.
        measured = (plus - minus) / 2.
        expected = float(result['solution']['basal_reference_derivative_n'].sum())
        np.testing.assert_allclose(measured, expected, rtol=2e-8)

    def test_missing_upper_strip_and_stale_geometry_fail_closed(self):
        s, lower_id, upper_id = admissible_native_contact_world()
        prepared = prepare_native_projected_pair(s, lower_id, upper_id)
        offsets = np.linspace(0., 160000., 65)
        with self.assertRaisesRegex(ValueError, 'material-attached source regions'):
            solve_native_projected_ray(s, prepared, 0, offsets,
                                       1e23, 1e23, 20000.)
        with self.assertRaisesRegex(ValueError, 'upper-mantle hinge strip'):
            solve_native_projected_ray(s, prepared, 0,
                                       np.linspace(0., 700000., 65),
                                       1e23, 1e23, 20000.)
        s.material_surface['geometry_revision'] += 1
        with self.assertRaisesRegex(ValueError, 'material geometry is stale'):
            solve_native_projected_ray(s, prepared, 3 * len(
                prepared['line_union']['line_weight_m']) // 4,
                offsets, 1e23, 1e23, 20000.)
        s.material_surface['geometry_revision'] -= 1
        s.collision_contacts[0]['last_seen_myr'] -= 1.
        with self.assertRaisesRegex(ValueError, 'not current'):
            solve_native_projected_ray(s, prepared, 3 * len(
                prepared['line_union']['line_weight_m']) // 4,
                offsets, 1e23, 1e23, 20000.)

    def test_material_column_and_phase_identity_changes_invalidate_prepared_loads(self):
        s, lower_id, upper_id = admissible_native_contact_world()
        prepared = prepare_native_projected_pair(s, lower_id, upper_id)
        ray = 3 * len(prepared['line_union']['line_weight_m']) // 4
        offsets = np.linspace(0., 160000., 65)
        def solve():
            return solve_native_projected_ray(
                s, prepared, ray, offsets, 1e23, 1e23, 20000.)
        record = s.channel_region_store['records'][0]
        record['history']['regions'][0]['column']['thickness_km'][0] += 1.
        s.structure['thickness_km'][0] += 1.
        with self.assertRaisesRegex(ValueError, 'material loads are stale'):
            solve()
        record['history']['regions'][0]['column']['thickness_km'][0] -= 1.
        s.structure['thickness_km'][0] -= 1.
        record['history']['regions'][0]['region_id'] = 'new-phase-identity'
        with self.assertRaisesRegex(ValueError, 'material loads are stale'):
            solve()


if __name__ == '__main__':
    unittest.main()
