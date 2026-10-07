"""Phase-localized entry energy versus the installed mean-face potential."""

import unittest

import numpy as np

import channel_region_native
import dense_crust
from channel_native_projected_ray import prepare_native_projected_pair
from channel_regional_entry_work import regional_contact_entry_work
from tests.admissible_native_contact import admissible_native_contact_world
from tests.test_channel_native_regional_contact_loads import regional_pair


class RegionalEntryWorkTests(unittest.TestCase):
    def mixed(self):
        s, geometry, normal = regional_pair()
        s.retained_dense_crust_version = 1
        s.collision_contacts = []
        return s, geometry, normal

    def test_localized_dense_phase_changes_work_and_reverses_with_position(self):
        s, geometry, normal = self.mixed()
        before = regional_contact_entry_work(s, 11, geometry, normal, 50.)
        self.assertGreater(before['contact_buoyancy_force_n'], 0.)
        self.assertLess(before['phase_localization_difference_j'], -8e21)
        self.assertAlmostEqual(
            before['total_work_difference_j'],
            before['phase_localization_difference_j']
            + before['geometry_measure_difference_j'], delta=1e8)
        record = s.channel_region_store['records'][0]
        first, second = record['history']['regions']
        for field in (dense_crust.DENSE, dense_crust.CONVERTED):
            first['column'][field][:], second['column'][field][:] = (
                second['column'][field].copy(), first['column'][field].copy())
        projected = channel_region_native._project(record, set(s.structure))
        for field in s.structure:
            s.structure[field][0] = projected[field][0]
        after = regional_contact_entry_work(s, 11, geometry, normal, 50.)
        self.assertGreater(after['phase_localization_difference_j'], 8e21)
        np.testing.assert_allclose(
            after['phase_localization_difference_j'],
            -before['phase_localization_difference_j'], rtol=1e-10)

    def test_spherical_coordinate_quadrature_converges(self):
        s, geometry, normal = self.mixed()
        values = [regional_contact_entry_work(
            s, 11, geometry, normal, 50., refinement=level)
            for level in (1, 2, 3)]
        self.assertGreater(values[0]['max_coordinate_refinement_difference_m'],
                           values[1]['max_coordinate_refinement_difference_m'])
        self.assertGreater(values[1]['max_coordinate_refinement_difference_m'],
                           values[2]['max_coordinate_refinement_difference_m'])
        self.assertLess(abs(values[2]['phase_resolved_work_j']
                            - values[1]['phase_resolved_work_j'])
                        / abs(values[2]['phase_resolved_work_j']), 2e-7)

    def test_uniform_phase_has_no_localization_term(self):
        s, lower, upper = admissible_native_contact_world()
        prepared = prepare_native_projected_pair(s, lower, upper)
        result = regional_contact_entry_work(
            s, lower, prepared['projected_pair'],
            prepared['hinge_normal'], prepared['dip_degrees'])
        self.assertLess(abs(result['phase_localization_difference_j'])
                        / abs(result['phase_resolved_work_j']), 1e-12)
        self.assertNotEqual(result['geometry_measure_difference_j'], 0.)

    def test_unrecognized_dense_phase_and_wrong_projection_fail_closed(self):
        s, geometry, normal = self.mixed()
        s.retained_dense_crust_version = 0
        with self.assertRaisesRegex(ValueError, 'phase mass disagrees'):
            regional_contact_entry_work(s, 11, geometry, normal, 50.)
        s.retained_dense_crust_version = 1
        with self.assertRaisesRegex(ValueError, 'hinge disagrees'):
            regional_contact_entry_work(s, 11, geometry, normal, 55.)
        with self.assertRaisesRegex(ValueError, 'unresolved'):
            regional_contact_entry_work(
                s, 11, geometry, normal, 50., refinement=0,
                coordinate_tolerance=1e-8)


if __name__ == '__main__':
    unittest.main()
