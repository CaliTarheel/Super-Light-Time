"""Pressure/thermal separation and inventory checks for a contact column."""

import unittest
import pickle
import json

import numpy as np

import crust_inventory
import dense_crust
import phase_evolution
from experiments.entry_channel_source_oracle import (
    advance_column, advance_ordered_column, project_face_regions,
    initialize_region_history,
    advance_region_history, remap_region_history, expand_zone_loads,
    coalesce_identical_region_histories)
from tests.test_entry_phase_depth import world


class EntryChannelSourceOracleTests(unittest.TestCase):
    def setUp(self):
        self.model = world()
        self.state = self.model.structure
        self.controls = self.model.retained_phase_parameters

    def test_mantle_only_matches_existing_pure_phase_kernel(self):
        result = advance_column(
            self.state, 0, 80000., 0., 0., 80., .25,
            constitutive_parameters=self.controls)
        direct = phase_evolution._advance_regions(
            self.state, np.array([0]), np.array([80.]),
            np.array([3300. * 80000.]),
            phase_evolution.parameters(self.controls), .25)
        for name, value in direct[0].items():
            np.testing.assert_array_equal(result['column'][name], value)
        self.assertEqual(result['returned_km'], direct[1][0])
        self.assertEqual(result['contraction_km'], direct[2][0])
        self.assertEqual(result['top_pressure_pa'], 9.81 * 3300. * 80000.)

    def test_ordered_multi_sheet_load_feeds_pure_phase_kernel_once(self):
        segments = [
            dict(kind='rock', top_depth_m=0., base_depth_m=20000.,
                 sheet_id=3, mass_kg_m2=2800. * 20000.),
            dict(kind='rock', top_depth_m=20000., base_depth_m=45000.,
                 sheet_id=2, mass_kg_m2=2900. * 25000.),
            dict(kind='mantle', top_depth_m=45000., base_depth_m=80000.),
        ]
        original = {key: value.copy() for key, value in self.state.items()}
        result = advance_ordered_column(
            self.state, 0, 80000., segments, (3, 2), 80., .25,
            constitutive_parameters=self.controls)
        mass = 2800. * 20000. + 2900. * 25000. + 3300. * 35000.
        direct = phase_evolution._advance_regions(
            self.state, np.array([0]), np.array([80.]),
            np.array([mass]), phase_evolution.parameters(self.controls), .25)
        for key in direct[0]:
            np.testing.assert_array_equal(result['column'][key], direct[0][key])
        self.assertEqual(result['top_pressure_pa'], 9.81 * mass)
        self.assertEqual(result['rock_sheet_ids'], (3, 2))
        for key, value in original.items():
            np.testing.assert_array_equal(self.state[key], value)

    def test_ordered_regional_history_retains_local_phase_and_rejects_mixed_schema(self):
        history = initialize_region_history(
            self.state, 0, [dict(region_id='contact', fraction=1.)])
        segments = [
            dict(kind='rock', top_depth_m=-1000., base_depth_m=19000.,
                 sheet_id=3, mass_kg_m2=2800. * 20000.),
            dict(kind='rock', top_depth_m=19000., base_depth_m=44000.,
                 sheet_id=2, mass_kg_m2=2900. * 25000.),
            dict(kind='mantle', top_depth_m=44000., base_depth_m=80000.),
        ]
        load = dict(region_id='contact', lower_top_depth_m=80000.,
                    segments=segments, covering_sheets_top_to_bottom=(3, 2),
                    effective_thermal_depth_km=80.)
        stage = advance_region_history(
            history, [load], .25, constitutive_parameters=self.controls)
        direct = advance_ordered_column(
            self.state, 0, 80000., segments, (3, 2), 80., .25,
            constitutive_parameters=self.controls)
        for name in direct['column']:
            np.testing.assert_array_equal(
                stage['history']['regions'][0]['column'][name],
                direct['column'][name])
        self.assertEqual(stage['top_pressure_pa'][0], direct['top_pressure_pa'])
        with self.assertRaisesRegex(ValueError, 'pressure and thermal path'):
            advance_region_history(history, [dict(load,
                upper_column_mass_kg_m2=1.)], .25)
        with self.assertRaisesRegex(ValueError, 'vertical order'):
            advance_region_history(history, [dict(load,
                covering_sheets_top_to_bottom=(2, 3))], .25)

    def test_layered_contact_preserves_mass_heat_and_input_state(self):
        original = {key: value.copy() for key, value in self.state.items()}
        result = advance_column(
            self.state, 0, 80000., 35000., 2800. * 35000., 80., .25,
            constitutive_parameters=self.controls)
        for key, value in original.items():
            np.testing.assert_array_equal(self.state[key], value)
        updated = result['column']
        before_mass = (dense_crust.mass_volume(self.state)[0]
                       + self.state[crust_inventory.RETURNED][0])
        after_mass = (dense_crust.mass_volume(updated)[0]
                      + updated[crust_inventory.RETURNED][0])
        np.testing.assert_allclose(after_mass, before_mass, rtol=2e-14)
        def heat(state, index):
            return (state[dense_crust.HEAT][index]
                    + state[dense_crust.HEAT_RETURNED][index]
                    - state[dense_crust.HEAT_BATH][index])
        np.testing.assert_allclose(heat(updated, 0), heat(self.state, 0),
                                   rtol=2e-14)
        self.assertEqual(result['top_pressure_pa'],
                         9.81 * (2800. * 35000. + 3300. * 45000.))

    def test_thermal_path_is_explicit_and_independent_of_top_pressure(self):
        controls = dict(self.controls, surface_temperature_c=0.,
                        mantle_temperature_c=1300., geotherm_c_per_km=15.)
        cold = advance_column(
            self.state, 0, 80000., 35000., 2800. * 35000., 30., .25,
            constitutive_parameters=controls)
        hot = advance_column(
            self.state, 0, 80000., 35000., 2800. * 35000., 80., .25,
            constitutive_parameters=controls)
        self.assertEqual(cold['top_pressure_pa'], hot['top_pressure_pa'])
        self.assertGreater(hot['column'][dense_crust.HEAT][0],
                           cold['column'][dense_crust.HEAT][0])

    def test_invalid_geometry_fails_without_mutating_column(self):
        original = {key: value.copy() for key, value in self.state.items()}
        with self.assertRaisesRegex(ValueError, 'nonpenetrating'):
            advance_column(
                self.state, 0, 34000., 35000., 2800. * 35000., 80., .25,
                constitutive_parameters=self.controls)
        for key, value in original.items():
            np.testing.assert_array_equal(self.state[key], value)

    def test_disjoint_regions_react_before_conservative_projection(self):
        regions = [
            dict(fraction=.5, lower_top_depth_m=0.,
                 current_upper_base_depth_m=0.,
                 upper_column_mass_kg_m2=0., effective_thermal_depth_km=0.),
            dict(fraction=.5, lower_top_depth_m=80000.,
                 current_upper_base_depth_m=35000.,
                 upper_column_mass_kg_m2=2800. * 35000.,
                 effective_thermal_depth_km=80.),
        ]
        original = {key: value.copy() for key, value in self.state.items()}
        result = project_face_regions(
            self.state, 0, regions, 1.,
            constitutive_parameters=self.controls, face_area_km2=1000.)
        shallow = advance_column(
            self.state, 0, 0., 0., 0., 0., 1.,
            constitutive_parameters=self.controls)
        deep = advance_column(
            self.state, 0, 80000., 35000., 2800. * 35000., 80., 1.,
            constitutive_parameters=self.controls)
        expected_dense = .5 * (shallow['column'][dense_crust.DENSE][0]
                                + deep['column'][dense_crust.DENSE][0])
        np.testing.assert_allclose(
            result['column'][dense_crust.DENSE][0], expected_dense,
            rtol=2e-14)
        averaged_first = advance_column(
            self.state, 0, 40000., 17500., 1400. * 35000., 40., 1.,
            constitutive_parameters=self.controls)
        self.assertGreater(
            abs(result['column'][dense_crust.DENSE][0]
                - averaged_first['column'][dense_crust.DENSE][0]), .1)
        before_mass = (dense_crust.mass_volume(self.state)[0]
                       + self.state[crust_inventory.RETURNED][0])
        after_mass = (dense_crust.mass_volume(result['column'])[0]
                      + result['column'][crust_inventory.RETURNED][0])
        np.testing.assert_allclose(after_mass, before_mass, rtol=2e-14)
        def heat(state, index):
            return (state[dense_crust.HEAT][index]
                    + state[dense_crust.HEAT_RETURNED][index]
                    - state[dense_crust.HEAT_BATH][index])
        np.testing.assert_allclose(
            heat(result['column'], 0), heat(self.state, 0), rtol=2e-14)
        self.assertEqual(result['returned_volume_km3'],
                         1000. * result['returned_km'])
        for key, value in original.items():
            np.testing.assert_array_equal(self.state[key], value)

    def test_single_region_matches_column_and_bad_partitions_fail(self):
        region = dict(fraction=1., lower_top_depth_m=80000.,
                      current_upper_base_depth_m=35000.,
                      upper_column_mass_kg_m2=2800. * 35000.,
                      effective_thermal_depth_km=80.)
        merged = project_face_regions(
            self.state, 0, [region], .25,
            constitutive_parameters=self.controls)['column']
        direct = advance_column(
            self.state, 0, 80000., 35000., 2800. * 35000., 80., .25,
            constitutive_parameters=self.controls)['column']
        for name in direct:
            np.testing.assert_array_equal(merged[name], direct[name])
        with self.assertRaisesRegex(ValueError, 'partition one face'):
            project_face_regions(
                self.state, 0, [dict(region, fraction=.9)], .25,
                constitutive_parameters=self.controls)
        missing = dict(region)
        del missing['effective_thermal_depth_km']
        with self.assertRaisesRegex(ValueError, 'explicit pressure or thermal'):
            project_face_regions(
                self.state, 0, [missing], .25,
                constitutive_parameters=self.controls)

    def test_fixed_regions_retain_distinct_histories_across_source_steps(self):
        history = initialize_region_history(self.state, 0, [
            dict(region_id='shallow', fraction=.5),
            dict(region_id='deep', fraction=.5)])
        loads = [
            dict(region_id='shallow', lower_top_depth_m=0.,
                 current_upper_base_depth_m=0., upper_column_mass_kg_m2=0.,
                 effective_thermal_depth_km=0.),
            dict(region_id='deep', lower_top_depth_m=80000.,
                 current_upper_base_depth_m=35000.,
                 upper_column_mass_kg_m2=2800. * 35000.,
                 effective_thermal_depth_km=80.),
        ]
        saved = pickle.dumps(history)
        first = advance_region_history(
            history, loads, 1., constitutive_parameters=self.controls,
            face_area_km2=1000.)
        self.assertEqual(pickle.dumps(history), saved)
        replay = advance_region_history(
            pickle.loads(saved), list(reversed(loads)), 1.,
            constitutive_parameters=self.controls)
        for key in first['column']:
            np.testing.assert_array_equal(first['column'][key], replay['column'][key])
        second = advance_region_history(
            first['history'], loads, 1., constitutive_parameters=self.controls)
        collapsed = project_face_regions(
            first['column'], 0,
            [dict(fraction=.5, **{key: value for key, value in load.items()
                                  if key != 'region_id'}) for load in loads],
            1., constitutive_parameters=self.controls)
        self.assertGreater(
            abs(second['column'][dense_crust.DENSE][0]
                - collapsed['column'][dense_crust.DENSE][0]), .01)
        for region in second['history']['regions']:
            before = (dense_crust.mass_volume(self.state)[0]
                      + self.state[crust_inventory.RETURNED][0])
            after = (dense_crust.mass_volume(region['column'])[0]
                     + region['column'][crust_inventory.RETURNED][0])
            np.testing.assert_allclose(after, before, rtol=2e-14)
        self.assertEqual(first['returned_volume_km3'],
                         1000. * first['returned_km'])
        with self.assertRaisesRegex(ValueError, 'per region ID'):
            advance_region_history(history, loads[:1], 1.,
                                   constitutive_parameters=self.controls)
        self.assertEqual(pickle.dumps(history), saved)

    def test_fixed_region_history_rejects_incomplete_or_duplicate_partition(self):
        with self.assertRaisesRegex(ValueError, 'partition one face'):
            initialize_region_history(self.state, 0, [
                dict(region_id='a', fraction=.4),
                dict(region_id='b', fraction=.4)])
        with self.assertRaisesRegex(ValueError, 'unique'):
            initialize_region_history(self.state, 0, [
                dict(region_id='a', fraction=.5),
                dict(region_id='a', fraction=.5)])

    def test_identical_same_zone_children_coalesce_without_source_change(self):
        original = initialize_region_history(self.state, 0, [
            dict(region_id='a', fraction=.5),
            dict(region_id='b', fraction=.5)])
        self.assertEqual(
            coalesce_identical_region_histories(original)['result_region_count'], 2)
        split = remap_region_history(
            original, [dict(zone_id='contact', fraction=1.)], [[.5], [.5]])
        saved = pickle.dumps(split)
        reduced = coalesce_identical_region_histories(split)
        self.assertEqual(reduced['original_region_count'], 2)
        self.assertEqual(reduced['result_region_count'], 1)
        self.assertEqual(reduced['history']['regions'][0]['fraction'], 1.)
        load = dict(zone_id='contact', lower_top_depth_m=80000.,
                    current_upper_base_depth_m=35000.,
                    upper_column_mass_kg_m2=2800. * 35000.,
                    effective_thermal_depth_km=80.)
        full = advance_region_history(
            split, expand_zone_loads(split, [load]), .5,
            constitutive_parameters=self.controls)
        compact = advance_region_history(
            reduced['history'], expand_zone_loads(reduced['history'], [load]),
            .5, constitutive_parameters=self.controls)
        for key in full['column']:
            np.testing.assert_array_equal(full['column'][key], compact['column'][key])
        self.assertEqual(full['returned_km'], compact['returned_km'])
        self.assertEqual(pickle.dumps(split), saved)

    def test_distinct_phase_histories_in_one_zone_are_not_coalesced(self):
        initial = initialize_region_history(self.state, 0, [
            dict(region_id='cold', fraction=.5),
            dict(region_id='buried', fraction=.5)])
        loads = [
            dict(region_id='cold', lower_top_depth_m=0.,
                 current_upper_base_depth_m=0., upper_column_mass_kg_m2=0.,
                 effective_thermal_depth_km=0.),
            dict(region_id='buried', lower_top_depth_m=80000.,
                 current_upper_base_depth_m=35000.,
                 upper_column_mass_kg_m2=2800. * 35000.,
                 effective_thermal_depth_km=80.),
        ]
        evolved = advance_region_history(
            initial, loads, 1., constitutive_parameters=self.controls)
        moved = remap_region_history(
            evolved['history'], [dict(zone_id='one', fraction=1.)],
            [[.5], [.5]])
        reduced = coalesce_identical_region_histories(moved)
        self.assertEqual(reduced['result_region_count'], 2)
        self.assertNotEqual(
            reduced['history']['regions'][0]['column'][dense_crust.DENSE][0],
            reduced['history']['regions'][1]['column'][dense_crust.DENSE][0])

    def test_moving_zone_intersections_split_without_erasing_material_history(self):
        history = initialize_region_history(self.state, 0, [
            dict(region_id='cold', fraction=.5),
            dict(region_id='buried', fraction=.5)])
        old_loads = [
            dict(region_id='cold', lower_top_depth_m=0.,
                 current_upper_base_depth_m=0., upper_column_mass_kg_m2=0.,
                 effective_thermal_depth_km=0.),
            dict(region_id='buried', lower_top_depth_m=80000.,
                 current_upper_base_depth_m=35000.,
                 upper_column_mass_kg_m2=2800. * 35000.,
                 effective_thermal_depth_km=80.),
        ]
        first = advance_region_history(
            history, old_loads, 1., constitutive_parameters=self.controls)
        saved = pickle.dumps(first['history'])
        zones = [dict(zone_id='open', fraction=.5),
                 dict(zone_id='contact', fraction=.5)]
        moved = remap_region_history(first['history'], zones,
                                     [[.25, .25], [.25, .25]])
        self.assertEqual(len(moved['regions']), 4)
        self.assertEqual(pickle.dumps(first['history']), saved)
        for child in moved['regions']:
            parent_id, zone_id = json.loads(child['region_id'])
            self.assertEqual(child['fraction'], .25)
            self.assertIn(parent_id, {'cold', 'buried'})
            self.assertEqual(child['zone_id'], zone_id)
        zone_loads = [dict(zone_id='open', lower_top_depth_m=0.,
                           current_upper_base_depth_m=0.,
                           upper_column_mass_kg_m2=0.,
                           effective_thermal_depth_km=0.),
                      dict(zone_id='contact', lower_top_depth_m=80000.,
                           current_upper_base_depth_m=35000.,
                           upper_column_mass_kg_m2=2800. * 35000.,
                           effective_thermal_depth_km=80.)]
        expanded = expand_zone_loads(moved, zone_loads)
        second = advance_region_history(
            moved, expanded, 1., constitutive_parameters=self.controls)
        expected = []
        for parent in first['history']['regions']:
            for zone in zone_loads:
                sample = advance_column(
                    parent['column'], 0, zone['lower_top_depth_m'],
                    zone['current_upper_base_depth_m'],
                    zone['upper_column_mass_kg_m2'],
                    zone['effective_thermal_depth_km'], 1.,
                    constitutive_parameters=self.controls)
                expected.append(sample['column'][dense_crust.DENSE][0])
        np.testing.assert_allclose(
            second['column'][dense_crust.DENSE][0], np.mean(expected),
            rtol=2e-14)
        before_mass = (dense_crust.mass_volume(self.state)[0]
                       + self.state[crust_inventory.RETURNED][0])
        for child in second['history']['regions']:
            after_mass = (dense_crust.mass_volume(child['column'])[0]
                          + child['column'][crust_inventory.RETURNED][0])
            np.testing.assert_allclose(after_mass, before_mass, rtol=2e-14)
        with self.assertRaisesRegex(ValueError, 'partition both'):
            remap_region_history(first['history'], zones,
                                 [[.25, .25], [.20, .30]])
        with self.assertRaisesRegex(ValueError, 'exactly once'):
            expand_zone_loads(moved, zone_loads[:1])
        self.assertEqual(pickle.dumps(first['history']), saved)


if __name__ == '__main__':
    unittest.main()
