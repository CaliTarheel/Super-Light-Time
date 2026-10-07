"""Opt-in native remesh transaction for material-attached phase regions."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

import channel_region_native
import checkpoint
import crust_inventory
import dense_crust
import mesh_coverage
import adaptive_material
import material_surface
from channel_region_history import advance_region_history
from experiments.channel_marker_binding import bind_marker_regions
from experiments.channel_region_history import initialize_region_history
from native_engine import Simulation
from native_material_adaptivity import adapt, _apply, _categories
from ridge_geometry import rotate
from tests.test_native_material_adaptivity import world, unit


def installed_world():
    s = world()
    crust_inventory.initialize(s.structure)
    dense_crust.initialize(s.structure, 1000.)
    crust_inventory.initialize(s.trace_structure)
    dense_crust.initialize(s.trace_structure, 1000.)
    face = 0
    triangle = s.material_surface['vertices'][s.material_surface['faces'][face]]
    a, b, c = triangle
    middle = unit(a+b)
    polygons = [np.array([a, middle, c]), np.array([middle, b, c])]
    area = np.array([mesh_coverage._polygon_area(polygon, 6371.) for polygon in polygons])
    fractions = area / area.sum()
    face_id = int(s.parcel_patch[face])
    history = initialize_region_history(s.structure, face, [
        dict(region_id='left', fraction=float(fractions[0])),
        dict(region_id='right', fraction=float(fractions[1]))], face_id=face_id)
    right = history['regions'][1]['column']
    right[dense_crust.HEAT] += 10.
    right[dense_crust.HEAT_ADDED] += 10.
    for field in (dense_crust.HEAT, dense_crust.HEAT_ADDED):
        s.structure[field][face] += 10. * fractions[1]
    record = dict(face_id=face_id, face_triangle=triangle.copy(),
                  history=history, polygons=polygons)
    bound = bind_marker_regions(
        s.material_surface, [record], s.trace_id[face:face+1],
        s.trace_patch[face:face+1], s.trace_xyz[face:face+1])[0]
    for name in s.trace_structure:
        s.trace_structure[name][face] = bound['column'][name][0]
    channel_region_native.install(s, [record])
    return s


def uniform_world():
    s = world()
    crust_inventory.initialize(s.structure)
    dense_crust.initialize(s.structure, 1000.)
    crust_inventory.initialize(s.trace_structure)
    dense_crust.initialize(s.trace_structure, 1000.)
    face = 0
    triangle = s.material_surface['vertices'][s.material_surface['faces'][face]]
    face_id = int(s.parcel_patch[face])
    history = initialize_region_history(
        s.structure, face, [dict(region_id='uniform', fraction=1.)], face_id=face_id)
    channel_region_native.install(s, [dict(
        face_id=face_id, face_triangle=triangle.copy(),
        history=history, polygons=[triangle.copy()])])
    return s


def nonuniform_area_world():
    s = installed_world()
    record = deepcopy(s.channel_region_store['records'][0])
    del s.channel_region_store
    for factor, region in zip((.8, 1.2), record['history']['regions']):
        column = region['column']
        column['area_factor'][:] = factor
        column['thickness_km'][:] /= factor
        crust_inventory.validate(column)
        dense_crust.validate(column)
    face = record['history']['face_index']
    projected = channel_region_native._project(record, set(s.structure))
    for name in s.structure:
        s.structure[name][face] = projected[name][0]
    fractions = np.array([region['fraction'] for region in record['history']['regions']])
    factors = np.array([region['column']['area_factor'][0]
                        for region in record['history']['regions']])
    s.mass[face] = s.material_surface['area_km2'][face] * np.sum(fractions / factors)
    s.material_surface['reference_area_km2'][face] = s.mass[face]
    bound = bind_marker_regions(
        s.material_surface, [record], s.trace_id[face:face+1],
        s.trace_patch[face:face+1], s.trace_xyz[face:face+1])[0]
    for name in s.trace_structure:
        s.trace_structure[name][face] = bound['column'][name][0]
    channel_region_native.install(s, [record])
    return s


class NativeChannelRegionTests(unittest.TestCase):
    def test_ordered_mantle_load_matches_legacy_native_source_stage(self):
        s = uniform_world()
        face_id = int(s.channel_region_store['records'][0]['face_id'])
        common = dict(region_id='uniform', lower_top_depth_m=80000.,
                      effective_thermal_depth_km=80.)
        legacy = channel_region_native.prepare_source(s, {face_id: [dict(
            common, current_upper_base_depth_m=0.,
            upper_column_mass_kg_m2=0.)]}, .25)
        ordered = channel_region_native.prepare_source(s, {face_id: [dict(
            common, segments=[dict(kind='mantle',
            top_depth_m=0., base_depth_m=80000.)],
            covering_sheets_top_to_bottom=())]}, .25)
        for name in s.structure:
            np.testing.assert_array_equal(ordered['structure'][name],
                                          legacy['structure'][name])
        self.assertEqual(ordered['returned_volume_km3'],
                         legacy['returned_volume_km3'])
        self.assertEqual(ordered['maximum_ordinary_pressure_pa'],
                         legacy['maximum_ordinary_pressure_pa'])

    def test_native_regional_source_accepts_two_measured_rock_layers(self):
        from entry_channel_column import advance_ordered_column
        s = nonuniform_area_world()
        face_id = int(s.channel_region_store['records'][0]['face_id'])
        old = s.channel_region_store['records'][0]['history']['regions'][1]['column']
        segments = [
            dict(kind='rock', top_depth_m=0., base_depth_m=20000.,
                 sheet_id=3, mass_kg_m2=2800. * 20000.),
            dict(kind='rock', top_depth_m=20000., base_depth_m=45000.,
                 sheet_id=2, mass_kg_m2=2900. * 25000.),
            dict(kind='mantle', top_depth_m=45000., base_depth_m=80000.),
        ]
        loads = [dict(region_id='left', lower_top_depth_m=0.,
                      current_upper_base_depth_m=0.,
                      upper_column_mass_kg_m2=0.,
                      effective_thermal_depth_km=0.),
                 dict(region_id='right', lower_top_depth_m=80000.,
                      segments=segments,
                      covering_sheets_top_to_bottom=(3, 2),
                      effective_thermal_depth_km=80.)]
        stage = channel_region_native.prepare_source(s, {face_id: loads}, .25)
        direct = advance_ordered_column(old, 0, 80000., segments, (3, 2),
                                        80., .25)
        evolved = stage['store']['records'][0]['history']['regions'][1]['column']
        for name in old:
            np.testing.assert_array_equal(evolved[name], direct['column'][name])
        self.assertGreater(stage['maximum_ordinary_pressure_pa'],
                           direct['top_pressure_pa'])
        self.assertEqual(stage['remapped_markers'], 1)
        np.testing.assert_array_equal(s.channel_region_store['records'][0]
                                      ['history']['regions'][1]['column'][dense_crust.HEAT],
                                      old[dense_crust.HEAT])

    def test_native_refinement_commits_child_histories_and_conserves_inventories(self):
        s = installed_world()
        before = deepcopy(s)
        self.assertTrue(adapt(s, 2.))
        records = s.channel_region_store['records']
        self.assertGreater(len(records), 1)
        self.assertEqual(s.channel_region_store['geometry_revision'],
                         s.material_surface['geometry_revision'])
        tracked = np.array([record['face_id'] for record in records])
        selected = np.isin(s.trace_patch, tracked)
        bound = bind_marker_regions(s.material_surface, records, s.trace_id[selected],
                                    s.trace_patch[selected], s.trace_xyz[selected])
        self.assertEqual(len(bound), 1)
        self.assertEqual(bound[0]['trace_id'], 0)
        for name in (*crust_inventory.FIELDS, *dense_crust.FIELDS):
            np.testing.assert_allclose(before.mass @ before.structure[name],
                                       s.mass @ s.structure[name], rtol=2e-10, atol=1e-6)
        self.assertEqual(before.channel_region_store['records'][0]['face_id'], 1000)
        self.assertEqual(len(before.channel_region_store['records']), 1)
        for name in before.trace_structure:
            np.testing.assert_array_equal(s.trace_structure[name], before.trace_structure[name])

    def test_nonuniform_area_factor_refinement_preserves_reference_mass_and_heat(self):
        s = nonuniform_area_world()
        before = deepcopy(s)
        self.assertTrue(adapt(s, 2.))
        self.assertGreater(len(s.channel_region_store['records']), 1)
        for record in s.channel_region_store['records']:
            index = record['history']['face_index']
            expected = s.material_surface['area_km2'][index] * sum(
                region['fraction'] / region['column']['area_factor'][0]
                for region in record['history']['regions'])
            np.testing.assert_allclose(s.mass[index], expected, rtol=2e-12)
            np.testing.assert_allclose(s.material_surface['reference_area_km2'][index],
                                       expected, rtol=2e-12)
        np.testing.assert_allclose(s.mass.sum(), before.mass.sum(), rtol=2e-12)
        for name in (*crust_inventory.FIELDS, *dense_crust.FIELDS):
            np.testing.assert_allclose(s.mass @ s.structure[name],
                                       before.mass @ before.structure[name], rtol=2e-12, atol=1e-6)

    def test_nonuniform_regions_stage_local_strain_without_spending_material(self):
        s = nonuniform_area_world()
        original = deepcopy(s.channel_region_store['records'][0])
        moved = deepcopy(s.material_surface)
        face = original['history']['face_index']
        vertex = moved['faces'][face, 1]
        moved['vertices'][vertex] = unit(moved['vertices'][vertex] + [.02, .01, .015])
        material_surface.refresh_geometry(moved)
        old_triangle = original['face_triangle']
        new_triangle = moved['vertices'][moved['faces'][face]]
        weights = np.linalg.solve(old_triangle.T, s.trace_xyz[face])
        weights /= weights.sum()
        moved_markers = s.trace_xyz.copy()
        moved_markers[face] = unit(weights @ new_triangle)
        stage = channel_region_native.prepare_motion(s, moved, moved_markers)
        record = stage['store']['records'][0]
        self.assertEqual(record['face_id'], original['face_id'])
        self.assertEqual(stage['remapped_markers'], 1)
        for old_region, old_polygon, new_region, new_polygon in zip(
                original['history']['regions'], original['polygons'],
                record['history']['regions'], record['polygons']):
            old_reference = (mesh_coverage._polygon_area(old_polygon, 6371.)
                             / old_region['column']['area_factor'][0])
            new_reference = (mesh_coverage._polygon_area(new_polygon, 6371.)
                             / new_region['column']['area_factor'][0])
            np.testing.assert_allclose(new_reference, old_reference, rtol=2e-10)
            np.testing.assert_allclose(
                new_reference * new_region['column'][dense_crust.HEAT][0],
                old_reference * old_region['column'][dense_crust.HEAT][0],
                rtol=2e-10)
        np.testing.assert_array_equal(s.channel_region_store['records'][0]['face_triangle'],
                                      old_triangle)
        selected = bind_marker_regions(
            moved, [record], s.trace_id[face:face+1],
            s.trace_patch[face:face+1], moved_markers[face:face+1])[0]
        for name in s.trace_structure:
            np.testing.assert_allclose(stage['trace_structure'][name][face],
                                       selected['column'][name][0], rtol=2e-10)
        np.testing.assert_allclose(s.mass[face],
            moved['area_km2'][face] * sum(
                region['fraction'] / region['column']['area_factor'][0]
                for region in record['history']['regions']), rtol=2e-10)
        np.testing.assert_allclose(
            s.material_surface['area_km2'][face] * s.structure['thickness_km'][face],
            moved['area_km2'][face] * stage['structure']['thickness_km'][face],
            rtol=2e-10)
        for name in (*crust_inventory.FIELDS, *dense_crust.FIELDS):
            np.testing.assert_allclose(s.mass @ s.structure[name],
                                       s.mass @ stage['structure'][name],
                                       rtol=2e-10, atol=1e-6)
        bad_markers = moved_markers.copy()
        bad_markers[face] *= -1.
        with self.assertRaises(ValueError):
            channel_region_native.prepare_motion(s, moved, bad_markers)
        bad_surface = deepcopy(moved)
        bad_surface['area_km2'][face] *= 1.01
        with self.assertRaisesRegex(ValueError, 'face areas disagree'):
            channel_region_native.prepare_motion(s, bad_surface, moved_markers)
        np.testing.assert_array_equal(s.channel_region_store['records'][0]['face_triangle'],
                                      old_triangle)

    def test_rigid_regional_motion_keeps_native_columns(self):
        s = nonuniform_area_world()
        moved = deepcopy(s.material_surface)
        angular_velocity = np.array([.01, -.02, .03])
        material_surface.advect_surface(moved, {0: angular_velocity}, 1.)
        markers = rotate(s.trace_xyz, angular_velocity)
        stage = channel_region_native.prepare_motion(s, moved, markers)
        for name in s.structure:
            np.testing.assert_allclose(stage['structure'][name], s.structure[name],
                                       rtol=2e-10, atol=1e-7)
        self.assertEqual(stage['remapped_markers'], 1)

    def test_native_regional_source_stage_keeps_local_phase_and_trace_history(self):
        s = nonuniform_area_world()
        original = deepcopy(s.channel_region_store['records'][0])
        face = original['history']['face_index']
        face_id = original['face_id']
        loads = [
            dict(region_id='left', lower_top_depth_m=0.,
                 current_upper_base_depth_m=0., upper_column_mass_kg_m2=0.,
                 effective_thermal_depth_km=0.),
            dict(region_id='right', lower_top_depth_m=80000.,
                 current_upper_base_depth_m=35000.,
                 upper_column_mass_kg_m2=2800. * 35000.,
                 effective_thermal_depth_km=80.),
        ]
        expected = advance_region_history(
            original['history'], loads, .25,
            face_area_km2=s.material_surface['area_km2'][face])
        stage = channel_region_native.prepare_source(s, {face_id: loads}, .25)
        self.assertEqual(stage['store']['epoch_myr'], s.t + .25)
        self.assertEqual(stage['remapped_markers'], 1)
        for name in s.structure:
            np.testing.assert_allclose(stage['structure'][name][face],
                                       expected['column'][name][0],
                                       rtol=2e-10, atol=1e-8)
        selected = bind_marker_regions(
            s.material_surface, stage['store']['records'],
            s.trace_id[face:face+1], s.trace_patch[face:face+1],
            s.trace_xyz[face:face+1])[0]
        for name in s.trace_structure:
            np.testing.assert_allclose(stage['trace_structure'][name][face],
                                       selected['column'][name][0], rtol=2e-10)
        np.testing.assert_allclose(stage['returned_volume_km3'],
                                   expected['returned_volume_km3'], rtol=2e-12)
        self.assertGreater(stage['maximum_ordinary_pressure_pa'], 0.)
        self.assertGreater(
            stage['trace_structure'][dense_crust.DENSE][face]
            - s.trace_structure[dense_crust.DENSE][face],
            stage['structure'][dense_crust.DENSE][face]
            - s.structure[dense_crust.DENSE][face])
        old_absolute = sum(
            mesh_coverage._polygon_area(polygon, 6371.)
            * (dense_crust.mass_volume(region['column'])[0]
               + region['column'][crust_inventory.RETURNED][0])
            / region['column']['area_factor'][0]
            for region, polygon in zip(original['history']['regions'],
                                       original['polygons']))
        new_record = stage['store']['records'][0]
        new_absolute = sum(
            mesh_coverage._polygon_area(polygon, 6371.)
            * (dense_crust.mass_volume(region['column'])[0]
               + region['column'][crust_inventory.RETURNED][0])
            / region['column']['area_factor'][0]
            for region, polygon in zip(new_record['history']['regions'],
                                       new_record['polygons']))
        np.testing.assert_allclose(new_absolute, old_absolute, rtol=2e-12)
        def conserved_heat(record):
            return sum(
                mesh_coverage._polygon_area(polygon, 6371.)
                * (region['column'][dense_crust.HEAT][0]
                   + region['column'][dense_crust.HEAT_RETURNED][0]
                   - region['column'][dense_crust.HEAT_BATH][0])
                / region['column']['area_factor'][0]
                for region, polygon in zip(record['history']['regions'],
                                           record['polygons']))
        np.testing.assert_allclose(conserved_heat(new_record),
                                   conserved_heat(original), rtol=2e-12)
        np.testing.assert_array_equal(s.channel_region_store['records'][0]
                                      ['history']['regions'][1]['column'][dense_crust.HEAT],
                                      original['history']['regions'][1]['column'][dense_crust.HEAT])
        np.testing.assert_array_equal(s.trace_structure[dense_crust.HEAT][face],
                                      original['history']['regions'][1]
                                      ['column'][dense_crust.HEAT][0])
        with self.assertRaisesRegex(ValueError, 'loads for every tracked face'):
            channel_region_native.prepare_source(s, {}, .25)
        invalid = deepcopy(loads)
        invalid[1]['lower_top_depth_m'] = 34000.
        with self.assertRaisesRegex(ValueError, 'nonpenetrating'):
            channel_region_native.prepare_source(s, {face_id: invalid}, .25)

    def test_native_zone_remap_preserves_old_material_histories_before_source(self):
        s = nonuniform_area_world()
        original = deepcopy(s.channel_region_store['records'][0])
        face_id = original['face_id']
        a, b, c = original['face_triangle']
        middle = unit(a + c)
        zones = [dict(zone_id='shallow', polygon=np.array([a, b, middle])),
                 dict(zone_id='deep', polygon=np.array([middle, b, c]))]
        stage = channel_region_native.prepare_zone_remap(s, {face_id: zones})
        record = stage['store']['records'][0]
        self.assertEqual(stage['remapped_markers'], 1)
        self.assertGreater(len(record['history']['regions']),
                           len(original['history']['regions']))
        np.testing.assert_allclose(
            sum(region['fraction'] for region in record['history']['regions']),
            1., atol=2e-10)
        for name in s.structure:
            old_absolute = sum(
                mesh_coverage._polygon_area(polygon, 6371.)
                * region['column'][name][0] / region['column']['area_factor'][0]
                for region, polygon in zip(original['history']['regions'],
                                           original['polygons']))
            new_absolute = sum(
                mesh_coverage._polygon_area(polygon, 6371.)
                * region['column'][name][0] / region['column']['area_factor'][0]
                for region, polygon in zip(record['history']['regions'],
                                           record['polygons']))
            np.testing.assert_allclose(new_absolute, old_absolute, rtol=2e-10,
                                       atol=1e-6)
        selected = bind_marker_regions(
            s.material_surface, [record], s.trace_id[:1],
            s.trace_patch[:1], s.trace_xyz[:1])[0]
        for name in s.trace_structure:
            np.testing.assert_allclose(selected['column'][name][0],
                                       s.trace_structure[name][0], rtol=2e-10)
        staged_model = deepcopy(s)
        staged_model.channel_region_store = stage['store']
        loads = [dict(region_id=region['region_id'],
                      lower_top_depth_m=80000. if region['zone_id'] == 'deep' else 0.,
                      current_upper_base_depth_m=35000. if region['zone_id'] == 'deep' else 0.,
                      upper_column_mass_kg_m2=2800. * 35000. if region['zone_id'] == 'deep' else 0.,
                      effective_thermal_depth_km=80. if region['zone_id'] == 'deep' else 0.)
                 for region in record['history']['regions']]
        source = channel_region_native.prepare_source(
            staged_model, {face_id: loads}, .25)
        self.assertGreater(source['maximum_ordinary_pressure_pa'], 0.)
        self.assertEqual(len(source['store']['records'][0]['history']['regions']),
                         len(record['history']['regions']))
        reverse_stage = channel_region_native.prepare_zone_remap(
            s, {face_id: zones[::-1]})
        reverse_model = deepcopy(s)
        reverse_model.channel_region_store = reverse_stage['store']
        reverse_loads = [dict(region_id=region['region_id'],
                              lower_top_depth_m=80000. if region['zone_id'] == 'deep' else 0.,
                              current_upper_base_depth_m=35000. if region['zone_id'] == 'deep' else 0.,
                              upper_column_mass_kg_m2=2800. * 35000. if region['zone_id'] == 'deep' else 0.,
                              effective_thermal_depth_km=80. if region['zone_id'] == 'deep' else 0.)
                         for region in reverse_stage['store']['records'][0]
                         ['history']['regions']]
        reversed_source = channel_region_native.prepare_source(
            reverse_model, {face_id: reverse_loads}, .25)
        for name in s.structure:
            np.testing.assert_allclose(reversed_source['structure'][name],
                                       source['structure'][name], rtol=2e-10,
                                       atol=1e-7)
        np.testing.assert_allclose(reversed_source['returned_volume_km3'],
                                   source['returned_volume_km3'], rtol=2e-10,
                                   atol=1e-6)
        with self.assertRaisesRegex(ValueError, 'do not cover'):
            channel_region_native.prepare_zone_remap(s, {face_id: zones[:1]})
        np.testing.assert_array_equal(s.channel_region_store['records'][0]
                                      ['face_triangle'], original['face_triangle'])

    def test_unbalanced_initial_store_is_rejected_without_mutation(self):
        s = installed_world()
        invalid = deepcopy(s.channel_region_store['records'])
        del s.channel_region_store
        invalid[0]['history']['regions'][0]['column'][dense_crust.HEAT] += 10.
        with self.assertRaisesRegex(ValueError, 'heat inventory'):
            channel_region_native.install(s, invalid)
        self.assertFalse(channel_region_native.enabled(s))

    def test_incorrect_reference_mass_is_rejected_before_install(self):
        s = installed_world()
        record = deepcopy(s.channel_region_store['records'][0])
        del s.channel_region_store
        s.mass[0] *= 1.01
        s.material_surface['reference_area_km2'][0] = s.mass[0]
        with self.assertRaisesRegex(ValueError, 'reference mass disagrees'):
            channel_region_native.install(s, [record])
        self.assertFalse(channel_region_native.enabled(s))

    def test_marker_phase_mismatch_is_rejected_before_install(self):
        s = installed_world()
        record = deepcopy(s.channel_region_store['records'][0])
        del s.channel_region_store
        s.trace_structure['rift_heat_m'][0] += 1.
        with self.assertRaisesRegex(ValueError, 'marker history disagrees'):
            channel_region_native.install(s, [record])
        self.assertFalse(channel_region_native.enabled(s))

    def test_tracked_sibling_coarsening_stays_atomic_until_a_merge_law_exists(self):
        s = installed_world()
        adapt(s, 2.)
        original = s.parcel_patch.copy()
        saved = deepcopy(s.channel_region_store)
        registry = s.material_adaptivity['registries'][-1]
        mapping = adaptive_material.coarsen(
            s.material_surface['vertices'], s.material_surface['faces'],
            s.parcel_plate, s.kind, registry, face_ids=s.parcel_patch,
            face_level=s.material_lineage['level'],
            active=np.zeros(len(s.mass), bool), categories=_categories(s))
        self.assertLess(len(mapping['faces']), len(s.mass))
        report = _apply(s, mapping, refining=False, registry=registry)
        self.assertFalse(report['changed'])
        self.assertIn('regional phase history coarsening', report['reason'])
        np.testing.assert_array_equal(s.parcel_patch, original)
        self.assertEqual(s.channel_region_store['geometry_revision'], saved['geometry_revision'])

    def test_exact_identical_siblings_coarsen_without_erasing_region_state(self):
        s = uniform_world()
        original = deepcopy(s)
        adapt(s, 2.)
        self.assertEqual(len(s.channel_region_store['records']), 4)
        registry = s.material_adaptivity['registries'][-1]
        mapping = adaptive_material.coarsen(
            s.material_surface['vertices'], s.material_surface['faces'],
            s.parcel_plate, s.kind, registry, face_ids=s.parcel_patch,
            face_level=s.material_lineage['level'],
            active=np.zeros(len(s.mass), bool), categories=_categories(s))
        report = _apply(s, mapping, refining=False, registry=registry)
        self.assertTrue(report['changed'])
        self.assertEqual(len(s.channel_region_store['records']), 1)
        region = s.channel_region_store['records'][0]['history']['regions'][0]
        self.assertEqual(region['fraction'], 1.)
        self.assertEqual(region['zone_id'], 'uniform')
        np.testing.assert_array_equal(region['column'][dense_crust.HEAT],
                                      original.channel_region_store['records'][0]
                                      ['history']['regions'][0]['column'][dense_crust.HEAT])
        np.testing.assert_array_equal(s.parcel_patch, original.parcel_patch)
        np.testing.assert_array_equal(s.trace_patch, original.trace_patch)

    def test_coarsening_keeps_distinct_subface_heat_histories(self):
        s = uniform_world()
        a, b, c = s.channel_region_store['records'][0]['face_triangle']
        s.trace_xyz[0] = unit(.22*a + .31*b + .47*c)
        adapt(s, 2.)
        for record in s.channel_region_store['records']:
            if record['face_id'] == int(s.trace_patch[0]):
                continue
            a, b, c = record['face_triangle']
            middle = unit(a+b)
            polygons = [np.array([a, middle, c]), np.array([middle, b, c])]
            areas = np.array([mesh_coverage._polygon_area(polygon, 6371.)
                              for polygon in polygons])
            fractions = areas / areas.sum()
            base = record['history']['regions'][0]['column']
            left, right = deepcopy(base), deepcopy(base)
            left[dense_crust.HEAT] += 10.
            left[dense_crust.HEAT_BATH] += 10.
            offset = 10. * fractions[0] / fractions[1]
            right[dense_crust.HEAT] -= offset
            right[dense_crust.HEAT_BATH] -= offset
            dense_crust.validate(left)
            dense_crust.validate(right)
            record['history']['regions'] = [
                dict(region_id='warm', zone_id='channel', fraction=float(fractions[0]),
                     column=left),
                dict(region_id='cool', zone_id='channel', fraction=float(fractions[1]),
                     column=right)]
            record['polygons'] = polygons
        saved = deepcopy(s.channel_region_store['records'])
        registry = s.material_adaptivity['registries'][-1]
        mapping = adaptive_material.coarsen(
            s.material_surface['vertices'], s.material_surface['faces'],
            s.parcel_plate, s.kind, registry, face_ids=s.parcel_patch,
            face_level=s.material_lineage['level'],
            active=np.zeros(len(s.mass), bool), categories=_categories(s))
        self.assertTrue(_apply(s, mapping, refining=False, registry=registry)['changed'])
        merged = s.channel_region_store['records']
        self.assertEqual(len(merged), 1)
        self.assertEqual(len(merged[0]['history']['regions']), 7)
        self.assertEqual({region['zone_id'] for region in merged[0]['history']['regions']},
                         {'channel', 'uniform'})
        old_heat = sum(mesh_coverage._polygon_area(polygon, 6371.)
                       * region['column'][dense_crust.HEAT][0]
                       / region['column']['area_factor'][0]
                       for record in saved for region, polygon in zip(
                           record['history']['regions'], record['polygons']))
        new_heat = sum(mesh_coverage._polygon_area(polygon, 6371.)
                       * region['column'][dense_crust.HEAT][0]
                       / region['column']['area_factor'][0]
                       for region, polygon in zip(merged[0]['history']['regions'],
                                                  merged[0]['polygons']))
        np.testing.assert_allclose(new_heat, old_heat, rtol=2e-12)
        self.assertEqual(s.trace_patch[0], 1000)

    def test_typed_checkpoint_keeps_the_native_store_and_source_hashes(self):
        import server
        self.assertTrue({'channel_region_native.py', 'channel_marker_binding.py',
                         'channel_region_refinement.py', 'channel_region_geometry.py',
                         'channel_region_history.py', 'entry_channel_column.py',
                         'entry_channel_pressure.py', 'channel_pair_zones.py',
                         'channel_entry_zones.py', 'channel_upper_region_zones.py',
                         'channel_region_native_source.py',
                         'channel_upper_rock_loads.py', 'channel_native_pressure.py',
                         'channel_native_source_preflight.py',
                         'channel_native_vertical_path.py',
                         'channel_upper_basal_reference.py',
                         'channel_native_upper_strip.py',
                         'channel_native_contact_strip.py',
                         'channel_contact_quadrature.py',
                         'channel_contact_line_quadrature.py',
                         'channel_contact_line_union.py',
                         'channel_two_plate_flexure.py',
                         'channel_contact_union_work.py',
                         'channel_horizontal_slab_oracle.py',
                         'channel_region_native_remap.py',
                         'channel_region_native_motion.py',
                         'channel_region_native_zones.py'}
                        <= set(server.AUXILIARY_SOURCES))
        s = installed_world()
        state = SimpleNamespace(t=s.t, config={}, rng=np.random.default_rng(2),
                                structure=s.structure, mass=s.mass,
                                material_surface=s.material_surface,
                                trace_id=s.trace_id, trace_patch=s.trace_patch,
                                trace_xyz=s.trace_xyz,
                                channel_region_store=s.channel_region_store)
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'regional-native.npz'
            checkpoint.write_checkpoint(path, state, dict(config=state.config), {})
            restored, _ = checkpoint.read_checkpoint(path, {}, SimpleNamespace)
        self.assertEqual(restored.channel_region_store['version'], 1)
        self.assertEqual(restored.channel_region_store['epoch_myr'], s.t)
        original = s.channel_region_store['records'][0]
        loaded = restored.channel_region_store['records'][0]
        np.testing.assert_array_equal(loaded['face_triangle'], original['face_triangle'])
        self.assertEqual(loaded['face_id'], original['face_id'])

    def test_ordinary_native_step_rejects_unadvanced_regional_state(self):
        s = Simulation.__new__(Simulation)
        s.channel_region_store = {}
        with self.assertRaisesRegex(ValueError, 'coupled native source law'):
            s.step(.1)
        with self.assertRaisesRegex(ValueError, 'coupled native source law'):
            s._advance_step(.1)
        with self.assertRaisesRegex(ValueError, 'coupled native source law'):
            s._step_once(.1)


if __name__ == '__main__':
    unittest.main()
