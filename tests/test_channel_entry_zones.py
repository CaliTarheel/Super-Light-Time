"""Native identity and contact checks for read-only entry-zone staging."""

from copy import deepcopy
import unittest

import numpy as np

import channel_entry_zones
import channel_region_native
import collision_contacts
import collision_surface
import crust_inventory
import dense_crust
import entry_regions
import material_surface
import mesh_coverage
import structure_engine
from channel_region_history import initialize_region_history
from channel_region_geometry import spherical_partition_intersections
from ridge_geometry import rotate
from tests.test_channel_region_native import uniform_world
from tests.test_collision_contacts import fixture as contact_fixture
from tests.test_entry_stack_work_oracle import FACES, NORMAL, geometry


def paired(*, overlap=True):
    vertices = geometry()
    if not overlap:
        vertices[3:] = rotate(vertices[3:], [0., 0., .3])
    surface = dict(vertices=vertices, faces=FACES.copy(),
                   face_id=np.array([100, 200]), radius_km=6371.)
    sheets = np.array([1, 2])
    owners = np.array([0, 1])
    plate_uids = np.array([11, 22])
    overlap_area = float(mesh_coverage.material_overlaps(
        vertices, FACES, sheets)['area_km2'].sum())
    contacts = [dict(top_sheet=2, under_sheet=1, state='active',
                     last_seen_myr=2., overlap_area_km2=overlap_area)]
    spec = dict(face_ids=np.array([100]), hinge_normals=NORMAL.copy(),
                finite_midpoints=np.zeros((1, 3)),
                finite_half_lengths_km=np.array([np.inf]),
                overriding_plate_uids=np.array([22]))
    return surface, sheets, owners, plate_uids, contacts, spec


def stacked():
    surface, sheets, owners, uids, _, spec = paired()
    upper = surface['vertices'][3:]
    centre = upper.sum(axis=0)
    centre /= np.linalg.norm(centre)
    top = .7 * upper + .3 * centre
    top /= np.linalg.norm(top, axis=1)[:, None]
    surface = dict(surface, vertices=np.vstack((surface['vertices'], top)),
                   faces=np.vstack((surface['faces'], [[6, 7, 8]])),
                   face_id=np.array([100, 200, 300]))
    sheets = np.array([1, 2, 3])
    owners = np.array([0, 1, 2])
    uids = np.array([11, 22, 33])
    areas = mesh_coverage.material_overlaps(
        surface['vertices'], surface['faces'], sheets)
    totals = {}
    for a, b, area in zip(areas['first'], areas['second'], areas['area_km2']):
        top_sheet, lower_sheet = sorted((int(sheets[a]), int(sheets[b])), reverse=True)
        key = top_sheet, lower_sheet
        totals[key] = totals.get(key, 0.) + float(area)
    contacts = [dict(top_sheet=top_sheet, under_sheet=lower_sheet,
                     state='active', last_seen_myr=2., overlap_area_km2=area)
                for (top_sheet, lower_sheet), area in totals.items()]
    return surface, sheets, owners, uids, contacts, spec


def upper_record(setup, index=1):
    surface = setup[0]
    face_id = int(surface['face_id'][index])
    triangle = surface['vertices'][surface['faces'][index]]
    a, b, c = triangle
    middle = a + b
    middle /= np.linalg.norm(middle)
    polygons = [np.array([a, middle, c]), np.array([middle, b, c])]
    fractions = spherical_partition_intersections(
        triangle, polygons, polygons)['old_fractions']
    return dict(face_id=face_id, face_triangle=triangle.copy(), polygons=polygons,
                history=dict(face_id=face_id, regions=[
                    dict(region_id='cool', fraction=float(fractions[0])),
                    dict(region_id='warm', fraction=float(fractions[1]))]))


def native_contact_world():
    """Synthetic active contact for read-only source preflights only."""
    s = contact_fixture(separated=True)
    s.relief[s.parcel_plate == 1] = 1200.
    structure_engine.initialize_parcels(s)
    crust_inventory.initialize(s.structure)
    dense_crust.initialize(s.structure, 1000.)
    s.trace_id = np.empty(0, np.int64)
    s.trace_structure = {name: np.empty(0, float) for name in s.structure}
    s.active = np.ones(2, bool)
    s.plate_balance_version = s.plate_resistance_version = s.material_mechanics_version = 1
    s.trench_systems = [dict(id=1, downgoing_plate_uid=101,
                             overriding_plate_uid=202)]
    upper_vertices = s.material_surface['vertex_owner'] == 1
    s.material_surface['vertices'][upper_vertices] = rotate(
        s.material_surface['vertices'][upper_vertices], [0., 0., -.08])
    material_surface.refresh_geometry(s.material_surface)
    s.pos = material_surface.face_centres(s.material_surface)
    collision_contacts.refresh(s)
    pairs = collision_surface.ordered_overlap_pairs(
        s.material_surface, s.parcel_collision_sheet,
        collision_surface.descendants(s.collision_contacts))
    selected = next((int(upper), int(lower)) for upper, lower in
                    zip(pairs['upper_face'], pairs['lower_face'])
                    if s.parcel_plate[upper] == 1 and s.parcel_plate[lower] == 0)
    upper, lower = selected
    lower_id = int(s.parcel_patch[lower])
    upper_id = int(s.parcel_patch[upper])
    s.parcel_entry_region = np.zeros(len(s.mass), np.int64)
    s.parcel_entry_region[lower] = 1
    hinge = s.pos[lower].copy()
    s.continental_entry_regions = dict(version=1, epoch_myr=float(s.t),
        face_ids=s.parcel_patch.copy(), regions=[dict(id=1, source_trench_id=1,
        overriding_plate_uid=202, hinge_normal=hinge, dip_degrees=50.,
        created_myr=float(s.t))])
    lower_triangle = s.material_surface['vertices'][s.material_surface['faces'][lower]]
    upper_triangle = s.material_surface['vertices'][s.material_surface['faces'][upper]]
    a, b, c = upper_triangle
    middle = a + b
    middle /= np.linalg.norm(middle)
    upper_polygons = [np.array([a, middle, c]), np.array([middle, b, c])]
    fractions = spherical_partition_intersections(
        upper_triangle, upper_polygons, upper_polygons)['old_fractions']
    records = [dict(face_id=lower_id, face_triangle=lower_triangle.copy(),
        polygons=[lower_triangle.copy()], history=initialize_region_history(
            s.structure, lower, [dict(region_id='incoming', fraction=1.)],
            face_id=lower_id)),
        dict(face_id=upper_id, face_triangle=upper_triangle.copy(),
        polygons=upper_polygons, history=initialize_region_history(
            s.structure, upper, [dict(region_id='cool', fraction=float(fractions[0])),
                                 dict(region_id='warm', fraction=float(fractions[1]))],
            face_id=upper_id))]
    warm = records[1]['history']['regions'][1]['column']
    warm['area_factor'] *= 1.2
    warm['thickness_km'] /= 1.2
    s.mass[upper] = s.material_surface['area_km2'][upper] * sum(
        region['fraction'] / region['column']['area_factor'][0]
        for region in records[1]['history']['regions'])
    s.material_surface['reference_area_km2'][upper] = s.mass[upper]
    upper_mean = channel_region_native._project(records[1], set(s.structure))
    for name in s.structure:
        s.structure[name][upper] = upper_mean[name][0]
    channel_region_native.install(s, records)
    return s, lower_id, upper_id


class NativeEntryZoneTests(unittest.TestCase):
    def zones(self, setup, tracked=(100, 200)):
        return channel_entry_zones.partition_native_entry_zones(
            *setup, tracked, epoch_myr=2.)

    def test_real_pair_identity_and_untracked_upper_column(self):
        result = self.zones(paired())
        lower = result['entry_faces'][100]
        self.assertGreater(lower['contact_area_km2'], 0.)
        self.assertAlmostEqual(sum(row['fraction'] for row in lower['zones']), 1., places=9)
        self.assertEqual({row['adjacent_upper_face_id'] for row in lower['zones']
                          if row['kind'] == 'entered-contact'}, {200})
        self.assertEqual(len(result['zones_by_face'][200]), 1)
        self.assertEqual(result['zones_by_face'][200][0]['zone_id'], 'unentered-0')
        self.assertEqual(set(result['zones_by_face'][100][0]), {'zone_id', 'polygon'})
        # A source store may track only the incoming face; the upper geometry
        # still comes from its native face, not from a supplied free triangle.
        self.assertEqual(set(self.zones(paired(), tracked=(100,))['zones_by_face']), {100})

    def test_free_entry_without_upper_face_remains_a_complete_partition(self):
        result = self.zones(paired(overlap=False))
        lower = result['entry_faces'][100]
        self.assertEqual(lower['contact_area_km2'], 0.)
        self.assertIn('entered-free', {zone['kind'] for zone in lower['zones']})
        self.assertAlmostEqual(sum(zone['fraction'] for zone in lower['zones']),
                               1., places=9)

    def test_three_sheet_contact_keeps_adjacent_overrider_and_checks_upper_contact(self):
        setup = stacked()
        result = self.zones(setup, tracked=(100,))
        both = [zone for zone in result['entry_faces'][100]['zones']
                if zone['kind'] == 'entered-contact'
                and zone['covering_sheets_top_to_bottom'] == (3, 2)]
        self.assertTrue(both)
        self.assertTrue(all(zone['adjacent_upper_face_id'] == 200 for zone in both))
        stale = list(deepcopy(setup))
        for row in stale[4]:
            if (row['top_sheet'], row['under_sheet']) == (3, 2):
                row['last_seen_myr'] = 1.
        with self.assertRaisesRegex(ValueError, 'stale upper-sheet contact'):
            self.zones(stale, tracked=(100,))

    def test_upper_local_histories_split_lower_contact_without_area_change(self):
        setup = paired()
        base = self.zones(setup, tracked=(100,))
        regional = channel_entry_zones.partition_native_entry_zones(
            *setup, (100,), epoch_myr=2.,
            upper_records_by_face_id={200: upper_record(setup)})
        original = base['entry_faces'][100]
        split = regional['entry_faces'][100]
        self.assertGreater(len(split['zones']), len(original['zones']))
        self.assertAlmostEqual(split['contact_area_km2'], original['contact_area_km2'])
        self.assertAlmostEqual(sum(zone['fraction'] for zone in split['zones']), 1., places=9)
        touched = [zone for zone in split['zones']
                   if zone['kind'] == 'entered-contact']
        self.assertEqual({zone['upper_region_bindings_top_to_bottom'][0][1]
                          for zone in touched}, {'cool', 'warm'})
        self.assertEqual(len({zone['zone_id'] for zone in split['zones']}),
                         len(split['zones']))
        self.assertEqual(len(regional['zones_by_face'][100]), len(split['zones']))
        bad = upper_record(setup)
        bad['history']['regions'][0]['fraction'] += .01
        with self.assertRaisesRegex(ValueError, 'fractions are stale'):
            channel_entry_zones.partition_native_entry_zones(
                *setup, (100,), epoch_myr=2., upper_records_by_face_id={200: bad})

    def test_two_upper_local_histories_remain_in_top_to_bottom_order(self):
        setup = stacked()
        result = channel_entry_zones.partition_native_entry_zones(
            *setup, (100,), epoch_myr=2.,
            upper_records_by_face_id={200: upper_record(setup),
                                      300: upper_record(setup, 2)})
        both = [zone for zone in result['entry_faces'][100]['zones']
                if zone['kind'] == 'entered-contact'
                and zone['covering_sheets_top_to_bottom'] == (3, 2)]
        self.assertTrue(both)
        self.assertEqual({tuple(face_id for face_id, _ in
                          zone['upper_region_bindings_top_to_bottom'])
                          for zone in both}, {(300, 200)})
        self.assertAlmostEqual(sum(zone['fraction'] for zone in
                                   result['entry_faces'][100]['zones']), 1., places=9)

    def test_current_native_contact_remaps_lower_and_upper_histories_together(self):
        s, lower_id, upper_id = native_contact_world()
        stage = channel_entry_zones.prepare_native_entry_zone_remap(s)
        self.assertEqual({record['face_id'] for record in stage['store']['records']},
                         {lower_id, upper_id})
        self.assertGreater(stage['geometry']['entry_faces'][lower_id]['contact_area_km2'], 0.)
        self.assertTrue(any(zone['upper_region_bindings_top_to_bottom']
                            for zone in stage['geometry']['entry_faces'][lower_id]['zones']))
        covered = [zone for zone in stage['upper_rock']['by_lower_face'][lower_id]
                   if zone['kind'] == 'entered-contact']
        self.assertTrue(covered)
        self.assertTrue(all(layer['rock_mass_kg_m2'] > 0.
                            for zone in covered
                            for layer in zone['rock_layers_top_to_bottom']))
        self.assertEqual({layer['face_id'] for zone in covered
                          for layer in zone['rock_layers_top_to_bottom']
                          if layer['region_id'] is not None}, {upper_id})
        by_region = {layer['region_id']: layer['rock_mass_kg_m2']
                     for zone in covered for layer in zone['rock_layers_top_to_bottom']
                     if layer['region_id'] is not None}
        self.assertGreater(by_region['cool'], by_region['warm'])
        self.assertEqual(len(s.channel_region_store['records'][0]['history']['regions']), 1)

    def test_stale_contact_and_wrong_upper_owner_fail_closed(self):
        setup = list(paired())
        setup[4] = [dict(setup[4][0], last_seen_myr=1.)]
        with self.assertRaisesRegex(ValueError, 'not current'):
            self.zones(setup)
        setup = list(paired())
        setup[4] = [dict(setup[4][0], overlap_area_km2=1.)]
        with self.assertRaisesRegex(ValueError, 'geometrically matched'):
            self.zones(setup)
        setup = list(paired())
        setup[5] = dict(setup[5], overriding_plate_uids=np.array([11]))
        with self.assertRaisesRegex(ValueError, 'overriding-owner handoff'):
            self.zones(setup)
        setup = list(paired())
        setup[4] = [dict(setup[4][0], top_sheet=1, under_sheet=2)]
        with self.assertRaisesRegex(ValueError, 'polarity handoff'):
            self.zones(setup)
        with self.assertRaisesRegex(ValueError, 'tracked hinge'):
            self.zones(paired(), tracked=(200,))

    def test_full_native_remap_stages_free_entry_without_changing_state(self):
        s = uniform_world()
        s.active = np.ones(2, bool)
        s.plate_balance_version = s.plate_resistance_version = s.material_mechanics_version = 1
        s.config['deforming_regions'] = 1
        s.trench_systems = [dict(id=1, downgoing_plate_uid=31,
                                 overriding_plate_uid=42)]
        s.parcel_collision_sheet = np.ones(len(s.mass), np.int64)
        s.collision_contacts = []
        face_id = int(s.parcel_patch[0])
        triangle = s.material_surface['vertices'][s.material_surface['faces'][0]]
        centre = triangle.sum(axis=0)
        centre /= np.linalg.norm(centre)
        normal = np.cross(centre, [0., 0., 1.]) + .01 * centre
        normal /= np.linalg.norm(normal)
        entry_regions.initialize(s, [dict(face_ids=np.array([face_id]),
            trench_id=1, overriding_plate_uid=42, hinge_normal=normal,
            dip_degrees=50.)])
        before = deepcopy(s.channel_region_store)
        stage = channel_entry_zones.prepare_native_entry_zone_remap(s)
        self.assertEqual(stage['remapped_markers'], 1)
        self.assertEqual(stage['store']['records'][0]['face_id'], face_id)
        self.assertAlmostEqual(sum(row['fraction'] for row in
            stage['store']['records'][0]['history']['regions']), 1., places=9)
        self.assertIn('entered-free', {row['kind'] for row in
            stage['geometry']['entry_faces'][face_id]['zones']})
        np.testing.assert_array_equal(s.channel_region_store['records'][0]['face_triangle'],
                                      before['records'][0]['face_triangle'])
        self.assertEqual(len(s.channel_region_store['records'][0]['history']['regions']), 1)


if __name__ == '__main__':
    unittest.main()
