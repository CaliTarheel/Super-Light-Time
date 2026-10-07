"""A pre-contact upper datum stays with its material through local histories."""

from copy import copy, deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

import channel_region_native
import checkpoint
import crustal_structure
import material_surface
from channel_region_history import (
    coalesce_identical_region_histories, remap_region_history)
from channel_region_refinement import split_history_to_children
from channel_upper_basal_reference import (
    REFERENCE, install_unloaded_upper_bases, prepare_unloaded_upper_bases)
from tests.test_channel_entry_zones import native_contact_world
from tests.test_channel_region_native import installed_world
from ridge_geometry import rotate


def precontact_world():
    s = installed_world()
    s.parcel_collision_sheet = np.arange(1, len(s.mass) + 1)
    return s


class UpperBasalReferenceTests(unittest.TestCase):
    def test_precontact_datum_is_local_and_does_not_rebase_after_source_or_split(self):
        s = precontact_world()
        face_id = s.channel_region_store['records'][0]['face_id']
        stage = prepare_unloaded_upper_bases(s, [face_id])
        old = s.channel_region_store['records'][0]['history']['regions']
        stored = stage['store']['records'][0]['history']['regions']
        for source, region in zip(old, stored):
            self.assertNotIn(REFERENCE, source)
            expected = (1000. * source['column']['thickness_km'][0]
                        - crustal_structure.elevation(source['column'])[0])
            self.assertAlmostEqual(region[REFERENCE]['depth_m'], expected)
            self.assertEqual(region[REFERENCE]['epoch_myr'], s.t)
        staged_world = copy(s)
        staged_world.channel_region_store = stage['store']
        loads = [dict(region_id=region['region_id'], lower_top_depth_m=80000.,
                      segments=[dict(kind='mantle', top_depth_m=0.,
                                     base_depth_m=80000.)],
                      covering_sheets_top_to_bottom=(),
                      effective_thermal_depth_km=80.) for region in stored]
        source = channel_region_native.prepare_source(
            staged_world, {face_id: loads}, 1.)
        for before, after in zip(stored,
                                 source['store']['records'][0]['history']['regions']):
            self.assertEqual(after[REFERENCE], before[REFERENCE])
        split = remap_region_history(stage['store']['records'][0]['history'],
            [dict(zone_id='contact', fraction=.5),
             dict(zone_id='free', fraction=.5)],
            np.outer([region['fraction'] for region in stored], [.5, .5]))
        self.assertEqual(len(split['regions']), 4)
        for child in split['regions']:
            self.assertIn(REFERENCE, child)
            self.assertEqual(child[REFERENCE]['epoch_myr'], s.t)
        record = stage['store']['records'][0]
        a, b, c = record['face_triangle']
        middle = a + b
        middle /= np.linalg.norm(middle)
        refined = split_history_to_children(record, [
            dict(face_id=1001, face_index=0, triangle=np.array([a, middle, c])),
            dict(face_id=1002, face_index=1, triangle=np.array([middle, b, c]))])
        self.assertEqual(len(refined['records']), 2)
        self.assertTrue(all(REFERENCE in region for row in refined['records']
                            for region in row['history']['regions']))

    def test_distinct_references_do_not_coalesce_and_checkpoint_round_trips(self):
        s = precontact_world()
        face_id = s.channel_region_store['records'][0]['face_id']
        stage = prepare_unloaded_upper_bases(s, [face_id])
        regions = stage['store']['records'][0]['history']['regions']
        for region in regions:
            region['zone_id'] = 'same-zone'
        regions[1]['column'] = {name: value.copy() for name, value in
                                regions[0]['column'].items()}
        regions[1][REFERENCE]['depth_m'] += 1.
        retained = coalesce_identical_region_histories(
            stage['store']['records'][0]['history'])
        self.assertEqual(retained['result_region_count'], 2)
        state = SimpleNamespace(t=s.t, config={}, rng=np.random.default_rng(2),
                                structure=s.structure, mass=s.mass,
                                material_surface=s.material_surface,
                                trace_id=s.trace_id, trace_patch=s.trace_patch,
                                trace_xyz=s.trace_xyz,
                                channel_region_store=stage['store'])
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'unloaded-base.npz'
            checkpoint.write_checkpoint(path, state, dict(config=state.config), {})
            restored, _ = checkpoint.read_checkpoint(path, {}, SimpleNamespace)
        loaded = restored.channel_region_store['records'][0]['history']['regions']
        self.assertEqual([row[REFERENCE] for row in loaded],
                         [row[REFERENCE] for row in regions])

    def test_rigid_material_motion_keeps_precontact_reference(self):
        s = precontact_world()
        face_id = s.channel_region_store['records'][0]['face_id']
        s.channel_region_store = prepare_unloaded_upper_bases(
            s, [face_id])['store']
        saved = [dict(row[REFERENCE]) for row in
                 s.channel_region_store['records'][0]['history']['regions']]
        moved = deepcopy(s.material_surface)
        angular_velocity = np.array([.01, -.02, .03])
        material_surface.advect_surface(moved, {0: angular_velocity}, 1.)
        markers = rotate(s.trace_xyz, angular_velocity)
        report = channel_region_native.prepare_motion(s, moved, markers)
        after = report['store']['records'][0]['history']['regions']
        self.assertEqual([row[REFERENCE] for row in after], saved)

    def test_current_overlap_and_silent_rebase_fail_closed(self):
        s, _, upper_id = native_contact_world()
        with self.assertRaisesRegex(ValueError, 'precede material overlap'):
            prepare_unloaded_upper_bases(s, [upper_id])
        s = precontact_world()
        face_id = s.channel_region_store['records'][0]['face_id']
        stage = prepare_unloaded_upper_bases(s, [face_id])
        staged_world = copy(s)
        staged_world.channel_region_store = stage['store']
        with self.assertRaisesRegex(ValueError, 'silently rebased'):
            prepare_unloaded_upper_bases(staged_world, [face_id])
        with self.assertRaisesRegex(ValueError, 'distinct tracked'):
            prepare_unloaded_upper_bases(s, [face_id, face_id])

    def test_one_time_install_is_atomic_and_changes_only_the_regional_store(self):
        s = precontact_world()
        face_id = s.channel_region_store['records'][0]['face_id']
        before = {name: value.copy() for name, value in s.structure.items()}
        previous_store = s.channel_region_store
        report = install_unloaded_upper_bases(s, [face_id])
        self.assertIsNot(s.channel_region_store, previous_store)
        self.assertEqual(set(report['reference_depth_m_by_face_region']), {face_id})
        self.assertTrue(all(REFERENCE in row for row in
                            s.channel_region_store['records'][0]['history']['regions']))
        for name in before:
            np.testing.assert_array_equal(s.structure[name], before[name])
        installed_store = s.channel_region_store
        with self.assertRaisesRegex(ValueError, 'silently rebased'):
            install_unloaded_upper_bases(s, [face_id])
        self.assertIs(s.channel_region_store, installed_store)
        contact, _, upper_id = native_contact_world()
        contact_store = contact.channel_region_store
        with self.assertRaisesRegex(ValueError, 'precede material overlap'):
            install_unloaded_upper_bases(contact, [upper_id])
        self.assertIs(contact.channel_region_store, contact_store)


if __name__ == '__main__':
    unittest.main()
