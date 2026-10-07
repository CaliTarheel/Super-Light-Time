"""Saved current-front metadata must not silently accept corrupted geometry."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np

import collision_contacts as contacts
import native_frame_sampling as sampling
from checkpoint import read_checkpoint, write_checkpoint
from orientation import orient_frame
from tests.test_collision_contacts import saved
from tests.test_collision_local_fronts import fixture


class CollisionFrontSchemaTests(unittest.TestCase):
    def test_nine_confirmed_corruptions_are_rejected_by_validator_and_sampler(self):
        frame = saved(fixture())
        mutations = {
            'wrong_parent_id': lambda row: row.update(parent_contact_id=999),
            'impossible_owner': lambda row: row.update(top_owner=999),
            'negative_area': lambda row: row.update(overlap_area_km2=-1.),
            'zero_center': lambda row: row.update(center=[0., 0., 0.]),
            'nonunit_normal': lambda row: row.update(normal=[0., 50., 0.]),
            'negative_radius': lambda row: row.update(footprint_radius_km=-100.),
            'impossible_overlap_index': lambda row: row.update(overlap_indices=[2147483647]),
            'mismatched_geometry_epoch': lambda row: row.update(geometry_epoch_myr=999.),
        }
        for name, change in mutations.items():
            bad = deepcopy(frame); change(bad['collision_contacts'][0]['local_fronts'][0])
            for check in (contacts.validate_frame, sampling.prepare):
                with self.subTest(name=name, check=check.__module__), self.assertRaises(ValueError):
                    check(bad)
        bad = deepcopy(frame); bad['collision_contacts'][0]['local_fronts_version'] = 999
        for check in (contacts.validate_frame, sampling.prepare):
            with self.subTest(name='unsupported_local_version', check=check.__module__), self.assertRaises(ValueError):
                check(bad)

    def test_shape_finiteness_tangency_and_resolved_contract_are_enforced(self):
        frame = saved(fixture())
        mutations = [
            lambda row: row.update(local_fronts_version=None),
            lambda row: row.update(local_fronts_version=True),
            lambda row: row.update(local_fronts={}),
            lambda row: row.update(local_fronts=[None]),
            lambda row: row['local_fronts'][0].update(center=[1., 0.]),
            lambda row: row['local_fronts'][0].update(center=[float('nan'), 0., 0.]),
            lambda row: row['local_fronts'][0].update(normal=row['local_fronts'][0]['center']),
            lambda row: row['local_fronts'][0].update(footprint_radius_km=float('inf')),
            lambda row: row['local_fronts'][0].update(length_km=-1.),
            lambda row: row['local_fronts'][0].update(normal_speed_km_myr=float('nan')),
            lambda row: row['local_fronts'][0].update(direction_resolved=False),
            lambda row: row['local_fronts'][0].update(direction_resolved='yes'),
            lambda row: row['local_fronts'][0].update(front_index=True),
            lambda row: row['local_fronts'][0].update(state='quiet'),
        ]
        for index, change in enumerate(mutations):
            bad = deepcopy(frame); change(bad['collision_contacts'][0])
            with self.subTest(case=index), self.assertRaises(ValueError): contacts.validate_frame(bad)
        for bad_contacts in ({}, [None]):
            bad = deepcopy(frame); bad['collision_contacts'] = bad_contacts
            with self.subTest(contacts=bad_contacts), self.assertRaises(ValueError): contacts.validate_frame(bad)

    def test_area_and_overlap_indices_partition_current_parent_geometry(self):
        frame = saved(fixture())
        mutations = [
            lambda row: row['local_fronts'][0].update(overlap_area_km2=1.),
            lambda row: row['local_fronts'][0].update(overlap_indices=[0, 0]),
            lambda row: row['local_fronts'][0].update(overlap_indices=[-1]),
            lambda row: row['local_fronts'][0].update(overlap_indices=[.5]),
            lambda row: row['local_fronts'][0].update(overlap_indices=[False]),
            lambda row: row['local_fronts'][0].update(overlap_indices=[]),
            lambda row: row['local_fronts'][0].update(overlap_indices=[0]),
            lambda row: row['local_fronts'][1].update(overlap_indices=row['local_fronts'][0]['overlap_indices']),
            lambda row: row['local_fronts'][1].update(front_index=0),
            lambda row: row.update(top_owner=999),
            lambda row: row.update(last_seen_myr=99.),
        ]
        for index, change in enumerate(mutations):
            bad = deepcopy(frame); change(bad['collision_contacts'][0])
            with self.subTest(case=index), self.assertRaises(ValueError): contacts.validate_frame(bad)
        # Validation compares sums with a numerical tolerance, not bit equality.
        frame['collision_contacts'][0]['local_fronts'][0]['overlap_area_km2'] += 1e-7
        contacts.validate_frame(frame)

    def test_generated_refined_rotated_and_checkpoint_records_remain_valid(self):
        for top, under in ((False, False), (True, False), (False, True), (True, True)):
            s = fixture(coarse=True, refine_top=top, refine_under=under)
            s.t = 2.; contacts.refresh(s, 2.)
            frame = saved(s); contacts.validate_frame(frame); sampling.prepare(frame)
            for angles in (dict(yaw=179.), dict(pitch=89.), dict(yaw=35., pitch=-63., roll=19.)):
                turned = orient_frame(frame, angles)
                contacts.validate_frame(turned); sampling.prepare(turned)
            s.config = {}; s.rng = np.random.default_rng(2)
            compatibility = dict(engine_sha256='local-front-schema-test', auxiliary_sources_sha256={}, numpy_version=np.__version__)
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary)/'checkpoint.npz'
                write_checkpoint(path, s, dict(config={}), compatibility)
                restored, _ = read_checkpoint(path, compatibility, SimpleNamespace)
            self.assertEqual(restored.collision_contacts, s.collision_contacts)
            contacts.validate_frame(saved(restored))
            # Cached geometry can be re-used after restoration without stale
            # time metadata or a second suture charge in the same epoch.
            contacts.refresh(restored, 2.)
            self.assertEqual(restored.collision_contacts, s.collision_contacts)
            contacts.validate_frame(saved(restored))

    def test_legacy_absent_version_and_quiet_history_keep_their_interpretation(self):
        frame = saved(fixture()); row = frame['collision_contacts'][0]
        row.pop('local_fronts'); row.pop('local_fronts_version')
        contacts.validate_frame(frame)
        row.update(state='quiet', overlap_area_km2=0., local_fronts=[])
        contacts.validate_frame(frame)
        row['local_fronts_version'] = 1
        frame['collision_diagnostics']['contact_pairs'] = 0
        contacts.validate_frame(frame)
        row['local_fronts'] = [dict(center=[1., 0., 0.])]
        with self.assertRaises(ValueError): contacts.validate_frame(frame)

    def test_unresolved_generated_footprint_requires_zero_direction_components(self):
        from tests.test_collision_polarity import fixture as coincident
        s = coincident(); contacts.refresh(s)
        frame = saved(s); contacts.validate_frame(frame)
        row = frame['collision_contacts'][0]['local_fronts'][0]
        self.assertFalse(row['direction_resolved'])
        for key, value in (('normal', [1., 0., 0.]), ('length_km', 1.), ('normal_speed_km_myr', -1.)):
            bad = deepcopy(frame); bad['collision_contacts'][0]['local_fronts'][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): contacts.validate_frame(bad)


if __name__ == '__main__': unittest.main()
