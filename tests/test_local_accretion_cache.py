"""Occupancy reuse accelerates the exact accretion rule, not an approximation."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from checkpoint import read_checkpoint, write_checkpoint
from crust_transport import deposit
import local_accretion as local
import structure_engine
from raster_engine import Simulation, _rotate
from tests.test_local_accretion import world


def rasterize_fixture(s):
    s.age = np.zeros(s.n)
    structure_engine.initialize_parcels(s)
    Simulation._rasterize(s)
    return s


class AccretionCacheTests(unittest.TestCase):
    def assert_exact(self, a, b, name='result'):
        if isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b, err_msg=name)
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys(), name)
            for key in a:
                self.assert_exact(a[key], b[key], name+'.'+str(key))
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b), name)
            for index, (left, right) in enumerate(zip(a, b)):
                self.assert_exact(left, right, name+f'[{index}]')
        else:
            self.assertEqual(a, b, name)

    def test_main_raster_occupancy_matches_full_owner_deposit_at_seam_and_poles(self):
        for angle in (0., 90., -90., 180.):
            with self.subTest(angle=angle):
                s = world()
                rotation = np.array([0., np.radians(angle), 0.])
                s.pos = _rotate(s.pos, rotation)
                s.parcel_east = _rotate(s.parcel_east, rotation)
                rasterize_fixture(s)
                self.assertIs(local._cached_occupancy(s), s._owner_occupancy)
                for owner in np.unique(s.parcel_plate):
                    legacy = local._material_components(s, int(owner))
                    cached = local._material_components(s, int(owner), s._owner_occupancy)
                    self.assert_exact(cached, legacy)

    def test_contact_plans_and_actual_transfers_are_exact_without_redeposition(self):
        cached = rasterize_fixture(world())
        legacy = deepcopy(cached)
        completed = False
        with patch.object(local, 'deposit', wraps=deposit) as calls:
            for _ in range(16):
                cached.t += 2.
                legacy.t += 2.
                with patch.object(local, '_cached_occupancy', return_value=None):
                    expected = local.plan_accretions(legacy, 2.)
                previous_calls = calls.call_count
                with patch.object(local, 'occupancy_signature', wraps=local.occupancy_signature) as signature:
                    actual = local.plan_accretions(cached, 2.)
                self.assertEqual(signature.call_count, 1, 'one signature per planning call, not per owner')
                self.assertEqual(calls.call_count, previous_calls, 'current cached geometry must not redeposit')
                self.assert_exact(actual, expected)
                self.assert_exact(cached.local_accretion_contacts, legacy.local_accretion_contacts)
                if actual:
                    self.assertTrue(local.apply_accretion(cached, actual[0]))
                    self.assertTrue(local.apply_accretion(legacy, expected[0]))
                    self.assertIsNone(local._cached_occupancy(cached), 'in-place owner transfer invalidates old occupancy')
                    completed = True
                    break
        self.assertTrue(completed)
        for name, old in vars(legacy).items():
            if isinstance(old, np.ndarray):
                np.testing.assert_array_equal(getattr(cached, name), old, err_msg=name)
        self.assert_exact(cached.events, legacy.events)
        self.assert_exact(cached.structure, legacy.structure)

    def test_every_geometry_input_detects_in_place_changes(self):
        original = rasterize_fixture(world())
        for name in ('pos', 'parcel_east', 'parcel_extent', 'mass', 'parcel_plate', 'xyz', 'cell_area'):
            with self.subTest(name=name):
                s = deepcopy(original)
                values = getattr(s, name)
                if np.issubdtype(values.dtype, np.integer):
                    values.flat[0] += 1
                else:
                    values.flat[0] = np.nextafter(values.flat[0], np.inf)
                self.assertIsNone(local._cached_occupancy(s))
        original.w += 1
        self.assertIsNone(local._cached_occupancy(original))

    def test_stale_mass_forces_original_deposit_and_same_contact_decision(self):
        cached = rasterize_fixture(world())
        cached.mass[cached.parcel_plate == 0] *= .3
        legacy = deepcopy(cached)
        with patch.object(local, '_cached_occupancy', return_value=None):
            expected = local.plan_accretions(legacy, 2.)
        with patch.object(local, 'deposit', wraps=deposit) as calls:
            actual = local.plan_accretions(cached, 2.)
        self.assertGreater(calls.call_count, 0)
        self.assert_exact(actual, expected)
        self.assert_exact(cached.local_accretion_contacts, legacy.local_accretion_contacts)

    def test_fresh_patch_craton_class_and_relief_details_are_never_cached(self):
        s = rasterize_fixture(world())
        signature = s._owner_occupancy_signature
        before = local._material_components(s, 0, s._owner_occupancy)
        self.assertEqual(len(before['components']), 2)
        # Protected group joins two hidden pieces; the occupancy still matches,
        # while the connected material plan must change immediately.
        s.parcel_craton[s.parcel_plate == 0] = 7
        s.parcel_patch += 100000
        s.kind[s.parcel_plate == 0] = 3
        s.relief += 17.
        s.suture[:] = .8
        self.assertEqual(local.occupancy_signature(s), signature)
        cached = local._material_components(s, 0, s._owner_occupancy)
        original = local._material_components(s, 0)
        self.assert_exact(cached, original)
        self.assertEqual(len(cached['components']), 1)
        self.assertTrue(all(row['anchor_patch'] >= 100000 for row in cached['components'].values()))

    def test_noncontiguous_and_empty_geometry_signatures_are_content_based(self):
        s = rasterize_fixture(world())
        signature = local.occupancy_signature(s)
        for name in ('pos', 'parcel_east', 'parcel_extent', 'mass', 'parcel_plate', 'xyz', 'cell_area'):
            old = getattr(s, name)
            backing = np.empty((len(old)*2, *old.shape[1:]), dtype=old.dtype)
            backing[::2] = old
            setattr(s, name, backing[::2])
        self.assertEqual(local.occupancy_signature(s), signature)
        empty = deepcopy(s)
        for name in ('pos', 'parcel_east', 'parcel_extent', 'mass', 'parcel_plate'):
            setattr(empty, name, getattr(empty, name)[:0])
        self.assertIsInstance(local.occupancy_signature(empty), str)
        self.assertNotEqual(local.occupancy_signature(empty), signature)

    def test_missing_cache_or_metadata_retains_original_path(self):
        s = world()
        self.assertIsNone(local._cached_occupancy(s))
        s._owner_occupancy = {0: np.empty(0, np.int32)}
        self.assertIsNone(local._cached_occupancy(s))
        s._owner_occupancy_signature = 'stale'
        self.assertIsNone(local._cached_occupancy(s))

    def test_actual_arc_append_and_in_place_growth_invalidate_cache(self):
        s = Simulation(dict(width=64, height=32, plate_count=4, seed=12))
        self.assertIsNotNone(local._cached_occupancy(s))
        cells = np.flatnonzero(s.crust == 0)[:1]
        s.steps = 1
        count = len(s.mass)
        s._add_arc_crust(cells, np.array([40.]))
        self.assertGreater(len(s.mass), count)
        self.assertIsNone(local._cached_occupancy(s))
        s._rasterize()
        self.assertIsNotNone(local._cached_occupancy(s))
        count = len(s.mass)
        s._add_arc_crust(cells, np.array([10.]))
        self.assertEqual(len(s.mass), count, 'exercise an existing parcel changing mass in place')
        self.assertIsNone(local._cached_occupancy(s))

    def test_short_actual_evolution_snapshots_are_identical_to_legacy_fallback(self):
        cached = Simulation(dict(width=64, height=32, plate_count=4, seed=12))
        legacy = deepcopy(cached)
        for _ in range(3):
            with patch.object(local, '_cached_occupancy', return_value=None):
                legacy.step()
            cached.step()
            self.assert_exact(cached.snapshot(), legacy.snapshot())
            self.assert_exact(cached.local_accretion_contacts, legacy.local_accretion_contacts)

    def test_cache_checkpoint_roundtrip_remains_valid_and_continuation_exact(self):
        s = Simulation(dict(width=64, height=32, plate_count=4, seed=12))
        compatibility = dict(engine_sha256='occupancy-cache-test', auxiliary_sources_sha256={}, numpy_version=np.__version__)
        scratch = Path(__file__).resolve().parents[1]/'tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as folder:
            path = Path(folder)/'checkpoint.npz'
            write_checkpoint(path, s, dict(config=s.config, time_myr=s.t), compatibility)
            loaded, _ = read_checkpoint(path, compatibility, Simulation)
        self.assertIsNotNone(local._cached_occupancy(loaded))
        self.assert_exact(s._owner_occupancy, loaded._owner_occupancy)
        for candidate in (s, loaded):
            candidate.step()
        self.assert_exact(s.snapshot(), loaded.snapshot())


if __name__ == '__main__':
    unittest.main()
