"""Projection cleanup may remove small map specks, never manufacture a cut.

Synthetic native/raster geometries exercise the real connectivity and width
gates. They are numerical boundary checks, not geological calibration.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import checkpoint
import mesh_geometry
import progressive_rifting as rifts


def native_scene():
    mesh = mesh_geometry.icosphere(3)
    state = SimpleNamespace(native_mesh=mesh, cell_area=mesh['area_km2'].copy())
    mask = mesh['xyz'][:, 0] > .3
    side = mask & (mesh['xyz'][:, 2] > 0.)
    return state, mask, side


def interior_speck(state, mask, side):
    neighbors = [set() for _ in mask]
    for a, b in state.native_mesh['edge_faces']:
        neighbors[a].add(b)
        neighbors[b].add(a)
    candidates = [cell for cell in np.flatnonzero(mask & ~side)
                  if all(mask[n] and not side[n] for n in neighbors[cell])]
    return min(candidates, key=lambda cell: np.linalg.norm(
        state.native_mesh['xyz'][cell]-np.array([.8, 0., -.6])))


def raster_scene(width=40, height=24):
    state = SimpleNamespace(w=width, h=height, cell_area=np.ones(width*height))
    mask = np.ones(width*height, bool)
    side = np.arange(width*height) % width < width//2
    return state, mask, side


def distance_field(side):
    # Distinct magnitudes make unwanted changes beyond sign flips observable.
    return np.where(side, 1., -1.)*(1.+np.arange(len(side))/len(side))


def canonical_metadata(value):
    """Compare typed checkpoint trees with exact values and unordered sets.

    Python set iteration order is not preserved by construction from decoded
    members. Only explicit codec set nodes are reordered here; lists, tuples,
    dictionary entries, discrete IDs and all floating values retain their order.
    """
    if isinstance(value, list):
        return [canonical_metadata(item) for item in value]
    if isinstance(value, dict):
        result = {key: canonical_metadata(item) for key, item in value.items()}
        if set(result) == {'t', 'v'} and result['t'] == 'set':
            result['v'] = sorted(result['v'], key=lambda item: json.dumps(item, sort_keys=True))
        return result
    return value


class ProjectionCleanupTests(unittest.TestCase):
    def call_immutable(self, state, mask, distance):
        before_mask, before_distance = mask.copy(), distance.copy()
        before_area = state.cell_area.copy()
        material = dict(mass=np.array([3., 4.]), positions=np.eye(3), owners=np.array([0, 1]))
        state.test_material = deepcopy(material)
        cleaned, report = rifts._consolidate_projection(state, mask, distance)
        np.testing.assert_array_equal(mask, before_mask)
        np.testing.assert_array_equal(distance, before_distance)
        np.testing.assert_array_equal(state.cell_area, before_area)
        for name, value in material.items():
            np.testing.assert_array_equal(state.test_material[name], value, err_msg=name)
        return cleaned, report

    def assert_refused_unchanged(self, state, mask, distance):
        cleaned, report = self.call_immutable(state, mask, distance)
        self.assertEqual(report['status'], 'refused')
        self.assertTrue(report['reason'])
        self.assertEqual(report['reassigned_cells'], 0)
        self.assertEqual(report['reassigned_area_km2'], 0.)
        np.testing.assert_array_equal(cleaned, distance)
        return report

    def test_large_native_lobe_is_refused_without_retagging(self):
        state, mask, _ = native_scene()
        side = mask & (np.abs(state.native_mesh['xyz'][:, 2]) > .3)
        before = rifts.partition_viability(state, mask, side)
        self.assertFalse(before['viable'])
        self.assertAlmostEqual(before['max_daughter_fragment_fraction'], .5)
        self.assertEqual(int(mask.sum()), 456)
        # The unrestricted helper discarded an entire cap, 154 cells/33.89%
        # of parent area, even though it was an equally sized resolved lobe.
        caps = []
        for sign in (-1, 1):
            caps.append(mask & (sign*state.native_mesh['xyz'][:, 2] > .3))
        removed = min(caps, key=lambda cap: state.cell_area[cap].sum())
        self.assertEqual(int(removed.sum()), 154)
        self.assertAlmostEqual(state.cell_area[removed].sum()/state.cell_area[mask].sum(),
                               .33891726113957876)
        self.assert_refused_unchanged(state, mask, distance_field(side))

    def test_native_ocean_speck_changes_only_its_ownership_sign(self):
        state, mask, clean_side = native_scene()
        speck = interior_speck(state, mask, clean_side)
        side = clean_side.copy()
        side[speck] = True
        self.assertTrue(rifts.partition_viability(state, mask, side)['viable'])
        distance = distance_field(side)
        cleaned, report = self.call_immutable(state, mask, distance)
        expected = distance.copy()
        expected[speck] *= -1.
        np.testing.assert_array_equal(cleaned, expected)
        self.assertEqual(report['status'], 'accepted')
        self.assertEqual(report['reassigned_cells'], 1)
        self.assertEqual(report['reassigned_area_km2'], state.cell_area[speck])
        self.assertTrue(report['before']['viable'])
        self.assertTrue(report['after']['viable'])
        self.assertFalse(report['committed'])

    def test_even_a_small_material_owned_speck_cannot_be_reassigned(self):
        state, mask, side = native_scene()
        speck = interior_speck(state, mask, side)
        side[speck] = True
        self.assertTrue(rifts.partition_viability(state, mask, side)['viable'])
        locked = np.zeros(len(mask), bool)
        locked[speck] = True
        distance = distance_field(side)
        before, locked_before = distance.copy(), locked.copy()
        cleaned, report = rifts._consolidate_projection(state, mask, distance, locked=locked)
        np.testing.assert_array_equal(distance, before)
        np.testing.assert_array_equal(cleaned, before)
        np.testing.assert_array_equal(locked, locked_before)
        self.assertEqual(report['status'], 'refused')
        self.assertEqual(report['reassigned_cells'], 0)
        self.assertEqual(report['reassigned_area_km2'], 0.)
        self.assertIn('material', report['reason'])

    def test_native_exact_area_limit_is_allowed_but_any_excess_is_refused(self):
        for excess in (0., 1.):
            with self.subTest(excess=excess):
                state, mask, clean_side = native_scene()
                speck = interior_speck(state, mask, clean_side)
                # Integer positive area weights isolate area, not cell count.
                # Main side=39*N; speck=N gives exactly 1/40, regardless of
                # how many cells the main side contains. N+1 exceeds the gate.
                main_count = int(clean_side.sum())
                state.cell_area[:] = 1.
                state.cell_area[clean_side] = 39.
                state.cell_area[speck] = main_count+excess
                side = clean_side.copy()
                side[speck] = True
                before = rifts.partition_viability(state, mask, side)
                self.assertEqual(before['viable'], excess == 0.)
                if excess == 0.:
                    self.assertEqual(before['max_daughter_fragment_fraction'], .025)
                    cleaned, report = self.call_immutable(state, mask, distance_field(side))
                    self.assertEqual(report['status'], 'accepted')
                    self.assertEqual(report['reassigned_area_km2'], main_count)
                    self.assertLess(cleaned[speck], 0.)
                else:
                    self.assertGreater(before['max_daughter_fragment_fraction'], .025)
                    self.assert_refused_unchanged(state, mask, distance_field(side))

    def test_raster_both_sides_use_original_masks_and_flip_simultaneously(self):
        state, mask, side = raster_scene()
        center = 12*state.w+10
        ring = np.array([center+dy*state.w+dx for dy in (-1, 0, 1)
                         for dx in (-1, 0, 1) if dx or dy])
        # Negative eight-cell island in the positive mainland, with a positive
        # one-cell speck inside it. Both original fragment fractions are small.
        # Sequential cleanup would flip the central cell twice; simultaneous
        # cleanup must flip it once, using the original component assignments.
        side[ring] = False
        self.assertTrue(rifts.partition_viability(state, mask, side)['viable'])
        distance = distance_field(side)
        cleaned, report = self.call_immutable(state, mask, distance)
        expected = distance.copy()
        changed = np.r_[ring, center]
        expected[changed] *= -1.
        np.testing.assert_array_equal(cleaned, expected)
        self.assertEqual(report['status'], 'accepted')
        self.assertEqual(report['reassigned_cells'], 9)
        self.assertEqual(report['reassigned_area_km2'], 9.)
        self.assertTrue(report['after']['viable'])

    def test_cleanup_cannot_manufacture_a_missing_daughter_interior(self):
        state, mask, _ = raster_scene()
        center = 12*state.w+10
        ring = np.array([center+dy*state.w+dx for dy in (-1, 0, 1)
                         for dx in (-1, 0, 1) if dx or dy])
        side = np.zeros(len(mask), bool)
        side[ring] = True
        before = rifts.partition_viability(state, mask, side)
        self.assertFalse(before['viable'])
        self.assertEqual(before['reason'], 'daughter has no resolved interior width')
        self.assertLess(before['max_daughter_fragment_fraction'], .025)
        manufactured = side.copy()
        manufactured[center] = True
        self.assertTrue(rifts.partition_viability(state, mask, manufactured)['viable'])
        self.assert_refused_unchanged(state, mask, distance_field(side))

    def test_existing_undivided_island_and_outside_cells_are_preserved(self):
        state, mask, side = native_scene()
        speck = interior_speck(state, mask, side)
        side[speck] = True
        island = state.native_mesh['xyz'][:, 0] < -.85
        mask |= island
        side |= island
        distance = distance_field(side)
        cleaned, report = self.call_immutable(state, mask, distance)
        self.assertEqual(report['status'], 'accepted')
        self.assertEqual(report['reassigned_cells'], 1)
        np.testing.assert_array_equal(cleaned[island | ~mask], distance[island | ~mask])
        self.assertLess(cleaned[speck], 0.)

    def test_large_other_component_cannot_dilute_a_local_fragment_violation(self):
        state, mask, side = native_scene()
        speck = interior_speck(state, mask, side)
        main_count = int(side.sum())
        state.cell_area[:] = 1.
        state.cell_area[side] = 39.
        state.cell_area[speck] = main_count+1.
        side[speck] = True
        island = state.native_mesh['xyz'][:, 0] < -.85
        mask |= island
        side |= island
        state.cell_area[island] = 1e12
        # Tiny as a percentage of total parent area, but over 2.5% of the
        # affected daughter's original area in the newly divided component.
        self.assertLess(state.cell_area[speck]/state.cell_area[mask].sum(), 1e-8)
        self.assert_refused_unchanged(state, mask, distance_field(side))

    def test_clean_partition_is_unchanged(self):
        state, mask, side = native_scene()
        distance = distance_field(side)
        cleaned, report = self.call_immutable(state, mask, distance)
        np.testing.assert_array_equal(cleaned, distance)
        self.assertEqual(report['status'], 'unchanged')
        self.assertEqual(report['reassigned_cells'], 0)
        self.assertEqual(report['reassigned_area_km2'], 0.)

    def test_distance_dtypes_cannot_silently_drop_a_zero_sign_change(self):
        state, mask, side = native_scene()
        speck = interior_speck(state, mask, side)
        side[speck] = True
        for dtype in (np.int64, np.uint8, np.bool_):
            with self.subTest(dtype=dtype):
                # A zero-valued negative sign cannot become a positive 1e-8
                # distance in an integer buffer. Refuse instead of silently
                # counting a cleanup which left an integer zero unchanged.
                distance = side.astype(dtype)
                report = self.assert_refused_unchanged(state, mask, distance)
                self.assertIsNone(report['before'])
        # A floating zero on the negative side may be a genuine isolated label.
        # Its positive replacement must remain representable in the input
        # dtype; 1e-8 alone underflows to zero in float16.
        for dtype in (np.float16, np.float32, np.float64):
            for zero in (0., -0.):
                with self.subTest(dtype=dtype, zero=zero):
                    state, mask, side = native_scene()
                    speck = interior_speck(state, mask, mask & ~side)
                    distance = np.where(side, 1., -1.).astype(dtype)
                    distance[speck] = zero
                    before_bytes = distance.tobytes()
                    cleaned, report = self.call_immutable(state, mask, distance)
                    self.assertEqual(report['status'], 'accepted')
                    self.assertEqual(report['reassigned_cells'], 1)
                    self.assertEqual(cleaned.dtype, distance.dtype)
                    self.assertGreater(cleaned[speck], 0.)
                    self.assertEqual(distance.tobytes(), before_bytes)

    def test_postcheck_failure_rolls_back_all_proposed_retagging(self):
        state, mask, side = native_scene()
        speck = interior_speck(state, mask, side)
        side[speck] = True
        distance = distance_field(side)
        original = rifts.partition_viability
        calls = []

        def gate(world, parent, selected):
            calls.append(selected.copy())
            result = original(world, parent, selected)
            if len(calls) == 2:
                result.update(viable=False, reason='deliberate postcheck rejection')
            return result

        with patch.object(rifts, 'partition_viability', side_effect=gate):
            report = self.assert_refused_unchanged(state, mask, distance)
        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[0][speck])
        self.assertFalse(calls[1][speck])
        self.assertFalse(report['after']['viable'])

    def test_actual_rift_commit_persists_projection_report_in_snapshot(self):
        from tests.test_progressive_rifting import loaded_continent, load_step
        state = loaded_continent()
        material = {name: getattr(state, name).copy()
                    for name in ('mass', 'pos', 'kind', 'parcel_patch', 'trace_xyz', 'trace_id')}
        for _ in range(150):
            load_step(state)
            if rifts.commit(state):
                break
        else:
            self.fail('existing resolved loading fixture must still reach breakthrough')
        for name, value in material.items():
            np.testing.assert_array_equal(getattr(state, name), value, err_msg=name)
        row = next(row for row in state.rift_systems if row['phase'] == 'broken_through')
        report = row['projection_cleanup']
        self.assertIn(report['status'], ('accepted', 'unchanged'))
        self.assertTrue(report['committed'])
        self.assertEqual(report, state.rift_mechanics['projection_cleanup'])
        saved = rifts.snapshot(state)
        json.dumps(saved, allow_nan=False)
        saved_row = next(r for r in saved['rift_systems'] if r['id'] == row['id'])
        self.assertEqual(saved_row['projection_cleanup'], report)
        self.assertEqual(saved['rift_mechanics']['projection_cleanup'], report)
        # Public diagnostics are independent copies, not mutable references to
        # the candidate or the material/rift history kept by the simulation.
        saved_row['projection_cleanup']['reassigned_cells'] = -99
        saved['rift_mechanics']['projection_cleanup']['committed'] = False
        self.assertGreaterEqual(report['reassigned_cells'], 0)
        self.assertTrue(state.rift_mechanics['projection_cleanup']['committed'])

        # Exercise the actual disk codec on the complete fresh fixture state,
        # including its RNG, material arrays, rift bonds and report metadata.
        original_arrays = {}
        original_metadata = checkpoint._encode(vars(state), original_arrays)
        manifest = dict(config=deepcopy(state.config), time_myr=float(state.t),
                        diagnostic_only=True)
        compatibility = dict(engine_sha256='projection-cleanup-fixture',
                             auxiliary_sources_sha256={}, numpy_version=np.__version__)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'checkpoint.npz'
            checkpoint.write_checkpoint(path, state, manifest, compatibility)
            with patch.object(type(state), '__init__', side_effect=AssertionError('No simulation restart')):
                restored, restored_manifest = checkpoint.read_checkpoint(path, compatibility, type(state))
        restored_arrays = {}
        restored_metadata = checkpoint._encode(vars(restored), restored_arrays)
        self.assertEqual(restored_manifest, manifest)
        self.assertEqual(
            json.dumps(canonical_metadata(restored_metadata), allow_nan=False, separators=(',', ':')),
            json.dumps(canonical_metadata(original_metadata), allow_nan=False, separators=(',', ':')))
        self.assertEqual(set(restored_arrays), set(original_arrays))
        for name, original in original_arrays.items():
            replayed = restored_arrays[name]
            self.assertEqual(replayed.dtype, original.dtype, name)
            self.assertEqual(replayed.shape, original.shape, name)
            self.assertEqual(replayed.tobytes(order='C'), original.tobytes(order='C'), name)
        restored_row = next(r for r in restored.rift_systems if r['id'] == row['id'])
        self.assertEqual(restored_row['projection_cleanup'], report)
        self.assertEqual(restored.rift_mechanics['projection_cleanup'], report)


if __name__ == '__main__':
    unittest.main()
