"""Explicit material-preserving ocean attachments and local initial polarity."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

import checkpoint
import native_engine
import primordial_ocean as ocean
import primordial_subduction as subduction
import slab_memory
from tests.test_primordial_ocean import fitted_fixture, edge_faces


class AttachmentConfigurationTests(unittest.TestCase):
    def test_explicit_unique_assignments_and_one_independent_ocean_required(self):
        entry = dict(region_index=1, continental_plate_uid=3)
        settings = ocean.normalize(dict(enabled=True, continental_attachments=[entry]))
        self.assertEqual(settings['continental_attachments'], [entry])
        entry['region_index'] = 0
        self.assertEqual(settings['continental_attachments'][0]['region_index'], 1)
        for attachments in (None, {}, [True], [dict(region_index=True, continental_plate_uid=3)],
                            [dict(region_index=4, continental_plate_uid=3)],
                            [dict(region_index=0, continental_plate_uid=-1)],
                            [dict(region_index=0, continental_plate_uid=3), dict(region_index=1, continental_plate_uid=3)],
                            [dict(region_index=0, continental_plate_uid=3), dict(region_index=0, continental_plate_uid=4)],
                            [dict(region_index=i, continental_plate_uid=i+1) for i in range(4)]):
            with self.subTest(attachments=attachments), self.assertRaises(ValueError):
                ocean.normalize(dict(enabled=True, continental_attachments=attachments))
        with self.assertRaisesRegex(ValueError, 'enabled'):
            ocean.normalize(dict(continental_attachments=[dict(region_index=0, continental_plate_uid=3)]))

    def test_disconnected_plate_or_nonadjacent_attachment_is_not_silently_accepted(self):
        fitted = fitted_fixture(2)
        partition = ocean.partition_water(fitted, 4)
        state = SimpleNamespace(active=np.array([True, True]), plate_uid=np.array([1, 2]))
        coast = set()
        for a, b in edge_faces(fitted['faces']).values():
            if fitted['face_kind'][a] != fitted['face_kind'][b]:
                coast.add(int(partition['face_labels'][a if fitted['face_kind'][a] == 0 else b]))
        attached = dict(region_index=min(coast), continental_plate_uid=2)
        self.assertEqual(ocean.attachment_slots(state, fitted, partition, [attached]), {min(coast): 1})
        nonadjacent = set(range(4))-coast
        self.assertTrue(nonadjacent)
        with self.assertRaisesRegex(ValueError, 'no shared coast'):
            ocean.attachment_slots(state, fitted, partition,
                [dict(region_index=min(nonadjacent), continental_plate_uid=2)])
        # Add a separated land island to the same continental owner in a
        # different water region. Joining the first region cannot connect it.
        changed = deepcopy(fitted)
        far = int(np.flatnonzero(partition['face_labels'] == min(nonadjacent))[0])
        changed['face_kind'][far] = 1
        changed['face_owner'][far] = 1
        changed_partition = deepcopy(partition)
        changed_partition['face_labels'][far] = -1
        with self.assertRaisesRegex(ValueError, 'disconnected'):
            ocean.attachment_slots(state, changed, changed_partition, [attached])

    def test_opposite_local_polarities_between_same_mixed_plate_pair(self):
        source = dict(owner_a=np.array([0, 0, 0]), owner_b=np.array([1, 1, 1]),
                      kind_a=np.array([0, 1, 0]), kind_b=np.array([1, 0, 0]),
                      length_km=np.array([20., 30., 40.]))
        state = SimpleNamespace(plate_uid=np.array([11, 22, 33]),
            primordial_ocean_carrier_uids=np.array([11, 22, 33]), primordial_connectivity_version=1,
            bp=np.array([0, 0, 0]), bq=np.array([1, 1, 1]), bl=np.array([20., 30., 40.]),
            native_boundary_geometry=dict(contact_index=np.array([0, 1, 2])))
        down, mask = subduction._material_targets(state, source, np.array([2]))
        np.testing.assert_array_equal(down, [0, 1, -1])
        np.testing.assert_array_equal(mask, [True, True, False])
        state.native_boundary_geometry['contact_index'] = np.array([0, 0, 2])
        with self.assertRaisesRegex(ValueError, 'opposite polarities'):
            subduction._material_targets(state, source, np.array([2]))


class MixedPlateStartupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = dict(width=48, height=24, mesh_level=4, coast_geometry_level=2,
            plate_count=4, mechanics_nodes=128, seed=37, physics_profile='reviewed_v1',
            primordial_subduction={'enabled': True}, primordial_ocean={'enabled': True})
        cls.separate = native_engine.Simulation(cls.config)
        cls.mixed_config = deepcopy(cls.config)
        cls.mixed_config['primordial_ocean']['continental_attachments'] = [
            dict(region_index=1, continental_plate_uid=3)]
        cls.mixed = native_engine.Simulation(cls.mixed_config)

    def test_material_age_ridge_geometry_and_volume_are_preserved(self):
        original, mixed = self.separate, self.mixed
        for name in ('mass', 'pos', 'kind', 'parcel_plate', 'parcel_patch', 'parcel_craton',
                     'relief', 'rift_id', 'rift_tangent', 'rift_extension_m', 'age'):
            np.testing.assert_array_equal(getattr(original, name), getattr(mixed, name), err_msg=name)
        for name in original.material_surface:
            if isinstance(original.material_surface[name], np.ndarray):
                np.testing.assert_array_equal(original.material_surface[name], mixed.material_surface[name], err_msg=name)
        for name in original.structure:
            np.testing.assert_array_equal(original.structure[name], mixed.structure[name], err_msg=name)
        self.assertEqual(original.original_mass, mixed.original_mass)
        self.assertEqual(original.original_craton_mass, mixed.original_craton_mass)
        for name in ('start', 'end', 'length_km'):
            np.testing.assert_array_equal(original.primordial_ocean_ridges[name], mixed.primordial_ocean_ridges[name])
        expected = original.support.copy()
        expected[:] = 0.
        for p in np.unique(original.parcel_plate):
            expected[p] = original.support[p]
        for old_uid, new_uid in zip(original.primordial_ocean_carrier_uids, mixed.primordial_ocean_carrier_uids):
            old = int(np.flatnonzero(original.plate_uid == old_uid)[0])
            new = int(np.flatnonzero(mixed.plate_uid == new_uid)[0])
            expected[new] += original.support[old]
        np.testing.assert_allclose(mixed.support, expected, rtol=2e-13, atol=2e-13)
        np.testing.assert_allclose(mixed.support.sum(axis=0), 1., rtol=0., atol=2e-10)
        self.assertEqual(mixed.count, original.count-1)
        self.assertNotIn(3, mixed.initial_ocean_plate_uids)
        self.assertEqual(mixed.primordial_ocean_carrier_uids[1], 3)
        for name in ('ocean_created_km2', 'ocean_consumed_km2', 'rift_events'):
            self.assertEqual(mixed.process_totals[name], 0.)

    def test_shared_coast_is_internal_and_far_coast_pulls_the_continent(self):
        original, mixed = self.separate, self.mixed
        down, coast = subduction.target_edges(mixed)
        np.testing.assert_array_equal(mixed.down[coast], down[coast])
        source = mixed.native_initial_owner_interfaces
        expected = (source['kind_a'] == 0) != (source['kind_b'] == 0)
        self.assertAlmostEqual(float(mixed.bl[coast].sum()), float(source['length_km'][expected].sum()), places=7)
        self.assertLess(float(mixed.bl[coast].sum()), original.primordial_subduction_diagnostics['initial_trench_length_km'])
        slot = int(np.flatnonzero(mixed.plate_uid == 3)[0])
        self.assertTrue(np.any(down[coast] == slot))
        self.assertTrue(any(row['downgoing_plate_uid'] == 3 and row['overriding_plate_uid'] == 4
                            for row in mixed.trench_systems))
        name = mixed.names[slot]
        self.assertEqual(original.plate_balance_diagnostics['driver_torque_n_m']['slab'][name], 0.)
        self.assertGreater(mixed.plate_balance_diagnostics['driver_torque_n_m']['slab'][name], 1e24)
        self.assertTrue(np.all(mixed.primordial_ocean_ridges['owner_uids'][:, 0] !=
                               mixed.primordial_ocean_ridges['owner_uids'][:, 1]))

    def test_coast_ridge_mixed_control_contact_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unresolved.*refine the control mesh'):
            native_engine.Simulation(dict(self.mixed_config, mesh_level=3))

    def test_restart_and_real_step_preserve_inventory_and_move_continental_material(self):
        s = deepcopy(self.mixed)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'checkpoint.npz'
            checkpoint.write_checkpoint(path, s, {'config': s.config}, {})
            restored, _ = checkpoint.read_checkpoint(path, {}, native_engine.Simulation)
        self.assertEqual(restored.config, s.config)
        self.assertEqual(restored.primordial_ocean_diagnostics, s.primordial_ocean_diagnostics)
        np.testing.assert_array_equal(restored.primordial_ocean_carrier_uids, s.primordial_ocean_carrier_uids)
        self.assertEqual(restored.snapshot()['primordial_connectivity_version'], 1)
        slot = int(np.flatnonzero(s.plate_uid == 3)[0])
        selected = s.trace_plate == slot
        marker_ids = s.trace_id[selected].copy()
        before = s.trace_xyz[selected].copy()
        s.step(.02)
        restored.step(.02)
        current = {int(uid): point for uid, point in zip(s.trace_id, s.trace_xyz)}
        after = np.asarray([current[int(uid)] for uid in marker_ids])
        self.assertGreater(float(np.max(np.linalg.norm(after-before, axis=1))), 1e-10)
        for name in ('support', 'plate', 'age', 'pos', 'omega', 'trace_xyz'):
            np.testing.assert_array_equal(getattr(restored, name), getattr(s, name), err_msg=name)
        for row in s.trench_systems:
            slab_memory.validate_row(row, require_mass=True)
        report = slab_memory.snapshot(s)['slab_memory_diagnostics']
        self.assertAlmostEqual((report['retained_area_km2']+report['retired_area_km2'])/
            (report[slab_memory.INITIAL_AREA_FIELD]+report['fed_area_km2']), 1., places=12)


if __name__ == '__main__':
    unittest.main()
