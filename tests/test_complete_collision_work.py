"""Force assembly must see boundary work absent from history bookkeeping."""
from types import SimpleNamespace
import unittest

import numpy as np

import gravitational_relaxation as gravity
import material_surface
import mesh_coverage
import plate_balance
from benchmarks.collision_architecture import runner
from tests.test_collision_architecture import overlapping_fixture, orientation
from tests.test_gravitational_relaxation import fixture, unit


class CompleteCollisionWorkTests(unittest.TestCase):
    def test_malformed_cached_inventory_is_rejected_before_reuse(self):
        points, faces, height, reference, sheets = fixture(overlap=True)
        good = mesh_coverage.material_overlaps(points, faces, sheets, include_touching=True)
        invalid = []
        for field, values in (('area_km2', [np.nan]), ('area_km2', [-1.]),
                              ('first', [.5]), ('second', [len(faces)])):
            damaged = dict(good)
            damaged[field] = np.asarray(values)
            invalid.append(damaged)
        duplicate = {key: np.repeat(good[key], 2) for key in ('first', 'second', 'area_km2')}
        duplicate['complete_force_stencil'] = True
        invalid.append(duplicate)
        for index, cache in enumerate(invalid):
            with self.subTest(index=index), self.assertRaises(ValueError):
                gravity.energy_gradient(points, faces, height, reference, sheets, overlap=cache)

    def test_stale_cache_cannot_claim_a_complete_force_stencil(self):
        separated, faces, height, reference, sheets = fixture(overlap=False)
        cache = mesh_coverage.material_overlaps(separated, faces, sheets, include_touching=True)
        points, _, _, _, _ = fixture(overlap=True)
        volumes = material_surface.spherical_face_areas(points, faces)*height
        with self.assertRaisesRegex(ValueError, 'Cached force stencil is incomplete'):
            gravity.energy_gradient(points, faces, height, reference, sheets, overlap=cache)
        with self.assertRaisesRegex(ValueError, 'Cached force stencil is incomplete'):
            gravity.reference_energy(points, faces, volumes, reference, sheets, overlap=cache)

    def test_plate_gpe_keeps_zero_area_birth_when_contact_ledger_is_empty(self):
        first = unit([[-.1, 0., 1.], [0., -.2, 1.], [.1, 0., 1.]])
        second = unit([[-.1, 0., 1.], [.1, 0., 1.], [0., .2, 1.]])
        points, faces = np.concatenate((first, second)), np.arange(6).reshape(2, 3)
        sheets, height = np.array([1, 2]), np.array([35., 35.])
        mesh = dict(vertices=points, faces=faces, vertex_owner=np.repeat([0, 1], 3),
                    area_km2=material_surface.spherical_face_areas(points, faces), radius_km=6371.)
        ledger = mesh_coverage.material_overlaps(points, faces, sheets)
        self.assertEqual(len(ledger['first']), 0)
        world = SimpleNamespace(material_surface=mesh, parcel_collision_sheet=sheets,
            structure=dict(thickness_km=height, reference_thickness_km=height), _collision_overlap=ledger)
        model = plate_balance.Balance.__new__(plate_balance.Balance)
        model.s, model.size, model.slot, model.notes = world, 6, {0: 0, 1: 1}, {}
        torque = model._collision_torque().reshape(2, 3)
        self.assertGreater(np.linalg.norm(torque), 0.)
        np.testing.assert_allclose(torque.sum(axis=0), 0., atol=1e-10*np.linalg.norm(torque))
        # An arbitrary old contact inventory is not manufactured by a force read.
        self.assertIs(world._collision_overlap, ledger)
        self.assertEqual(len(ledger['first']), 0)

    def test_legacy_connected_patch_weld_keeps_complete_rotation_work(self):
        q = orientation()
        results = []
        for world, basis in ((overlapping_fixture(), np.eye(3)),
                             (overlapping_fixture(rotation=q), q)):
            world.suture_weld_coordinate_version = 0
            model, _, _, _ = runner.assemble(world)
            element = next(item for item in model.elements if item['kind'] == 'suture_weld')
            results.append(element['rows'][0].reshape(-1, 2, 3)@basis)
        np.testing.assert_allclose(results[1], results[0], rtol=2e-10, atol=1e-15)


if __name__ == '__main__':
    unittest.main()
