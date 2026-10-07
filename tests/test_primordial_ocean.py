"""Conservative initial ocean ownership, represented ridges and coastal slabs."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import numpy as np

import checkpoint
import mesh_geometry
import native_engine
import normal_partition
import primordial_ocean as ocean
import primordial_subduction
import slab_memory
import trench_history


def fitted_fixture(level=1):
    mesh = mesh_geometry.icosphere(level)
    kind = (mesh['xyz'][:, 0] > .45).astype(np.uint8)*2
    return dict(vertices=mesh['vertices'].copy(), faces=mesh['faces'].copy(), face_kind=kind,
                face_owner=(kind > 0).astype(np.int16), face_craton=np.where(kind > 0, 7, -1),
                source_seed_cells=np.arange(len(kind)), area_km2=mesh['area_km2'].copy(), diagnostics={})


def edge_faces(faces):
    adjacent = {}
    for face, nodes in enumerate(faces):
        for a, b in zip(nodes, np.roll(nodes, -1)):
            adjacent.setdefault(tuple(sorted((int(a), int(b)))), []).append(face)
    return adjacent


class OceanGeometryTests(unittest.TestCase):
    def test_config_is_opt_in_and_finite(self):
        self.assertFalse(native_engine.validate_config()['primordial_ocean']['enabled'])
        for bad in (True, {'enabled': 1}, {'plate_count': True}, {'plate_count': 2.5},
                    {'plate_count': 9}, {'half_spreading_rate_cm_yr': 0.},
                    {'maximum_age_myr': float('inf')}, {'unrecognized': 1}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ocean.normalize(bad)
        with self.assertRaisesRegex(ValueError, 'reviewed_v1'):
            native_engine.validate_config(dict(primordial_ocean={'enabled': True}))

    def test_partition_is_connected_deterministic_and_never_changes_land(self):
        fitted = fitted_fixture()
        original = deepcopy(fitted)
        result = ocean.partition_water(fitted, 4)
        repeated = ocean.partition_water(fitted, 4)
        for name in ('vertices', 'faces', 'face_kind', 'face_owner', 'face_craton', 'source_seed_cells', 'area_km2'):
            np.testing.assert_array_equal(fitted[name], original[name], err_msg=name)
        for name in ('face_labels', 'seed_faces', 'ridge_vertex_indices', 'cooling_distance_km'):
            np.testing.assert_array_equal(result[name], repeated[name], err_msg=name)
        labels = result['face_labels']
        np.testing.assert_array_equal(labels[fitted['face_kind'] > 0], -1)
        adjacency = edge_faces(fitted['faces'])
        for plate, seed in enumerate(result['seed_faces']):
            reached, todo = {int(seed)}, [int(seed)]
            while todo:
                here = todo.pop()
                for pair in adjacency.values():
                    if here not in pair:
                        continue
                    for other in pair:
                        if labels[other] == plate and other not in reached:
                            reached.add(other)
                            todo.append(other)
            self.assertEqual(reached, set(np.flatnonzero(labels == plate)))
        for a, b in result['ridge_face_indices']:
            self.assertEqual(fitted['face_kind'][a], 0)
            self.assertEqual(fitted['face_kind'][b], 0)
            self.assertNotEqual(labels[a], labels[b])

    def test_cooling_distance_matches_independent_within_plate_shortest_paths(self):
        fitted = fitted_fixture()
        result = ocean.partition_water(fitted, 3)
        faces, vertices = fitted['faces'], fitted['vertices']
        centers = vertices[faces].sum(axis=1)
        centers /= np.linalg.norm(centers, axis=1)[:, None]
        count = len(faces)
        graph = np.full((count, count), np.inf)
        np.fill_diagonal(graph, 0.)
        seeds = np.full(count, np.inf)
        labels = result['face_labels']
        for edge, (a, b) in edge_faces(faces).items():
            if labels[a] < 0 or labels[b] < 0:
                continue
            midpoint = vertices[list(edge)].sum(axis=0)
            midpoint /= np.linalg.norm(midpoint)
            da = 6371.*np.arccos(np.clip(centers[a]@midpoint, -1., 1.))
            db = 6371.*np.arccos(np.clip(centers[b]@midpoint, -1., 1.))
            if labels[a] == labels[b]:
                graph[a, b] = graph[b, a] = da+db
            else:
                seeds[a], seeds[b] = min(seeds[a], da), min(seeds[b], db)
        for intermediate in range(count):
            graph = np.minimum(graph, graph[:, intermediate, None]+graph[None, intermediate, :])
        expected = np.min(graph+seeds[None, :], axis=1)[result['water_faces']]
        np.testing.assert_allclose(result['cooling_distance_km'], expected, rtol=3e-13, atol=3e-10)
        self.assertTrue(np.all(expected > 0.))  # centres have finite age; ridge edges have zero age

    def test_disconnected_basins_must_each_receive_a_seed(self):
        fitted = fitted_fixture()
        fitted['face_kind'][:] = 1
        neighbors = edge_faces(fitted['faces'])
        selected = []
        for face in range(len(fitted['faces'])):
            if all(not (face in pair and any(previous in pair for previous in selected)) for pair in neighbors.values()):
                selected.append(face)
            if len(selected) == 3:
                break
        fitted['face_kind'][selected] = 0
        with self.assertRaisesRegex(ValueError, '3 disconnected water components'):
            ocean.partition_water(fitted, 2)


class OceanStartupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # One extra control level resolves the newly added coastal triple points.
        cls.config = dict(width=48, height=24, mesh_level=3, coast_geometry_level=2,
                          plate_count=4, mechanics_nodes=128, seed=37, physics_profile='reviewed_v1',
                          primordial_subduction={'enabled': True})
        cls.baseline = native_engine.Simulation(cls.config)
        cls.startup = native_engine.Simulation(dict(cls.config, primordial_ocean={'enabled': True}))

    def test_land_material_and_total_ocean_support_are_unchanged(self):
        baseline, split = self.baseline, self.startup
        for name in ('mass', 'pos', 'kind', 'parcel_plate', 'parcel_patch', 'parcel_craton',
                     'relief', 'rift_id', 'rift_tangent', 'rift_extension_m'):
            np.testing.assert_array_equal(getattr(split, name), getattr(baseline, name), err_msg=name)
        for name in ('vertices', 'faces', 'face_kind', 'face_owner', 'face_id', 'reference_area_km2'):
            np.testing.assert_array_equal(split.material_surface[name], baseline.material_surface[name], err_msg=name)
        for name in baseline.structure:
            np.testing.assert_array_equal(split.structure[name], baseline.structure[name], err_msg=name)
        parent = int(np.flatnonzero(baseline.plate_uid == baseline.initial_ocean_plate_uid)[0])
        slots = np.flatnonzero(np.isin(split.plate_uid, split.initial_ocean_plate_uids))
        np.testing.assert_allclose(split.support[slots].sum(axis=0), baseline.support[parent], atol=2e-12, rtol=2e-12)
        continents = np.unique(baseline.parcel_plate)
        np.testing.assert_array_equal(split.support[continents], baseline.support[continents])
        np.testing.assert_allclose(split.support.sum(axis=0), 1., atol=2e-10, rtol=0.)
        self.assertEqual(split.original_mass, baseline.original_mass)
        self.assertEqual(split.original_craton_mass, baseline.original_craton_mass)
        self.assertEqual(split.steps, 0)
        self.assertEqual(split.t, 0.)
        for name in ('ocean_created_km2', 'ocean_consumed_km2', 'rift_events'):
            self.assertEqual(split.process_totals[name], 0., name)

    def test_uid_lifecycle_age_integral_and_ridge_geometry(self):
        s = self.startup
        self.assertEqual(s.initial_ocean_plate_uid, self.baseline.initial_ocean_plate_uid)
        self.assertEqual(len(np.unique(s.initial_ocean_plate_uids)), 4)
        self.assertEqual(s.count, self.baseline.count+3)
        slots = np.flatnonzero(np.isin(s.plate_uid, s.initial_ocean_plate_uids))
        self.assertTrue(np.all(s.active[slots]))
        self.assertGreater(s.next_plate_uid, max(s.plate_uid))
        for slot in slots:
            if s.plate_uid[slot] != s.initial_ocean_plate_uid:
                self.assertEqual(s.plate_parent_uid[slot], s.initial_ocean_plate_uid)
                self.assertEqual(s.plate_created[slot], 0.)
        diagnostics = s.primordial_ocean_diagnostics
        self.assertAlmostEqual(diagnostics['initial_control_water_age_area_myr_km2']/
                               diagnostics['initial_water_age_area_myr_km2'], 1., places=13)
        source = s.native_initial_owner_interfaces
        ocean_owners = np.isin(source['owner_a'], slots) & np.isin(source['owner_b'], slots)
        self.assertTrue(np.all(source['ocean'][ocean_owners]))
        self.assertAlmostEqual(float(source['length_km'][ocean_owners].sum()), diagnostics['ridge_length_km'], places=7)
        self.assertGreater(diagnostics['ridge_segments'], 0)
        self.assertGreater(np.ptp(s.age[s.crust == 0]), 100.)

    def test_all_original_coasts_seed_local_incoming_ocean_without_ridge_subduction(self):
        s = self.startup
        down, coast = primordial_subduction.target_edges(s)
        original_length = self.baseline.primordial_subduction_diagnostics['initial_trench_length_km']
        self.assertAlmostEqual(float(s.bl[coast].sum()), original_length, places=7)
        np.testing.assert_array_equal(s.down[coast], down[coast])
        self.assertTrue(np.all(s.trench_id[coast] > 0))
        self.assertTrue(np.all(np.isin(s.plate_uid[down[coast]], s.initial_ocean_plate_uids)))
        ocean_slots = np.flatnonzero(np.isin(s.plate_uid, s.initial_ocean_plate_uids))
        ridges = np.isin(s.bp, ocean_slots) & np.isin(s.bq, ocean_slots)
        self.assertFalse(np.any(coast[ridges]))
        self.assertTrue(np.all(s.trench_id[ridges] == 0))
        expected = original_length*100./np.sin(np.radians(50.))
        self.assertAlmostEqual(sum(row[slab_memory.INITIAL_AREA_FIELD] for row in s.trench_systems)/expected, 1., places=13)
        for row in s.trench_systems:
            self.assertIn(row['downgoing_plate_uid'], s.initial_ocean_plate_uids)
            self.assertNotIn(row['overriding_plate_uid'], s.initial_ocean_plate_uids)
            slab_memory.validate_row(row, require_mass=True)
        opening = deepcopy(s)
        opening.normal_speed[:] = 5.
        opening.bcode[:] = 3
        opening.down[:] = -1
        trench_history.prepare(opening)
        owners, load, _ = slab_memory.line_load(opening)
        np.testing.assert_array_equal(owners[coast], down[coast])
        self.assertTrue(np.all(load[coast] > 0.))
        self.assertFalse(np.any(normal_partition.subduction(opening)))

    def test_output_resolution_and_restart_preserve_the_initial_condition(self):
        s = self.startup
        other = native_engine.Simulation(dict(self.config, width=64, height=32, primordial_ocean={'enabled': True}))
        for name in ('support', 'plate', 'age', 'mass', 'pos', 'initial_ocean_plate_uids'):
            np.testing.assert_array_equal(getattr(s, name), getattr(other, name), err_msg=name)
        self.assertEqual(s.primordial_ocean_diagnostics, other.primordial_ocean_diagnostics)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'checkpoint.npz'
            checkpoint.write_checkpoint(path, s, {'config': s.config}, {})
            restored, _ = checkpoint.read_checkpoint(path, {}, native_engine.Simulation)
        self.assertEqual(restored.config, s.config)
        self.assertEqual(restored.primordial_ocean_diagnostics, s.primordial_ocean_diagnostics)
        self.assertEqual(restored.trench_systems, s.trench_systems)
        for name in ('support', 'plate', 'age', 'initial_ocean_plate_uids'):
            np.testing.assert_array_equal(getattr(restored, name), getattr(s, name))
        self.assertEqual(restored.snapshot()['primordial_ocean_version'], 1)
        # Resume does not execute either initial condition a second time.
        self.assertTrue(primordial_subduction.initialize(restored))
        self.assertEqual(restored.trench_systems, s.trench_systems)

    def test_coarse_unresolved_coastal_junctions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unresolved.*refine the control mesh'):
            native_engine.Simulation(dict(self.config, mesh_level=2, primordial_ocean={'enabled': True}))

    def test_real_small_step_conserves_initial_and_consumed_slab_inventory(self):
        s = deepcopy(self.startup)
        s.step(.02)
        self.assertEqual(s.t, .02)
        for row in s.trench_systems:
            slab_memory.validate_row(row, require_mass=True)
        report = slab_memory.snapshot(s)['slab_memory_diagnostics']
        self.assertAlmostEqual((report['retained_area_km2']+report['retired_area_km2'])/
            (report[slab_memory.INITIAL_AREA_FIELD]+report['fed_area_km2']), 1., places=12)


if __name__ == '__main__':
    unittest.main()
