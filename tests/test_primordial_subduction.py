"""Inherited primordial margins carry explicit geological baselines, not fake time."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import numpy as np

import checkpoint
import native_engine
import normal_partition
import primordial_subduction as initial
import slab_memory as slab
import trench_history as history


class PrimordialSubductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = dict(width=48, height=24, mesh_level=2, coast_geometry_level=2,
                          plate_count=4, mechanics_nodes=128, seed=37, physics_profile='reviewed_v1')
        cls.baseline = native_engine.Simulation(cls.config)
        cls.seeded = native_engine.Simulation(dict(cls.config, primordial_subduction={'enabled': True}))

    def test_explicit_configuration_preserves_default_and_rejects_bad_settings(self):
        self.assertFalse(native_engine.validate_config()['primordial_subduction']['enabled'])
        self.assertFalse(hasattr(self.baseline, 'primordial_subduction_version'))
        self.assertEqual(self.baseline.physics_profile_diagnostics['initial_slab_mass_kg'], 0.)
        for bad in (True, {'enabled': 1}, {'initial_slab_depth_km': 0},
                    {'initial_slab_depth_km': 661}, {'initial_slab_depth_km': float('nan')},
                    {'target_margin_fraction': 0.}, {'target_margin_fraction': 1.1},
                    {'selection_seed': -1}, {'unrecognized': 1}):
            with self.assertRaises(ValueError):
                initial.normalize(bad)
        with self.assertRaisesRegex(ValueError, 'reviewed_v1'):
            native_engine.validate_config(dict(primordial_subduction={'enabled': True}))

    def test_every_real_coast_has_ocean_polarity_and_declared_initial_inventory(self):
        s = self.seeded
        ocean, mask = initial.target_edges(s)
        np.testing.assert_array_equal(s.down[mask], ocean)
        frame = s.snapshot()
        np.testing.assert_array_equal(frame['native_boundary_code'][mask], 2)
        np.testing.assert_array_equal(frame['native_boundary_kinematic_code'], s.bcode)
        self.assertTrue(np.all(s.trench_id[mask] > 0))
        source = s.native_initial_owner_interfaces
        target = ((source['owner_a'] == ocean) | (source['owner_b'] == ocean)) & ~source['ocean']
        self.assertAlmostEqual(float(source['length_km'][target].sum()), float(s.bl[mask].sum()), places=7)
        expected_area = s.bl[mask].sum()*100/np.sin(np.radians(50))
        self.assertAlmostEqual(sum(r[slab.INITIAL_AREA_FIELD] for r in s.trench_systems)/expected_area, 1., places=14)
        self.assertAlmostEqual(s.primordial_subduction_diagnostics['initial_incoming_ocean_fraction_min'], 1.)
        for row in s.trench_systems:
            self.assertEqual(row['downgoing_plate_uid'], s.initial_ocean_plate_uid)
            self.assertEqual(row['maturity'], 1.)
            self.assertEqual(row['active_myr'], 0.)
            self.assertEqual(row['shortening_km'], 0.)
            self.assertEqual(row['slab_fed_area_km2'], 0.)
            self.assertEqual(row['slab_fed_excess_mass_kg'], 0.)
            slab.validate_row(row, require_mass=True)
        self.assertEqual(s.process_totals['ocean_consumed_km2'], 0.)
        self.assertEqual(s.t, 0.)
        self.assertEqual(s.steps, 0)

    def test_highland65_world_selects_connected_subset_without_edge_checkerboard(self):
        config = dict(width=64, height=32, mesh_level=3, coast_geometry_level=3,
                      plate_count=12, mechanics_nodes=128, seed=41,
                      physics_profile='reviewed_v1')
        initial_world = native_engine.make_initial(config, preset='highland65')
        self.assertEqual(initial_world['initial_subduction']['target_margin_fraction'], .45)
        s = native_engine.Simulation(config, initial_world)
        self.assertTrue(s.initial_subduction_adopted)
        self.assertTrue(s.primordial_subduction_diagnostics['enabled'])
        down, candidate = initial.target_edges(s)
        _, selected, report = initial.selected_target_edges(s)
        self.assertTrue(np.all(selected <= candidate))
        self.assertGreater(candidate.sum(), selected.sum())
        self.assertGreater(report['candidate_components'], report['selected_components'])
        self.assertGreater(report['selected_length_km'], 0.)
        self.assertLess(report['selected_length_km'], report['candidate_length_km'])
        self.assertTrue(np.all(s.trench_id[selected] > 0))
        self.assertTrue(np.all(s.trench_id[candidate & ~selected] == 0))
        self.assertAlmostEqual(s.primordial_subduction_diagnostics['target_margin_fraction'], .45)

    def test_initialization_changes_no_surface_and_does_not_repeat_on_resume(self):
        s = deepcopy(self.seeded)
        for name in ('mass', 'pos', 'plate', 'age', 'crust', 'support'):
            np.testing.assert_array_equal(getattr(s, name), getattr(self.baseline, name), err_msg=name)
        rows = deepcopy(s.trench_systems)
        self.assertTrue(initial.initialize(s))
        self.assertEqual(s.trench_systems, rows)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'checkpoint.npz'
            checkpoint.write_checkpoint(path, s, {'config': s.config}, {})
            restored, _ = checkpoint.read_checkpoint(path, {}, native_engine.Simulation)
        self.assertEqual(initial.snapshot(restored), initial.snapshot(s))
        self.assertEqual(restored.trench_systems, rows)

    def test_force_is_present_without_fabricating_convergent_consumption(self):
        s = deepcopy(self.seeded)
        self.assertGreater(sum(s.plate_balance_diagnostics['driver_torque_n_m']['slab'].values()), 0.)
        ocean, mask = initial.target_edges(s)
        s.normal_speed[:] = 5.
        s.bcode[:] = 3
        s.down[:] = -1
        history.prepare(s)
        owners, load, _ = slab.line_load(s)
        self.assertTrue(np.all(load[mask] > 0.))
        np.testing.assert_array_equal(owners[mask], ocean)
        self.assertFalse(np.any(normal_partition.subduction(s)))
        np.testing.assert_array_equal(s.trench_maturity, 0.)

    def test_initial_area_mass_survive_partition_join_and_retirement(self):
        parent = deepcopy(self.seeded.trench_systems[0])
        original = deepcopy(parent)
        child = dict(phase='mature', length_km=parent['length_km']*.3)
        slab.ensure(child, conservative=True)
        parent['length_km'] *= .7
        slab.refresh_line_load(parent)
        slab.partition(parent, child, .3)
        for field in (*slab.FIELDS, *slab.MASS_FIELDS, slab.INITIAL_AREA_FIELD):
            self.assertAlmostEqual((parent[field]+child[field])/max(original[field], 1.), original[field]/max(original[field], 1.), places=14)
        slab.join(child, parent)
        parent['length_km'] = original['length_km']
        slab.refresh_line_load(parent)
        slab.validate_row(parent, require_mass=True)
        self.assertEqual(child[slab.INITIAL_AREA_FIELD], 0.)
        self.assertAlmostEqual(parent[slab.INITIAL_AREA_FIELD]/original[slab.INITIAL_AREA_FIELD], 1.)
        s = deepcopy(self.seeded)
        s.t = 2.
        s.native_subduction_diagnostics = dict(step_end_myr=2., step_duration_myr=2.,
                                               removed_area_by_trench=[], removed_area_km2=0.)
        slab.advance(s, 2.)
        report = slab.snapshot(s)['slab_memory_diagnostics']
        self.assertEqual(report['fed_area_km2'], 0.)
        self.assertAlmostEqual((report['retained_area_km2']+report['retired_area_km2'])/report[slab.INITIAL_AREA_FIELD], 1.)
        self.assertLess(report['retained_area_km2'], report[slab.INITIAL_AREA_FIELD])
        for row in s.trench_systems:
            slab.validate_row(row, require_mass=True)

    def test_shutdown_does_not_resurrect_inherited_maturity(self):
        from tests.test_trench_history import fixture, step
        s = fixture()
        s.normal_partition_version = 1
        s.slab_memory_version = 2
        history.initialize(s)
        row = s.trench_systems[0]
        row.update(maturity=1., initial_maturity=1., phase='mature', initial_subduction={'version': 1})
        s.normal_speed[:] = 5.
        s.bcode[:] = 1
        for _ in range(5):
            step(s)
        self.assertEqual(row['phase'], 'shutdown')
        self.assertEqual(row['initial_maturity'], 0.)
        s.normal_speed[:] = -1.
        s.bcode[:] = 2
        s.down[:] = 0
        step(s)
        self.assertEqual(row['episode'], 2)
        self.assertAlmostEqual(row['maturity'], .02)

    def test_small_real_step_closes_initial_and_runtime_inventories(self):
        s = deepcopy(self.seeded)
        frame = s.snapshot()
        self.assertTrue(frame['primordial_subduction_diagnostics']['enabled'])
        report = frame['slab_memory_diagnostics']
        self.assertEqual(report['retained_area_km2'], report[slab.INITIAL_AREA_FIELD])
        s.step(.05)
        self.assertEqual(s.t, .05)
        for row in s.trench_systems:
            slab.validate_row(row, require_mass=True)
        report = slab.snapshot(s)['slab_memory_diagnostics']
        self.assertAlmostEqual((report['retained_area_km2']+report['retired_area_km2'])/
                               (report[slab.INITIAL_AREA_FIELD]+report['fed_area_km2']), 1.)
        before = s.bcode.copy()
        frame = s.snapshot()
        np.testing.assert_array_equal(s.bcode, before)
        np.testing.assert_array_equal(frame['native_boundary_kinematic_code'], before)
        self.assertFalse(frame['primordial_subduction_review']['declared_initial_mechanism'])
        self.assertTrue(any(r.get('mechanism') == 'inherited_subduction' for r in frame['boundary_segments']))


if __name__ == '__main__':
    unittest.main()
