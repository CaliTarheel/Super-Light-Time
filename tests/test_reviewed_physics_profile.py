"""Fresh profile isolation and historical subduction stalls, not forced reversal."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

import checkpoint
import native_engine
import physics_profile
import slab_memory
import trench_history
from tests.test_trench_history import fixture, step


class PolarityStallTests(unittest.TestCase):
    def world(self):
        s = fixture()
        s.normal_partition_version = 1
        s.subduction_polarity_version = 1
        s.slab_memory_version = 2
        trench_history.initialize(s)
        for _ in range(5):
            step(s)
        return s

    def test_continental_arrival_shuts_down_without_inventing_opposite_trench(self):
        s = self.world()
        row = s.trench_systems[0]
        row.update(slab_fed_area_km2=100., slab_retained_area_km2=100.,
                   slab_initial_excess_mass_kg=1000., slab_retained_excess_mass_kg=1000.)
        slab_memory.refresh_line_load(row)
        original_polarity = s.polarity.copy()
        s.crust[s.ba] = 1
        s.down[:] = 1  # A fallback or stale classification must not create reversal.
        s.bcode[:] = 3  # Oblique convergence remains physically active under PR 9.
        trench_history.prepare(s)
        np.testing.assert_array_equal(s.down, -1)
        np.testing.assert_array_equal(s.bcode, 4)
        np.testing.assert_array_equal(trench_history.weights(s), 0.)
        for _ in range(3):
            s.t += 2.
            s.native_subduction_diagnostics = dict(step_end_myr=s.t, step_duration_myr=2.,
                removed_area_by_trench=[], removed_area_km2=0.)
            slab_memory.advance(s, 2.)
            trench_history.update(s, 2.)
        self.assertEqual(len(s.trench_systems), 1)
        row = s.trench_systems[0]
        self.assertEqual(row['downgoing_plate_uid'], 11)
        self.assertEqual(row['phase'], 'shutdown')
        self.assertEqual(row['episodes'][-1]['shutdown_reason'], 'buoyant_collision')
        self.assertEqual(row['slab_fed_excess_mass_kg'], 0.)
        self.assertAlmostEqual(row['slab_retained_excess_mass_kg'] + row['slab_retired_excess_mass_kg'], 1000.)
        self.assertGreater(row['slab_retained_excess_mass_kg'], 0.)
        slab_memory.validate_row(row, require_mass=True)
        np.testing.assert_array_equal(s.polarity, original_polarity)
        self.assertFalse(any(e.get('reason') == 'polarity_reversal' for e in s.events))

    def test_surviving_oceanic_part_keeps_original_polarity_and_maturity(self):
        s = self.world()
        s.crust[s.ba] = 1
        s.down[:] = 1
        ocean_fraction = np.full(len(s.ba), .1)
        ocean_fraction[0] = 0.
        with patch('native_subduction.enabled', return_value=True), patch(
                'native_subduction.edge_ocean_fraction', return_value=ocean_fraction):
            trench_history.prepare(s)
        self.assertEqual(s.down[0], -1)
        self.assertEqual(s.bcode[0], 4)
        np.testing.assert_array_equal(s.down[1:], 0)
        np.testing.assert_array_equal(s.trench_maturity[1:], 1.)
        self.assertEqual(len(s.trench_systems), 1)

    def test_same_polarity_can_resume_after_temporary_block(self):
        s = self.world()
        s.crust[s.ba] = 1
        trench_history.prepare(s)
        step(s)
        self.assertEqual(s.trench_systems[0]['phase'], 'quiet')
        s.crust[s.ba] = 0
        s.down[:] = 0
        s.bcode[:] = 2
        step(s)
        self.assertEqual(len(s.trench_systems), 1)
        self.assertEqual(s.trench_systems[0]['episode'], 1)
        self.assertEqual(s.trench_systems[0]['phase'], 'mature')

    def test_legacy_policy_keeps_historical_reversal_contract(self):
        s = self.world()
        s.subduction_polarity_version = 0
        s.crust[s.ba] = 1
        s.down[:] = 1
        step(s)
        self.assertEqual(len(s.trench_systems), 2)
        self.assertEqual(s.trench_systems[1]['downgoing_plate_uid'], 22)


class ReviewedProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = dict(width=48, height=24, mesh_level=2, coast_geometry_level=2,
                          plate_count=4, mechanics_nodes=128, seed=37)
        cls.legacy = native_engine.Simulation(cls.config)
        cls.reviewed = native_engine.Simulation(dict(cls.config, physics_profile='reviewed_v1'))

    def test_config_requires_explicit_profile_and_rejects_unknown_values(self):
        self.assertEqual(native_engine.validate_config()['physics_profile'], 'legacy')
        for invalid in (None, True, 1, {}, 'reviewed_v2'):
            with self.assertRaises(ValueError):
                native_engine.validate_config(dict(physics_profile=invalid))
        self.assertFalse(hasattr(self.legacy, 'physics_profile_version'))
        self.assertFalse(hasattr(self.legacy, 'plate_balance_version'))
        self.assertFalse(hasattr(self.legacy, 'plate_resistance_version'))

    def test_same_starting_world_without_invented_mass_or_elapsed_history(self):
        a, b = self.legacy, self.reviewed
        for name in ('plate', 'crust', 'age', 'mass', 'pos', 'parcel_plate', 'kind'):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)
        for name in ('thickness_km', 'area_factor', 'reference_thickness_km'):
            np.testing.assert_array_equal(a.structure[name], b.structure[name], err_msg=name)
        self.assertEqual(a.rng.bit_generator.state, b.rng.bit_generator.state)
        self.assertEqual(b.t, 0.)
        self.assertEqual(b.steps, 0)
        self.assertGreater(len(b.trench_systems), 0)
        trench_events = [event for event in b.events if event['type'].startswith('trench_')]
        self.assertEqual(len(trench_events), len(b.trench_systems))
        self.assertEqual({event['details']['trench_id'] for event in trench_events},
                         {row['id'] for row in b.trench_systems})
        self.assertFalse(any(event['type'] == 'breakup' for event in b.events))
        self.assertEqual([event['id'] for event in b.events], list(range(1, len(b.events)+1)))
        for row in b.trench_systems:
            self.assertEqual(row['maturity'], 0.)
            self.assertEqual(row['active_myr'], 0.)
            self.assertEqual(row['shortening_km'], 0.)
            self.assertEqual(row['slab_retained_excess_mass_kg'], 0.)
        self.assertLessEqual(b.plate_balance_diagnostics['scaled_force_residual'], 1e-7)
        self.assertLess(b.physics_profile_diagnostics['initial_max_speed_cm_yr'], .1)
        self.assertFalse(np.array_equal(a.omega, b.omega))

    def test_all_selected_mechanisms_and_snapshot_are_explicit(self):
        s = self.reviewed
        for name, value in dict(physics_profile_version=1, plate_balance_version=1,
                plate_resistance_version=1,
                normal_partition_version=1, suture_weld_version=1,
                suture_weld_coordinate_version=1, collision_interface_version=1,
                foundering_version=2, foundering_depth_version=1,
                subduction_polarity_version=1, erosion_relief_version=1).items():
            self.assertEqual(getattr(s, name), value, name)
        self.assertFalse(s.foreland_loading_enabled)
        frame = s.snapshot()
        self.assertEqual(frame['physics_profile']['name'], 'reviewed_v1')
        self.assertEqual(frame['plate_resistance_version'], 1)
        self.assertEqual(frame['physics_profile']['plate_resistance_version'], 1)
        self.assertFalse(frame['physics_profile']['foreland_loading_enabled'])
        self.assertIn('nearly stationary', frame['physics_profile_diagnostics']['startup'])

    def test_fresh_reviewed_world_selects_the_accretion_policies(self):
        import local_accretion
        import localized_accretion
        s = self.reviewed
        self.assertTrue(localized_accretion.enabled(s))
        self.assertEqual(s.accretion_age_policy_version, local_accretion.AGE_POLICY_VERSION)
        self.assertEqual(s.accretion_welded_stack_version, local_accretion.WELDED_STACK_VERSION)
        self.assertTrue(local_accretion.welded_stack_enabled(s))
        diagnostics = s.physics_profile_diagnostics
        self.assertEqual(diagnostics['accretion_age_policy_version'], 1)
        self.assertEqual(diagnostics['accretion_welded_stack_version'], 1)
        frame = s.snapshot()
        self.assertEqual(frame['accretion_welded_stack_version'], 1)
        localized_accretion.validate_frame(frame)
        # Saved and legacy worlds keep their recorded edge-connected terranes.
        self.assertFalse(hasattr(self.legacy, 'accretion_welded_stack_version'))
        self.assertFalse(local_accretion.welded_stack_enabled(self.legacy))
        self.assertNotIn('accretion_welded_stack_version', self.legacy.snapshot())

    def test_existing_profile_is_not_a_resistance_migration(self):
        historical = deepcopy(self.reviewed)
        del historical.plate_resistance_version
        before = historical.omega.copy()
        physics_profile.initialize(historical)
        self.assertFalse(hasattr(historical, 'plate_resistance_version'))
        self.assertEqual(physics_profile.snapshot(historical)['plate_resistance_version'], 0)
        np.testing.assert_array_equal(historical.omega, before)

    def test_fresh_reviewed_world_selects_breakup_extension_rupture_criterion(self):
        import mesh_history
        import progressive_rifting
        self.assertFalse(hasattr(self.legacy, 'rupture_criterion_version'))
        legacy = self.legacy.snapshot()
        self.assertNotIn('rupture_criterion_version', legacy)
        self.assertEqual(legacy['rift_mechanics']['rupture_strain_threshold'], .15)
        self.assertNotIn('breakup_extension_km', legacy['rift_mechanics'])
        self.assertEqual(progressive_rifting.validate_frame(legacy), 0)
        s = self.reviewed
        self.assertEqual(s.rupture_criterion_version, 1)
        self.assertEqual(s.physics_profile_diagnostics['rupture_criterion_version'], 1)
        self.assertNotIn('rupture_strain_threshold', s.rift_mechanics)
        self.assertEqual(s.rift_mechanics['breakup_extension_km'], 100.)
        self.assertEqual(s.rift_mechanics['realized_extension_fraction'], .25)
        frame = s.snapshot()
        self.assertEqual(frame['rupture_criterion_version'], 1)
        self.assertEqual(frame['physics_profile']['rupture_criterion_version'], 1)
        self.assertEqual(progressive_rifting.validate_frame(frame), 1)
        mesh_history.arrays(frame)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'checkpoint.npz'
            checkpoint.write_checkpoint(path, s, dict(config=s.config), {})
            restored, _ = checkpoint.read_checkpoint(path, {}, native_engine.Simulation)
        self.assertEqual(restored.rupture_criterion_version, 1)
        self.assertEqual(restored.rift_mechanics, s.rift_mechanics)
        historical = deepcopy(s)
        del historical.rupture_criterion_version
        physics_profile.initialize(historical)
        self.assertFalse(hasattr(historical, 'rupture_criterion_version'))
        self.assertEqual(progressive_rifting.rupture_criterion_version(historical), 0)
        # A saved reviewed world keeps its earlier profile block exactly.
        self.assertNotIn('rupture_criterion_version', physics_profile.snapshot(historical)['physics_profile'])

    def test_fresh_reviewed_world_selects_rift_commit_policy_one(self):
        import progressive_rifting
        self.assertFalse(hasattr(self.legacy, 'rift_commit_version'))
        legacy = self.legacy.snapshot()
        self.assertNotIn('rift_commit_version', legacy)
        self.assertFalse(any(key.startswith('commit_') for key in legacy['rift_mechanics']))
        s = deepcopy(self.reviewed)
        self.assertEqual(s.rift_commit_version, 1)
        self.assertEqual(s.physics_profile_diagnostics['rift_commit_version'], 1)
        self.assertEqual(s.rift_mechanics['commit_policy'], progressive_rifting.RIFT_COMMIT_POLICY)
        self.assertEqual(s.rift_mechanics['commit_loading_unit'], progressive_rifting.PROXY_LOADING_UNIT)
        frame = s.snapshot()
        self.assertEqual(frame['rift_commit_version'], 1)
        self.assertEqual(frame['physics_profile']['rift_commit_version'], 1)
        progressive_rifting.validate_frame(frame)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'checkpoint.npz'
            checkpoint.write_checkpoint(path, s, dict(config=s.config), {})
            restored, _ = checkpoint.read_checkpoint(path, {}, native_engine.Simulation)
        self.assertEqual(restored.rift_commit_version, 1)
        self.assertIsInstance(restored.rift_commit_version, int)
        self.assertEqual(restored.rift_mechanics, s.rift_mechanics)
        self.assertEqual(progressive_rifting.rift_commit_version(restored), 1)
        # The per-step mechanics rebuild keeps the labels on a resumed world.
        restored.step(.05)
        self.assertEqual(restored.rift_mechanics['commit_policy_version'], 1)
        progressive_rifting.validate_frame(restored.snapshot())
        historical = deepcopy(self.reviewed)
        del historical.rift_commit_version
        physics_profile.initialize(historical)
        self.assertFalse(hasattr(historical, 'rift_commit_version'))
        self.assertEqual(progressive_rifting.rift_commit_version(historical), 0)
        self.assertNotIn('rift_commit_version', physics_profile.snapshot(historical)['physics_profile'])

    def test_enhanced_breakup_extension_pending_work_survives_a_checkpoint(self):
        import progressive_rifting
        from tests.test_rupture_calibration import observed_cut_candidates
        s = native_engine.Simulation(dict(self.config, physics_profile='reviewed_v1',
                                          enhanced_rifting={'enabled': True}))
        self.assertEqual(s.rift_mechanics['realized_extension_fraction'], 1.)
        self.assertEqual(s.rift_mechanics['commit_loading_unit'], progressive_rifting.REALIZED_LOADING_UNIT)
        # Material refinement in the first step reindexes faces and drops that
        # step's pending work (native_material_adaptivity); take a second step.
        for _ in range(2):
            s.step(s.config['dt_myr'])
        pending = s.rift_pending
        self.assertEqual(pending['rupture_criterion_version'], 1)
        self.assertIn('extension_km', pending)
        # Policy-1 loading records its material image (each patch once, every
        # node key present); assert_same below checks it survives the checkpoint.
        ids = np.asarray(pending['material_patch_ids'])
        self.assertEqual(len(ids), len(np.unique(ids)))
        self.assertEqual(set(zip(np.asarray(pending['material_patch_bases']).tolist(),
                                 np.asarray(pending['material_patch_owner_uids']).tolist())),
                         set(progressive_rifting._node_keys(pending['mesh'])))
        progressive_rifting.validate_frame(s.snapshot())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'checkpoint.npz'
            checkpoint.write_checkpoint(path, s, dict(config=s.config), {})
            restored, _ = checkpoint.read_checkpoint(path, {}, native_engine.Simulation)
        self.assertEqual(restored.rupture_criterion_version, 1)
        self.assertEqual(restored.rift_mechanics, s.rift_mechanics)

        def assert_same(a, b, path='rift_pending'):
            if isinstance(a, dict):
                self.assertEqual(set(a), set(b), path)
                for key in a:
                    assert_same(a[key], b[key], f'{path}.{key}')
            elif isinstance(a, np.ndarray):
                np.testing.assert_array_equal(a, b, err_msg=path)
                self.assertEqual(a.dtype, np.asarray(b).dtype, path)
            else:
                self.assertEqual(a, b, path)
                self.assertEqual(type(a), type(b), path)
        assert_same(pending, restored.rift_pending)
        # The restored work reaches the same km decision: at 100 km every
        # damaged, opening link is a candidate; just below it none is.
        for extension, expected in ((100., True), (np.nextafter(100., 0.), False)):
            decisions = []
            for state in (s, restored):
                trial = deepcopy(state)
                trial.rift_pending['damage'][:] = 1.
                trial.rift_pending['edge_extension'][:] = 1.
                trial.rift_pending['extension_km'][:] = extension
                decisions.append(observed_cut_candidates(trial, trial.rift_pending['mesh']))
            for decision in decisions:
                if expected:
                    self.assertTrue(np.all(decision))
                else:
                    self.assertIsNone(decision)

    def test_checkpoint_preserves_profile_and_actual_step_continuation(self):
        s = deepcopy(self.reviewed)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'checkpoint.npz'
            checkpoint.write_checkpoint(path, s, dict(config=s.config), {})
            restored, _ = checkpoint.read_checkpoint(path, {}, native_engine.Simulation)
        self.assertEqual(physics_profile.snapshot(restored), physics_profile.snapshot(s))
        s.step(.05)
        restored.step(.05)
        self.assertEqual(restored.t, .05)
        for name in ('pos', 'mass', 'omega', 'support', 'age'):
            np.testing.assert_array_equal(getattr(s, name), getattr(restored, name), err_msg=name)
        self.assertIn('timestep_diagnostics', s.snapshot())
        for row in s.trench_systems:
            slab_memory.validate_row(row, require_mass=True)


if __name__ == '__main__':
    unittest.main()
