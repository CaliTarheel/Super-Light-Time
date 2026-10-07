"""Explicit startup selection and real native continuation of moving-hinge work."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

import backarc
import checkpoint
import mesh_history
import native_engine
import native_frame_sampling
import normal_partition
import physics_profile
import plate_balance
import slab_memory


class SubductionResponseProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = dict(width=48, height=24, mesh_level=2, coast_geometry_level=2,
                          plate_count=4, mechanics_nodes=128, seed=37,
                          physics_profile='reviewed_v1', primordial_subduction={'enabled': True})
        cls.fixed = native_engine.Simulation(cls.config)
        cls.moving = native_engine.Simulation(dict(cls.config, subduction_response='moving_hinge_v1'))

    def test_config_selection_is_explicit_and_strict(self):
        self.assertEqual(native_engine.validate_config()['subduction_response'], 'fixed_trench')
        for value in (None, True, 1, {}, 'moving_hinge_v2'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                native_engine.validate_config(dict(subduction_response=value))
        with self.assertRaisesRegex(ValueError, 'reviewed_v1'):
            native_engine.validate_config(dict(subduction_response='moving_hinge_v1'))

    def test_only_force_closure_changes_not_initial_inventory(self):
        a, b = self.fixed, self.moving
        for name in ('plate', 'crust', 'age', 'support', 'mass', 'pos', 'kind', 'parcel_plate'):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)
        for left, right in zip(a.trench_systems, b.trench_systems):
            for name in (slab_memory.INITIAL_AREA_FIELD, 'slab_initial_excess_mass_kg',
                         'slab_retained_excess_mass_kg', 'downgoing_plate_uid', 'overriding_plate_uid'):
                self.assertEqual(left[name], right[name], name)
        self.assertEqual(b.snapshot()['physics_profile']['subduction_response_version'], 1)
        self.assertEqual(a.snapshot()['subduction_response_version'], 0)
        self.assertFalse(np.array_equal(a.omega, b.omega))
        self.assertGreater(b.plate_balance_diagnostics['slab_power_w']['overriding'], 0.)
        self.assertLessEqual(b.plate_balance_diagnostics['scaled_force_residual'], 1e-7)

    def test_existing_checkpoint_does_not_read_new_config_as_migration(self):
        s = deepcopy(self.fixed)
        del s.subduction_response_version
        del s.backarc_driver_version
        s.config['subduction_response'] = 'moving_hinge_v1'
        before = s.omega.copy()
        physics_profile.initialize(s)
        self.assertEqual(plate_balance.subduction_response_version(s), 0)
        self.assertEqual(physics_profile.snapshot(s)['subduction_response_version'], 0)
        self.assertEqual(backarc.driver_version(s), 0)
        self.assertFalse(hasattr(s, 'backarc_driver_version'))
        np.testing.assert_array_equal(s.omega, before)

    def test_backarc_driver_follows_the_fresh_subduction_response(self):
        self.assertEqual(self.moving.backarc_driver_version, backarc.DRIVER_VERSION)
        self.assertEqual(self.fixed.backarc_driver_version, 0)
        for world, version in ((self.moving, 1), (self.fixed, 0)):
            self.assertEqual(world.physics_profile_diagnostics['backarc_driver_version'], version)
            self.assertEqual(physics_profile.snapshot(world)['physics_profile']['backarc_driver_version'], version)
        # A saved world from before the field existed keeps the swept driver.
        legacy = deepcopy(self.moving)
        del legacy.backarc_driver_version
        self.assertEqual(backarc.driver_version(legacy), 0)
        self.assertEqual(backarc.snapshot_fields(legacy), {})

    def test_driver_frames_validate_and_fixed_trench_frames_carry_none(self):
        frame = deepcopy(self.moving).snapshot()
        self.assertEqual(frame['backarc_driver_version'], 1)
        self.assertEqual(frame['subduction_response_version'], 1)
        report = frame['backarc_driver_diagnostics']
        self.assertEqual((report['frame'], report['law']), (backarc.DRIVER_FRAME, backarc.DRIVER_LAW))
        self.assertGreater(report['anchored_length_km'], 0.)
        mesh_history.arrays(frame)
        native_frame_sampling.prepare(frame)
        fixed = deepcopy(self.fixed).snapshot()
        self.assertNotIn('backarc_driver_version', fixed)
        self.assertNotIn('backarc_driver_diagnostics', fixed)
        mesh_history.arrays(fixed)
        stray = dict(fixed, backarc_driver_diagnostics=report)
        for validator in (mesh_history.arrays, native_frame_sampling.prepare):
            with self.subTest(validator=validator.__module__), self.assertRaises(ValueError):
                validator(stray)
        mismatched = dict(frame, subduction_response_version=0)
        with self.assertRaises(ValueError):
            mesh_history.arrays(mismatched)

    def _at_trench_migration(self):
        """Native intermediate state before the inherited migration stage."""
        s = deepcopy(self.moving)
        s._advect(.02)
        s.t += .02
        s._rasterize()
        s._boundaries()
        return s

    def test_moving_hinge_does_not_remap_intermediate_ocean_support(self):
        s = self._at_trench_migration()
        support, owners = s.support.copy(), s.plate.copy()
        totals = dict(s.process_totals)
        # The same geometry must expose the old remapping. Checking only the
        # final retreat array misses it: the final boundary rebuild zeroes it.
        legacy = deepcopy(s)
        legacy.subduction_response_version = 0
        legacy._migrate_trenches(.02)
        self.assertGreater(np.max(np.abs(legacy.support-support)), 0.)
        self.assertGreater(legacy.process_totals['trench_swept_km2'],
                           totals['trench_swept_km2'])
        self.assertGreater(legacy.process_totals['trench_arc_sweep_km2'],
                           totals['trench_arc_sweep_km2'])

        s._migrate_trenches(.02)
        np.testing.assert_array_equal(s.support, support)
        np.testing.assert_array_equal(s.plate, owners)
        self.assertEqual(s.process_totals, totals)
        np.testing.assert_array_equal(s.trench_retreat_speed, 0.)

    def test_missing_response_version_preserves_legacy_remapping(self):
        fixed = self._at_trench_migration()
        fixed.subduction_response_version = 0
        historical = deepcopy(fixed)
        del historical.subduction_response_version
        before = fixed.support.copy()
        for world in (fixed, historical):
            world._migrate_trenches(.02)
        self.assertGreater(np.max(np.abs(fixed.support-before)), 0.)
        for name in ('support', 'plate', 'trench_retreat_speed'):
            np.testing.assert_array_equal(getattr(fixed, name), getattr(historical, name))
        self.assertEqual(fixed.process_totals, historical.process_totals)

    def test_native_step_and_checkpoint_agree_and_consume_finite_ocean(self):
        s = deepcopy(self.moving)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'checkpoint.npz'
            checkpoint.write_checkpoint(path, s, dict(config=s.config), {})
            restored, _ = checkpoint.read_checkpoint(path, {}, native_engine.Simulation)
        for world in (s, restored):
            sweep = {name: world.process_totals[name] for name in
                     ('trench_swept_km2', 'trench_arc_sweep_km2')}
            world.step(.02)
            self.assertEqual(world.subduction_response_version, 1)
            self.assertEqual(world.backarc_driver_version, 1)
            self.assertEqual(world.t, .02)
            # The sea-anchor driver never writes the swept retreat, so the
            # force balance's moving-hinge guard keeps holding.
            np.testing.assert_array_equal(world.trench_retreat_speed, 0.)
            self.assertEqual(world.backarc_driver_diagnostics['version'], 1)
            self.assertEqual({name: world.process_totals[name] for name in sweep}, sweep)
            for row in world.trench_systems:
                slab_memory.validate_row(row, require_mass=True)
            self.assertGreater(world.process_totals['ocean_consumed_km2'], 0.)
            report = world.plate_balance_diagnostics
            self.assertLessEqual(report['scaled_force_residual'], 1e-7)
            self.assertLess(abs(report['power_balance_error_w'])/report['driver_work_w'], 1e-6)
        for name in ('pos', 'mass', 'omega', 'support', 'age', 'plate'):
            np.testing.assert_array_equal(getattr(s, name), getattr(restored, name), err_msg=name)
        self.assertEqual(s.trench_systems, restored.trench_systems)
        self.assertEqual(s.backarc_driver_diagnostics, restored.backarc_driver_diagnostics)
        self.assertEqual(s.backarc_basins, restored.backarc_basins)

    def _upper_plate_moving_away(self, version):
        """The moving world with its longest attached margin's upper plate
        moving 5 km/Myr (0.5 cm/yr) inland at the trench while the ocean
        converges at 20.

        Motion is prescribed (forces held) only while the original plates
        exist; once a plate splits, every solve is the real force balance.
        """
        s = deepcopy(self.moving)
        s.backarc_driver_version = version
        owners, load, _ = slab_memory.line_load(s)
        attached = np.flatnonzero(normal_partition.subduction(s) & (owners == s.down) & (load > 0.))
        over_of = np.where(s.bp[attached] == s.down[attached], s.bq[attached], s.bp[attached])
        pairs = {}
        for k, over in zip(attached, over_of):
            pairs.setdefault((int(s.down[k]), int(over)), []).append(k)
        (down, over), edges = max(pairs.items(), key=lambda item: s.bl[item[1]].sum())
        edges = np.asarray(edges)
        centre = np.sum(s.bmid[edges]*s.bl[edges, None], axis=0)
        centre /= np.linalg.norm(centre)
        inland = np.where((s.bq[edges] == over)[:, None], s.bn[edges], -s.bn[edges]).sum(axis=0)
        inland -= centre*(inland@centre)
        inland /= np.linalg.norm(inland)
        axis = np.cross(centre, inland)/backarc.RADIUS_KM
        s.omega[over], s.omega[down] = 5.*axis, 25.*axis
        return s, {int(u) for u, on in zip(s.plate_uid, s.active) if on}

    def _advance(self, s, original, steps, stop=None):
        real = native_engine.Simulation._forces
        solves = []

        def forces(world, dt):
            if {int(u) for u, on in zip(world.plate_uid, world.active) if on} <= original:
                return None
            solves.append(float(world.t))
            return real(world, dt)
        with patch.object(native_engine.Simulation, '_forces', forces):
            for _ in range(steps):
                s.step(2.)
                if stop is not None and stop(s):
                    break
        return solves

    def test_sea_anchor_loads_and_ruptures_an_arc_in_the_native_engine(self):
        s, original = self._upper_plate_moving_away(backarc.DRIVER_VERSION)
        ruptured = lambda world: any(r['arc_plate_uid'] is not None for r in world.backarc_basins)
        solves = self._advance(s, original, 25, ruptured)
        self.assertTrue(ruptured(s), [(r['phase'], r['loading_km']) for r in s.backarc_basins])
        row = next(r for r in s.backarc_basins if r['arc_plate_uid'] is not None)
        self.assertGreaterEqual(row['loading_km'], backarc.RUPTURE_LOADING_KM)
        self.assertGreaterEqual(row['rupture_myr']-row['started_myr'], 8.)
        self.assertIn(row['arc_plate_uid'], {int(u) for u, on in zip(s.plate_uid, s.active) if on})
        # The new geometry is re-solved by the force balance in the rupture step.
        self.assertEqual(solves, [float(s.t)])
        self.assertTrue(any('anchored slab fractured' in e['description'] for e in s.events
                            if e['type'] == 'backarc_rifting'))
        np.testing.assert_array_equal(s.trench_retreat_speed, 0.)
        frame = s.snapshot()
        mesh_history.arrays(frame)
        self.assertGreater(frame['backarc_driver_diagnostics']['anchored_length_km'], 0.)

    def test_swept_driver_cannot_load_under_the_moving_hinge(self):
        # Driver 0 reads trench_retreat_speed, identically zero without the
        # forearc sweep: the N1 failure the sea anchor replaces.
        # The same world and window under driver 1 does load, so the empty
        # driver-0 ledger is not an artefact of immature trenches or the
        # sector gate.
        anchor, original = self._upper_plate_moving_away(backarc.DRIVER_VERSION)
        self._advance(anchor, original, 3)
        rows = [r for r in anchor.backarc_basins if r['arc_plate_uid'] is None]
        self.assertTrue(rows)
        self.assertGreater(max(r['loading_rate_km_myr'] for r in rows), backarc.LOADING_EDGE_KM_MYR)
        s, original = self._upper_plate_moving_away(0)
        self._advance(s, original, 3)
        self.assertEqual(s.backarc_basins, [])
        self.assertNotIn('backarc_driver_version', s.snapshot())


if __name__ == '__main__':
    unittest.main()
