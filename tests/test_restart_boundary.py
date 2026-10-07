"""Unevolved restart provenance cannot masquerade as an accepted gravity step."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import threading
import unittest

import checkpoint
import gravity_constraints_schema as schema
import mesh_history
import native_engine
import native_frame_sampling
import production_restart_migration as migration
import restart_boundary as boundary
import server


def frame_for(s):
    return dict(time_myr=s.t, stats=dict(integration_steps=s.steps),
        mesh_version=1, material_mechanics_version=1, gravity_constraint_version=1,
        production_restart_migration=deepcopy(s.production_restart_migration),
        deformation_diagnostics=deepcopy(s.deformation_diagnostics),
        plate_balance_diagnostics=deepcopy(getattr(s, 'plate_balance_diagnostics', {})))


class RestartBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        initial = native_engine.Simulation(dict(width=48, height=24,
            mesh_level=2, coast_geometry_level=2, plate_count=4, mechanics_nodes=128,
            seed=37, physics_profile='reviewed_v1', primordial_subduction={'enabled': True}))
        initial.t = 145.98
        initial.steps = 74
        initial.step(.02)
        assert initial.t == 146. and initial.steps == 75
        # The historical checkpoint contains precisely this incomplete-interval
        # discriminator. No fallback is installed or executed by this fixture.
        initial.deformation_diagnostics = dict(model='historical fallback',
            gravitational_relaxation=dict(gravity_constraint_version=1,
                requested_dt_myr=2., completed_dt_myr=0., accepted_time_fraction=0.,
                skipped_gravity_myr=2., internal_steps=[], substeps=0))
        cls.state, _ = migration.migrate_production(initial)

    def test_boundary_preserves_failed_history_and_does_not_mutate_state(self):
        s = deepcopy(self.state)
        before = deepcopy(vars(s))
        frame = boundary.project_snapshot(s, frame_for(s))
        schema.validate_frame(frame)
        schema.validate_gravity_policies([frame])
        old = frame[boundary.FIELD]['inherited_diagnostics']['deformation_diagnostics']
        self.assertEqual(old, before['deformation_diagnostics'])
        self.assertEqual(old['gravitational_relaxation']['completed_dt_myr'], 0.)
        self.assertEqual(s.deformation_diagnostics, before['deformation_diagnostics'])
        self.assertEqual(s.plate_balance_diagnostics, before['plate_balance_diagnostics'])
        self.assertEqual(s.rng.bit_generator.state, before['rng'].bit_generator.state)
        self.assertEqual((s.t, s.steps), (146., 75))
        self.assertNotIn('gravitational_relaxation', frame['deformation_diagnostics'])
        self.assertNotIn('plate_balance_diagnostics', frame)
        json.dumps(frame, allow_nan=False)

    def test_boundary_rejects_mutated_time_steps_receipt_and_interval_claims(self):
        original = boundary.project_snapshot(self.state, frame_for(self.state))
        changes = (
            lambda f: f.update(time_myr=148.),
            lambda f: f['stats'].update(integration_steps=76),
            lambda f: f['production_restart_migration'].update(epoch_myr=144.),
            lambda f: f['production_restart_migration'].update(epoch_steps=74),
            lambda f: f['deformation_diagnostics'].update(gravitational_relaxation={}),
            lambda f: f.update(plate_balance_diagnostics={}),
            lambda f: f[boundary.FIELD]['inherited_diagnostics'].clear(),
            lambda f: f.update(gravity_constraint_version=0),
        )
        for change in changes:
            with self.subTest(change=change):
                frame = deepcopy(original)
                change(frame)
                with self.assertRaises(ValueError):
                    schema.validate_frame(frame)
                with self.assertRaises(ValueError):
                    schema.validate_gravity_policies([frame])

    def test_same_epoch_with_advanced_counter_cannot_emit_boundary(self):
        s = deepcopy(self.state)
        s.steps += 1
        with self.assertRaisesRegex(ValueError, 'different step counter'):
            boundary.project_snapshot(s, frame_for(s))

    def test_later_frame_has_no_exception_and_incomplete_gravity_still_fails(self):
        s = deepcopy(self.state)
        s.t += 2.
        s.steps += 1
        frame = frame_for(s)
        self.assertIs(boundary.project_snapshot(s, frame), frame)
        self.assertNotIn(boundary.FIELD, frame)
        with self.assertRaisesRegex(ValueError, 'full physical interval'):
            schema.validate_frame(frame)

    def test_native_snapshot_saved_read_and_sampler_retain_boundary_record(self):
        s = deepcopy(self.state)
        inherited = deepcopy(s.deformation_diagnostics)
        # Match the full seed preparation's derived-cache refresh order.
        s._rasterize()
        s._boundaries()
        frame = s.snapshot()
        mesh_history.arrays(frame)
        native_frame_sampling.prepare(frame)
        self.assertEqual(s.deformation_diagnostics, inherited)
        with tempfile.TemporaryDirectory() as directory:
            manager = server.SimulationManager.__new__(server.SimulationManager)
            manager.root = Path(directory)
            manager.current = dict(run_id='boundary', frames=[], frame_count=1)
            manager.lock = threading.RLock()
            (Path(directory)/'boundary').mkdir()
            manager.save_frame(0, frame)
            restored = manager.frame(0, include_native=True)
            self.assertEqual(restored[boundary.FIELD], frame[boundary.FIELD])
            self.assertEqual(restored['production_restart_migration'], frame['production_restart_migration'])
            native_frame_sampling.prepare(restored)
            rotated = manager.frame(0, orientation=dict(yaw=23., pitch=7., roll=0.), include_native=True)
            self.assertEqual(rotated[boundary.FIELD], frame[boundary.FIELD])
            native_frame_sampling.prepare(rotated)
            checkpoint.write_checkpoint(Path(directory)/'state.npz', s, dict(config=s.config), {})
            resumed, _ = checkpoint.read_checkpoint(Path(directory)/'state.npz', {}, native_engine.Simulation)
            self.assertEqual(resumed.deformation_diagnostics, inherited)
            self.assertEqual(resumed.snapshot()[boundary.FIELD], frame[boundary.FIELD])

    def test_first_actual_evolution_automatically_returns_to_strict_schema(self):
        s = deepcopy(self.state)
        s.step(.02)
        frame = s.snapshot()
        self.assertNotIn(boundary.FIELD, frame)
        schema.validate_frame(frame)
        schema.validate_gravity_policies([frame])
        gravity = frame['deformation_diagnostics']['gravitational_relaxation']
        self.assertEqual(gravity['accepted_time_fraction'], 1.)
        gravity['completed_dt_myr'] = 0.
        with self.assertRaisesRegex(ValueError, 'full physical interval'):
            schema.validate_frame(frame)

    def test_source_capture_includes_boundary_validator_and_projection(self):
        self.assertIn('restart_boundary.py', server.AUXILIARY_SOURCES)


if __name__ == '__main__':
    unittest.main()
