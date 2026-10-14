"""Attachment history, finite ocean support and coupled breakup work."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np

import balance_force_ledger as forces
import breakup_mode
import persistent_ocean_intake as intake
import plate_balance as pb
import checkpoint
from tests.test_effective_subduction_forces import fixture, uniform_basis
from tests.test_lite_force_breakup import paid_state_and_mode
from tests._native_spreading_oracle import material_rectangles


def attached_fixture():
    s = fixture()
    s.t = 0.
    s.config['effective_subduction']['persistent_intake'] = dict(enabled=True)
    intake.initialize(s)
    ends = np.array([[np.cos(.1), 0., -np.sin(.1)], [np.cos(.1), 0., np.sin(.1)]])
    s.native_boundary_geometry = dict(segments_start=ends[:1], segments_end=ends[1:],
        segment_normals=np.array([[0., 1., 0.]]), contact_index=np.array([0]))
    s.material_surface = dict(vertices=np.empty((0, 3)), faces=np.empty((0, 3), int))
    s.parcel_plate = np.empty(0, int)
    return s


class PersistentOceanIntakeTests(unittest.TestCase):
    def test_declaration_and_motion_alone_do_not_latch_at_zero(self):
        s = attached_fixture()
        s.normal_speed[:] = -10.
        self.assertEqual(len(pb.Balance(s, 1.).intake_constraints['rows']), 0)
        s.t = 1.
        s.native_subduction_diagnostics = dict(step_end_myr=1., removed_area_by_trench=[
            dict(trench_id=1, area_km2=0.)])
        intake.update(s)
        self.assertNotIn(intake.LATCH, s.trench_systems[0]['effective_subduction'])
        s.native_subduction_diagnostics['removed_area_by_trench'][0]['area_km2'] = 2.
        intake.update(s)
        self.assertEqual(s.trench_systems[0]['effective_subduction'][intake.LATCH], 1.)
        self.assertEqual(len(pb.Balance(s, 1.).intake_constraints['rows']), 2)

    def test_latched_trench_does_not_disappear_when_velocity_reverses(self):
        s = attached_fixture(); s.t = 1.
        s.trench_systems[0]['effective_subduction'][intake.LATCH] = 1.
        before = pb.Balance(s, 1.).intake_constraints['rows']
        s.normal_speed[:] = 20.; s.t = 2.
        s.native_subduction_diagnostics = dict(step_end_myr=2., removed_area_by_trench=[])
        intake.update(s)
        np.testing.assert_array_equal(pb.Balance(s, 1.).intake_constraints['rows'], before)

    def test_actual_incoming_continent_releases_the_wet_constraint(self):
        s = attached_fixture(); s.t = 1.
        s.trench_systems[0]['effective_subduction'][intake.LATCH] = 1.
        vertices, faces = material_rectangles([(-1000., 1000., -1000., 1000.)])
        s.material_surface = dict(vertices=vertices, faces=faces, vertex_owner=np.zeros(len(vertices), int))
        s.parcel_plate = np.zeros(len(faces), int)
        self.assertEqual(len(pb.Balance(s, 1.).intake_constraints['rows']), 0)
        # Upper-plate material must not block incoming ocean.
        s.parcel_plate[:] = 1
        self.assertEqual(len(pb.Balance(s, 1.).intake_constraints['rows']), 2)

    def test_common_rotation_and_boundary_storage_reversal_leave_intake_unchanged(self):
        s = attached_fixture(); s.t = 1.
        s.trench_systems[0]['effective_subduction'][intake.LATCH] = 1.
        base = pb.Balance(s, 1.).intake_constraints['rows']
        np.testing.assert_allclose(base@np.tile([.4, -.2, .6], 2), 0., atol=1e-15)
        s.bp, s.bq = s.bq.copy(), s.bp.copy()
        s.ba, s.bb = s.bb.copy(), s.ba.copy()
        s.bn *= -1.
        s.native_boundary_geometry['segment_normals'] *= -1.
        np.testing.assert_array_equal(pb.Balance(s, 1.).intake_constraints['rows'], base)

    def test_missing_policy_preserves_historical_motion_and_refuses_silent_switch(self):
        s = fixture(); b = pb.Balance(s, 1.); b.solve()
        self.assertEqual(len(b.intake_constraints['rows']), 0)
        self.assertFalse(hasattr(s, intake.FIELD))
        s = attached_fixture()
        s.config['effective_subduction']['persistent_intake']['enabled'] = False
        with self.assertRaisesRegex(ValueError, 'silently disabled'):
            pb.Balance(s, 1.)

    def test_attachment_policy_and_latch_survive_checkpoint_and_frame_validation(self):
        s = attached_fixture(); s.t = 1.; s.rng = np.random.default_rng(42)
        s.trench_systems[0]['effective_subduction'][intake.LATCH] = 1.
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'intake.npz'
            checkpoint.write_checkpoint(path, s, {'config': s.config}, {})
            restored, _ = checkpoint.read_checkpoint(path, {}, SimpleNamespace)
        self.assertEqual(intake.snapshot(restored), intake.snapshot(s))
        frame = dict(time_myr=1., **intake.snapshot(restored))
        intake.validate_frame(frame)
        frame['persistent_ocean_intake_trenches'][0]['started_myr'] = 2.
        with self.assertRaises(ValueError):
            intake.validate_frame(frame)
        with self.assertRaises(ValueError):
            intake.validate_frame(dict(persistent_ocean_intake_version=1))

    def test_reactions_are_exported_and_recomputed_for_a_daughter_mode(self):
        s, model, _ = paid_state_and_mode()
        normal, _ = model._relative_rows(model.trench_edges)
        model.intake_constraints = dict(rows=-normal, minimum=np.full(len(normal), .1), ports=[
            dict(down=int(model.slab_owner[e]), over=int(model.bq[e]), point=model.radial[e].tolist())
            for e in model.trench_edges])
        model.solve(); s.omega = model.rotation()
        full = model._evaluate(model.x, .01)
        first = model._evaluate(model.x, .01, want_hessian=False)
        self.assertEqual(full[0], first[0])
        np.testing.assert_array_equal(full[1], first[1])
        self.assertIsNone(first[2])
        self.assertGreater(float(model.intake_reaction.sum()), 0.)
        ledger = forces.export(model, model.x)
        projected = forces.compile_mode(ledger, uniform_basis(model))
        np.testing.assert_allclose(projected.intake_constraints['rows'], -normal, atol=1e-14)
        self.assertLess(ledger['diagnostics']['gradient_agreement'], 1e-9)
        self.assertLess(np.linalg.norm(ledger['generalized_force'])/np.linalg.norm(model._force_reference), 1e-8)
        cells = np.arange(6); piece = np.array([1, 1, 1, 0, 0, 0], bool)
        cut = dict(edges=np.array([[0, 3]]), mid=np.array([[1., 0., 0.]]),
                   length_m=np.array([1e6]), cut=np.array([True]))
        found = breakup_mode.solve(s, 0, cells, piece, np.array([0., 0., 1.]), cut,
            tuple(np.array([1e11]) for _ in range(3)), model, ledger=ledger)
        self.assertTrue(found['opening_mode_supported'], found)
        self.assertGreater(found['predicted_opening_cm_yr_mean'], 0.)
        self.assertLess(found['mode_work']['closure_relative'], 1e-8)
        self.assertIn('persistent_intake_source_power_w', found['mode_work'])


if __name__ == '__main__':
    unittest.main()
