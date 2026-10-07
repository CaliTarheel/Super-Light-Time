"""Portable collision-harness contracts and deliberately failing controls.

These small component tests do not establish native whole-world collision
fidelity. The subprocess cases exercise actual typed checkpoint continuation.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import checkpoint
import collision_contacts as contacts
import enhanced_rifting
import material_surface
import native_material_evolution as evolution
from benchmarks.collision_architecture import runner
from ridge_geometry import rotate


def orientation():
    """Independent Rodrigues construction, with column-vector convention."""
    axis = np.array([.3, -.7, .2])
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    angle = 1.13
    return np.eye(3)*np.cos(angle)+(1.-np.cos(angle))*np.outer(axis, axis)+np.sin(angle)*skew


def overlapping_fixture(*, rotation=None):
    """A declared already-contacting initial state, made by rigid displacement.

    Both bodies move equally from the separated builder without changing rock
    inventory. This is not a claim that the component trajectory crossed its
    first contact in one admissible 4.5 Myr step.
    """
    s = runner.fixture(nx=3, ny=3, rotation=rotation)
    mesh = s.material_surface
    mesh['vertices'] = rotate(mesh['vertices'], s.omega[mesh['vertex_owner']]*4.5)
    material_surface.refresh_geometry(mesh)
    s.pos = material_surface.face_centres(mesh)
    contacts.refresh(s)
    return s


class CollisionArchitectureTests(unittest.TestCase):
    def assert_tree_equal(self, actual, expected):
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                with self.subTest(field=key):
                    self.assert_tree_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for a, e in zip(actual, expected):
                self.assert_tree_equal(a, e)
        else:
            self.assertEqual(actual, expected)

    def test_fixture_has_matched_mirrored_geometry_columns_and_loading(self):
        s = runner.fixture(nx=4, ny=3)
        mesh = s.material_surface
        nv, nf = len(mesh['vertices'])//2, len(mesh['faces'])//2
        np.testing.assert_array_equal(mesh['vertices'][nv:], mesh['vertices'][:nv]@runner.MIRROR)
        np.testing.assert_array_equal(mesh['faces'][nf:]-nv, mesh['faces'][:nf])
        np.testing.assert_allclose(mesh['area_km2'][nf:], mesh['area_km2'][:nf], rtol=2e-15)
        for value in s.structure.values():
            np.testing.assert_array_equal(value[:nf], value[nf:])
        np.testing.assert_array_equal(s.omega[1], s.omega[0]@runner.MIRROR)
        self.assertEqual(s.collision_contacts, [])
        triangles = mesh['vertices'][mesh['faces']]
        signed = np.einsum('ij,ij->i', triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2]))
        self.assertTrue(np.all(signed > 0.))
        self.assertEqual(len(np.unique(s.parcel_patch)), len(s.parcel_patch))

    def test_force_free_rest_advances_without_inventing_motion_or_material(self):
        s = runner.fixture('rest', nx=3, ny=3)
        before = deepcopy(s)
        initial = runner.observe(s)
        np.testing.assert_array_equal(s.omega, 0.)
        self.assertTrue(all(initial['gates'].values()), initial['gates'])
        runner.advance(s, .5)
        row = runner.observe(s)
        self.assertTrue(all(row['gates'].values()), row['gates'])
        self.assertEqual(s.t, .5)
        np.testing.assert_allclose(s.material_surface['vertices'], before.material_surface['vertices'], rtol=0., atol=5e-12)
        np.testing.assert_allclose(s.structure['thickness_km'], before.structure['thickness_km'], rtol=0., atol=1e-8)
        np.testing.assert_array_equal(s.parcel_plate, before.parcel_plate)
        np.testing.assert_array_equal(s.parcel_patch, before.parcel_patch)
        np.testing.assert_array_equal(s.parcel_burial_myr, before.parcel_burial_myr)
        np.testing.assert_array_equal(s.parcel_collision_suture, before.parcel_collision_suture)
        self.assertEqual(s.collision_contacts, [])

    def test_authored_strength_reaches_native_viscosity_without_initial_force_change(self):
        states = [runner.fixture(case, nx=3, ny=3) for case in ('symmetric', 'weak0', 'weak1')]
        for index, s in enumerate(states):
            expected = np.ones(len(s.mass))
            if index:
                expected[s.parcel_plate == index-1] = .4
            np.testing.assert_array_equal(enhanced_rifting.material_viscosity(s), expected)
            np.testing.assert_array_equal(s.kind, 1)
            for name, value in s.structure.items():
                np.testing.assert_array_equal(value, states[0].structure[name])
        rotations = [runner.assemble(s)[0].rotation() for s in states]
        for actual in rotations[1:]:
            # Current plate assembly does not consume sheet viscosity before
            # geometric/GPE feedback; this documents the split's actual scope.
            np.testing.assert_array_equal(actual, rotations[0])

    def test_local_inventory_gate_rejects_small_face_corruption_hidden_globally(self):
        s = runner.fixture(nx=5, ny=5)
        row = runner.observe(s)
        self.assertTrue(row['gates']['local_volume'])
        volume = s.benchmark['initial_volume']
        face = int(np.argmin(volume))
        local_limit = runner.TOL['volume_atol_km3']+runner.TOL['volume_rtol']*volume[face]
        # One face exceeds its local allowance but stays inside owner/global
        # allowances. This ensures the test detects missing locality checks.
        s.structure['thickness_km'][face] += 4.*local_limit/s.material_surface['area_km2'][face]
        result = runner.gates(s, row)
        self.assertFalse(result['local_volume'])
        self.assertTrue(result['aggregate_volume'])

    def test_real_contact_deformation_requires_exactly_one_column_update(self):
        s = overlapping_fixture()
        self.assertGreater(s.collision_diagnostics['pair_overlap_area_km2'], 0.)
        row = runner.observe(s)
        good = deepcopy(s)
        runner.advance(good, .1)
        self.assertTrue(runner.gates(good, row)['local_volume'])
        # Deliberately omit the column transaction after real production
        # contact/gravity deformation. Never mutate production checkpoints.
        evolution.advect(s, .1)
        self.assertGreater(float(np.max(np.abs(s.geometric_log_area))), 1e-8)
        self.assertFalse(runner.gates(s, row)['local_volume'])

    def test_generic_orientation_rotates_the_force_solution_and_preserves_work(self):
        q = orientation()
        first = overlapping_fixture()
        # Rotate the represented contacting geometry, keeping the same
        # nonsmooth incidence. Rebuilding it via an independent sequence of
        # rigid rotations has a separate known-failure regression below.
        turned = deepcopy(first)
        turned.material_surface['vertices'] = first.material_surface['vertices']@q.T
        material_surface.refresh_geometry(turned.material_surface)
        turned.pos = material_surface.face_centres(turned.material_surface)
        turned.omega = first.omega@q.T
        turned.benchmark['free_omega'] = first.benchmark['free_omega']@q.T
        contacts.refresh(turned)
        self.assertGreater(first.collision_diagnostics['pair_overlap_area_km2'], 0.)
        model, external, gpe, basal = runner.assemble(first)
        other, ext2, gpe2, basal2 = runner.assemble(turned)
        self.assertGreater(float(np.linalg.norm(gpe)), 0.)
        self.assertGreater(float(np.linalg.norm(model.interface_stiffness)), 0.)
        np.testing.assert_allclose(other.rotation(), model.rotation()@q.T, rtol=2e-12, atol=1e-14)
        np.testing.assert_allclose(ext2.reshape(2, 3), external.reshape(2, 3)@q.T, rtol=2e-13, atol=1e-5)
        self.assertAlmostEqual(float(ext2@other.x)/float(external@model.x), 1., delta=2e-13)
        self.assertAlmostEqual(float(other.x@basal2@other.x)/float(model.x@basal@model.x), 1., delta=2e-13)
        np.testing.assert_allclose(gpe2.reshape(2, 3), gpe.reshape(2, 3)@q.T, rtol=2e-10, atol=1e-5)

    def test_rebuilt_rotated_contact_preserves_gpe_at_near_coincident_edges(self):
        # Regression: physically equivalent construction in a
        # rotated frame changes represented geometry by ulps near coincident
        # clipping edges. At 86a79b3 this changes omega by 1.6898e-5 rad/Myr
        # (up to 3.75% componentwise), despite unchanged ordering, matching
        # overlap area, interface covariance and converged force solves.
        # Differentiated endpoint clipping now preserves the cancelling work.
        # Keep the original strict oracle; do not widen its tolerance.
        q = orientation()
        first, turned = overlapping_fixture(), overlapping_fixture(rotation=q)
        model, _, gpe, _ = runner.assemble(first)
        other, _, changed, _ = runner.assemble(turned)
        np.testing.assert_allclose(changed.reshape(2, 3), gpe.reshape(2, 3)@q.T, rtol=2e-10, atol=1e-5)
        np.testing.assert_allclose(other.rotation(), model.rotation()@q.T, rtol=2e-12, atol=1e-14)

    def test_cached_area_cannot_hide_a_corrupt_column_inventory(self):
        s = runner.fixture(nx=3, ny=3)
        row = runner.observe(s)
        s.material_surface['area_km2'][0] *= 1.01
        s.structure['thickness_km'][0] /= 1.01
        # Cached A*T is unchanged, but independently recomputed spherical area
        # reveals both the stale cache and missing material.
        result = runner.gates(s, row)
        self.assertFalse(result['area_cache'])
        self.assertFalse(result['local_volume'])

    def test_missing_poststep_solver_or_policy_is_not_accepted(self):
        s = runner.fixture('rest', nx=3, ny=3)
        runner.observe(s)
        runner.advance(s, .1)
        row = runner.observe(s)
        self.assertTrue(row['gates']['sheet_stationarity'])
        s.benchmark['last_step']['solver'] = None
        self.assertFalse(runner.gates(s, row)['sheet_stationarity'])
        del s.same_sheet_nonpenetration_version
        self.assertFalse(runner.gates(s, row)['policies'])

    def test_failed_inventory_gate_stops_before_advancing_or_writing_checkpoint(self):
        s = runner.fixture('rest', nx=3, ny=3)
        s.structure['thickness_km'][0] *= 1.00001
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(runner, 'advance', side_effect=AssertionError('Failed state advanced')) as advance:
                report = runner.run_case(s, dt=.1, end=.2, output=directory)
            advance.assert_not_called()
            self.assertFalse(report['complete'])
            self.assertFalse(report['numerical_gates_pass'])
            self.assertIn('local_volume', report['error']['message'])
            self.assertFalse((Path(directory)/'state.npz').exists())
            self.assertTrue((Path(directory)/'report.json').exists())
            self.assertEqual(s.t, 0.)

    def test_checkpoint_write_failure_does_not_report_success(self):
        s = runner.fixture('rest', nx=3, ny=3)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(runner.checkpoint, 'write_checkpoint', side_effect=OSError('Injected storage failure')):
                report = runner.run_case(s, dt=.1, end=0., output=directory)
            self.assertFalse(report['complete'])
            self.assertFalse(report['numerical_gates_pass'])
            self.assertNotIn('checkpoint_written', report)
            self.assertEqual(report['error']['message'], 'Injected storage failure')
            saved = json.loads((Path(directory)/'report.json').read_text())
            self.assertFalse(saved['complete'])

    def checkpoint_continuation(self, s):
        runner.observe(s)
        runner.advance(s, .1)
        before = deepcopy(s)
        expected = deepcopy(s)
        runner.observe(expected)
        runner.advance(expected, .1)
        runner.observe(expected)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory/'source.npz'
            output = directory/'continued'
            manifest = runner.source_manifest()
            checkpoint.write_checkpoint(source, s, dict(config=s.config), runner.compatibility(manifest))
            environment = os.environ.copy()
            environment.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
            result = subprocess.run([sys.executable, '-B', '-m', 'benchmarks.collision_architecture.runner',
                '--resume', str(source), '--dt', '.1', '--end', str(expected.t), '--output', str(output)],
                cwd=runner.ROOT, env=environment, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout+'\n'+result.stderr)
            report = json.loads((output/'report.json').read_text())
            self.assertTrue(report['complete'])
            self.assertTrue(report['numerical_gates_pass'])
            restored, _ = checkpoint.read_checkpoint(output/'state.npz', runner.compatibility(runner.source_manifest()), type(s))
            # Wall times and observation rows legitimately differ. Compare the
            # actual physical fields and persistent contact history exactly.
            for key in ('material_surface', 'structure', 'omega', 'pos', 'mass', 'relief',
                        'parcel_patch', 'parcel_plate', 'parcel_collision_sheet',
                        'parcel_burial_myr', 'parcel_collision_suture', 'parcel_exposed_fraction',
                        'parcel_rift_seed_weakness', 'collision_contacts', 't'):
                with self.subTest(state_field=key):
                    self.assert_tree_equal(getattr(restored, key), getattr(expected, key))
            self.assert_tree_equal(restored.rng.bit_generator.state, expected.rng.bit_generator.state)
            for key in runner.POLICIES:
                self.assertEqual(getattr(restored, key), getattr(expected, key))
        # Saving/continuing in the child cannot alter the parent source state.
        self.assert_tree_equal(s.structure, before.structure)
        np.testing.assert_array_equal(s.material_surface['vertices'], before.material_surface['vertices'])

    def test_precontact_checkpoint_continuation_in_fresh_process(self):
        self.checkpoint_continuation(runner.fixture(nx=3, ny=3))

    def test_postcontact_checkpoint_continuation_in_fresh_process(self):
        s = overlapping_fixture()
        self.assertGreater(s.collision_diagnostics['pair_overlap_area_km2'], 0.)
        self.checkpoint_continuation(s)


if __name__ == '__main__':
    unittest.main()
