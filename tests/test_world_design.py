"""Authored controls preserve physical constraints and portable provenance."""
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import zipfile

import numpy as np

import world_design as design
from deforming_regions import deform
from orientation import orient_initial, rotation_matrix
try:
    from .test_deforming_regions import fixture
except ImportError:
    from test_deforming_regions import fixture


def region(**changes):
    return dict(dict(id='interior', kind='protected', lon_deg=-2., lat_deg=0., radius_deg=1.,
                     strength=1., start_myr=1., end_myr=3., reason='Keep this interior coherent.'), **changes)


def state(surface=None, **changes):
    return SimpleNamespace(config=dict(world_design=design.normalize(dict(enabled=True, interventions=[region(**changes)]))),
                           material_surface=surface, t=2.)


class WorldDesignTests(unittest.TestCase):
    def test_portable_round_trip_is_deep_and_rejects_silent_input_changes(self):
        original = design.normalize(dict(enabled=True, revision=7, interventions=[region(lon_deg=180.)]))
        restored = design.normalize(json.loads(json.dumps(original)))
        self.assertEqual(original, restored)
        self.assertEqual(restored['interventions'][0]['lon_deg'], -180.)
        restored['interventions'][0]['reason'] = 'Edited'
        self.assertNotEqual(restored, original)
        invalid = [dict(version=2), dict(enabled=1), dict(extra=True), dict(revision=.5),
                   dict(interventions=[region(strength=float('nan'))]), dict(interventions=[region(radius_deg=0.)]),
                   dict(interventions=[region(end_myr=1.)]), dict(interventions=[region(reason=' ')]),
                   dict(interventions=[region(), region()])]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                design.normalize(value)

    def test_date_line_and_polar_caps_use_true_spherical_geometry(self):
        def xyz(lon, lat):
            return design.center_xyz(dict(lon_deg=lon, lat_deg=lat))
        seam = region(lon_deg=179., radius_deg=5.)
        weights = design.cap_weights(np.array([xyz(-179, 0), xyz(177, 0), xyz(150, 0)]), seam)
        np.testing.assert_array_equal(weights, [1., 1., 0.])
        pole = region(lat_deg=90., radius_deg=5.)
        weights = design.cap_weights(np.array([xyz(lon, 88.) for lon in [-179, -90, 0, 90, 179]]), pole)
        np.testing.assert_array_equal(weights, 1.)
        matrix = rotation_matrix(dict(pitch=90, roll=27, yaw=145))
        points = np.array([xyz(-179, 0), xyz(174.5, 0), xyz(150, 0)])
        np.testing.assert_allclose(design.cap_weights(points, seam),
            design.cap_weights(points@matrix, design.rotate_region(seam, matrix)), atol=1e-12)

    def test_disabled_and_out_of_interval_are_bitwise_baseline(self):
        surface, omega, boundaries = fixture()
        baseline = deform(surface, omega, boundaries, 2.)
        sim = state(surface)
        for mode in ['disabled', 'before', 'ended', 'zero']:
            sim.config['world_design']['enabled'] = mode != 'disabled'
            sim.t = 0. if mode == 'before' else 3. if mode == 'ended' else 2.
            sim.config['world_design']['interventions'][0]['strength'] = 0. if mode == 'zero' else 1.
            options = design.deformation_options(sim)
            self.assertEqual(options, {})
            result = deform(surface, omega, boundaries, 2., **options)
            for key in ('vertices', 'areal_strain', 'residual_velocity_km_myr'):
                np.testing.assert_array_equal(baseline[key], result[key])

    def test_full_protection_keeps_affected_faces_rigid_and_preserves_area_volume(self):
        surface, omega, boundaries = fixture()
        before = copy.deepcopy(surface)
        sim = state(surface)
        options = design.deformation_options(sim)
        result = deform(surface, omega, boundaries, 2., **options)
        protected = options['vertex_protected']
        self.assertGreater(int(protected.sum()), 0)
        np.testing.assert_array_equal(result['vertices'][protected], result['rigid_vertices'][protected])
        np.testing.assert_array_equal(result['areal_strain'][protected[surface['faces']].all(axis=1)], 0.)
        np.testing.assert_allclose(result['face_area_km2']*35./result['area_ratio'], before['area_km2']*35., rtol=3e-13)
        for key in before:
            if isinstance(before[key], np.ndarray):
                np.testing.assert_array_equal(before[key], surface[key])

    def test_small_protected_cap_inside_a_face_is_not_lost_between_vertices(self):
        surface, _, _ = fixture()
        triangle = surface['vertices'][surface['faces'][50]]
        centre = triangle.sum(axis=0)
        centre /= np.linalg.norm(centre)
        row = region(lon_deg=float(np.degrees(np.arctan2(centre[1], centre[0]))),
                     lat_deg=float(np.degrees(np.arcsin(centre[2]))), radius_deg=.25)
        np.testing.assert_array_equal(design.cap_weights(triangle, row), 0.)
        self.assertTrue(design.protected_face_mask(surface, row)[50])

    def test_weak_regions_only_scale_real_boundary_response_and_cannot_soften_cratons(self):
        surface, omega, boundaries = fixture(craton=True)
        sim = state(surface, kind='weak', radius_deg=8.)
        options = design.deformation_options(sim)
        self.assertLessEqual(options['vertex_response'].max(), 1.5)
        baseline = deform(surface, omega, boundaries, 2.)
        result = deform(surface, omega, boundaries, 2., **options)
        craton = np.unique(surface['faces'][surface['face_kind'] == 2])
        np.testing.assert_array_equal(result['vertices'][craton], result['rigid_vertices'][craton])
        # Geometry/column caps may saturate the realized strain in either case;
        # the admitted target response still grows, without relaxing those caps.
        original_response = np.linalg.norm(baseline['commanded_residual_velocity_km_myr'], axis=1).max()
        authored_response = np.linalg.norm(result['commanded_residual_velocity_km_myr'], axis=1).max()
        self.assertGreater(authored_response, original_response*1.4)
        omega[:] = [.002, .003, .004]
        quiet = deform(surface, omega, boundaries, 2., **options)
        np.testing.assert_array_equal(quiet['areal_strain'], 0.)
        np.testing.assert_array_equal(quiet['vertices'], quiet['rigid_vertices'])

    def test_author_events_transitions_and_snapshot_provenance_survive_restore(self):
        events = []
        sim = state()
        sim.t = 0.
        sim._record = lambda kind, message, **extra: events.append(dict(kind=kind, time=sim.t, **extra))
        design.initialize(sim)
        self.assertEqual(design.next_transition(sim), 1.)
        sim.t = 1.
        design.record_transitions(sim)
        checkpoint = copy.deepcopy(sim)
        design.record_transitions(checkpoint)
        self.assertEqual([event['kind'] for event in events].count('authored_region_started'), 1)
        self.assertEqual(design.next_transition(checkpoint), 3.)
        sim.t = 3.
        design.record_transitions(sim)
        self.assertEqual(events[-1]['kind'], 'authored_region_ended')
        self.assertEqual(events[-1]['details']['origin'], 'authored')
        self.assertEqual(design.snapshot_metadata(sim)['world_design_diagnostics']['active_at_epoch_ids'], [])
        self.assertEqual(design.snapshot_metadata(sim)['world_design'], sim.config['world_design'])

    def test_initial_globe_rotation_preserves_region_footprint_and_ids(self):
        control = design.normalize(dict(enabled=True, interventions=[region(lon_deg=179., lat_deg=45.)]))
        world = dict(width=8, height=4, crust=np.zeros(32, np.uint8), world_design=control)
        angles = dict(yaw=70., pitch=50., roll=-17.)
        rotated = orient_initial(world, angles)
        row = rotated['world_design']['interventions'][0]
        self.assertEqual(row['id'], 'interior')
        self.assertEqual(row['start_myr'], 1.)
        np.testing.assert_allclose(design.center_xyz(row), design.center_xyz(control['interventions'][0])@rotation_matrix(angles), atol=1e-14)
        self.assertEqual(world['world_design'], control)

    def test_deformation_rejects_misaligned_or_unbounded_design_options(self):
        surface, omega, boundaries = fixture()
        for options in (dict(vertex_protected=[True]), dict(vertex_response=[1.]),
                        dict(vertex_response=np.full(len(surface['vertices']), 1.51))):
            with self.assertRaises(ValueError):
                deform(surface, omega, boundaries, 2., **options)

    def test_native_manager_records_exact_intervals_and_reloads_portable_design(self):
        from server import SimulationManager
        from tectonics import make_initial
        control = design.normalize(dict(enabled=True, revision=3, interventions=[
            region(kind='weak', start_myr=.75, end_myr=1.25, radius_deg=12.)]))
        config = dict(width=48, height=24, mesh_level=2, coast_geometry_level=2,
                      mechanics_nodes=128, adaptive_refinement=0, duration_myr=2,
                      dt_myr=2, snapshot_myr=2, plate_count=6, seed=12)
        initial = dict(make_initial(config), world_design=control)
        scratch = Path(__file__).resolve().parents[1]/'tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            self.assertTrue(Path(directory).resolve().is_relative_to(scratch.resolve()))
            manager = SimulationManager(directory)
            result = manager.start(config, initial)
            worker = manager.worker
            if worker:
                worker.join(timeout=40)
                self.assertFalse(worker.is_alive(), 'Focused 2 Myr fixture did not complete.')
            self.assertEqual(manager.status()['state'], 'complete', manager.status().get('error'))
            path = manager.path()
            self.assertEqual(json.loads((path/'initial.json').read_text())['world_design'], control)
            self.assertEqual(json.loads((path/'config.json').read_text())['world_design'], control)
            self.assertIn('world_design.py', manager.status()['auxiliary_sources_sha256'])
            frame = manager.frame(-1)
            self.assertEqual(frame['world_design'], control)
            transitions = [(event['type'], event['time_myr']) for event in frame['events']
                           if event['type'] in ('authored_region_started', 'authored_region_ended')]
            self.assertEqual(transitions, [('authored_region_started', .75), ('authored_region_ended', 1.25)])
            restored = SimulationManager(directory)
            self.assertEqual(restored.status()['run_id'], result['run_id'])
            self.assertEqual(restored.frame(-1)['world_design'], control)
            rotated = restored.frame(-1, dict(yaw=80, pitch=20))
            self.assertNotEqual(rotated['world_design']['interventions'][0]['lon_deg'], control['interventions'][0]['lon_deg'])
            with zipfile.ZipFile(io.BytesIO(restored.export(-1, dict(yaw=80, pitch=20)))) as archive:
                exported_config = json.loads(archive.read('config.json'))
                exported_initial = json.loads(archive.read('initial.json'))
                exported_manifest = json.loads(archive.read('manifest.json'))
                self.assertEqual(exported_config['world_design'], exported_initial['world_design'])
                self.assertEqual(exported_manifest['config'], exported_config)
                self.assertEqual(json.loads(archive.read('source/config.json'))['world_design'], control)

    def test_branch_design_accepts_future_controls_and_rejects_past_reinterpretation(self):
        baseline = design.normalize()
        future = design.normalize(dict(enabled=True, interventions=[region(start_myr=50, end_myr=100)]))
        self.assertEqual(design.validate_branch(baseline, future, 50), future)
        with self.assertRaises(ValueError):
            design.validate_branch(baseline, future, 51)
        historical = design.normalize(dict(enabled=True, interventions=[region(start_myr=0, end_myr=100)]))
        augmented = copy.deepcopy(historical)
        augmented['interventions'].append(region(id='future', start_myr=50, end_myr=100))
        self.assertEqual(design.validate_branch(historical, augmented, 50), augmented)
        for invalid in (baseline, future, dict(historical, enabled=False)):
            with self.assertRaises(ValueError):
                design.validate_branch(historical, invalid, 50)


if __name__ == '__main__':
    unittest.main()
