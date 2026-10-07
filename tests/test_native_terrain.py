"""Final pixels must query actual spherical geometry, never the review grid."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

import mesh_geometry
import native_frame_sampling
import terrain
import terrain_jobs
import continental_margin


def triangle(centre, radius):
    centre = np.asarray(centre, float); centre /= np.linalg.norm(centre)
    axis = [1., 0., 0.] if abs(centre[0]) < .9 else [0., 0., 1.]
    u = np.cross(centre, axis); u /= np.linalg.norm(u)
    v = np.cross(centre, u)
    theta = np.arange(3)*2*np.pi/3
    points = centre+radius*(np.cos(theta)[:, None]*u+np.sin(theta)[:, None]*v)
    return points/np.linalg.norm(points, axis=1)[:, None]


def native_frame():
    control = mesh_geometry.icosphere(1)
    tiny = terrain.output_points(128, 33*128+75, 33*128+76)[0]
    material = np.concatenate((triangle(tiny, .003), triangle([0., 0., 1.], .09), triangle([-1., 0., 0.], .09)))
    n = len(control['faces'])
    return dict(mesh_version=1, width=16, height=8, time_myr=200.,
        plates=[dict(id=0, uid=1), dict(id=1, uid=2)],
        elevation=np.full(128, -9999.), crust=np.zeros(128, np.uint8), boundary=np.zeros(128, np.uint8),
        mesh_vertices=control['vertices'], mesh_faces=control['faces'], mesh_area_km2=control['area_km2'],
        mesh_plate=np.zeros(n, np.int32), mesh_crust=np.zeros(n, np.uint8),
        mesh_boundary=np.zeros(n, np.uint8), mesh_age_myr=np.full(n, 80.), mesh_ocean_relief_m=np.zeros(n),
        material_vertices=material, material_faces=np.arange(9, dtype=np.int32).reshape(3, 3),
        material_face_id=np.array([30, 31, 32], np.int64), material_owner=np.ones(3, np.int32),
        material_kind=np.array([3, 2, 1], np.uint8), material_height_m=np.array([1500., 2200., 3300.]),
        material_erosion_rate_m_myr=np.zeros(3))


class NativeTerrainTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1]/'tmp'
        scratch.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def build(self, frame, name, **kwargs):
        metadata = terrain.build_terrain(frame, self.root/name, width=128, **kwargs)
        return np.load(self.root/name/'elevation_m.npy'), metadata

    def test_direct_baseline_retains_tiny_polar_and_seam_islands_absent_from_review(self):
        frame = native_frame()
        elevation, metadata = self.build(frame, 'direct', detail=0.)
        expected = native_frame_sampling.sample_frame(frame, terrain.output_points(128, 0, 128*64))['elevation']
        np.testing.assert_array_equal(elevation.ravel(), expected.astype(np.float32))
        self.assertEqual(elevation[33, 75], 1500.)
        self.assertTrue(np.any(elevation[0] == 2200.))
        self.assertTrue(np.any(elevation[:, 0] == 3300.))
        self.assertTrue(np.any(elevation[:, -1] == 3300.))
        self.assertEqual(int(frame['crust'].sum()), 0)
        self.assertFalse(metadata['baseline']['review_raster_used'])
        self.assertFalse(metadata['baseline']['regional_bias_correction'])

    def test_review_resolution_and_values_cannot_change_any_final_pixel(self):
        first = native_frame()
        second = deepcopy(first)
        second.update(width=128, height=64, elevation=np.full(8192, np.nan),
                      crust=np.full(8192, 2, np.uint8), boundary=np.full(8192, 4, np.uint8))
        a, _ = self.build(first, 'small-review', detail=1.3, band_rows=3)
        b, _ = self.build(second, 'large-review', detail=1.3, band_rows=7)
        np.testing.assert_array_equal(a, b)

    def test_rotated_pixels_sample_inverse_rotated_native_geometry_directly(self):
        frame = native_frame()
        orientation = dict(yaw=42., pitch=80., roll=-17.)
        with patch.object(terrain, 'reproject_raster', side_effect=AssertionError('No finished-map reprojection')):
            elevation, _ = self.build(frame, 'rotated', detail=0., orientation=orientation)
        points = terrain.output_points(128, 0, 8192, orientation)
        expected = native_frame_sampling.sample_frame(frame, points)['elevation'].astype(np.float32)
        np.testing.assert_array_equal(elevation.ravel(), expected)

    def test_historical_margin_reconstruction_requires_explicit_opt_in_and_records_it(self):
        frame = native_frame()
        # A continuous inherited surface supports the versioned margin adapter.
        frame.update(surface_reconstruction_version=1, arc_material_version=1, arc_surface_version=2,
                     material_arc_id=np.array([1, 0, 0], np.int64), material_arc_basal_m=np.full(3, -4000.))
        preserved = {key: value.copy() for key, value in frame.items() if isinstance(value, np.ndarray)}
        original, original_metadata = self.build(frame, 'unchanged-margins', detail=0.)
        reconstructed, metadata = self.build(frame, 'upgraded-margins', detail=0., reconstruct_margins=True)
        revised = continental_margin.upgrade_frame(frame)
        expected = native_frame_sampling.sample_frame(revised, terrain.output_points(128, 0, 8192))['elevation']
        np.testing.assert_array_equal(reconstructed.ravel(), expected.astype(np.float32))
        self.assertFalse(original_metadata['source']['reconstruct_margins'])
        self.assertTrue(metadata['source']['reconstruct_margins'])
        self.assertEqual(metadata['source']['continental_margin_version'], 1)
        self.assertTrue(metadata['source']['continental_margin_revision']['surface_only'])
        self.assertFalse(np.array_equal(original, reconstructed))
        self.assertNotIn('continental_margin_version', frame)
        for name, values in preserved.items():
            np.testing.assert_array_equal(frame[name], values)

    def test_bounded_sampling_and_texture_preserve_source_zero_contour(self):
        counts = []
        def sample(points):
            counts.append(len(points))
            return 3000.*points[:, 2]
        for name, chunk, band in [('one', 19, 1), ('seven', 32768, 7)]:
            terrain.build_sampled_terrain(sample, self.root/name, source={'source_type': 'fixture'},
                width=128, detail=2., sample_chunk=chunk, band_rows=band)
        a = np.load(self.root/'one/elevation_m.npy')
        b = np.load(self.root/'seven/elevation_m.npy')
        np.testing.assert_array_equal(a, b)
        baseline = sample(terrain.output_points(128, 0, 8192)).reshape(64, 128)
        np.testing.assert_array_equal(a >= 0, baseline >= 0)
        self.assertLessEqual(max(counts), terrain.SAMPLE_CHUNK)
        self.assertGreater(float(np.max(abs(a-baseline))), 1.)

    def test_sampled_output_cancellation_and_png_range_do_not_modify_numeric_heights(self):
        calls = 0
        def sample(points):
            nonlocal calls
            calls += 1
            return np.full(len(points), -15000.)
        with self.assertRaises(terrain.TerrainCancelled):
            terrain.build_sampled_terrain(sample, self.root/'cancel', source={}, width=128,
                                          sample_chunk=19, cancel=lambda: calls >= 2)
        self.assertFalse((self.root/'cancel/terrain_metadata.json').exists())
        result = terrain.build_sampled_terrain(sample, self.root/'range', source={}, width=32, detail=0.)
        np.testing.assert_array_equal(np.load(self.root/'range/elevation_m.npy'), -15000.)
        self.assertEqual(result['files']['heightmap_16bit.png']['clipped_pixels'], 512)

    def test_job_archives_exact_epoch_and_recursive_sampler_sources(self):
        run = self.root/'native'; run.mkdir()
        frame = native_frame()
        arrays = {key: value for key, value in frame.items() if isinstance(value, np.ndarray)}
        metadata = {key: value for key, value in frame.items() if key not in arrays}
        np.savez(run/'frame_0000.npz', **arrays)
        (run/'frame_0000.json').write_text(json.dumps(metadata))
        (run/'config.json').write_text('{"seed":37}')
        (run/'engine.py').write_text('# immutable fixture engine')
        class History:
            def _history_context(self, run_id=None):
                return run, dict(run_id='native', frame_count=1, config={'seed': 37}, engine_sha256='fixture')
        jobs = terrain_jobs.TerrainManager(self.root/'jobs')
        with patch.object(terrain_jobs, 'OUTPUT_WIDTHS', (128,)):
            job = jobs.start(History(), 0, width=128, detail=0.)
        jobs.worker.join(30)
        self.assertFalse(jobs.worker.is_alive())
        self.assertEqual(jobs.status()['state'], 'complete', jobs.status())
        with zipfile.ZipFile(jobs.download(job['job_id'], 'zip')) as archive:
            saved = json.loads(archive.read('terrain_metadata.json'))
            for suffix, digest in saved['source_frame_sha256'].items():
                self.assertEqual(hashlib.sha256(archive.read('source/frame'+suffix)).hexdigest(), digest)
            for name, digest in saved['terrain_builder_sources_sha256'].items():
                self.assertEqual(hashlib.sha256(archive.read('builder/'+name)).hexdigest(), digest)
            for helper in ('native_frame_sampling.py', 'material_reconstruction.py', 'mesh_geometry.py', 'arc_surface.py'):
                self.assertIn('builder/'+helper, archive.namelist())
        # All imports needed to interpret the selected source are available in
        # the package, even when the original application is not on sys.path.
        job_path = jobs.path(job['job_id'])
        environment = dict(os.environ); environment.pop('PYTHONPATH', None)
        code = "import json,numpy as np; from pathlib import Path; import terrain; f=json.loads(Path('../source/frame.json').read_text()); d=np.load('../source/frame.npz',allow_pickle=False); f.update({k:d[k] for k in d.files}); terrain.build_terrain(f,'../replayed',width=128,detail=0.)"
        result = subprocess.run([sys.executable, '-c', code], cwd=job_path/'builder', env=environment,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        np.testing.assert_array_equal(np.load(job_path/'replayed/elevation_m.npy'),
                                      np.load(job_path/'elevation_m.npy'))

    def test_capture_walks_transitive_local_imports_without_substituting_saved_model(self):
        source = self.root/'closure'; source.mkdir()
        for name, content in {'entry.py': 'import first\n', 'first.py': 'from second import x\n',
                              'second.py': 'import entry\nx=1\n'}.items():
            (source/name).write_text(content)
        captured = terrain_jobs.capture_builder_sources(source, ('entry.py',))
        self.assertEqual(set(captured), {'entry.py', 'first.py', 'second.py'})


if __name__ == '__main__':
    unittest.main()
