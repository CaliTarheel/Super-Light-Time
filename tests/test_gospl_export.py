"""Independent geometry, material-budget and native goSPL package regressions.

These check the adapter contract without pretending to execute the PETSc solver.
"""
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

import numpy as np

from gospl_export import (ExportCancelled, RADIUS_M, build_history, cells_at,
                         checked_times, icosphere, interval_forcing, load_frame,
                         matched_traces, plate_rotations, sample)
from validate_gospl import validate_package


# Row-vector positive 90-degree rotation about the geographic north pole.
QUARTER_TURN = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])


def grid_points(width=32, height=16):
    lon, lat = np.meshgrid((np.arange(width)+.5)*2*np.pi/width-np.pi,
                          np.pi/2-(np.arange(height)+.5)*np.pi/height)
    return np.column_stack((np.cos(lat).ravel()*np.cos(lon).ravel(),
                            np.cos(lat).ravel()*np.sin(lon).ravel(),
                            np.sin(lat).ravel()))


def synthetic_frame(time=0., height=1000., loss=0., turn=0, field=None):
    width, rows = 32, 16
    points = grid_points(width, rows)
    transform = np.linalg.matrix_power(QUARTER_TURN, turn)
    return dict(width=width, height=rows, time_myr=float(time),
                plates=[dict(id=0, uid=71, name='Fixture continent',
                             angular_velocity=[0., 0., np.pi/4])],
                elevation=np.full(width*rows, height, np.float32) if field is None
                else np.asarray(field, np.float32).ravel(),
                plate=np.zeros(width*rows, np.int32),
                crust=np.ones(width*rows, np.uint8),
                boundary=np.zeros(width*rows, np.uint8),
                trace_id=np.arange(len(points), dtype=np.int64),
                trace_xyz=points@transform,
                trace_plate_uid=np.full(len(points), 71, np.int64),
                trace_erosion_m=np.full(len(points), loss, np.float64))


class MeshAndForcingTests(unittest.TestCase):
    def test_icosphere_is_closed_outward_connected_and_has_bounded_valence(self):
        for level in range(4):
            with self.subTest(level=level):
                vertices, faces = icosphere(level)
                self.assertEqual(vertices.shape, (10*4**level+2, 3))
                self.assertEqual(faces.shape, (20*4**level, 3))
                np.testing.assert_allclose(np.linalg.norm(vertices, axis=1), 1., atol=3e-15)
                self.assertEqual(len(np.unique(vertices, axis=0)), len(vertices))
                self.assertTrue(np.issubdtype(faces.dtype, np.integer))
                self.assertGreaterEqual(faces.min(), 0)
                self.assertLess(faces.max(), len(vertices))
                edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]],
                                                 faces[:, [2, 0]]]), axis=1)
                unique, count = np.unique(edges, axis=0, return_counts=True)
                np.testing.assert_array_equal(count, 2)
                degree = np.bincount(unique.ravel(), minlength=len(vertices))
                self.assertEqual(int(np.count_nonzero(degree == 5)), 12)
                self.assertTrue(np.all((degree == 5)|(degree == 6)))
                self.assertEqual(len(vertices)-len(unique)+len(faces), 2)
                a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
                self.assertTrue(np.all(np.einsum('ij,ij->i', np.cross(b-a, c-a), a) > 0))
                neighbors = [set() for _ in vertices]
                for a, b in unique:
                    neighbors[a].add(int(b)); neighbors[b].add(int(a))
                visited, pending = {0}, [0]
                while pending:
                    for node in neighbors[pending.pop()]-visited:
                        visited.add(node); pending.append(node)
                self.assertEqual(len(visited), len(vertices))

    def test_longitude_convention_seam_and_unique_pole_sampling(self):
        w, h = 32, 16
        field = np.arange(w*h, dtype=float).reshape(h, w)
        centers = grid_points(w, h)
        np.testing.assert_allclose(sample(field, centers), field.ravel(), atol=1e-10)
        axes = np.array([[1., 0., 0.], [0., 1., 0.], [-1., 0., 0.], [0., -1., 0.]])
        np.testing.assert_array_equal(cells_at(axes, w, h), [h//2*w+w//2, h//2*w+3*w//4,
                                                          h//2*w, h//2*w+w//4])
        poles = np.array([[0., 0., 1.], [-0., -0., 1.], [0., -0., -1.], [-0., 0., -1.]])
        np.testing.assert_allclose(sample(field, poles), [field[0].mean()]*2+[field[-1].mean()]*2)
        seam = np.array([[-1., 1e-12, 0.], [-1., -1e-12, 0.]])
        np.testing.assert_allclose(sample(field, seam)[0], sample(field, seam)[1], atol=2e-9)

    def test_finite_rigid_rotation_chord_lands_on_sphere_at_known_destination(self):
        first, second = synthetic_frame(), synthetic_frame(2, turn=1)
        vertices, _ = icosphere(2)
        fields, info = interval_forcing(vertices, first, second, dict(dt_myr=2, erosion=0))
        destination = vertices*RADIUS_M + fields['hdisp']*2_000_000.
        np.testing.assert_allclose(destination, vertices@QUARTER_TURN*RADIUS_M, atol=.7, rtol=0)
        np.testing.assert_allclose(np.linalg.norm(destination, axis=1), RADIUS_M, atol=.7, rtol=0)
        self.assertEqual(info['motion'][0]['method'], 'material_rigid_fit')
        self.assertLess(info['motion'][0]['rigid_fit_rms_m'], 1e-7)
        np.testing.assert_allclose(fields['upsub'], 0., atol=1e-12)

    def test_recorded_signed_relaxation_removes_decay_from_vertical_forcing(self):
        vertices, _ = icosphere(1)
        for signed_loss in (40., -20.):
            with self.subTest(signed_loss=signed_loss):
                # Actual prescribed uplift is +100 m, and the source independently
                # loses signed_loss to relaxation. Their combined height is saved.
                first = synthetic_frame(height=1000., loss=500.)
                second = synthetic_frame(2, height=1100.-signed_loss,
                                         loss=500.+signed_loss, turn=1)
                fields, info = interval_forcing(vertices, first, second, dict(dt_myr=2, erosion=1))
                np.testing.assert_allclose(fields['upsub']*2_000_000., 100., atol=2e-5)
                np.testing.assert_allclose(fields['geometric_rate']*2_000_000., 100.-signed_loss, atol=2e-5)
                np.testing.assert_allclose(fields['relaxation_correction']*2_000_000., signed_loss, atol=2e-5)
                self.assertTrue(np.all(fields['correction_supported']))
                self.assertEqual(info['relaxation_fallback_nodes'], 0)

    def test_advected_nonuniform_topography_does_not_become_false_uplift(self):
        points = grid_points()
        old = (1000.+400*points[:, 0]+100*points[:, 2]).reshape(16, 32)
        new = np.roll(old, 8, axis=1)
        self.assertGreater(float(np.max(np.abs(new-old))), 500.)
        first = synthetic_frame(field=old)
        second = synthetic_frame(2, turn=1, field=new)
        vertices, _ = icosphere(2)
        fields, _ = interval_forcing(vertices, first, second, dict(dt_myr=2, erosion=0))
        # Saved float32 rows can differ by an ULP when their polar mean is
        # summed after a cyclic reordering; allow 0.2 mm, not geological relief.
        np.testing.assert_allclose(fields['geometric_rate']*2_000_000., 0., atol=2e-4)
        np.testing.assert_allclose(fields['upsub']*2_000_000., 0., atol=2e-4)

    def test_nonrigid_motion_and_ocean_fallback_are_honestly_identified(self):
        first, second = synthetic_frame(), synthetic_frame(2, turn=1)
        second['trace_xyz'][:100] = first['trace_xyz'][:100]
        a, b = matched_traces(first, second)
        _, info = plate_rotations(first, second, a, b, dict(dt_myr=2))
        self.assertEqual(info[0]['method'], 'nonrigid_material_fit_approximation')
        self.assertGreater(info[0]['rigid_fit_rms_m'], 100_000.)
        first['plates'].append(dict(id=1, uid=99, name='Untracked ocean', angular_velocity=[0., 0., np.pi/4]))
        matrices, info = plate_rotations(first, second, a, b, dict(dt_myr=2))
        self.assertEqual(info[1]['method'], 'recorded_euler_approximation')
        np.testing.assert_allclose(matrices[1], QUARTER_TURN, atol=1e-14)


class NativePackageTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1]/'tmp'
        scratch.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='test-gospl-', dir=scratch)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root/'source-run'
        self.run.mkdir()
        self.manifest = dict(run_id='synthetic-history', config=dict(dt_myr=2, erosion=1),
                             frames=[dict(index=i, time_myr=10+i*2) for i in range(3)],
                             auxiliary_sources_sha256={})
        for name, content in [('engine.py', b'# Saved engine fixture\n'), ('material_geometry.py', b'# Saved helper fixture\n')]:
            (self.run/name).write_bytes(content)
            digest = hashlib.sha256(content).hexdigest()
            if name == 'engine.py': self.manifest['engine_sha256'] = digest
            else: self.manifest['auxiliary_sources_sha256'][name] = digest
        for i in range(3):
            self.write_frame(i, synthetic_frame(10+i*2, height=1000.+i*80, loss=500.+i*20, turn=i))

    def write_frame(self, index, frame):
        arrays = {key: value for key, value in frame.items() if isinstance(value, np.ndarray)}
        metadata = {key: value for key, value in frame.items() if key not in arrays}
        stem = self.run/f'frame_{index:04d}'
        np.savez_compressed(stem.with_suffix('.npz'), **arrays)
        stem.with_suffix('.json').write_text(json.dumps(metadata), encoding='utf8')

    def build(self, **kwargs):
        output = self.root/kwargs.pop('folder', 'package')
        metadata = build_history(self.run, output, self.manifest, subdivisions=1, **kwargs)
        return output, metadata

    def test_full_native_package_paths_units_history_and_provenance(self):
        output, metadata = self.build(dt_years=50_000)
        validation = validate_package(output)
        self.assertTrue(validation['passed'])
        self.assertFalse(validation['solver_run'])
        self.assertEqual(validation['intervals'], 2)
        config = json.loads((output/'input.yml').read_text())
        self.assertEqual(config['domain']['npdata'], ['input/mesh', 'v', 'c', 'z'])
        self.assertEqual(config['time'], dict(start=0, end=4_000_000, dt=50_000, tout=2_000_000))
        self.assertIs(config['domain']['fast'], False)
        self.assertIs(config['output']['makedir'], True)
        self.assertIsInstance(config['spl']['K'], float)
        fast = json.loads((output/'forcing-check.yml').read_text())
        self.assertIs(fast['domain']['fast'], True)
        self.assertEqual(fast['tectonics'], config['tectonics'])
        self.assertEqual(metadata['start_myr'], 10.)
        self.assertEqual(metadata['end_myr'], 14.)
        self.assertEqual(metadata['source_engine_sha256'], self.manifest['engine_sha256'])
        self.assertFalse(metadata['solver_run_here'])
        with zipfile.ZipFile(output/'gospl-history.zip') as archive:
            self.assertIsNone(archive.testzip())
            names = set(archive.namelist())
            self.assertTrue({'input.yml', 'forcing-check.yml', 'input/mesh.npz',
                             'input/forcing_0000.npz', 'input/forcing_0001.npz',
                             'source/manifest.json', 'source/frame-index.csv',
                             'source/model/engine.py', 'source/model/material_geometry.py',
                             'validate_export.py', 'export_metadata.json', 'validation.json'} <= names)
            self.assertFalse(any(name.endswith('.tmp') or name.startswith('/') for name in names))
            for interval, entry in enumerate(config['tectonics']):
                self.assertEqual(entry['start'], interval*2_000_000)
                self.assertEqual(entry['end'], (interval+1)*2_000_000)
                self.assertEqual(entry['hdisp'][0], entry['upsub'][0])
                with np.load(io.BytesIO(archive.read(entry['hdisp'][0]+'.npz')), allow_pickle=False) as data:
                    self.assertEqual(data['hdisp'].shape, (42, 3))
                    self.assertEqual(data['upsub'].shape, (42,))
                    np.testing.assert_allclose(data['upsub']*2_000_000., 100., atol=2e-5)
                    self.assertEqual(int(data['start_year']), entry['start'])
                    self.assertEqual(int(data['end_year']), entry['end'])
            for row in metadata['source_frames']:
                for suffix in ('npz', 'json'):
                    payload = (self.run/f'frame_{row["index"]:04d}.{suffix}').read_bytes()
                    self.assertEqual(row[suffix+'_sha256'], hashlib.sha256(payload).hexdigest())
        self.assertEqual((output/'source/model/engine.py').read_bytes(), (self.run/'engine.py').read_bytes())

    def test_timestep_must_divide_every_interval_and_times_are_whole_years(self):
        for step in (300_000, 100_000.5, 0, -2, True):
            with self.subTest(step=step), self.assertRaises(ValueError):
                checked_times(self.manifest, 0, 2, step)
        self.assertEqual(checked_times(self.manifest, 1, 2, 500_000).tolist(), [12_000_000, 14_000_000])
        for times in ([10., 12., 11.], [10., 12., 12.], [10., 12.0000005, 14.]):
            manifest = copy.deepcopy(self.manifest)
            for row, time in zip(manifest['frames'], times): row['time_myr'] = time
            with self.subTest(times=times), self.assertRaises(ValueError):
                checked_times(manifest, 0, 2, 50_000)
        with self.assertRaisesRegex(ValueError, 'divide'):
            self.build(dt_years=300_000)
        self.assertFalse((self.root/'package/gospl-history.zip').exists())

    def test_legacy_counterless_history_fails_without_an_accepted_archive(self):
        old = synthetic_frame(10.)
        del old['trace_erosion_m']
        self.write_frame(0, old)
        with self.assertRaisesRegex(ValueError, 'predates|erosion counters'):
            self.build()
        self.assertFalse((self.root/'package/gospl-history.zip').exists())

    def test_tampered_source_and_incompatible_frame_metadata_are_rejected(self):
        (self.run/'engine.py').write_text('# Unexpected source change', encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            self.build(folder='bad-source')
        self.assertFalse((self.root/'bad-source/gospl-history.zip').exists())
        self.write_frame(1, synthetic_frame(12.5, turn=1))
        with self.assertRaisesRegex(ValueError, 'times do not match'):
            self.build(folder='bad-time')
        invalid = synthetic_frame(10.)
        invalid['width'] = 30
        self.write_frame(0, invalid)
        with self.assertRaisesRegex(ValueError, '2:1'):
            load_frame(self.run, 0)

    def test_cancellation_during_conversion_and_zip_never_commits_archive(self):
        for index, stage in enumerate(('Converting interval 2', 'Validating mesh and forcing intervals', 'Packaging goSPL files')):
            cancelled = [False]
            def progress(percent, message):
                if message.startswith(stage): cancelled[0] = True
            folder = f'cancel-stage-{index}'
            with self.subTest(stage=stage), self.assertRaises(ExportCancelled):
                self.build(folder=folder, progress=progress, cancel=lambda: cancelled[0])
            self.assertTrue(cancelled[0])
            self.assertFalse((self.root/folder/'gospl-history.zip').exists())
            self.assertFalse((self.root/folder/'gospl-history.zip.tmp').exists())

    def test_validator_rejects_a_separate_unchecked_vertical_archive(self):
        output, _ = self.build()
        path = output/'input.yml'
        config = json.loads(path.read_text())
        config['tectonics'][0]['upsub'][0] = 'input/unvalidated-vertical'
        path.write_text(json.dumps(config), encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'same interval archive'):
            validate_package(output)

    def test_standalone_validator_rejects_corrupt_mesh_under_python_optimization(self):
        output, _ = self.build()
        path = output/'input/mesh.npz'
        with np.load(path, allow_pickle=False) as data:
            fields = {key: data[key] for key in data.files}
        fields['v'][0] *= .5
        np.savez_compressed(path, **fields)
        result = subprocess.run([sys.executable, '-O', str(output/'validate_export.py')],
                                capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Vertices do not lie on the declared sphere', result.stderr)


if __name__ == '__main__':
    unittest.main()
