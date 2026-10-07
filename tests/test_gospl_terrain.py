"""Evolved solver elevations enter the same final-terrain jobs without noise."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

import gospl_results
import mesh_geometry
import terrain
import terrain_jobs


class GosplTerrainTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1]/'tmp'; scratch.mkdir(exist_ok=True)
        self.scratch = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        mesh = mesh_geometry.icosphere(1)
        self.source = dict(source_type='gospl_result', vertices=mesh['vertices'], faces=mesh['faces'],
                           elevation_m=2000*mesh['vertices'][:, 0]-500.,
                           metadata=dict(time_years=4_000_000., selected_epoch=2, radius_m=6_371_000.,
                                         source_files=[dict(path='source.h5', sha256='original-hdf-fingerprint')],
                                         source_export={'orientation': dict(yaw=80., pitch=30., roll=0.)},
                                         orientation_in_coordinates=True))
        self.inspected = dict(source_type='gospl_result', root=str(self.root),
                              epochs=[dict(index=2, time_years=4_000_000., parts=1, descriptor='source.xmf')], default_epoch=2)

    def test_solver_default_exact_height_archive_and_viewer(self):
        jobs = terrain_jobs.TerrainManager(self.root/'jobs')
        loaded = []
        def loader(path, epoch, **kwargs):
            loaded.append((path, epoch))
            kwargs['progress'](1.)
            return deepcopy(self.source)
        with patch.object(gospl_results, 'inspect_result', return_value=self.inspected), \
             patch.object(gospl_results, 'load_result', side_effect=loader), \
             patch.object(terrain_jobs, 'OUTPUT_WIDTHS', (128,)):
            job = jobs.start_gospl_result(self.root, width=128)
            jobs.worker.join(30)
        self.assertFalse(jobs.worker.is_alive())
        state = jobs.status()
        self.assertEqual(state['state'], 'complete', state)
        self.assertEqual(loaded, [(str(self.root.resolve()), 2)])
        self.assertEqual(state['detail'], 0.)
        self.assertEqual(state['time_myr'], 4.)
        # The source's existing export rotation must not be applied again.
        expected = gospl_results.sample_height(self.source, terrain.output_points(128, 0, 8192)).astype(np.float32)
        values = np.load(jobs.download(job['job_id'], 'npy'))
        np.testing.assert_array_equal(values.ravel(), expected)
        self.assertEqual(jobs.sample(job['job_id'], 15, 8)['elevation_m'], float(values[8, 15]))
        with zipfile.ZipFile(jobs.download(job['job_id'], 'zip')) as archive:
            metadata = json.loads(archive.read('terrain_metadata.json'))
            canonical = archive.read('source/gospl_result.npz')
            self.assertEqual(hashlib.sha256(canonical).hexdigest(), metadata['source_result']['canonical_result_sha256'])
            for name in ('gospl_results.py', 'gospl_hdf_reader.py', 'mesh_geometry.py'):
                self.assertIn('builder/'+name, archive.namelist())
            self.assertEqual(metadata['source_result']['source_files'][0]['sha256'], 'original-hdf-fingerprint')
            self.assertEqual(metadata['baseline']['maximum_added_detail_m'], 0.)

    def test_explicit_repositioning_is_direct_sampling_of_evolved_mesh(self):
        orientation = dict(yaw=-50., pitch=81., roll=17.)
        source = self.source
        prepared = gospl_results.prepare(source)
        terrain.build_sampled_terrain(lambda points: gospl_results.sample_height(source, points, prepared),
            self.root/'rotated', source=source['metadata'], width=128, orientation=orientation)
        values = np.load(self.root/'rotated/elevation_m.npy')
        expected = gospl_results.sample_height(source, terrain.output_points(128, 0, 8192, orientation))
        np.testing.assert_array_equal(values.ravel(), expected.astype(np.float32))

    def test_import_cancellation_has_no_completed_derived_products(self):
        jobs = terrain_jobs.TerrainManager(self.root/'jobs')
        def cancel(*args, **kwargs):
            jobs.stop.set()
            raise gospl_results.ResultCancelled('cancelled fixture')
        with patch.object(gospl_results, 'inspect_result', return_value=self.inspected), \
             patch.object(gospl_results, 'load_result', side_effect=cancel), \
             patch.object(terrain_jobs, 'OUTPUT_WIDTHS', (128,)):
            job = jobs.start_gospl_result(self.root, width=128)
            jobs.worker.join(30)
        self.assertEqual(jobs.status()['state'], 'cancelled')
        self.assertFalse((jobs.path(job['job_id'])/'terrain_metadata.json').exists())
        self.assertFalse((jobs.path(job['job_id'])/'terrain.zip').exists())


if __name__ == '__main__':
    unittest.main()
