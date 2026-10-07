"""Exercise a real run, saved restart, and complete history export together."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import hashlib

import numpy as np
from PIL import Image

import server
from server import SimulationManager, capture_auxiliary_sources
from terrain_jobs import TerrainManager, read_saved_auxiliary_sources


class PersistenceTests(unittest.TestCase):
    def test_run_reload_and_export_preserve_time_units_and_initial_map(self):
        scratch = Path(__file__).resolve().parents[1] / 'tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            manager = SimulationManager(directory)
            config = dict(width=48, height=24, mesh_level=2, mechanics_nodes=128,
                          duration_myr=4, dt_myr=2, snapshot_myr=2, plate_count=6)
            painting = server.native(server.make_initial(config))
            result = manager.start(config, painting)
            manager.worker.join(timeout=30)
            self.assertFalse(manager.worker.is_alive())
            state = manager.status()
            self.assertEqual(state['state'], 'complete', state.get('error'))
            self.assertEqual([item['time_myr'] for item in state['frames']], [0, 2, 4])
            reloaded = SimulationManager(directory)
            self.assertEqual(reloaded.status()['run_id'], result['run_id'])
            np.testing.assert_array_equal(manager.frame(2)['elevation'], reloaded.frame(2)['elevation'])
            with reloaded.export_file(1) as exported, zipfile.ZipFile(exported) as archive:
                self.assertEqual(len([n for n in archive.namelist() if n.endswith('.npz')]), 3)
                metadata = json.loads(archive.read('raster_metadata.json'))
                self.assertEqual(metadata['time_myr'], 2)
                values = np.load(io.BytesIO(archive.read('elevation_m.npy')), allow_pickle=False)
                image = np.asarray(Image.open(io.BytesIO(archive.read('heightmap_16bit.png')))).astype(np.int32)
                np.testing.assert_allclose(image - 12000, values, atol=.5)
                self.assertEqual(hashlib.sha256(archive.read('model/tectonics.py')).hexdigest(), state['engine_sha256'])
                self.assertEqual(set(state['auxiliary_sources_sha256']), set(server.AUXILIARY_SOURCES))
                for name, fingerprint in state['auxiliary_sources_sha256'].items():
                    payload = archive.read('model/' + name)
                    self.assertEqual(payload, server.AUXILIARY_SOURCES[name])
                    self.assertEqual(hashlib.sha256(payload).hexdigest(), fingerprint)
                initial = json.loads(archive.read('initial.json'))
                # Input artwork is preserved exactly. Its native mesh import
                # and review raster are separate spatial representations.
                self.assertEqual(initial, painting)

    def test_capture_follows_only_local_imports_and_freezes_dependency_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fracture = b'from crust_transport import remap\nimport numpy\n'
            transport = b'def remap():\n    return 1\n'
            (root / 'fracture.py').write_bytes(fracture)
            (root / 'crust_transport.py').write_bytes(transport)
            (root / 'unrelated.py').write_bytes(b'UNRELATED = True\n')
            captured = capture_auxiliary_sources(b'from fracture import split\n', root)
            self.assertEqual(captured, {'fracture.py': fracture, 'crust_transport.py': transport})
            (root / 'fracture.py').write_bytes(b'# later version\n')
            self.assertEqual(captured['fracture.py'], fracture)
            self.assertEqual(capture_auxiliary_sources(b'import numpy\n', root), {})

    def test_history_and_terrain_packages_use_saved_helpers_and_keep_legacy_empty(self):
        scratch = Path(__file__).resolve().parents[1] / 'tmp'
        scratch.mkdir(exist_ok=True)
        original_sources = {'fracture.py': b'# recorded fracture dependency\n',
                            'crust_transport.py': b'# recorded transport dependency\n'}
        for legacy in (False, True):
            with self.subTest(legacy=legacy), tempfile.TemporaryDirectory(dir=scratch) as directory:
                root = Path(directory)
                manager = SimulationManager(root / 'runs')
                with patch.object(server, 'AUXILIARY_SOURCES', original_sources):
                    manager.start(dict(width=48, height=24, mesh_level=2,
                                       mechanics_nodes=128, duration_myr=2,
                                       snapshot_myr=2, plate_count=6), None)
                    manager.worker.join(timeout=30)
                self.assertFalse(manager.worker.is_alive())
                self.assertEqual(manager.status()['state'], 'complete', manager.status().get('error'))
                if legacy:
                    # A pre-helper history can coexist with unrelated files. Its
                    # absent provenance mapping must not discover those files.
                    manager.current.pop('auxiliary_sources_sha256')
                    old_engine = b'# legacy self-contained engine\n'
                    (manager.path() / 'engine.py').write_bytes(old_engine)
                    manager.current['engine_sha256'] = hashlib.sha256(old_engine).hexdigest()
                    manager.persist()
                recorded = manager.status().get('auxiliary_sources_sha256', {})
                replacement = {'fracture.py': b'# newly installed version\n',
                               'future_helper.py': b'# added after this run\n'}
                with patch.object(server, 'AUXILIARY_SOURCES', replacement):
                    with manager.export_file(1) as stream, zipfile.ZipFile(stream) as archive:
                        packaged = {name for name in archive.namelist()
                                    if name.startswith('model/') and name.endswith('.py')
                                    and name != 'model/tectonics.py'}
                        self.assertEqual(packaged, {'model/' + name for name in recorded})
                        for name, fingerprint in recorded.items():
                            payload = archive.read('model/' + name)
                            self.assertEqual(payload, original_sources[name])
                            self.assertEqual(hashlib.sha256(payload).hexdigest(), fingerprint)
                    jobs = TerrainManager(root / 'terrain')
                    jobs.start(manager, 1, width=2048, detail=0)
                    jobs.worker.join(timeout=30)
                self.assertFalse(jobs.worker.is_alive())
                job = jobs.status()
                self.assertEqual(job['state'], 'complete', job.get('error'))
                self.assertEqual(job['source_auxiliary_sources_sha256'], recorded)
                with zipfile.ZipFile(jobs.download(job['job_id'], 'zip')) as archive:
                    metadata = json.loads(archive.read('terrain_metadata.json'))
                    self.assertEqual(metadata['source_auxiliary_sources_sha256'], recorded)
                    self.assertEqual(metadata['source_engine_sha256'], manager.status()['engine_sha256'])
                    self.assertEqual(metadata['source_frame'], 1)
                    self.assertEqual(archive.read('source/frame.npz'),
                                     (manager.path() / 'frame_0001.npz').read_bytes())
                    packaged = {name for name in archive.namelist()
                                if name.startswith('source/') and name.endswith('.py')
                                and name != 'source/engine.py'}
                    self.assertEqual(packaged, {'source/' + name for name in recorded})
                    for name, fingerprint in recorded.items():
                        self.assertEqual(hashlib.sha256(archive.read('source/' + name)).hexdigest(), fingerprint)

    def test_missing_tampered_or_unsafe_saved_dependencies_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fingerprint = hashlib.sha256(b'# recorded\n').hexdigest()
            expected = {'fracture.py': fingerprint}
            with self.assertRaisesRegex(ValueError, 'missing or unreadable'):
                read_saved_auxiliary_sources(root, expected)
            (root / 'fracture.py').write_bytes(b'# altered\n')
            with self.assertRaisesRegex(ValueError, 'does not match'):
                read_saved_auxiliary_sources(root, expected)
            for invalid in ({'../fracture.py': fingerprint}, {'engine.py': fingerprint},
                            {'fracture.py': 'wrong'}, {12: fingerprint}, None):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    read_saved_auxiliary_sources(root, invalid)
            self.assertEqual(read_saved_auxiliary_sources(root, {}), {})


if __name__ == '__main__':
    unittest.main()
