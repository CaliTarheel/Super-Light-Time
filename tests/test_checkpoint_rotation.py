"""A damaged latest generation cannot displace the last valid backup."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from checkpoint import METADATA_KEY, previous_checkpoint, read_checkpoint, write_checkpoint


class CheckpointRotationTests(unittest.TestCase):
    def test_valid_header_with_inconsistent_state_does_not_replace_last_good_backup(self):
        compatibility = dict(engine_sha256='test', auxiliary_sources_sha256={}, numpy_version=np.__version__)
        simulation = SimpleNamespace(t=0., config=dict(seed=12), rng=np.random.default_rng(12),
                                     material=np.arange(24, dtype=float))
        manifest = dict(config=simulation.config)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'checkpoint.npz'
            write_checkpoint(target, simulation, manifest, compatibility)
            original = target.read_bytes()
            simulation.t = 1.
            write_checkpoint(target, simulation, manifest, compatibility)
            self.assertEqual(previous_checkpoint(target).read_bytes(), original)
            with np.load(target, allow_pickle=False) as archive:
                arrays = dict(archive)
            metadata = json.loads(arrays[METADATA_KEY].tobytes())
            metadata['time_myr'] = 99.  # Valid ZIP/JSON, invalid state agreement.
            arrays[METADATA_KEY] = np.frombuffer(json.dumps(metadata).encode(), np.uint8)
            np.savez_compressed(target, **arrays)
            simulation.t = 2.
            write_checkpoint(target, simulation, manifest, compatibility)
            self.assertEqual(previous_checkpoint(target).read_bytes(), original)
            self.assertEqual(read_checkpoint(target, compatibility, SimpleNamespace)[0].t, 2.)
            self.assertEqual(read_checkpoint(previous_checkpoint(target), compatibility, SimpleNamespace)[0].t, 0.)


if __name__ == '__main__':
    unittest.main()
