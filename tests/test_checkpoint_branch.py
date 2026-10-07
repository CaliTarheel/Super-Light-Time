"""A branch changes future authored metadata, never the saved parent past."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import server
from checkpoint import read_checkpoint
from server import SimulationManager
from tectonics import Simulation


class BranchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.manager = SimulationManager(self.temporary.name)
        self.config = dict(width=48, height=24, mesh_level=2, coast_geometry_level=2,
                           mechanics_nodes=128, seed=12, duration_myr=4,
                           dt_myr=1, snapshot_myr=2, plate_count=4)

    def paused_parent(self):
        step = Simulation.step
        def pause(simulation, dt=None):
            step(simulation, dt)
            self.manager.pause()
        with patch.object(Simulation, 'step', pause):
            self.manager.start(self.config, None)
            self.manager.worker.join(900)
        self.assertEqual(self.manager.status()['state'], 'paused')
        return self.manager.path()

    def future_design(self, start=1.):
        return dict(version=1, enabled=True, revision=1, interventions=[dict(
            id='future-weakness', kind='weak', lon_deg=179., lat_deg=72., radius_deg=8.,
            strength=.5, start_myr=start, end_myr=4., reason='Compare future regional response')])

    def hashes(self, root):
        return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in root.iterdir() if path.is_file()}

    def test_branch_copies_history_preserves_physics_parent_and_exact_pending_scheduler(self):
        parent = self.paused_parent()
        before = self.hashes(parent)
        original, manifest = read_checkpoint(parent/'checkpoint.npz', self.manager.compatibility(), Simulation)
        branch = self.manager.branch(parent.name, self.future_design())
        root = self.manager.path()
        self.assertNotEqual(root, parent)
        self.assertEqual(branch['state'], 'paused')
        self.assertTrue(branch['can_resume'])
        self.assertEqual(branch['branch_origin']['run_id'], parent.name)
        self.assertEqual(branch['branch_origin']['time_myr'], 1.)
        derived, saved = read_checkpoint(root/'checkpoint.npz', self.manager.compatibility(), Simulation)
        self.assertEqual(saved['next_output_myr'], manifest['next_output_myr'])
        for name, value in vars(original).items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(getattr(derived, name), value, err_msg=name)
        for name in ('vertices', 'faces', 'reference_area_km2'):
            np.testing.assert_array_equal(derived.material_surface[name], original.material_surface[name], err_msg=name)
        self.assertEqual(derived.rng.bit_generator.state, original.rng.bit_generator.state)
        for path in parent.glob('frame_*'):
            self.assertEqual((root/path.name).read_bytes(), path.read_bytes())
        self.assertEqual(json.loads((root/'initial.json').read_text())['world_design'], self.future_design())
        self.assertEqual(self.hashes(parent), before)
        self.manager.resume()
        self.manager.worker.join(900)
        self.assertEqual(self.manager.status()['state'], 'complete', self.manager.status().get('error'))
        self.assertEqual([row['time_myr'] for row in self.manager.status()['frames']], [0., 1., 2., 4.])
        self.assertEqual(self.hashes(parent), before)
        self.assertEqual(self.manager.frame(-1)['world_design'], self.future_design())

    def test_past_design_and_incompatible_source_are_rejected_before_creating_branch(self):
        parent = self.paused_parent()
        folders = set(Path(self.temporary.name).iterdir())
        with self.assertRaisesRegex(ValueError, 'at or after'):
            self.manager.branch(parent.name, self.future_design(start=0.))
        with patch.object(server, 'ENGINE_SOURCE', b'# incompatible source'):
            with self.assertRaisesRegex(ValueError, 'engine source'):
                self.manager.branch(parent.name, self.future_design())
        self.assertEqual(set(Path(self.temporary.name).iterdir()), folders)
        self.assertEqual(self.manager.path(), parent)

    def test_saved_source_tampering_is_rejected_before_publishing_branch(self):
        parent = self.paused_parent()
        (parent/'engine.py').write_bytes(b'# damaged saved source')
        with self.assertRaisesRegex(ValueError, 'source is damaged'):
            self.manager.branch(parent.name, self.future_design())
        self.assertEqual(len(list(Path(self.temporary.name).iterdir())), 1)


if __name__ == '__main__':
    unittest.main()
