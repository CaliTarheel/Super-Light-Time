"""Frozen native backend survives a real short pause/resume transaction."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import json
import numpy as np
import server
from tectonics import Simulation

class NativeCheckpointResumeTests(unittest.TestCase):
    def test_saved_frames_match_continuous_integration_after_fresh_manager_resume(self):
        config=dict(width=48,height=24,mesh_level=2,coast_geometry_level=2,
                    mechanics_nodes=128,adaptive_refinement=0,seed=12,
                    duration_myr=4,dt_myr=1,snapshot_myr=2,plate_count=4)
        config=server.fresh_world.fresh_world_request(config)
        expected=list(Simulation(config,server.native(server.make_initial(config))).snapshots())
        scratch=Path(__file__).resolve().parents[1]/'tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temporary:
            manager=server.SimulationManager(temporary,workers=1)
            real_step=Simulation.step
            def pause(sim,dt=None):
                real_step(sim,dt)
                if sim.t>=1:manager.pause()
            with patch.object(Simulation,'step',pause):
                manager.start(config,None)
                manager.worker.join(120)
            self.assertEqual(manager.status()['state'],'paused',manager.status())
            old_files={p.name:p.read_bytes() for p in manager.path().glob('frame_*')}
            restored=server.SimulationManager(temporary,workers=1)
            self.assertTrue(restored.status()['can_resume'])
            self.assertEqual(restored.status()['flow_backend'],manager.compatibility()['flow_backend'])
            restored.resume()
            restored.worker.join(120)
            self.assertEqual(restored.status()['state'],'complete',restored.status())
            times=[frame['time_myr'] for frame in restored.status()['frames']]
            self.assertEqual(times,[0.,1.,2.,4.])
            for name,payload in old_files.items():self.assertEqual((restored.path()/name).read_bytes(),payload)
            for snapshot in expected:
                index=times.index(snapshot['time_myr'])
                with np.load(restored.path()/f'frame_{index:04d}.npz',allow_pickle=False) as data:
                    for key in data.files:
                        np.testing.assert_array_equal(data[key],np.asarray(snapshot[key],dtype=data[key].dtype),err_msg=key)
                saved=json.loads((restored.path()/f'frame_{index:04d}.json').read_text())
                for key,value in saved.items():
                    if key not in ('index','pause_frame'):
                        self.assertEqual(value,server.native(snapshot[key]),key)

if __name__=='__main__':unittest.main()
