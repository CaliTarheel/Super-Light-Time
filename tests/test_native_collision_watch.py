"""Durable native probe resumes exact accepted state and keeps failed chunks out."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

import checkpoint
from experiments import native_collision_watch as watch
from native_engine import Simulation


class NativeCollisionWatchTests(unittest.TestCase):
    def test_two_continent_preset_has_distinct_checkpoint_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'two-continent.npz'
            result=watch.run(path,.01,chunk_myr=.01,max_source_step_myr=.01,
                preset='two-continent')
            self.assertEqual(result['accepted_chunks'],1)
            self.assertLess(abs(result['history'][0]['slab_excess_mass_relative_residual']),1e-12)
            self.assertEqual(result['history'][0]['active_contact_count'],0)
            state,manifest=checkpoint.read_checkpoint(path,watch.compatibility(),Simulation)
            self.assertEqual(manifest['preset'],'two-continent')
            self.assertGreater(float(state.material_surface['area_km2'][
                state.parcel_plate==1].sum()),1e6)
            with self.assertRaisesRegex(ValueError,'different experiment or step policy'):
                watch.run(path,.02,chunk_myr=.01,max_source_step_myr=.01,
                    preset='single-incoming')

    def test_final_chunk_absorbs_roundoff_residue(self):
        near=.44999999999999996
        self.assertEqual(watch._chunk_limit(near,.5,.05),.5-near)

    def test_resume_from_accepted_chunk_and_preserve_checkpoint_on_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'native-watch.npz'
            first=watch.run(path,.01,chunk_myr=.01,max_source_step_myr=.01)
            self.assertEqual(first['accepted_chunks'],1)
            self.assertTrue(path.is_file())
            state,manifest=checkpoint.read_checkpoint(path,watch.compatibility(),Simulation)
            self.assertEqual(state.t,.01)
            self.assertEqual(manifest['history'][-1]['end_myr'],state.t)
            second=watch.run(path,.02,chunk_myr=.01,max_source_step_myr=.01)
            self.assertEqual(second['accepted_chunks'],2)
            self.assertEqual(second['time_myr'],.02)
            self.assertEqual(second['history'][0],first['history'][0])
            direct_path=Path(folder)/'direct.npz'
            direct=watch.run(direct_path,.02,chunk_myr=.01,max_source_step_myr=.01)
            for resumed,direct_chunk in zip(second['history'],direct['history']):
                self.assertEqual({k:v for k,v in resumed.items() if k!='wall_seconds'},
                                 {k:v for k,v in direct_chunk.items() if k!='wall_seconds'})
            resumed_state,_=checkpoint.read_checkpoint(path,watch.compatibility(),Simulation)
            direct_state,_=checkpoint.read_checkpoint(direct_path,watch.compatibility(),Simulation)
            np.testing.assert_array_equal(resumed_state.material_surface['vertices'],
                                          direct_state.material_surface['vertices'])
            np.testing.assert_array_equal(resumed_state.omega,direct_state.omega)
            self.assertEqual(resumed_state.trench_systems,direct_state.trench_systems)
            previous,_=checkpoint.read_checkpoint(checkpoint.previous_checkpoint(path),
                                                  watch.compatibility(),Simulation)
            self.assertEqual(previous.t,.01)
            with patch.object(watch,'advance',side_effect=ValueError('injected next-chunk failure')):
                with self.assertRaisesRegex(ValueError,'injected next-chunk failure'):
                    watch.run(path,.03,chunk_myr=.01,max_source_step_myr=.01)
            durable,manifest=checkpoint.read_checkpoint(path,watch.compatibility(),Simulation)
            self.assertEqual(durable.t,.02)
            self.assertEqual(len(manifest['history']),2)
            with self.assertRaisesRegex(ValueError,'different experiment or step policy'):
                watch.run(path,.03,chunk_myr=.02,max_source_step_myr=.01)


if __name__=='__main__':unittest.main()
