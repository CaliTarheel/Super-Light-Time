"""Marked arc policy, pending inventory and source closure survive persistence."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import arc_emplacement_geometry
import checkpoint
import mesh_history
import native_arc_material
import native_frame_sampling
import orientation
import server
from test_native_processes import ocean_fixture
from test_arc_source_geometry import add,positions_in_cell


class ArcEmplacementHistoryTests(unittest.TestCase):
    def test_initial_and_emplaced_snapshots_validate_through_shared_consumers(self):
        s=ocean_fixture()
        for emitted in (False,True):
            if emitted:
                add(s,positions_in_cell(s)[:1],[100.])
                # Complete the engine's normal post-emplacement derived
                # coverage/domain stages before asking for a public epoch.
                s._rasterize();s._boundaries();s._update_surface_domains()
            frame=s.snapshot()
            self.assertEqual(frame['arc_emplacement_version'],1)
            arc_emplacement_geometry.validate_frame(frame)
            mesh_history.arrays(frame)
            native_frame_sampling.prepare(frame)
            rotated=orientation.orient_frame(frame,dict(yaw=52.,pitch=37.,roll=-19.))
            arc_emplacement_geometry.validate_frame(rotated)
            if emitted:
                matrix=orientation.rotation_matrix(dict(yaw=52.,pitch=37.,roll=-19.))
                first=frame['arc_material_diagnostics']['emplacement_geometry']['sources'][0]['geometry_xyz']
                second=rotated['arc_material_diagnostics']['emplacement_geometry']['sources'][0]['geometry_xyz']
                np.testing.assert_allclose(second,np.asarray(first)@matrix,atol=2e-16)
            for version in (True,False,1.,'1',2,-1):
                bad=dict(frame,material_mechanics_version=version)
                with self.subTest(mechanics=version):
                    with self.assertRaises(ValueError): mesh_history.arrays(bad)
                    with self.assertRaises(ValueError): native_frame_sampling.prepare(bad)

    def test_real_typed_checkpoint_and_recursive_capture_retain_new_policy(self):
        s=ocean_fixture();point=positions_in_cell(s)[:1]
        add(s,point,[.0004])
        compatibility=dict(engine_sha256='fixture',auxiliary_sources_sha256={},numpy_version=np.__version__)
        manifest=dict(run_id='arc-policy-fixture',config=s.config,frames=[dict(time_myr=0.)],frame_count=1,state='paused')
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'checkpoint.npz'
            checkpoint.write_checkpoint(path,s,manifest,compatibility)
            restored,_=checkpoint.read_checkpoint(path,compatibility,type(s))
        self.assertEqual(restored.native_arc_emplacement_version,1)
        self.assertEqual(restored.native_arc_pending['emplacement_version'],1)
        self.assertEqual(restored.native_arc_pending['source_column_km'],25.)
        for key in ('xyz','owner','area'):
            np.testing.assert_array_equal(restored.native_arc_pending[key],s.native_arc_pending[key])
        for world in (s,restored): add(world,point,[.0007])
        np.testing.assert_array_equal(restored.mass,s.mass)
        np.testing.assert_array_equal(restored.material_surface['vertices'],s.material_surface['vertices'])
        sources=server.capture_auxiliary_sources((server.ROOT/'tectonics.py').read_bytes())
        for name in ('native_arc_material.py','arc_emplacement_geometry.py'):
            self.assertEqual(sources[name],(server.ROOT/name).read_bytes())


if __name__=='__main__': unittest.main()
