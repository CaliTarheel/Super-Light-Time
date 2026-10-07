"""The live map and saved goSPL path share the repaired arc surface."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from native_engine import Simulation
import native_frame_sampling as sampling
import gospl_export
import server
from orientation import rotation_matrix
from validate_gospl import validate_package


def island():
    s=Simulation(dict(width=96,height=48,mesh_level=2,plate_count=4,dt_myr=2,
                      snapshot_myr=2,duration_myr=4,seed=37),
        dict(width=48,height=24,crust=np.zeros(48*24,np.uint8)))
    cell=int(np.argmin(np.linalg.norm(s.xyz-[-1.,0.,0.],axis=1)))
    s._add_arc_crust(np.array([cell]),np.array([100000.]))
    s._rasterize();s._boundaries();s._update_surface_domains()
    return s,cell


class ArcPipelineTests(unittest.TestCase):
    def test_live_snapshot_and_native_sampling_agree_including_submerged_aprons(self):
        s,_=island()
        for step in range(2):
            frame=s.snapshot()
            lon,lat=np.meshgrid((np.arange(s.w)+.5)*2*np.pi/s.w-np.pi,
                np.pi/2-(np.arange(s.h)+.5)*np.pi/s.h)
            points=np.column_stack((np.cos(lat.ravel())*np.cos(lon.ravel()),
                np.cos(lat.ravel())*np.sin(lon.ravel()),np.sin(lat.ravel())))
            sampled=sampling.sample_frame(frame,points)
            np.testing.assert_array_equal(sampled['elevation'].astype(np.float32),frame['elevation'])
            np.testing.assert_array_equal(sampled['crust'],frame['crust'])
            self.assertTrue(np.any((sampled['crust']==3)&(sampled['elevation']<0)))
            probes=np.vstack((s.pos,s.material_surface['vertices']))
            self.assertTrue(np.any(sampling.sample_frame(frame,probes)['elevation']>0))
            if not step:s.step()

    def test_rotated_package_retains_real_arc_geometry_profiles_and_finite_forcing(self):
        s,cell=island();first=s.snapshot()
        s.step();s._add_arc_crust(np.array([cell]),np.array([50000.]))
        s._rasterize();s._boundaries();s._update_surface_domains();second=s.snapshot()
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);manager=server.SimulationManager(root/'runs')
            manager.current=dict(run_id='arc-pipeline',frame_count=2,
                frames=[dict(time_myr=0.),dict(time_myr=2.)],config=s.config)
            manager.path().mkdir(parents=True)
            manager.save_frame(0,first);manager.save_frame(1,second)
            pose=dict(yaw=31.,pitch=71.,roll=-14.)
            metadata=gospl_export.build_history(manager.path(),root/'gospl',manager.current,
                subdivisions=2,dt_years=100000,orientation=pose)
            self.assertTrue(validate_package(root/'gospl')['passed'])
            self.assertIn('arc_surface.py',metadata['native_sampling_sources_sha256'])
            points,_=gospl_export.icosphere(2)
            expected=sampling.sample_frame(first,points@rotation_matrix(pose).T)['elevation'].astype(np.float32)
            with np.load(root/'gospl/input/mesh.npz') as saved:np.testing.assert_array_equal(saved['z'],expected)
            with np.load(root/'gospl/input/forcing_0000.npz') as saved:
                self.assertTrue(all(np.isfinite(saved[key]).all() for key in saved.files))
                np.testing.assert_allclose(saved['upsub'],saved['geometric_rate']+saved['relaxation_correction'],rtol=2e-6,atol=1e-10)


if __name__ == '__main__':unittest.main()
