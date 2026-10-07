"""Native geometry, actual columns, ancestry, and restart evolve together."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import numpy as np
from tests.test_native_engine import world
from native_engine import Simulation
from checkpoint import write_checkpoint,read_checkpoint
import mesh_history
import crustal_structure
import structure_engine


class HybridEngineTests(unittest.TestCase):
    def test_empty_ocean_world_can_advance_without_material_vertices(self):
        s=Simulation(dict(width=96,height=48,mesh_level=2,plate_count=4,mechanics_nodes=128),
            dict(width=48,height=24,crust=np.zeros(48*24,np.uint8)))
        s.step()
        self.assertEqual(len(s.mass),0)
        mesh_history.arrays(s.snapshot())

    def test_refined_history_preserves_roots_and_actual_column_area(self):
        s=world();roots=s.parcel_patch.copy();mass=s.mass.copy()
        craton_mass=float(s.mass[s.kind==2].sum())
        initial_count=len(s.mass)
        for _ in range(8):
            s.step()
            self.assertGreaterEqual(len(s.mass),initial_count)
            np.testing.assert_allclose(s.mass*s.structure['area_factor'],s.material_surface['area_km2'],
                rtol=2e-10,atol=1e-5)
            np.testing.assert_allclose(crustal_structure.elevation(s.structure),
                structure_engine.material_height(s.kind,s.relief),atol=2e-9)
            self.assertAlmostEqual(float(s.mass[s.kind==2].sum())/craton_mass,1.,places=12)
            for uid,reference in zip(roots,mass):
                self.assertAlmostEqual(float(s.mass[s.material_lineage['root_id']==uid].sum())/reference,1.,places=11)
            mesh_history.arrays(s.snapshot())
        self.assertGreater(len(s.mass),initial_count)
        self.assertGreater(s.material_lineage['level'].max(),0)

    def test_checkpoint_after_refinement_resumes_exactly(self):
        s=world();s.step();s.step()
        compatibility=dict(engine_sha256='test',auxiliary_sources_sha256={},numpy_version=np.__version__)
        manifest=dict(config=s.config,time_myr=s.t,next_output_myr=s.t+2.)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'checkpoint.npz'
            write_checkpoint(path,s,manifest,compatibility)
            restored,_=read_checkpoint(path,compatibility,Simulation)
            s.step();restored.step()
            resumed_frame=restored.snapshot()
            for key,value in mesh_history.arrays(s.snapshot()).items():
                np.testing.assert_array_equal(value,resumed_frame[key],err_msg=key)
            self.assertEqual(s.rift_bonds,restored.rift_bonds)


if __name__=='__main__':unittest.main()
