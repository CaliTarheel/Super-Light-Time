"""Ocean resolution must not change the physical starting world or RNG draw."""
from copy import deepcopy
import unittest
import numpy as np
from native_engine import Simulation,make_initial


class InitialControlIndependenceTests(unittest.TestCase):
    def test_fixed_coast_artwork_keeps_material_and_seeded_motion_exact(self):
        config=dict(width=48,height=24,coast_geometry_level=2,mesh_level=2,
                    plate_count=4,seed=37,mechanics_nodes=128,adaptive_refinement=0)
        initial=make_initial(config)
        original=deepcopy(initial)
        states=[Simulation(dict(config,mesh_level=level),initial) for level in (2,3,4)]
        baseline=states[0]
        for other in states[1:]:
            self.assertNotEqual(other.n,baseline.n)
            self.assertEqual(other.initial_geometry_diagnostics['coast_geometry_level'],2)
            self.assertTrue(other.initial_geometry_diagnostics['material_seed_independent_of_control'])
            for name in ('vertices','faces','face_id','face_owner','face_kind','reference_area_km2'):
                np.testing.assert_array_equal(other.material_surface[name],baseline.material_surface[name],err_msg=name)
            for name in ('mass','pos','kind','relief','parcel_plate','parcel_patch','parcel_craton',
                         'trace_xyz','trace_patch','trace_kind','omega','mantle','centres'):
                np.testing.assert_array_equal(getattr(other,name),getattr(baseline,name),err_msg=name)
            for name in baseline.structure:
                np.testing.assert_array_equal(other.structure[name],baseline.structure[name],err_msg=name)
            self.assertEqual(other.rng.bit_generator.state,baseline.rng.bit_generator.state)
        self.assertEqual(set(initial),set(original))
        for name in initial:
            if isinstance(initial[name],np.ndarray):
                np.testing.assert_array_equal(initial[name],original[name],err_msg=name)
            else:self.assertEqual(initial[name],original[name])

if __name__=='__main__':unittest.main()
