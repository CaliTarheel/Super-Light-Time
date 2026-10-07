"""Force-derived slab traction drives the existing continental rift membrane."""
from copy import deepcopy
import math
import unittest

import numpy as np

import native_engine
import progressive_rifting
import rift_material
import rift_traction
import slab_memory


def traction_world(*,mobility=12.):
    config=dict(width=48,height=24,mesh_level=2,coast_geometry_level=2,
                plate_count=4,mechanics_nodes=128,seed=37,
                physics_profile='reviewed_v1',subduction_response='moving_hinge_v1',
                primordial_subduction={'enabled':True},
                rift_traction=dict(enabled=True,reference_strength_n_per_m=1e15,
                    mobility_km_myr=mobility,reach_km=3000.,
                    max_equivalent_speed_km_myr=200.,boundary_motion_fraction=0.))
    return native_engine.Simulation(config)


class RiftTractionTests(unittest.TestCase):
    def test_config_requires_reviewed_moving_hinge_and_single_rift_mechanics_mode(self):
        with self.assertRaisesRegex(ValueError,'reviewed_v1'):
            native_engine.validate_config(dict(rift_traction={'enabled':True}))
        with self.assertRaisesRegex(ValueError,'moving_hinge_v1'):
            native_engine.validate_config(dict(physics_profile='reviewed_v1',
                rift_traction={'enabled':True}))
        with self.assertRaisesRegex(ValueError,'enhanced_rifting'):
            native_engine.validate_config(dict(physics_profile='reviewed_v1',
                subduction_response='moving_hinge_v1',
                rift_traction={'enabled':True},enhanced_rifting={'enabled':True}))

    def test_actual_conserved_slab_line_force_loads_continental_material(self):
        s=traction_world()
        mesh=rift_material.refresh(s)
        loads,weights,report=rift_traction.loading(s,mesh)
        owners,line_mass,_=slab_memory.line_load(s)
        active=(owners>=0)&(line_mass>0.)&(np.asarray(s.bl)>1e-10)
        expected=float(np.max(9.81*line_mass[active]*math.sin(
            math.radians(slab_memory.SUBDUCTION_DIP_DEG)),initial=0.))
        self.assertTrue(report['enabled'])
        self.assertTrue(report['moving_hinge_reaction'])
        self.assertGreater(report['trench_edges'],0)
        self.assertGreater(report['loaded_nodes'],0)
        self.assertGreater(np.count_nonzero(weights),0)
        self.assertAlmostEqual(report['max_line_force_n_per_m']/expected,1.,places=13)
        self.assertGreater(report['integrated_slab_force_n'],0.)
        loaded=weights>0.
        np.testing.assert_allclose(np.sum(loads[loaded]*mesh['xyz'][loaded],axis=1),0.,atol=2e-13)
        # The inherited slabs descend on the primordial ocean; moving-hinge
        # reciprocal work is what loads the overriding continental material.
        incoming=set(map(int,np.unique(owners[active])))
        loaded_owners=set(map(int,np.unique(mesh['owners'][loaded])))
        self.assertTrue(loaded_owners-incoming)

    def test_compliance_scales_equivalent_loading_without_changing_physical_force(self):
        s=traction_world(mobility=12.)
        mesh=rift_material.refresh(s)
        a,_,first=rift_traction.loading(s,mesh)
        s.config['rift_traction']['mobility_km_myr']=6.
        b,_,second=rift_traction.loading(s,mesh)
        self.assertEqual(first['max_line_force_n_per_m'],second['max_line_force_n_per_m'])
        self.assertEqual(first['integrated_slab_force_n'],second['integrated_slab_force_n'])
        self.assertAlmostEqual(second['max_equivalent_speed_km_myr']/
                               first['max_equivalent_speed_km_myr'],.5,places=12)
        np.testing.assert_allclose(b,a*.5,rtol=2e-12,atol=2e-14)

    def test_progressive_rifting_records_force_loading_and_uses_it_as_soft_constraint(self):
        s=traction_world()
        zero=np.zeros(s.n)
        progressive_rifting.update(s,.01,zero)
        report=s.rift_mechanics['traction_loading']
        self.assertTrue(report['enabled'])
        self.assertTrue(report['applied'])
        self.assertEqual(s.rift_mechanics['loading_source'],
                         'conserved slab traction + reduced boundary-motion proxy')
        self.assertGreater(report['traction_soft_weight_sum'],0.)
        self.assertEqual(report['boundary_soft_weight_sum'],0.)
        self.assertGreater(np.count_nonzero(s.rift_pending['velocity']),0)

    def test_highland_metadata_adopts_moving_hinge_unless_user_explicitly_overrides(self):
        config=dict(width=64,height=32,mesh_level=3,coast_geometry_level=3,
                    plate_count=12,mechanics_nodes=128,seed=41,
                    physics_profile='reviewed_v1')
        initial=native_engine.make_initial(config,preset='highland65')
        self.assertTrue(initial['rift_traction']['enabled'])
        adopted=native_engine.Simulation(config,deepcopy(initial))
        self.assertTrue(adopted.initial_rift_traction_adopted)
        self.assertTrue(adopted.config['rift_traction']['enabled'])
        self.assertEqual(adopted.config['subduction_response'],'moving_hinge_v1')
        fixed=native_engine.Simulation(dict(config,subduction_response='fixed_trench'),deepcopy(initial))
        self.assertFalse(fixed.initial_rift_traction_adopted)
        self.assertFalse(fixed.config['rift_traction']['enabled'])
        self.assertEqual(fixed.config['subduction_response'],'fixed_trench')


if __name__=='__main__':
    unittest.main()
