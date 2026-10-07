"""New geological records retain geometry, unknowns and forcing semantics."""
from copy import deepcopy
import unittest
import numpy as np

from orientation import orient_frame, rotation_matrix
import gospl_export as export
from tests.test_ridge_gospl import thermal_frame
from tests.test_gospl_export import grid_points


class FoundationExportTests(unittest.TestCase):
    def test_rotated_trench_geometry_and_unknown_rift_age(self):
        w, h = 32, 16
        age = np.full(w*h, -1., np.float32)
        age.reshape(h,w)[4:12,8:24] = 80.
        frame = dict(width=w, height=h, rift_cooling_age_myr=age,
                     trench=np.ones(w*h,np.int32)*3,
                     trench_systems=[dict(id=3, center=[1.,0.,0.],
                         geometry_xyz=[[1.,0.,0.],[0.,1.,0.]],
                         history=[dict(time_myr=6.,phase='initiating')])])
        pose=dict(yaw=33,pitch=61,roll=-12)
        result=orient_frame(frame,pose)
        np.testing.assert_allclose(result['trench_systems'][0]['geometry_xyz'],
            np.array(frame['trench_systems'][0]['geometry_xyz'])@rotation_matrix(pose))
        valid=result['rift_cooling_age_myr'] >= 0
        self.assertTrue(valid.any());self.assertTrue((~valid).any())
        np.testing.assert_allclose(result['rift_cooling_age_myr'][valid],80.,atol=2e-5)
        self.assertEqual(result['trench_systems'][0]['history'],frame['trench_systems'][0]['history'])
        np.testing.assert_array_equal(frame['rift_cooling_age_myr'],age)

    def test_unmarked_new_columns_use_net_rate_not_legacy_relief_decay(self):
        first=thermal_frame(8.,relief=1200.,supported=False)
        first.update(structure_version=1,erosion_rate_m_myr=np.full(first['width']*first['height'],1.25))
        second=deepcopy(first);second['time_myr']=10.
        fields, diagnostics=export.interval_forcing(grid_points(128,64),first,second,dict(erosion=3.))
        np.testing.assert_allclose(fields['relaxation_correction']*2e6,2.5,atol=1e-6)
        self.assertFalse(fields['correction_supported'].any())
        self.assertEqual(diagnostics['erosion_convention'],'net denudation after rebound')

    def test_recorded_net_erosion_does_not_add_rebound_a_second_time(self):
        first=thermal_frame(8.,relief=1200.,supported=True)
        first.update(structure_version=1,erosion_rate_m_myr=np.full(128*64,99.),
                     trace_denudation_m=np.zeros(128*64),trace_rebound_m=np.zeros(128*64))
        second=deepcopy(first);second['time_myr']=10.
        second['trace_erosion_m'] += 3.
        second['trace_denudation_m'] += 20.
        second['trace_rebound_m'] += 17.
        fields,_=export.interval_forcing(grid_points(128,64),first,second,dict(erosion=1.))
        np.testing.assert_allclose(fields['relaxation_correction']*2e6,3.,atol=1e-6)
        self.assertTrue(fields['correction_supported'].all())

    def test_missing_net_rate_is_rejected_instead_of_inventing_old_physics(self):
        first=thermal_frame(8.);first['structure_version']=1
        second=deepcopy(first);second['time_myr']=10.
        with self.assertRaisesRegex(ValueError,'net erosion-rate'):
            export.interval_forcing(grid_points(128,64),first,second,{})


if __name__=='__main__':unittest.main()
