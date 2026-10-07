"""Small native rigid-motion check; no Simulation constructor or world step."""
from copy import deepcopy
import unittest
from unittest.mock import patch
import numpy as np
import coarse_history
import crustal_structure as columns
import native_material_evolution as evolution
import numerical_accuracy as accuracy
from ridge_geometry import rotate
from tests.test_native_material_evolution import fixture


class CoarseMaterialMotionTests(unittest.TestCase):
    def test_full_interval_rotates_markers_and_preserves_columns_without_fine_solvers(self):
        s=fixture()
        s.t=121.;s.steps=121
        s.collision_surface_version=1;s.material_mechanics_version=1
        s.deformation_diagnostics={'model':'synthetic previous detailed stage','iterations':32}
        accuracy.initialize_columns(s.structure,s.material_surface['area_km2'])
        s.numerical_accuracy_version=1
        s.trace_structure=deepcopy(s.structure)
        # A nonzero old spent value must survive; this is not fresh credit.
        s.structure[accuracy.SPENT][:]=s.structure[accuracy.BUDGET]*.25
        s.trace_structure[accuracy.SPENT][:]=s.structure[accuracy.SPENT]
        before=deepcopy(s)
        coarse_history.activate(s,source_receipt={'run_id':'synthetic-test',
            'checkpoint_file':'synthetic-fixture-no-file-read','checkpoint_sha256':'a'*64})
        self.assertEqual(s.t,121.)
        self.assertEqual(s.steps,121)
        np.testing.assert_array_equal(s.material_surface['vertices'],before.material_surface['vertices'])
        dt=1.
        with patch('deforming_regions.deform',side_effect=AssertionError('fine contact called')), \
                patch('gravitational_relaxation.relax',side_effect=AssertionError('fine gravity called')):
            evolution.advect(s,dt)
        np.testing.assert_allclose(s.material_surface['vertices'],
            rotate(before.material_surface['vertices'],before.omega[0]*dt),rtol=0,atol=2e-15)
        np.testing.assert_allclose(s.trace_xyz,rotate(before.trace_xyz,before.omega[0]*dt),rtol=0,atol=2e-15)
        self.assertGreater(float(np.linalg.norm(s.trace_xyz-before.trace_xyz)),1e-5)
        np.testing.assert_allclose(s.material_surface['area_km2'],before.material_surface['area_km2'],rtol=2e-12)
        np.testing.assert_array_equal(s.material_deformation['numerical_accuracy_endpoint'][accuracy.SPENT],before.structure[accuracy.SPENT])
        endpoint=accuracy.column_endpoint(s)
        after,_=columns.evolve_structure(s.structure,dt,geometric_log_area=s.geometric_log_area,geometric_accuracy=endpoint)
        old_volume=before.material_surface['area_km2']*before.structure['thickness_km']
        new_volume=s.material_surface['area_km2']*after['thickness_km']
        np.testing.assert_allclose(new_volume,old_volume,rtol=2e-13)
        np.testing.assert_array_equal(after[accuracy.SPENT],before.structure[accuracy.SPENT])
        self.assertEqual(s.deformation_diagnostics['model'],'rigid material transport')
        self.assertEqual(s.deformation_diagnostics['deforming_vertices'],0)
        self.assertNotIn('gravitational_relaxation',s.deformation_diagnostics)
        frame={'deformation_diagnostics':deepcopy(s.deformation_diagnostics)}
        coarse_history.annotate_snapshot(s,frame)
        self.assertFalse(frame['coarse_history_policy']['internal_p1_strain_resolved'])
        self.assertFalse(frame['coarse_history_policy']['gravitational_relaxation_resolved'])
        # This substage does not itself advance the world clock.
        self.assertEqual(s.t,121.)
        self.assertEqual(s.steps,121)


if __name__=='__main__':unittest.main()
