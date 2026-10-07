"""Actual spherical strain must be the only column-strain source."""
import unittest
import numpy as np
from copy import deepcopy
import structure_engine
from crustal_structure import initialize_structure, evolve_structure


class GeometricColumnsTests(unittest.TestCase):
    def test_actual_area_conserves_volume_and_overrides_boundary_proxy(self):
        old = initialize_structure([1, 2, 3], [220., 440., 970.])
        ratios = np.array([1.2, 1., .9])
        new, budget = evolve_structure(old, 2., geometric_log_area=np.log(ratios),
                                      shortening_km_myr=100., extension_km_myr=50.)
        np.testing.assert_allclose(new['area_factor'], ratios, atol=1e-14)
        np.testing.assert_allclose(new['thickness_km']*ratios, old['thickness_km'], atol=1e-13)
        self.assertEqual(new['thickness_km'][1], old['thickness_km'][1])
        self.assertGreater(budget['extension_loss_m'][0], 0.)
        self.assertGreater(budget['shortening_gain_m'][2], 0.)

    def test_rigid_or_pure_shear_does_not_thin_under_boundary_loading(self):
        old = initialize_structure([1], [220.])
        new, budget = evolve_structure(old, 2., geometric_log_area=[0.], extension_km_myr=100.)
        for key in old:
            np.testing.assert_array_equal(new[key], old[key], err_msg=key)
        np.testing.assert_array_equal(budget['mechanical_delta_m'], [0.])

    def test_unrealizable_geometry_fails_instead_of_clipping_a_column(self):
        old = initialize_structure([1], [220.])
        with self.assertRaises(ValueError):
            evolve_structure(old, 2., geometric_log_area=[np.log(10.)])

    def test_erosion_volume_is_accounted_after_geometric_stretch(self):
        old = initialize_structure([1], [220.])
        new, budget = evolve_structure(old, 2., geometric_log_area=[np.log(1.2)], denudation_m=100.)
        removed = budget['removed_volume_km_per_reference_km2']
        np.testing.assert_allclose(new['thickness_km']*new['area_factor']+removed,
                                   old['thickness_km']*old['area_factor'], atol=1e-13)

    def test_rift_inversion_tags_compression_without_a_second_area_change(self):
        from tests.test_rift_inversion_engine import compressional_world
        s=compressional_world()
        s.geometric_log_area=np.full(len(s.mass),-.02)
        s.trace_geometric_log_area=np.full(len(s.trace_xyz),-.02)
        twin=deepcopy(s)
        twin.rift_extension_m[:]=0.;twin.trace_rift_extension_m[:]=0.
        for model in (s,twin):
            model.t+=2.
            structure_engine.deform(model,*(np.zeros(model.n) for _ in range(3)),2.)
        self.assertGreater(s.trace_inversion_uplift_m.max(),0.)
        np.testing.assert_allclose(s.structure['area_factor'],np.exp(-.02),atol=1e-14)
        np.testing.assert_allclose(s.trace_relief_m,twin.trace_relief_m,atol=1e-12)
        np.testing.assert_allclose(s.trace_uplift_m,twin.trace_uplift_m,atol=1e-12)

    def test_new_fault_does_not_thin_unstretched_geometry(self):
        from tests.test_structure_engine import world
        s=world();before=deepcopy(s.structure)
        s.geometric_log_area=np.zeros(len(s.mass));s.trace_geometric_log_area=np.zeros(len(s.trace_xyz))
        structure_engine.cut_rift(s,s.kind==1,s.trace_kind==1,250.)
        for key in before:np.testing.assert_array_equal(s.structure[key],before[key],err_msg=key)


if __name__ == '__main__':
    unittest.main()
