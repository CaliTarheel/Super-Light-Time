"""Coupled runtime mechanics preserve material while admitting lateral escape."""
from copy import deepcopy
import unittest
import numpy as np

import deforming_regions
import material_surface
import native_material_evolution
from ridge_geometry import rotate
try:
    from .test_deforming_regions import fixture
    from .test_native_material_evolution import fixture as marker_fixture
except ImportError:
    from test_deforming_regions import fixture
    from test_native_material_evolution import fixture as marker_fixture


def head_on():
    surface, omega, boundary = fixture(-20.)
    points, faces = surface['vertices'], surface['faces']
    right = points.copy(); right[:, 1] *= -1.
    paired = material_surface.initialize_surface(np.vstack((points, right)),
        np.vstack((faces, faces[:, [0, 2, 1]]+len(points))),
        np.repeat([0, 1], len(faces)), np.ones(2*len(faces), np.uint8))
    return paired, omega, boundary


class MaterialMechanicsTests(unittest.TestCase):
    def test_point_only_contact_does_not_transmit_a_material_force(self):
        state=marker_fixture()
        points=np.vstack((state.material_surface['vertices'],[1.,-.02,.06],[1.,-.06,.02]))
        faces=np.array([[0,1,2],[0,2,3],[3,4,5]])
        mesh=material_surface.initialize_surface(points,faces,np.zeros(3,int),np.array([2,1,1]))
        boundary={key:getattr(state,key) for key in ('bmid','bn','bl','bp','bq')}
        result=deforming_regions.deform(mesh,state.omega,boundary,2.,mechanics_version=1,
            require_material_contact=True,iterations=1024)
        self.assertEqual(result['diagnostics']['material_contact']['material_components'],2)
        self.assertTrue(result['rigid_mask'][3])
        np.testing.assert_array_equal(result['commanded_residual_velocity_km_myr'],0.)
        np.testing.assert_allclose(result['vertices'],result['rigid_vertices'],atol=0.)

    def test_new_arc_mechanical_reference_retains_independent_imported_relief(self):
        state=marker_fixture();state.parcel_arc_id=np.array([0,9])
        before={key:value.copy() for key,value in state.structure.items()}
        np.testing.assert_array_equal(native_material_evolution.gravitational_reference(state),
            state.structure['reference_thickness_km'])
        state.native_arc_birth_profile_version=1
        reference=native_material_evolution.gravitational_reference(state)
        self.assertEqual(reference[1],8.)
        self.assertEqual(reference[0],before['reference_thickness_km'][0])
        for key,value in before.items():np.testing.assert_array_equal(state.structure[key],value)

    def test_observed_strain_diagnostic_matches_independent_affine_metric(self):
        radius = 1e5
        xy = np.array([[0.,0.],[1.,0.],[0.,1.]])
        unit = lambda x: x/np.linalg.norm(x, axis=1)[:,None]
        points = unit(np.column_stack((np.ones(3), xy/radius)))
        moved = unit(np.column_stack((np.ones(3), xy*np.array([.9,1.05])/radius)))
        faces = np.array([[0,1,2]])
        fronts = dict(bn=np.array([[0.,1.,0.]]))
        result = deforming_regions.finite_strain_summary(points, moved, faces, np.array([0]),
            np.zeros(3,int), np.ones(3), fronts, radius)
        row = result['owner_rows'][0]
        self.assertAlmostEqual(row['mean_across_front_log_strain'], np.log(.9), places=9)
        self.assertAlmostEqual(row['mean_transverse_log_strain'], np.log(1.05), places=9)
        axis = np.array([.1,1.5,.7])
        rigid = deforming_regions.finite_strain_summary(points, rotate(points,axis), faces, np.array([0]),
            np.zeros(3,int), np.ones(3), fronts, radius)
        self.assertLess(abs(rigid['maximum_principal_log_extension']), 1e-10)
        self.assertLess(abs(rigid['minimum_principal_log_extension']), 1e-10)

    def test_head_on_contact_shortens_and_escapes_symmetrically_with_actual_volume(self):
        surface, omega, boundary = head_on()
        result = deforming_regions.deform(surface, omega, boundary, 2., mechanics_version=1,
            iterations=1024, tolerance=1e-9)
        points, moved = surface['vertices'], result['vertices']
        half = len(points)//2
        np.testing.assert_allclose(moved[:half, 1], -moved[half:, 1], atol=2e-12)
        np.testing.assert_allclose(moved[:half, [0, 2]], moved[half:, [0, 2]], atol=2e-12)
        # These are observed finite geometric widths, not solver strain fields.
        self.assertLess(np.ptp(moved[:half, 1]), np.ptp(points[:half, 1]))
        self.assertGreater(np.ptp(moved[:half, 2]), np.ptp(points[:half, 2]))
        new_area = material_surface.spherical_face_areas(moved, surface['faces'])
        old_area = material_surface.spherical_face_areas(points, surface['faces'])
        np.testing.assert_allclose(new_area*35./result['area_ratio'], old_area*35., rtol=5e-14)
        self.assertTrue(result['diagnostics']['solver']['normal_only_driven'])
        self.assertEqual(result['diagnostics']['accepted_residual_fraction'], 1.)

    def test_runtime_loading_and_geometry_are_equivariant_through_the_pole(self):
        surface, omega, boundary = head_on()
        first = deforming_regions.deform(surface, omega, boundary, 2., mechanics_version=1,
            iterations=1024, tolerance=1e-10)
        axis = np.array([0., -np.pi/2, .3])
        turned = deepcopy(surface); turned['vertices'] = rotate(surface['vertices'], axis)
        fronts = deepcopy(boundary)
        for key in ('bmid', 'bn'): fronts[key] = rotate(fronts[key], axis)
        second = deforming_regions.deform(turned, rotate(omega, axis), fronts, 2., mechanics_version=1,
            iterations=1024, tolerance=1e-10)
        np.testing.assert_allclose(second['vertices'], rotate(first['vertices'], axis), atol=3e-11)
        np.testing.assert_allclose(second['area_ratio'], first['area_ratio'], atol=2e-10)

    def test_runtime_gravity_and_loading_share_the_same_marker_and_volume_map(self):
        state = marker_fixture()
        state.material_mechanics_version = 1
        state.parcel_collision_sheet = np.array([1, 1])
        old_area = state.material_surface['area_km2'].copy()
        columns = state.structure['thickness_km'].copy()
        native_material_evolution.advect(state, 2.)
        gravity = state.deformation_diagnostics['gravitational_relaxation']
        self.assertEqual(gravity['completed_dt_myr'], 2.)
        self.assertLess(gravity['energy_after_km4'], gravity['energy_before_km4'])
        self.assertGreater(gravity['maximum_displacement_km'], 0.)
        np.testing.assert_allclose(state.trace_xyz, material_surface.face_centres(state.material_surface), atol=2e-14)
        np.testing.assert_allclose(state.material_surface['area_km2']*columns*np.exp(-state.geometric_log_area),
            old_area*columns, rtol=3e-14)
        np.testing.assert_array_equal(state.trace_geometric_log_area, state.geometric_log_area)
        np.testing.assert_array_equal(state.structure['thickness_km'], columns)


if __name__ == '__main__': unittest.main()
