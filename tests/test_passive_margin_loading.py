"""Opening cannot hold an already cut continental edge at the rift axis."""
from copy import deepcopy
import unittest

import numpy as np

from deforming_regions import deform
from material_surface import refresh_geometry, reassign_owners
from ridge_geometry import rotate
from tests.test_deformation_contacts import R, patch, combine, loading, subdivide


def opening():
    omega, boundary = loading()
    return -omega, boundary


def solve(surface, omega, boundary, **options):
    return deform(surface, omega, boundary, 2., require_material_contact=True,
                  mechanics_version=1, iterations=1024, tolerance=1e-9, **options)


def margin(last=0., owner=0):
    return patch(np.linspace(-1600., last, 9), np.linspace(-1000., 1000., 11), owner=owner)


class PassiveMarginLoadingTests(unittest.TestCase):
    def test_separating_free_edge_follows_its_plate_without_thinning(self):
        omega, boundary = opening()
        surface = margin()
        result = solve(surface, omega, boundary)
        np.testing.assert_array_equal(result['vertices'], result['rigid_vertices'])
        np.testing.assert_allclose(result['areal_strain'], 0., atol=2e-13)
        np.testing.assert_array_equal(result['commanded_residual_velocity_km_myr'], 0.)
        self.assertEqual(result['diagnostics']['material_contact']['rejected_open_free_edge_pairs'], 1)

    def test_matching_cut_faces_separate_instead_of_staying_at_the_axis(self):
        omega, boundary = opening()
        left = margin()
        right = patch(np.linspace(0., 1600., 9), np.linspace(-1000., 1000., 11), owner=1)
        surface = combine(left, right)
        old_area = surface['area_km2'].copy()
        thickness = np.full(len(old_area), 35.)
        # Forty Myr of continued opening must not grow a continental tail.
        for _ in range(20):
            result = solve(surface, omega, boundary)
            thickness /= result['area_ratio']
            surface['vertices'] = result['vertices']
            refresh_geometry(surface)
        np.testing.assert_allclose(surface['area_km2'], old_area, rtol=2e-12)
        np.testing.assert_allclose(thickness, 35., atol=2e-10)
        left_side = surface['vertices'][surface['vertex_owner'] == 0]
        right_side = surface['vertices'][surface['vertex_owner'] == 1]
        gap = (np.min(np.arctan2(right_side[:, 1], right_side[:, 0]))
               - np.max(np.arctan2(left_side[:, 1], left_side[:, 0])))*R
        self.assertAlmostEqual(gap, 1600., places=8)

    def test_unbroken_material_crossing_front_can_still_extend(self):
        omega, boundary = opening()
        result = solve(margin(200.), omega, boundary)
        self.assertGreater(result['area_ratio'].max(), 1.01)
        self.assertGreater(result['diagnostics']['material_contact']['anchored_front_pairs'], 0)

    def test_ownership_cut_releases_previously_continuous_material(self):
        omega, boundary = opening()
        surface = patch(np.arange(-1600., 1601., 200.), np.linspace(-1000., 1000., 11))
        before = solve(surface, omega, boundary)
        self.assertGreater(before['area_ratio'].max(), 1.01)
        points, faces = surface['vertices'], surface['faces']
        owners = (points[faces, 1].mean(axis=1) > 0.).astype(np.int32)
        reassign_owners(surface, owners)
        self.assertGreater(len(surface['vertices']), len(points))
        after = solve(surface, omega, boundary)
        np.testing.assert_array_equal(after['vertices'], after['rigid_vertices'])
        np.testing.assert_allclose(after['area_ratio'], 1., atol=2e-13)

    def test_separated_margins_allow_existing_ocean_birth_path(self):
        import native_spreading
        from tests.test_native_spreading_pairing import scene, expected_area
        state, axis = scene()
        state.material_surface = combine(margin(), patch(
            np.linspace(0., 1600., 9), np.linspace(-1000., 1000., 11), owner=1))
        _, boundary = opening()
        result = solve(state.material_surface, state.omega, boundary)
        state.material_surface['vertices'] = result['vertices']
        refresh_geometry(state.material_surface)
        before = state.material_surface['vertices'].copy()
        # The next source epoch has an ocean interval between the cut edges.
        birth = native_spreading.advance(state, state.support.copy(), 2.)
        self.assertAlmostEqual(float(birth@state.cell_area), expected_area(axis, 2.), delta=1e-6)
        self.assertGreater(state.spreading_diagnostics['generated_area_km2'], 0.)
        np.testing.assert_array_equal(state.material_surface['vertices'], before)

    def test_internal_mesh_edge_is_not_a_free_edge(self):
        omega, boundary = opening()
        # The axis lies exactly along indexed internal edges in both meshes.
        surface = patch(np.arange(-1600., 401., 200.), np.linspace(-1000., 1000., 11))
        for mesh in (surface, subdivide(surface)):
            result = solve(mesh, omega, boundary)
            self.assertGreater(result['area_ratio'].max(), 1.01)
            self.assertEqual(result['diagnostics']['material_contact']['anchored_front_pairs'], 1)

    def test_detached_same_owner_patch_across_axis_does_not_make_a_bridge(self):
        omega, boundary = opening()
        surface = combine(margin(), patch(np.linspace(0., 1600., 9), np.linspace(-1000., 1000., 11)))
        result = solve(surface, omega, boundary)
        np.testing.assert_array_equal(result['vertices'], result['rigid_vertices'])
        np.testing.assert_allclose(result['areal_strain'], 0., atol=2e-13)
        self.assertEqual(result['diagnostics']['material_contact']['rejected_open_free_edge_pairs'], 2)

    def test_converging_free_edge_still_shortens(self):
        omega, boundary = loading()
        result = solve(margin(), omega, boundary)
        self.assertLess(result['area_ratio'].min(), .98)
        self.assertEqual(result['diagnostics']['material_contact']['anchored_front_pairs'], 1)

    def test_oblique_separation_cannot_drag_a_free_edge(self):
        omega, boundary = opening()
        omega[0, 1], omega[1, 1] = -.003, .003
        result = solve(margin(), omega, boundary)
        np.testing.assert_array_equal(result['vertices'], result['rigid_vertices'])
        np.testing.assert_allclose(result['areal_strain'], 0., atol=2e-13)

    def test_pure_transform_loading_is_retained(self):
        omega, boundary = opening()
        omega[:] = 0.
        omega[0, 1], omega[1, 1] = -.003, .003
        result = solve(margin(), omega, boundary)
        self.assertGreater(np.linalg.norm(result['commanded_residual_velocity_km_myr']), 1.)
        self.assertEqual(result['diagnostics']['material_contact']['anchored_front_pairs'], 1)

    def test_admission_is_invariant_under_rotation_owner_order_and_refinement(self):
        omega, boundary = opening()
        for surface in (margin(), margin(200.),
                        patch(np.arange(-1600., 401., 200.), np.linspace(-1000., 1000., 11))):
            original = solve(surface, omega, boundary)
            accepted = original['diagnostics']['material_contact']['anchored_front_pairs']
            refined = solve(subdivide(surface), omega, boundary)
            self.assertEqual(refined['diagnostics']['material_contact']['anchored_front_pairs'], accepted)
            for axis in (np.array([0., -np.pi/2, 0.]), np.array([.7, -.4, 1.2])):
                mesh, fronts = deepcopy(surface), deepcopy(boundary)
                mesh['vertices'] = rotate(mesh['vertices'], axis)
                for key in ('bmid', 'bn'):
                    fronts[key] = rotate(fronts[key], axis)
                fronts['bp'], fronts['bq'] = fronts['bq'], fronts['bp']
                fronts['bn'] *= -1.
                result = solve(mesh, rotate(omega, axis), fronts)
                self.assertEqual(result['diagnostics']['material_contact']['anchored_front_pairs'], accepted)
                np.testing.assert_allclose(result['vertices'], rotate(original['vertices'], axis), atol=3e-10)


if __name__ == '__main__':
    unittest.main()
