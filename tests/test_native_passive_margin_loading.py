"""An ocean ownership contour cannot reattach an already separated continent."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np

import deforming_regions
import native_material_evolution as evolution
import native_spreading
from material_surface import refresh_geometry
from ridge_geometry import rotate
from tests.test_deformation_contacts import R, patch, combine, subdivide
from tests.test_passive_margin_loading import margin, opening, solve
from tests.test_native_spreading_pairing import scene, expected_area


def native_state(surface, omega, boundary, *, split=1):
    """Represent the prescribed finite arc as native support geometry."""
    state = SimpleNamespace(material_surface=surface, omega=omega,
                            **{key: np.asarray(value).copy() for key, value in boundary.items()})
    middle, normal = boundary['bmid'][0], boundary['bn'][0]
    tangent = np.cross(normal, middle)
    half = boundary['bl'][0]/(2.*R)
    angles = np.linspace(-half, half, split+1)
    points = np.cos(angles)[:, None]*middle+np.sin(angles)[:, None]*tangent
    state.native_boundary_geometry = dict(segments_start=points[:-1], segments_end=points[1:],
        segment_normals=np.tile(normal, (split, 1)), contact_index=np.zeros(split, np.int32))
    state._valid_loading_edges = lambda: np.ones(len(state.bl), bool)
    return state


def displaced_shores():
    # The support contour is at x=0, inside the left continent. Its free edge
    # is x=200; the other owner's shore starts at x=300. No material crosses
    # the 100 km gap, regardless of the ocean-support contour's position.
    return combine(margin(200.), patch(np.linspace(300., 1700., 8),
        np.linspace(-1000., 1000., 11), owner=1))


class NativePassiveMarginLoadingTests(unittest.TestCase):
    def assert_rigid(self, result):
        # Rotated fixture vertices acquire one normalization roundoff before
        # the returned endpoint; commanded non-rigid velocity is exactly zero.
        np.testing.assert_allclose(result['vertices'], result['rigid_vertices'], atol=3e-15, rtol=0.)
        np.testing.assert_array_equal(result['commanded_residual_velocity_km_myr'], 0.)
        np.testing.assert_allclose(result['area_ratio'], 1., atol=3e-12, rtol=0.)
        np.testing.assert_allclose(35./result['area_ratio'], 35., atol=2e-10, rtol=0.)

    def test_displaced_native_contour_cannot_thin_only_one_detached_shore(self):
        omega, boundary = opening()
        surface = displaced_shores()
        # This deliberately prescribed raw front exhibits the old false bridge:
        # one body crosses the contour even though the opposing shore is gone.
        raw = solve(surface, omega, boundary)
        left = surface['face_owner'] == 0
        self.assertGreater(raw['area_ratio'][left].max(), 1.05)
        np.testing.assert_allclose(raw['area_ratio'][~left], 1., atol=3e-12)

        state = native_state(surface, omega, boundary)
        original_valid = state._valid_loading_edges().copy()
        fronts, diagnostic = evolution.material_loading_boundaries(state, 1)
        self.assert_rigid(solve(surface, omega, fronts))
        self.assertEqual(diagnostic['unbonded_opening_segments'], 1)
        self.assertAlmostEqual(diagnostic['unbonded_opening_length_km'], boundary['bl'][0], places=9)
        np.testing.assert_array_equal(state._valid_loading_edges(), original_valid)
        np.testing.assert_array_equal(state.native_boundary_geometry['contact_index'], [0])

    def test_mirrored_detached_shores_release_the_right_continent_too(self):
        omega, boundary = opening()
        surface = combine(patch(np.linspace(-1700., -300., 8),
            np.linspace(-1000., 1000., 11), owner=0),
            patch(np.linspace(-200., 1600., 9), np.linspace(-1000., 1000., 11), owner=1))
        raw = solve(surface, omega, boundary)
        right = surface['face_owner'] == 1
        self.assertGreater(raw['area_ratio'][right].max(), 1.05)
        np.testing.assert_allclose(raw['area_ratio'][~right], 1., atol=3e-12)
        state = native_state(surface, omega, boundary)
        fronts, diagnostic = evolution.material_loading_boundaries(state, 1)
        self.assert_rigid(solve(surface, omega, fronts))
        self.assertEqual(diagnostic['unbonded_opening_segments'], 1)

    def test_very_slow_opening_releases_normal_and_oblique_targets(self):
        from ridge_geometry import normal_motion_threshold
        for oblique in (False, True):
            with self.subTest(oblique=oblique):
                omega, boundary = opening()
                # Full separation is 0.001 km/Myr: material release must not
                # wait for the independent seafloor-production speed gate.
                omega[:] = 0.
                omega[0, 2], omega[1, 2] = -.0005/R, .0005/R
                if oblique:
                    omega[0, 1], omega[1, 1] = -.003, .003
                relative = np.cross(omega[1]-omega[0], boundary['bmid'][0])*R
                normal_speed = float(relative@boundary['bn'][0])
                shear_speed = np.linalg.norm(relative-normal_speed*boundary['bn'][0])
                self.assertAlmostEqual(normal_speed, .001)
                self.assertLess(normal_speed, normal_motion_threshold(shear_speed))
                if oblique:
                    self.assertGreater(shear_speed, 30.)
                state = native_state(displaced_shores(), omega, boundary)
                fronts, diagnostic = evolution.material_loading_boundaries(state, 1)
                target, weight, _ = deforming_regions._targets(state.material_surface['vertices'],
                    state.material_surface['vertex_owner'], omega, fronts, R, 400., 1.)
                np.testing.assert_array_equal(target, 0.)
                np.testing.assert_array_equal(weight, 0.)
                self.assert_rigid(solve(state.material_surface, omega, fronts))
                self.assertEqual(diagnostic['unbonded_opening_segments'], 1)

    def test_release_survives_obliquity_rotation_owner_reversal_and_refinement(self):
        for refined in (False, True):
            for oblique in (False, True):
                with self.subTest(refined=refined, oblique=oblique):
                    omega, boundary = opening()
                    if oblique:
                        omega[0, 1], omega[1, 1] = -.003, .003
                    surface = displaced_shores()
                    if refined:
                        surface = subdivide(surface)
                    state = native_state(surface, omega, boundary, split=3)
                    # Move the scene through a pole and reverse the owner/normal
                    # convention without changing its actual physical motion.
                    axis = np.array([.7, -np.pi/2., 1.2])
                    state.material_surface['vertices'] = rotate(surface['vertices'], axis)
                    state.omega = rotate(omega, axis)
                    state.bp, state.bq = state.bq.copy(), state.bp.copy()
                    for name in ('segments_start', 'segments_end', 'segment_normals'):
                        state.native_boundary_geometry[name] = rotate(state.native_boundary_geometry[name], axis)
                    state.native_boundary_geometry['segment_normals'] *= -1.
                    fronts, diagnostic = evolution.material_loading_boundaries(state, 1)
                    self.assert_rigid(solve(state.material_surface, state.omega, fronts))
                    self.assertEqual(diagnostic['unbonded_opening_segments'], 3)
                    self.assertAlmostEqual(diagnostic['unbonded_opening_length_km'], boundary['bl'][0], places=8)

    def test_convergence_and_transform_keep_their_material_response(self):
        for mode in ('convergence', 'transform'):
            with self.subTest(mode=mode):
                omega, boundary = opening()
                if mode == 'convergence':
                    omega *= -1.
                else:
                    omega[:] = 0.
                    omega[0, 1], omega[1, 1] = -.003, .003
                surface = margin()
                state = native_state(surface, omega, boundary)
                fronts, diagnostic = evolution.material_loading_boundaries(state, 1)
                result = solve(surface, omega, fronts)
                reference = solve(surface, omega, boundary)
                self.assertEqual(diagnostic['unbonded_opening_segments'], 0)
                self.assertEqual(diagnostic['unbonded_opening_length_km'], 0.)
                self.assertGreater(np.linalg.norm(result['commanded_residual_velocity_km_myr']), 1.)
                np.testing.assert_allclose(result['vertices'], reference['vertices'], atol=3e-12)
                if mode == 'convergence':
                    self.assertLess(result['area_ratio'].min(), .98)

    def test_generic_prescribed_intact_front_and_legacy_mechanics_still_extend(self):
        omega, boundary = opening()
        surface = margin(200.)
        for version, finite_geometry in ((1, False), (0, True)):
            with self.subTest(version=version, finite_geometry=finite_geometry):
                state = native_state(surface, omega, boundary)
                if not finite_geometry:
                    del state.native_boundary_geometry
                fronts, _ = evolution.material_loading_boundaries(state, version)
                result = solve(surface, omega, fronts)
                self.assertGreater(result['area_ratio'].max(), 1.05)

    def test_stale_opening_is_not_counted_or_reactivated(self):
        omega, boundary = opening()
        state = native_state(displaced_shores(), omega, boundary, split=2)
        state._valid_loading_edges = lambda: np.array([False])
        fronts, diagnostic = evolution.material_loading_boundaries(state, 1)
        np.testing.assert_array_equal(fronts['valid'], [False, False])
        self.assertEqual(diagnostic['unbonded_opening_segments'], 0)
        self.assertEqual(diagnostic['unbonded_opening_length_km'], 0.)
        self.assert_rigid(solve(state.material_surface, omega, fronts))

    def test_real_convergent_collision_front_survives_opening_support_rejection(self):
        omega, boundary = opening()
        state = native_state(margin(), omega, boundary)
        # This independent front faces the opposite way, so the actual relative
        # motion closes it even while opening the unrelated support contour.
        state.collision_contacts = [dict(state='active', top_owner=0, under_owner=1,
            center=[1., 0., 0.], normal=[0., -1., 0.], overlap_area_km2=100., length_km=1000.)]
        fronts, diagnostic = evolution.material_loading_boundaries(state, 1)
        np.testing.assert_array_equal(fronts['valid'], [False, True])
        self.assertEqual(diagnostic['unbonded_opening_segments'], 1)
        self.assertEqual(diagnostic['total_loading_fronts'], 2)
        target, _, _ = deforming_regions._targets(state.material_surface['vertices'],
            state.material_surface['vertex_owner'], omega, fronts, R, 400., 1.)
        self.assertGreater(np.linalg.norm(target), 1.)

    def test_releasing_continents_preserves_locally_paired_ocean_production(self):
        state, axis = scene(corridor=True)
        state.material_surface = combine(
            margin(-30.), patch(np.linspace(30., 1600., 9), np.linspace(-1000., 1000., 11), owner=1))
        state.bmid = np.array([[1., 0., 0.]])
        state.bn = np.array([[0., 1., 0.]])
        state.bl = np.array([R*np.arctan2(np.linalg.norm(np.cross(*axis)), axis[0]@axis[1])])
        state._valid_loading_edges = lambda: np.array([True])
        fronts, diagnostic = evolution.material_loading_boundaries(state, 1)
        result = solve(state.material_surface, state.omega, fronts)
        self.assert_rigid(result)
        state.material_surface['vertices'] = result['vertices']
        refresh_geometry(state.material_surface)
        before = deepcopy(state.material_surface)
        birth = native_spreading.advance(state, state.support.copy(), 2.)
        self.assertEqual(diagnostic['unbonded_opening_segments'], 1)
        self.assertAlmostEqual(float(birth@state.cell_area), expected_area(axis, 2.), delta=1e-6)
        self.assertGreater(state.spreading_diagnostics['generated_area_km2'], 0.)
        self.assertAlmostEqual(state.spreading_diagnostics['side_p_area_km2'],
                               state.spreading_diagnostics['side_q_area_km2'], delta=1e-6)
        for name, value in before.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(state.material_surface[name], value)


if __name__ == '__main__':
    unittest.main()
