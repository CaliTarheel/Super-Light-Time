"""Shared-vertex reconstruction removes scalar seams without changing crust."""
from copy import deepcopy
import unittest

import numpy as np

import gospl_export
import material_reconstruction as reconstruction
import mesh_geometry
import native_frame_sampling as sampling
try:
    from .test_native_gospl_sampling import frame
except ImportError:
    from test_native_gospl_sampling import frame


def patch_geometry():
    vertices = np.array([[1., -.2, -.2], [1., .2, -.2],
                         [1., .2, .2], [1., -.2, .2]])
    vertices /= np.linalg.norm(vertices, axis=1)[:, None]
    return vertices, np.array([[0, 1, 2], [0, 2, 3]])


def continuous_frame():
    source = frame(centers=[])
    vertices, faces = patch_geometry()
    source.update(surface_reconstruction_version=1,
                  material_vertices=vertices, material_faces=faces,
                  material_face_id=np.array([100, 101]), material_owner=np.array([1, 1]),
                  material_kind=np.array([1, 2], np.uint8),
                  material_height_m=np.array([1000., 2200.]),
                  material_erosion_rate_m_myr=np.array([3., 11.]),
                  trace_id=np.arange(6), trace_xyz=vertices[faces].reshape((-1, 3)),
                  trace_patch=np.repeat([100, 101], 3), trace_plate_uid=np.full(6, 20),
                  trace_erosion_m=np.zeros(6))
    return source


def edge_queries(vertices, faces, epsilon=1e-8):
    middle = vertices[0]+vertices[2]
    middle /= np.linalg.norm(middle)
    centers = vertices[faces].sum(axis=1)
    centers /= np.linalg.norm(centers, axis=1)[:, None]
    points = middle+epsilon*centers
    return points/np.linalg.norm(points, axis=1)[:, None]


class MaterialReconstructionTests(unittest.TestCase):
    def test_area_weights_preserve_constants_bounds_and_actual_connectivity(self):
        vertices, faces = patch_geometry()
        stencil = reconstruction.prepare(vertices, faces, [1, 1], area_km2=[1., 3.])
        nodal = reconstruction.vertex_values(stencil, [100., 900.])
        np.testing.assert_allclose(nodal, [700., 100., 700., 900.])
        np.testing.assert_allclose(reconstruction.vertex_values(stencil, [413., 413.]), 413.)
        self.assertEqual(stencil['vertex_count'], 4)
        self.assertGreaterEqual(nodal.min(), 100.)
        self.assertLessEqual(nodal.max(), 900.)

    def test_barycentric_interpolator_reproduces_constant_and_linear_nodal_fields(self):
        vertices, faces = patch_geometry()
        stencil = reconstruction.prepare(vertices, faces, [1, 1])
        weights = np.array([[.2, .3, .5], [.6, .1, .3]])
        query = np.einsum('ij,ijk->ik', weights, vertices[faces])
        query /= np.linalg.norm(query, axis=1)[:, None]
        hits, actual_weights = mesh_geometry.locate_points(query, mesh_geometry.build_locator(vertices, faces))
        np.testing.assert_array_equal(hits, [0, 1])
        np.testing.assert_allclose(actual_weights, weights, atol=1e-14)
        linear_nodes = vertices@np.array([110., -230., 770.])+80.
        actual = reconstruction.sample_hits(stencil, linear_nodes, hits, actual_weights)
        np.testing.assert_allclose(actual, np.einsum('ij,ij->i', weights, linear_nodes[faces]), atol=1e-11)
        np.testing.assert_allclose(reconstruction.sample_hits(stencil, np.full(4, 57.), hits, actual_weights), 57.)

    def test_shared_edge_has_one_limit_from_both_faces(self):
        source = continuous_frame()
        vertices, faces = source['material_vertices'], source['material_faces']
        points = edge_queries(vertices, faces)
        result = sampling.sample_frame(source, points)
        np.testing.assert_array_equal(result['material_face'], [0, 1])
        np.testing.assert_array_equal(result['crust'], [1, 2])
        self.assertLess(abs(np.diff(result['elevation'])[0]), 1e-4)
        self.assertLess(abs(np.diff(result['erosion_rate_m_myr'])[0]), 1e-6)
        np.testing.assert_allclose(result['elevation'], 1600., atol=1e-4)

    def test_disconnected_overlapping_triangles_with_opposed_height_never_blend(self):
        vertices, faces = patch_geometry()
        separate = np.concatenate([vertices[faces[0]], vertices[faces[0]]])
        triangles = np.arange(6).reshape((2, 3))
        stencil = reconstruction.prepare(separate, triangles, [1, 1])
        nodal = reconstruction.vertex_values(stencil, [-800., 1800.])
        result = reconstruction.sample_hits(stencil, nodal, np.array([0, 1]), np.full((2, 3), 1./3))
        np.testing.assert_allclose(result, [-800., 1800.])
        self.assertEqual(stencil['vertex_count'], 6)

    def test_distinct_owners_never_share_values_even_if_input_reuses_vertex_indices(self):
        vertices, faces = patch_geometry()
        stencil = reconstruction.prepare(vertices, faces, [1, 2])
        nodal = reconstruction.vertex_values(stencil, [-800., 1800.])
        result = reconstruction.sample_hits(stencil, nodal, np.array([0, 1]), np.full((2, 3), 1./3))
        np.testing.assert_allclose(result, [-800., 1800.])
        self.assertEqual(stencil['vertex_count'], 6)

    def test_rotation_over_pole_and_seam_preserves_height_and_ids(self):
        source = continuous_frame()
        points = np.concatenate([edge_queries(source['material_vertices'], source['material_faces']),
                                 [[-1., 0., 0.]]])
        first = sampling.sample_frame(source, points)
        for matrix in (gospl_export.rotation([0., 1., 0.], np.pi/2),
                       gospl_export.rotation([0., 0., 1.], np.pi)):
            moved = deepcopy(source)
            for key in ('mesh_vertices', 'material_vertices'):
                moved[key] = moved[key]@matrix
            result = sampling.sample_frame(moved, points@matrix)
            for key in ('elevation', 'erosion_rate_m_myr', 'material_face', 'plate', 'crust'):
                np.testing.assert_allclose(result[key], first[key], atol=1e-9, err_msg=key)

    def test_single_arc_triangle_keeps_constant_height_and_does_not_fill_ocean(self):
        source = frame(heights=[1234.], rates=[17.])
        source['surface_reconstruction_version'] = 1
        result = sampling.sample_frame(source, np.array([[0., 0., 1.], [0., 1., 0.]]))
        self.assertEqual(result['elevation'][0], 1234.)
        self.assertEqual(result['erosion_rate_m_myr'][0], 17.)
        np.testing.assert_array_equal(result['material_face'], [0, -1])
        self.assertEqual(result['crust'][1], 0)

    def test_projection_is_not_a_column_ledger_or_claim_of_face_mean_conservation(self):
        vertices, faces = patch_geometry()
        values = np.array([100., 900.]); original = values.copy()
        stencil = reconstruction.prepare(vertices, faces, [1, 1])
        nodal = reconstruction.vertex_values(stencil, values)
        displayed_centres = reconstruction.sample_hits(stencil, nodal, np.arange(2), np.full((2, 3), 1./3))
        self.assertFalse(np.allclose(displayed_centres, values))
        np.testing.assert_array_equal(values, original)

    def test_old_native_frames_keep_face_constant_heights_and_erosion(self):
        source = continuous_frame(); source.pop('surface_reconstruction_version')
        points = edge_queries(source['material_vertices'], source['material_faces'])
        result = sampling.sample_frame(source, points)
        np.testing.assert_array_equal(result['elevation'], [1000., 2200.])
        np.testing.assert_array_equal(result['erosion_rate_m_myr'], [3., 11.])

    def test_continuous_native_sampling_is_independent_of_display_raster(self):
        source = continuous_frame()
        points = edge_queries(source['material_vertices'], source['material_faces'])
        expected = sampling.sample_frame(source, points)
        source.update(width=800, height=400, elevation=np.full(320000, -9999.),
                      crust=np.zeros(320000), plate=np.zeros(320000), boundary=np.zeros(320000))
        result = sampling.sample_frame(source, points)
        for key in expected:
            np.testing.assert_array_equal(result[key], expected[key], err_msg=key)

    def test_thermal_support_is_evaluated_once_after_material_reconstruction(self):
        source = continuous_frame(); source['time_myr'] = 2.
        point = np.array([[1., 0., 0.]])
        baseline = sampling.sample_frame(source, point)['elevation']
        source['ridge_episodes'] = [dict(started_myr=0., peak_myr=2., end_myr=12., center=[1., 0., 0.],
            radius_km=1000., thermal_peak_m=500., volcanic_budget_m=200., overriding_plate_uid=20)]
        result = sampling.sample_frame(source, point)
        np.testing.assert_allclose(result['elevation'], baseline+500.)
        np.testing.assert_allclose(result['thermal_support_m'], 500.)

    def test_varying_pure_erosion_does_not_become_tectonic_uplift(self):
        first = continuous_frame(); second = deepcopy(first); second['time_myr'] = 2.
        loss = np.array([40., 160.])
        second['material_height_m'] -= loss
        second['trace_erosion_m'] += np.repeat(loss, 3)
        before = deepcopy((first, second))
        points = np.concatenate([edge_queries(first['material_vertices'], first['material_faces']),
                                 first['trace_xyz']])
        fields, _ = gospl_export.interval_forcing(points, first, second, {'dt_myr': 2.})
        np.testing.assert_allclose(fields['upsub'], 0., atol=1e-11)
        self.assertTrue(np.all(fields['relaxation_correction'] > 0.))
        np.testing.assert_array_equal(fields['correction_supported'], 1)
        a, b = gospl_export.matched_traces(first, second)
        actual_loss, _ = sampling.material_erosion_correction(first, second, a, b)
        np.testing.assert_array_equal(actual_loss, loss)
        for source, original in zip((first, second), before):
            for key in ('material_height_m', 'material_erosion_rate_m_myr', 'trace_erosion_m'):
                np.testing.assert_array_equal(source[key], original[key])

    def test_correction_support_requires_support_from_every_contributing_column(self):
        source = continuous_frame(); context = sampling.prepare(source)
        points = np.array([source['material_vertices'][1], [1., 0., 0.]])
        sampled = sampling.sample_frame(source, points, context)
        correction = sampling.prepare_material_correction(context, [40., 160.], [True, False])
        loss, supported = sampling.sample_material_correction(context, sampled, correction)
        np.testing.assert_allclose(loss, [40., 100.])
        np.testing.assert_array_equal(supported, [True, False])

    def test_varying_erosion_on_rigid_polar_motion_preserves_the_export_budget(self):
        first = continuous_frame(); second = deepcopy(first); second['time_myr'] = 2.
        second['material_height_m'] -= [40., 160.]
        second['trace_erosion_m'] += np.repeat([40., 160.], 3)
        matrix = gospl_export.rotation([0., 1., 0.], np.pi/2)
        second['material_vertices'] = second['material_vertices']@matrix
        second['trace_xyz'] = second['trace_xyz']@matrix
        points = first['material_vertices'][first['material_faces']].sum(axis=1)
        points /= np.linalg.norm(points, axis=1)[:, None]
        fields, _ = gospl_export.interval_forcing(points, first, second, {'dt_myr': 2.})
        np.testing.assert_allclose(fields['upsub'], 0., atol=1e-11)
        expected = (points@matrix-points)*gospl_export.RADIUS_M/2_000_000.
        np.testing.assert_allclose(fields['hdisp'], expected, rtol=1e-6, atol=1e-7)

    def test_empty_surfaces_and_uncovered_queries_never_gain_a_reconstructed_face(self):
        stencil = reconstruction.prepare(np.empty((0, 3)), np.empty((0, 3), int), np.empty(0, int))
        nodal = reconstruction.vertex_values(stencil, [])
        self.assertEqual(reconstruction.sample_hits(stencil, nodal, np.empty(0, int), np.empty((0, 3))).size, 0)
        vertices, faces = patch_geometry()
        stencil = reconstruction.prepare(vertices, faces, [1, 1])
        nodal = reconstruction.vertex_values(stencil, [1., 2.])
        with self.assertRaisesRegex(ValueError, 'containing'):
            reconstruction.sample_hits(stencil, nodal, np.array([-1]), [[1., 0., 0.]])
        source = continuous_frame(); source['surface_reconstruction_version'] = 2
        with self.assertRaisesRegex(ValueError, 'version'):
            sampling.prepare(source)


if __name__ == '__main__':
    unittest.main()
