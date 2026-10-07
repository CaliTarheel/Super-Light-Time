"""Resolved volcanic columns meet their own submerged boundary continuously."""
from copy import deepcopy
import unittest
from unittest.mock import patch
import numpy as np
import arc_surface
import material_reconstruction
import native_arc_material
import native_frame_sampling as sampling
import gospl_export
from orientation import orient_frame, rotation_matrix
try:
    from .test_native_gospl_sampling import frame
except ImportError:
    from test_native_gospl_sampling import frame


def unit(x): return x/np.linalg.norm(x, axis=-1, keepdims=True)


def volcanic_frame():
    source = frame(centers=[])
    geometry = native_arc_material._patch(np.array([1., 0., 0.]), np.array([0., 1., 0.]), 25000.)
    basal = -5651.+2473.*np.exp(-.0278*60.)+150.
    profile = arc_surface.birth_profile(geometry['profile_radius'], basal)
    n = len(geometry['faces'])
    source.update(surface_reconstruction_version=1, arc_material_version=1, arc_surface_version=1,
        material_vertices=geometry['vertices'], material_faces=geometry['faces'],
        material_owner=np.ones(n, np.int32), material_kind=np.full(n, 3, np.uint8),
        material_face_id=np.arange(n, dtype=np.int64)+100,
        material_height_m=profile['height_m'], material_erosion_rate_m_myr=np.zeros(n),
        material_arc_id=np.ones(n, np.int64), material_arc_basal_m=np.full(n, basal))
    return source


def edge_probes(source, epsilon):
    context = sampling.prepare(source); stencil = context['material_reconstruction']
    faces = source['material_faces']; v = source['material_vertices']
    edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]], axis=2).reshape(-1, 2)
    edges, counts = np.unique(edges, axis=0, return_counts=True); edges = edges[counts == 1]
    a, b = v[edges].transpose(1, 0, 2); middle = unit(a+b); normal = unit(np.cross(a, b))
    center = unit(v.sum(axis=0)); normal *= np.sign(normal@center)[:, None]
    inside = middle*np.cos(epsilon)+normal*np.sin(epsilon)
    outside = middle*np.cos(epsilon)-normal*np.sin(epsilon)
    return sampling.sample_frame(source, inside, context), sampling.sample_frame(source, outside, context)


class ArcSurfaceTests(unittest.TestCase):
    def test_birth_has_emerged_summit_submerged_flanks_and_bounded_actual_columns(self):
        profile = arc_surface.birth_profile(np.linspace(0, 1, 101), -6200.)
        self.assertEqual(profile['height_m'][0], 970.)
        self.assertEqual(profile['height_m'][-1], -6200.)
        self.assertTrue(np.all(np.diff(profile['height_m']) <= 0.))
        self.assertTrue(np.any(profile['height_m'] < 0.))
        self.assertEqual((profile['thickness_km'].min(), profile['thickness_km'].max()), (8., 25.))
        source = volcanic_frame(); value = sampling.sample_frame(source, np.array([[1., 0., 0.]]))
        np.testing.assert_allclose(value['elevation'], 970., atol=1e-9)

    def test_every_open_arc_edge_converges_to_ocean_not_a_flat_sided_plateau(self):
        source = volcanic_frame()
        coarse = edge_probes(source, 1e-5); fine = edge_probes(source, 1e-8)
        for inner, outer in (coarse, fine):
            np.testing.assert_array_equal(inner['crust'], 3)
            np.testing.assert_array_equal(outer['crust'], 0)
        coarse_jump = np.max(abs(coarse[0]['elevation']-coarse[1]['elevation']))
        fine_jump = np.max(abs(fine[0]['elevation']-fine[1]['elevation']))
        self.assertLess(fine_jump, .05)
        self.assertLess(fine_jump, coarse_jump*.002)

    def test_curved_surrounding_bathymetry_matches_along_edges_not_only_vertices(self):
        source = volcanic_frame()
        centers = unit(source['mesh_vertices'][source['mesh_faces']].sum(axis=1))
        source['mesh_age_myr'] = 60.+35.*centers[:, 1]
        source['mesh_ocean_relief_m'] = 150.+300.*centers[:, 2]
        inner, outer = edge_probes(source, 1e-8)
        self.assertLess(np.max(abs(inner['elevation']-outer['elevation'])), .05)
        self.assertGreater(np.ptp(outer['elevation']), 1.)

    def test_material_columns_interior_values_and_erosion_ledgers_are_unchanged(self):
        source = volcanic_frame(); original = deepcopy(source)
        context = sampling.prepare(source)
        self.assertEqual(context['arc_surface']['open_edge_count'], 8)
        faces = np.flatnonzero(~context['arc_surface']['affected_faces'])
        weights = np.full((len(faces), 3), 1/3)
        points = unit(source['material_vertices'][source['material_faces'][faces]].sum(axis=1))
        base = material_reconstruction.sample_hits(context['material_reconstruction'],
            context['material_height_m_vertices'], faces, weights)
        actual = sampling.sample_frame(source, points, context)
        np.testing.assert_allclose(actual['elevation'], base, atol=1e-9)
        np.testing.assert_array_equal(source['material_height_m'], original['material_height_m'])
        np.testing.assert_array_equal(source['material_erosion_rate_m_myr'], original['material_erosion_rate_m_myr'])

    def test_legacy_arc_profiles_keep_the_exact_prior_sampling(self):
        source = volcanic_frame(); old = deepcopy(source)
        for field in ('arc_surface_version', 'arc_material_version', 'material_arc_id', 'material_arc_basal_m'):
            del old[field]
        inner, outer = edge_probes(old, 1e-8)
        self.assertGreater(np.max(abs(inner['elevation']-outer['elevation'])), 500.)
        flagged_but_disabled = deepcopy(source); flagged_but_disabled['arc_surface_version'] = 0
        other, _ = edge_probes(flagged_but_disabled, 1e-8)
        np.testing.assert_array_equal(other['elevation'], inner['elevation'])

    def test_disconnected_overlapping_patches_and_nonarc_faces_do_not_share_nodes(self):
        source = volcanic_frame(); v, f = source['material_vertices'], source['material_faces']
        n = len(f); vertices = np.r_[v, v]; faces = np.r_[f, f+len(v)]
        stencil = material_reconstruction.prepare(vertices, faces, np.ones(2*n, int))
        prepared = arc_surface.prepare(stencil, np.r_[np.ones(n, int), np.zeros(n, int)])
        self.assertTrue(prepared['affected_faces'][:n].any())
        self.assertFalse(prepared['affected_faces'][n:].any())
        nodal = material_reconstruction.vertex_values(stencil, np.r_[source['material_height_m'], np.full(n, 8888.)])
        indices = np.arange(n, 2*n); weights = np.full((n, 3), 1/3)
        values = arc_surface.apply(prepared, nodal, indices, weights, np.full(n, -6000.))
        np.testing.assert_allclose(values, 8888.)

    def test_connected_neighbour_is_not_misidentified_as_open_water(self):
        source = volcanic_frame(); stencil = material_reconstruction.prepare(source['material_vertices'],
            source['material_faces'], source['material_owner'])
        arc_ids = source['material_arc_id'].copy(); arc_ids[0] = 0
        prepared = arc_surface.prepare(stencil, arc_ids)
        # Relabeling an interior triangle cannot create three phantom coasts.
        self.assertEqual(prepared['open_edge_count'], 8)
        self.assertFalse(prepared['boundary_nodes'][stencil['face_vertices'][0]].any())

    def test_pole_and_seam_rotation_preserve_profile_and_exact_edge_ties(self):
        source = volcanic_frame(); angles = dict(yaw=179., pitch=88., roll=-17.)
        rotated = orient_frame(source, angles)
        points = unit(source['material_vertices'][source['material_faces']].sum(axis=1))
        a = sampling.sample_frame(source, points)
        b = sampling.sample_frame(rotated, points@rotation_matrix(angles))
        np.testing.assert_allclose(a['elevation'], b['elevation'], atol=2e-8)
        inner, outer = edge_probes(rotated, 1e-8)
        self.assertLess(np.max(abs(inner['elevation']-outer['elevation'])), .05)

    def test_differing_material_and_ocean_thermal_hosts_do_not_leave_a_heat_cliff(self):
        source = volcanic_frame(); source['time_myr'] = 2.
        source['ridge_episodes'] = [dict(started_myr=0., peak_myr=2., end_myr=12., center=[1., 0., 0.],
            radius_km=1000., thermal_peak_m=500., volcanic_budget_m=200., overriding_plate_uid=20)]
        inner, outer = edge_probes(source, 1e-8)
        self.assertGreater(inner['thermal_support_m'].max(), 450.)
        self.assertLess(inner['display_thermal_support_m'].max(), .01)
        self.assertLess(np.max(abs(inner['elevation']-outer['elevation'])), .05)

    def test_schema_requires_explicit_profile_version_and_complete_fields(self):
        for kind in ('missing_id', 'missing_basal', 'negative_id', 'wrong_kind', 'unflagged', 'legacy_surface'):
            with self.subTest(kind=kind):
                source = volcanic_frame()
                if kind == 'missing_id': del source['material_arc_id']
                if kind == 'missing_basal': del source['material_arc_basal_m']
                if kind == 'negative_id': source['material_arc_id'][0] = -1
                if kind == 'wrong_kind': source['material_kind'][0] = 2
                if kind == 'unflagged': del source['arc_material_version']
                if kind == 'legacy_surface': del source['surface_reconstruction_version']
                with self.assertRaises(ValueError): sampling.prepare(source)

    def test_accretion_keeps_inherited_arc_profile_after_juvenile_kind_becomes_continent(self):
        before = volcanic_frame(); after = deepcopy(before); after['material_kind'][:] = 1
        points = unit(before['material_vertices'][before['material_faces']].sum(axis=1))
        a, b = sampling.sample_frame(before, points), sampling.sample_frame(after, points)
        np.testing.assert_array_equal(a['elevation'], b['elevation'])
        np.testing.assert_array_equal(b['crust'], 1)

    def test_actual_engine_snapshot_and_native_export_use_identical_arc_reconstruction(self):
        from native_engine import Simulation
        s = Simulation(dict(width=128, height=64, mesh_level=2, mechanics_nodes=128, plate_count=4,
                            duration_myr=2), dict(width=48, height=24, crust=np.zeros(48*24, np.uint8)))
        cell = int(np.argmax(s.xyz[:, 0]))
        native_arc_material.add_arc_crust(s, np.array([cell]), np.array([500000.]))
        s._rasterize(); s._boundaries(); s._update_surface_domains()
        saved = s.snapshot()
        lon, lat = np.meshgrid((np.arange(s.w)+.5)*2*np.pi/s.w-np.pi,
                              np.pi/2-(np.arange(s.h)+.5)*np.pi/s.h)
        points = np.column_stack((np.cos(lat).ravel()*np.cos(lon).ravel(),
                                   np.cos(lat).ravel()*np.sin(lon).ravel(), np.sin(lat).ravel()))
        reconstructed = sampling.sample_frame(saved, points)
        self.assertGreater(np.count_nonzero(saved['crust'] == 3), 0)
        self.assertTrue(np.any(saved['elevation'][saved['crust'] == 3] < 0.))
        np.testing.assert_array_equal(reconstructed['plate'], saved['plate'])
        np.testing.assert_array_equal(reconstructed['crust'], saved['crust'])
        np.testing.assert_allclose(reconstructed['elevation'], saved['elevation'], atol=.001, rtol=0.)

    def test_query_is_independent_of_display_raster_and_export_retains_raw_column_budget(self):
        first = volcanic_frame(); second = deepcopy(first); second['time_myr'] = 2.
        for value in (first, second):
            count = len(value['material_faces']); value.update(material_transport_version=1,
                material_root_id=value['material_face_id'].copy(), material_reference_corners=np.tile(np.eye(3), (count, 1, 1)),
                material_erosion_total_m=np.zeros(count))
        second['mesh_ocean_relief_m'] -= 250.
        context = sampling.prepare(first); faces = np.flatnonzero(context['arc_surface']['affected_faces'])
        points = unit(first['material_vertices'][first['material_faces'][faces]].sum(axis=1))
        with patch.object(gospl_export, 'sample', side_effect=AssertionError('No display raster lookup')):
            fields, _ = gospl_export.interval_forcing(points, first, second, {})
        np.testing.assert_array_equal(fields['upsub'], 0.)
        self.assertGreater(np.max(abs(fields['display_geometric_rate'])), 0.)
        np.testing.assert_array_equal(fields['display_geometric_rate'], fields['reconstruction_difference_rate'])


if __name__ == '__main__': unittest.main()
