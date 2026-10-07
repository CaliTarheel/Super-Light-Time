"""Actual contained volcanic unions have no artificial overlap-edge cliffs."""
from copy import deepcopy
import unittest
from unittest.mock import patch
import numpy as np
import adaptive_material
import native_frame_sampling as sampling
import mesh_geometry
import gospl_export
from orientation import orient_frame, rotation_matrix
try:
    from .test_arc_surface import volcanic_frame, unit
    from .test_native_gospl_sampling import triangle
except ImportError:
    from test_arc_surface import volcanic_frame, unit
    from test_native_gospl_sampling import triangle


SCALARS = ('material_owner', 'material_kind', 'material_height_m',
           'material_erosion_rate_m_myr', 'material_arc_basal_m')


def overlapping_arcs(different_owners=False):
    source = volcanic_frame(); v = source['material_vertices'].copy(); f = source['material_faces'].copy()
    angle = .013
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    source['material_vertices'] = np.r_[v, v@rotation.T]
    source['material_faces'] = np.r_[f, f+len(v)]
    for name in SCALARS: source[name] = np.r_[source[name], source[name]]
    source['material_face_id'] = np.arange(2*len(f), dtype=np.int64)+100
    source['material_arc_id'] = np.r_[np.ones(len(f), np.int64), np.full(len(f), 2, np.int64)]
    if different_owners: source['material_owner'][:len(f)] = 0
    source['arc_surface_version'] = 2
    return source


def probes(source, epsilon):
    vertices, faces = source['material_vertices'], source['material_faces']
    edges = faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    _, first, count = np.unique(np.sort(edges, axis=1), axis=0, return_index=True, return_counts=True)
    chosen = first[count == 1]; owner = np.repeat(np.arange(len(faces)), 3)[chosen]
    edges = edges[chosen]
    middle = unit(vertices[edges].sum(axis=1)); center = unit(vertices[faces[owner]].sum(axis=1))
    inward = unit(center-middle*np.sum(middle*center, axis=1)[:, None])
    return unit(middle+epsilon*inward), unit(middle-epsilon*inward)


class ArcSurfaceEnvelopeTests(unittest.TestCase):
    def assert_continuous(self, source):
        context = sampling.prepare(source); jumps = []
        for epsilon in (1e-6, 1e-8):
            inside, outside = probes(source, epsilon)
            a, b = sampling.sample_frame(source, inside, context), sampling.sample_frame(source, outside, context)
            jumps.append(float(np.max(np.abs(a['elevation']-b['elevation']))))
        self.assertLess(jumps[1], .1)
        self.assertLess(jumps[1], jumps[0]*.011)

    def test_same_owner_overlap_is_continuous_and_preserves_legacy_version_one(self):
        source = overlapping_arcs(); self.assert_continuous(source)
        legacy = deepcopy(source); legacy['arc_surface_version'] = 1
        a, b = probes(legacy, 1e-8)
        jump = np.abs(sampling.sample_frame(legacy, a)['elevation']-sampling.sample_frame(legacy, b)['elevation'])
        self.assertGreater(jump.max(), 3300.)
        # Native columns and identities are never rewritten by the selector.
        for name in SCALARS+('material_faces', 'material_face_id', 'material_arc_id'):
            np.testing.assert_array_equal(source[name], legacy[name])

    def test_different_owner_overlap_selects_height_even_against_control_owner(self):
        source = overlapping_arcs(different_owners=True); self.assert_continuous(source)
        inside, _ = probes(source, 1e-8)
        result = sampling.sample_frame(source, inside)
        self.assertTrue(np.any(result['plate'] == 1))
        self.assertTrue(np.all(source['mesh_plate'] == 0))
        chosen = result['material_face'] >= 0
        np.testing.assert_array_equal(result['plate'][chosen], source['material_owner'][result['material_face'][chosen]])

    def test_open_apron_meets_actual_nonarc_bedrock_including_below_ocean(self):
        for base in (250., -7000.):
            with self.subTest(base=base):
                source = volcanic_frame(); n = len(source['material_faces']); nv = len(source['material_vertices'])
                source['material_vertices'] = np.r_[source['material_vertices'], triangle([1., 0., 0.], .1)]
                source['material_faces'] = np.r_[source['material_faces'], [[nv, nv+1, nv+2]]]
                values = dict(material_owner=0, material_kind=1, material_height_m=base,
                              material_erosion_rate_m_myr=0., material_arc_basal_m=0., material_arc_id=0, material_face_id=999)
                for name, value in values.items(): source[name] = np.r_[source[name], np.asarray(value, dtype=source[name].dtype)]
                source['arc_surface_version'] = 2
                # Probe only the arc's actual edges, not the unrelated legacy continent coastline.
                arc_only = deepcopy(source); arc_only['material_faces'] = source['material_faces'][:n]
                a, b = probes(arc_only, 1e-8)
                inside, outside = sampling.sample_frame(source, a), sampling.sample_frame(source, b)
                self.assertLess(np.max(np.abs(inside['elevation']-outside['elevation'])), .1)
                np.testing.assert_allclose(outside['elevation'], base, atol=1e-9)

    def test_buried_arc_does_not_replace_ocean_or_fill_outside_real_geometry(self):
        source = volcanic_frame(); source['arc_surface_version'] = 2; source['material_height_m'][:] = -8000.
        points = unit(source['material_vertices'][source['material_faces']].sum(axis=1))
        points = np.r_[points, [[0., 1., 0.], [0., 0., 1.]]]
        result = sampling.sample_frame(source, points)
        np.testing.assert_array_equal(result['material_face'], -1)
        np.testing.assert_array_equal(result['crust'], 0)
        np.testing.assert_allclose(result['elevation'], -5651.+2473.*np.exp(-.0278*60.)+150.)

    def test_height_ties_use_stable_identity_independent_of_candidate_order(self):
        source = overlapping_arcs(different_owners=True); n = len(source['material_faces'])//2
        source['material_vertices'][17:] = source['material_vertices'][:17]
        source['material_face_id'][:n] += 1000
        point = np.array([[1., 0., 0.]])
        context = sampling.prepare(source); cells, _ = mesh_geometry.locate_points(point, context['locator'])
        hits = mesh_geometry.locate_points(point, context['material_locator'], all_hits=True)
        a = sampling.select_exposed_material(context, point, cells, np.array([0]), hits)
        b = sampling.select_exposed_material(context, point, cells, np.array([0]), tuple(x[::-1] for x in hits))
        np.testing.assert_array_equal(a['material_face'], b['material_face'])
        self.assertGreaterEqual(a['material_face'][0], n)

    def test_refined_overlapping_patches_retain_continuity_at_all_actual_open_edges(self):
        source = overlapping_arcs(different_owners=True)
        mapping = adaptive_material.refine(source['material_vertices'], source['material_faces'],
            source['material_owner'], source['material_kind'], desired_edge_km=10., max_level=1, max_faces=1000)
        parents = mapping['source_face']
        self.assertGreater(len(parents), len(source['material_faces']))
        for name in SCALARS+('material_arc_id',): source[name] = source[name][parents].copy()
        source['material_face_id'] = np.arange(len(parents), dtype=np.int64)+1000
        source['material_vertices'], source['material_faces'] = mapping['vertices'], mapping['faces']
        self.assert_continuous(source)

    def test_pole_rotation_and_display_grid_independence(self):
        source = overlapping_arcs(different_owners=True); inside, outside = probes(source, 1e-7)
        points = np.r_[inside, outside]; original = sampling.sample_frame(source, points)
        rotation = dict(yaw=174., pitch=89., roll=-43.)
        rotated = orient_frame(source, rotation)
        moved = points@rotation_matrix(rotation)
        with patch.object(gospl_export, 'cells_at', side_effect=AssertionError('No review raster lookup')):
            result = sampling.sample_frame(rotated, moved)
        np.testing.assert_allclose(result['elevation'], original['elevation'], atol=2e-7)
        np.testing.assert_array_equal(result['material_face'], original['material_face'])

    def test_thermal_transition_is_continuous_but_raw_owner_heat_is_not_diluted(self):
        source = overlapping_arcs(different_owners=True); source['time_myr'] = 2.
        source['ridge_episodes'] = [dict(started_myr=0., peak_myr=2., end_myr=12., center=[1., 0., 0.],
            radius_km=1000., thermal_peak_m=500., volcanic_budget_m=200., overriding_plate_uid=20)]
        self.assert_continuous(source)
        points = unit(source['material_vertices'][source['material_faces']].sum(axis=1))
        result = sampling.sample_frame(source, points)
        selected = result['plate'] == 1
        self.assertTrue(selected.any())
        self.assertGreater(result['thermal_support_m'][selected].max(), 450.)
        self.assertGreater(np.max(result['thermal_support_m']-result['display_thermal_support_m']), 100.)

    def test_underlying_surface_change_does_not_create_false_column_uplift(self):
        first = overlapping_arcs(different_owners=True); second = deepcopy(first); second['time_myr'] = 2.
        for value in (first, second):
            count = len(value['material_faces'])
            value.update(material_transport_version=1, material_root_id=value['material_face_id'].copy(),
                material_reference_corners=np.tile(np.eye(3), (count, 1, 1)), material_erosion_total_m=np.zeros(count))
        second['mesh_ocean_relief_m'] -= 250.
        context = sampling.prepare(first); faces = np.flatnonzero(context['arc_surface']['affected_faces'])
        points = unit(first['material_vertices'][first['material_faces'][faces]].sum(axis=1))
        with patch.object(gospl_export, 'sample', side_effect=AssertionError('No display raster lookup')):
            fields, _ = gospl_export.interval_forcing(points, first, second, {})
        np.testing.assert_array_equal(fields['upsub'], 0.)
        self.assertGreater(np.max(abs(fields['display_geometric_rate'])), 0.)
        np.testing.assert_array_equal(fields['display_geometric_rate'], fields['reconstruction_difference_rate'])


if __name__ == '__main__': unittest.main()
