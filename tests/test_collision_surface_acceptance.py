"""Independent analytic oracles for the explicit collision support surface.

Expected support comes from known sheet columns and spherical footprint areas,
never from the production support diagnostic or surface residual. These fixtures
do not run the tectonic simulation or reinterpret an archived world in place.
"""
from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np

import collision_contacts
import collision_surface
import material_surface
import native_frame_sampling as sampling
import structure_engine
from orientation import orient_frame, rotation_matrix
from tests.test_native_gospl_sampling import frame as ocean_frame


RADIUS_KM = 6371.
# Independent statement of the agreed densities and units, not the tested
# module's imported coefficient. A changed constitutive law needs a new oracle.
AIRY = 1000. * (3300. - 2800.) / 3300.


def unit(x):
    x = np.asarray(x, float)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def triangle():
    return unit([[1., -.08, -.05], [1., .08, -.05], [1., 0., .08]])


def triangle_area(corners):
    """Independent scalar spherical-excess calculation for analytic fixtures."""
    a, b, c = np.asarray(corners, float)
    numerator = abs(float(np.linalg.det(np.array([a, b, c]))))
    denominator = 1. + a @ b + b @ c + c @ a
    return 2. * np.arctan2(numerator, denominator) * RADIUS_KM**2


def divide(corners):
    a, b, c = corners
    ab, bc, ca = unit([a+b, b+c, c+a])
    vertices = np.array([a, b, c, ab, bc, ca])
    faces = np.array([[0, 3, 5], [3, 1, 4], [5, 4, 2], [3, 4, 5]])
    return vertices, faces


def world(specs, *, relations=None):
    """Each spec is (corners, thickness, raw height, conformingly refined)."""
    surface = None
    roots, thick, heights = [], [], []
    for owner, (corners, thickness, height, refined) in enumerate(specs):
        vertices, faces = divide(corners) if refined else (corners, np.array([[0, 1, 2]]))
        count = len(faces)
        args = (vertices, faces, np.full(count, owner), np.ones(count, int))
        if surface is None:
            surface = material_surface.initialize_surface(*args)
        else:
            material_surface.append_surface(surface, *args)
        roots.extend([1000 + owner] * count)
        thick.extend([thickness] * count)
        heights.extend([height] * count)
    count = len(roots)
    s = SimpleNamespace(
        collision_surface_version=1, material_surface=surface,
        parcel_patch=surface['face_id'].copy(),
        parcel_plate=surface['face_owner'].copy(), kind=surface['face_kind'].copy(),
        mass=surface['reference_area_km2'].copy(), pos=material_surface.face_centres(surface),
        relief=np.asarray(heights) - 220., suture=np.zeros(count),
        material_lineage=dict(root_id=np.asarray(roots, np.int64)),
        t=0., plate_uid=np.arange(len(specs), dtype=np.int64) + 101,
        trace_xyz=np.empty((0, 3)), trace_patch=np.empty(0, np.int64))
    structure_engine.initialize_parcels(s)
    s.structure['thickness_km'][:] = thick
    s.structure['reference_thickness_km'][:] = thick
    collision_contacts.ensure_fields(s)
    if relations is None:
        relations = [(i+1, i+2) for i in range(len(specs)-1)]
    s.collision_contacts = [dict(id=i+1, top_sheet=top, under_sheet=under, state='active')
                            for i, (top, under) in enumerate(relations)]
    return s


def saved(s, *, new=True, margins=False):
    f = ocean_frame(centers=[])
    n = len(s.mass)
    f.update(
        plates=[dict(id=i, uid=int(uid), angular_velocity=[0., 0., 0.])
                for i, uid in enumerate(s.plate_uid)],
        material_vertices=s.material_surface['vertices'].copy(),
        material_faces=s.material_surface['faces'].copy(),
        material_face_id=s.parcel_patch.copy(), material_owner=s.parcel_plate.copy(),
        material_kind=s.kind.copy(), material_height_m=s.relief.copy()+220.,
        material_crustal_thickness_km=s.structure['thickness_km'].copy(),
        material_erosion_rate_m_myr=2.+s.parcel_plate.astype(float),
        surface_reconstruction_version=1, arc_material_version=1, arc_surface_version=2,
        material_arc_id=np.zeros(n, np.int64), material_arc_basal_m=np.zeros(n),
        **collision_contacts.snapshot_fields(s))
    if new:
        f.update(collision_surface.snapshot_fields(s))
    if margins:
        f.update(continental_margin_version=1,
                 continental_margin_parameters=dict(width_km=150., shelf_depth_m=180.))
    return f


def physical_inventory(s):
    area = np.array([triangle_area(s.material_surface['vertices'][f])
                     for f in s.material_surface['faces']])
    return dict(vertices=s.material_surface['vertices'].copy(),
                faces=s.material_surface['faces'].copy(), ids=s.parcel_patch.copy(),
                root=s.material_lineage['root_id'].copy(), mass=s.mass.copy(),
                relief=s.relief.copy(), structure=deepcopy(s.structure),
                area=area, volume=float(area @ s.structure['thickness_km']))


class CollisionSurfaceAcceptanceTests(unittest.TestCase):
    def assert_inventory_equal(self, first, second):
        for name in first:
            if name == 'structure':
                for key in first[name]:
                    np.testing.assert_array_equal(first[name][key], second[name][key])
            else:
                np.testing.assert_array_equal(first[name], second[name])

    def test_two_coincident_sheets_have_analytic_support_without_added_rock(self):
        tri = triangle()
        s = world([(tri, 35., 500., False), (tri, 27., 3000., False)])
        before = physical_inventory(s)
        collision_surface.refresh(s)
        np.testing.assert_allclose(s.parcel_collision_support_m, [27.*AIRY, 0.], rtol=2e-12)
        np.testing.assert_allclose(s.parcel_collision_load_thickness_km, [27., 0.], rtol=2e-12)
        first = s.parcel_collision_support_m.copy()
        collision_surface.refresh(s)
        sampling.sample_frame(saved(s), unit(tri.sum(axis=0))[None])
        np.testing.assert_array_equal(s.parcel_collision_support_m, first)
        self.assert_inventory_equal(before, physical_inventory(s))

    def test_three_sheet_transitive_support_counts_each_lower_column_once(self):
        tri = triangle()
        specs = [(tri, 35., 500., False), (tri, 23., 800., False), (tri, 41., 900., False)]
        s = world(specs, relations=[(1, 2), (2, 3), (1, 3)])
        collision_surface.refresh(s)
        expected = np.array([64., 41., 0.]) * AIRY
        np.testing.assert_allclose(s.parcel_collision_support_m, expected, rtol=2e-12)
        s.collision_contacts.reverse()
        collision_surface.refresh(s)
        np.testing.assert_allclose(s.parcel_collision_support_m, expected, rtol=2e-12)
        # A redundant direct edge and a transitive path are one buried sheet.
        s.collision_contacts = [dict(id=1, top_sheet=1, under_sheet=2),
                                dict(id=2, top_sheet=2, under_sheet=3)]
        collision_surface.refresh(s)
        np.testing.assert_allclose(s.parcel_collision_support_m, expected, rtol=2e-12)

    def test_partial_root_load_integral_is_invariant_to_either_sheet_refinement(self):
        tri = triangle()
        v, f = divide(tri)
        quarter = v[f[0]]
        full_area, lower_area = triangle_area(tri), triangle_area(quarter)
        expected = AIRY * 29. * lower_area / full_area
        total_volume = None
        for top_refined, under_refined in ((False, False), (True, False), (False, True), (True, True)):
            s = world([(tri, 35., 500., top_refined), (quarter, 29., 700., under_refined)])
            collision_surface.refresh(s)
            top = s.parcel_plate == 0
            np.testing.assert_allclose(s.parcel_collision_support_m[top], expected, rtol=3e-12)
            np.testing.assert_array_equal(s.parcel_collision_support_m[~top], 0.)
            inventory = physical_inventory(s)
            integrated = inventory['area'][top] @ s.parcel_collision_support_m[top]
            self.assertAlmostEqual(integrated / (AIRY*29.*lower_area), 1., places=11)
            if total_volume is None:
                total_volume = inventory['volume']
            else:
                self.assertAlmostEqual(inventory['volume']/total_volume, 1., places=12)

    def test_retained_relationship_without_geometric_overlap_has_no_support(self):
        tri = triangle()
        turned = tri @ rotation_matrix(dict(yaw=40.))
        s = world([(tri, 35., 500., False), (turned, 29., 700., False)])
        collision_surface.refresh(s)
        np.testing.assert_array_equal(s.parcel_collision_support_m, 0.)

    def test_absent_middle_sheet_does_not_break_transitive_local_order(self):
        tri = triangle()
        distant = tri @ rotation_matrix(dict(yaw=40.))
        s = world([(tri, 35., 500., False), (distant, 23., 800., False),
                   (tri, 41., 900., False)], relations=[(1, 2), (2, 3)])
        collision_surface.refresh(s)
        np.testing.assert_allclose(s.parcel_collision_support_m, [41.*AIRY, 0., 0.], rtol=2e-12)
        result = sampling.sample_frame(saved(s), unit(tri.sum(axis=0))[None])
        self.assertEqual(s.parcel_collision_sheet[result['material_face'][0]], 1)
        self.assertAlmostEqual(result['elevation'][0], 500.+41.*AIRY, places=7)

    def test_public_sampler_matches_independent_identity_and_unclipped_height(self):
        tri = triangle()
        s = world([(tri, 35., 500., False), (tri, 40., 3000., False), (tri, 45., 1000., False)])
        collision_surface.refresh(s)
        query = unit(.2*tri[0]+.3*tri[1]+.5*tri[2])[None]
        result = sampling.sample_frame(saved(s), query)
        expected_support = 85.*AIRY
        self.assertEqual(s.parcel_collision_sheet[result['material_face'][0]], 1)
        self.assertEqual(result['plate'][0], 0)
        self.assertAlmostEqual(result['erosion_rate_m_myr'][0], 2., places=10)
        self.assertAlmostEqual(result['selected_sheet_height_m'][0], 500., places=7)
        self.assertAlmostEqual(result['physical_stack_support_m'][0], expected_support, places=7)
        self.assertAlmostEqual(result['display_thermal_support_m'][0], 0., places=7)
        self.assertAlmostEqual(result['raw_elevation_m'][0], 500.+expected_support, places=7)
        self.assertAlmostEqual(result['elevation'][0], 500.+expected_support, places=7)
        self.assertGreater(result['elevation'][0], 9000.)

    def test_legacy_gate_keeps_known_height_crossing_case(self):
        tri = triangle()
        s = world([(tri, 35., 1000., False), (tri, 27., 4000., False)])
        collision_surface.refresh(s)
        query = unit(tri.sum(axis=0))[None]
        old = sampling.sample_frame(saved(s, new=False), query)
        self.assertAlmostEqual(old['elevation'][0], 4000., places=7)
        self.assertAlmostEqual(old['collision_surface_offset_m'][0], 3000., places=7)
        self.assertEqual(s.parcel_collision_sheet[old['material_face'][0]], 1)
        repaired = sampling.sample_frame(saved(s), query)
        self.assertAlmostEqual(repaired['elevation'][0], 1000.+27.*AIRY, places=7)
        self.assertEqual(s.parcel_collision_sheet[repaired['material_face'][0]], 1)

    def test_versioned_support_must_be_present_aligned_finite_and_nonnegative(self):
        tri = triangle()
        s = world([(tri, 35., 500., False), (tri, 27., 3000., False)])
        collision_surface.refresh(s)
        frame = saved(s)
        for bad in (None, np.array([0.]), np.array([-1., 0.]), np.array([np.nan, 0.])):
            copy = deepcopy(frame)
            if bad is None:
                copy.pop('material_collision_support_m')
            else:
                copy['material_collision_support_m'] = bad
            with self.assertRaises(ValueError):
                sampling.prepare(copy)
        copy = deepcopy(frame)
        copy['collision_surface_version'] = 123
        with self.assertRaises(ValueError):
            sampling.prepare(copy)
        copy = deepcopy(frame)
        copy.pop('collision_surface_version')
        with self.assertRaises(ValueError):
            sampling.prepare(copy)
        for bad in (None, np.array([0.]), np.array([-1., 0.]), np.array([np.inf, 0.])):
            copy = deepcopy(frame)
            if bad is None:
                copy.pop('material_collision_load_thickness_km')
            else:
                copy['material_collision_load_thickness_km'] = bad
            with self.assertRaises(ValueError):
                sampling.prepare(copy)
        mismatched = deepcopy(frame)
        mismatched['material_collision_support_m'][0] += 137.
        with self.assertRaises(ValueError):
            sampling.prepare(mismatched)

    def test_explicit_support_perturbation_changes_height_and_cannot_be_hidden_by_zero_residual(self):
        tri = triangle()
        s = world([(tri, 35., 500., False), (tri, 27., 3000., False)])
        collision_surface.refresh(s)
        frame = saved(s)
        query = unit(tri.sum(axis=0))[None]
        initial = sampling.sample_frame(frame, query)
        changed = deepcopy(frame)
        changed['material_collision_support_m'][0] += 137.
        changed['material_collision_load_thickness_km'][0] += 137./AIRY
        result = sampling.sample_frame(changed, query)
        self.assertAlmostEqual(result['elevation'][0]-initial['elevation'][0], 137., places=7)
        self.assertAlmostEqual(result['physical_stack_support_m'][0]-initial['physical_stack_support_m'][0], 137., places=7)
        np.testing.assert_array_equal(changed['material_height_m'], frame['material_height_m'])

    def test_invalid_order_cycle_and_alignment_fail_before_changing_physical_inventory(self):
        tri = triangle()
        for problem in ('missing_order', 'cycle', 'root_alignment', 'sheet_alignment'):
            s = world([(tri, 35., 500., False), (tri, 27., 3000., False)])
            if problem == 'missing_order':
                s.collision_contacts = []
            elif problem == 'cycle':
                s.collision_contacts.append(dict(id=2, top_sheet=2, under_sheet=1))
            elif problem == 'root_alignment':
                s.material_lineage['root_id'] = s.material_lineage['root_id'][:1].copy()
            else:
                s.parcel_collision_sheet = s.parcel_collision_sheet[:1].copy()
            before = physical_inventory(s)
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                collision_surface.refresh(s)
            self.assert_inventory_equal(before, physical_inventory(s))

    def test_upper_free_edge_and_lower_emergence_share_a_continuous_interface(self):
        lower = triangle()
        vertices, children = divide(lower)
        upper = vertices[children[0]]
        s = world([(upper, 35., 1000., False), (lower, 27., 4000., False)])
        collision_surface.refresh(s)
        # This edge is strictly inside the lower footprint. It is both the
        # upper free edge and the lower material's re-emergence boundary.
        a, b = upper[1], upper[2]
        middle = unit(a+b)
        normal = unit(np.cross(a, b))
        if normal @ upper[0] < 0:
            normal *= -1
        frame = saved(s, margins=True)
        differences = []
        for distance in (.1, .01, .001):
            angle = distance/RADIUS_KM
            points = np.array([np.cos(angle)*middle+np.sin(angle)*normal,
                               np.cos(angle)*middle-np.sin(angle)*normal])
            result = sampling.sample_frame(frame, points)
            np.testing.assert_array_equal(result['plate'], [0, 1])
            differences.append(abs(float(np.diff(result['elevation'])[0])))
            np.testing.assert_allclose(
                result['raw_elevation_m'], result['selected_sheet_height_m']
                + result['physical_stack_support_m'] + result['display_thermal_support_m'],
                atol=1e-8, rtol=0.)
        self.assertLess(differences[-1], .1)
        self.assertLess(differences[-1], max(1e-6, differences[-2]*.12))
        self.assertLess(differences[-2], max(1e-6, differences[-3]*.12))

    def test_support_and_public_surface_are_equivariant_at_poles_and_dateline(self):
        tri = triangle()
        s = world([(tri, 35., 500., False), (tri, 27., 3000., True)])
        collision_surface.refresh(s)
        source = saved(s)
        query = unit(np.array([.2*tri[0]+.3*tri[1]+.5*tri[2], tri.mean(axis=0)]))
        baseline = sampling.sample_frame(source, query)
        for angles in (dict(yaw=180.), dict(pitch=90.), dict(pitch=-90.), dict(yaw=43., pitch=-61., roll=27.)):
            matrix = rotation_matrix(angles)
            turned = deepcopy(s)
            turned.material_surface['vertices'] = s.material_surface['vertices'] @ matrix
            material_surface.refresh_geometry(turned.material_surface)
            collision_surface.refresh(turned)
            np.testing.assert_allclose(turned.parcel_collision_support_m, s.parcel_collision_support_m, rtol=2e-11, atol=1e-8)
            oriented = orient_frame(source, angles)
            result = sampling.sample_frame(oriented, query @ matrix)
            for name in ('selected_sheet_height_m', 'physical_stack_support_m', 'display_thermal_support_m', 'raw_elevation_m', 'elevation'):
                np.testing.assert_allclose(result[name], baseline[name], atol=1e-7, rtol=0.)
            np.testing.assert_array_equal(result['plate'], baseline['plate'])


if __name__ == '__main__':
    unittest.main()
