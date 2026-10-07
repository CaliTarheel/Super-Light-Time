"""Actual disconnected spherical contacts, including triangulation adversaries."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
import numpy as np

import collision_contacts as contacts
import collision_fronts
import material_surface
import mesh_coverage
import structure_engine
from deforming_regions import deform
from orientation import rotation_matrix
from ridge_geometry import rotate

R = 6371.


def unit(x):
    x = np.asarray(x, float)
    return x/np.linalg.norm(x, axis=-1, keepdims=True)


def patch(x, y, owner, keep=lambda x, y: True):
    xx, yy = np.meshgrid(x, y)
    vertices = unit(np.column_stack((np.ones(xx.size), xx.ravel()/R, yy.ravel()/R)))
    faces = []
    for j in range(len(y)-1):
        for i in range(len(x)-1):
            if keep((x[i]+x[i+1])/2, (y[j]+y[j+1])/2):
                a = j*len(x)+i
                faces.extend(((a, a+1, a+1+len(x)), (a, a+1+len(x), a+len(x))))
    return vertices, np.array(faces), np.full(len(faces), owner), np.ones(len(faces), int)


def refine(p):
    vertices, faces, owners, kinds = p
    points = list(vertices); edges = {}; output = []
    for a, b, c in faces:
        mid = []
        for i, j in ((a, b), (b, c), (c, a)):
            key = tuple(sorted((int(i), int(j))))
            if key not in edges:
                edges[key] = len(points); points.append(unit(vertices[i]+vertices[j]))
            mid.append(edges[key])
        ab, bc, ca = mid
        output.extend(((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)))
    return np.array(points), np.array(output), np.repeat(owners, 4), np.repeat(kinds, 4)


def fixture(*, coarse=False, refine_top=False, refine_under=False):
    # Both sheets are connected. Only the two legs of the inverted U overlap
    # the horizontal sheet; the bridge is outside, with a 1200 km wide gap.
    a = patch([-1100, 1100] if coarse else [-1100, -900, -600, -300, 0, 300, 600, 900, 1100],
              [-200, 200] if coarse else [-200, 0, 200], 0)
    b = patch([-900, -600, -300, 0, 300, 600, 900], [0, 200, 400, 600, 800], 1,
              lambda x, y: abs(x) >= 600 or y >= 600)
    mesh = material_surface.initialize_surface(*(refine(a) if refine_top else a))
    material_surface.append_surface(mesh, *(refine(b) if refine_under else b))
    n = len(mesh['faces'])
    s = SimpleNamespace(material_surface=mesh, parcel_patch=mesh['face_id'].copy(),
        parcel_plate=mesh['face_owner'].copy(), kind=mesh['face_kind'].copy(),
        mass=mesh['reference_area_km2'].copy(), pos=material_surface.face_centres(mesh),
        relief=np.where(mesh['face_owner'] == 0, 2780., 780.), suture=np.zeros(n),
        plate_uid=np.array([101, 202]), t=0., omega=np.array([[0., 0., 0.], [.015, 0., 0.]]))
    structure_engine.initialize_parcels(s)
    contacts.refresh(s)
    return s


def fronts(s):
    return sorted(contacts.iter_motion_fronts(s.collision_contacts), key=lambda r: r['center'][1])


def empty_boundaries():
    return dict(bmid=np.empty((0, 3)), bn=np.empty((0, 3)), bl=np.empty(0),
                bp=np.empty(0, int), bq=np.empty(0, int))


class CollisionLocalFrontTests(unittest.TestCase):
    def test_disconnected_closing_and_opening_patches_cannot_cancel(self):
        s = fixture(); rows = fronts(s)
        self.assertEqual(len(np.unique(s.parcel_collision_sheet)), 2)
        self.assertEqual(len(s.collision_contacts), 1)
        self.assertEqual(len(rows), 2)
        self.assertLess(rows[0]['normal_speed_km_myr'], -11.)
        self.assertGreater(rows[1]['normal_speed_km_myr'], 11.)
        self.assertLess(abs(s.collision_contacts[0]['normal_speed_km_myr']), 1e-10)
        # Independent finite differences of actual clipped area establish
        # which physical patch is closing, without using the front normal.
        areas = []
        for dt in (-.005, .005):
            vertices = s.material_surface['vertices'].copy()
            selected = s.material_surface['vertex_owner'] == 1
            vertices[selected] = rotate(vertices[selected], s.omega[1]*dt)
            overlap = mesh_coverage.material_overlaps(vertices, s.material_surface['faces'], s.parcel_collision_sheet)
            center_x = s.pos[overlap['first'], 1]+s.pos[overlap['second'], 1]
            areas.append([overlap['area_km2'][center_x < 0].sum(), overlap['area_km2'][center_x > 0].sum()])
        derivative = (np.array(areas[1])-areas[0])/.01
        self.assertGreater(derivative[0], 3298.)
        self.assertLess(derivative[1], -3298.)
        # One reduced normal per connected patch approximates its integrated
        # area motion closely here; this is a measured geometric bound.
        np.testing.assert_allclose([-r['normal_speed_km_myr']*r['length_km'] for r in rows], derivative, rtol=.001)
        boundaries = contacts.deformation_boundaries(s, empty_boundaries())
        self.assertEqual(len(boundaries['bl']), 1)
        deformation = deform(s.material_surface, s.omega, boundaries, 2., belt_width_km=200., require_material_contact=True)
        self.assertGreater(float(np.max(np.abs(np.log(deformation['face_area_km2']/s.material_surface['area_km2'])))), .01)
        before = s.omega.copy(); contacts.resist_motion(s, 2.)
        # Plate balance owns velocities; this retired API records diagnostics.
        # Actual weld forces are tested in test_collision_force_continuity.
        np.testing.assert_array_equal(s.omega,before)
        self.assertEqual(s.collision_resistance_diagnostics['distinct_motion_fronts'],2)

    def test_coarse_face_cannot_bridge_separate_clipped_polygons(self):
        s = fixture(coarse=True); rows = fronts(s)
        self.assertEqual(len(rows), 2)
        overlap = s._collision_overlap
        face_sets = [set(overlap['first'][r['overlap_indices']]) for r in rows]
        self.assertTrue(face_sets[0] & face_sets[1], 'Adversary must share an actual coarse upper triangle.')
        self.assertLess(rows[0]['center'][1], -.1)
        self.assertGreater(rows[1]['center'][1], .1)

    def test_refining_either_sheet_preserves_geometry_motion_and_polarity(self):
        baseline = fixture(coarse=True); reference = fronts(baseline)
        contacts.resist_motion(baseline, 2.)
        for a, b in ((True, False), (False, True), (True, True)):
            with self.subTest(top=a, under=b):
                s = fixture(coarse=True, refine_top=a, refine_under=b)
                rows = fronts(s); self.assertEqual(len(rows), 2)
                for row, expected in zip(rows, reference):
                    for key in ('center', 'normal', 'overlap_area_km2', 'length_km', 'normal_speed_km_myr', 'footprint_radius_km'):
                        np.testing.assert_allclose(row[key], expected[key], rtol=2e-10, atol=2e-10, err_msg=key)
                contacts.resist_motion(s, 2.)
                np.testing.assert_allclose(s.omega, baseline.omega, rtol=2e-12, atol=1e-14)

    def test_polar_dateline_rotation_preserves_local_geometry(self):
        original = fixture(); reference = fronts(original)
        for angles in (dict(yaw=179.), dict(pitch=89.), dict(yaw=37., pitch=-63., roll=18.)):
            s = deepcopy(original); matrix = rotation_matrix(angles)
            s.material_surface['vertices'] = s.material_surface['vertices']@matrix
            s.pos = s.pos@matrix; s.omega = s.omega@matrix
            material_surface.refresh_geometry(s.material_surface); contacts.refresh(s)
            rows = list(contacts.iter_motion_fronts(s.collision_contacts))
            for expected in reference:
                row = min(rows, key=lambda r: np.linalg.norm(np.array(r['center'])-np.array(expected['center'])@matrix))
                for key in ('center', 'normal'):
                    np.testing.assert_allclose(row[key], np.array(expected[key])@matrix, atol=5e-11)
                for key in ('length_km', 'normal_speed_km_myr', 'footprint_radius_km', 'overlap_area_km2'):
                    np.testing.assert_allclose(row[key], expected[key], rtol=2e-10, atol=2e-9)

    def test_suture_only_advances_on_closing_patch_and_once_per_epoch(self):
        s = fixture(); before = deepcopy(s); rows = fronts(s)
        opening = np.unique(np.r_[s._collision_overlap['first'][rows[1]['overlap_indices']],
                                  s._collision_overlap['second'][rows[1]['overlap_indices']]])
        closing = np.unique(np.r_[s._collision_overlap['first'][rows[0]['overlap_indices']],
                                  s._collision_overlap['second'][rows[0]['overlap_indices']]])
        self.assertFalse(set(opening) & set(closing))
        s.parcel_collision_suture[opening] = .3; s.suture[opening] = .3
        s.t = 2.; contacts.refresh(s, 2.)
        np.testing.assert_array_equal(s.parcel_collision_suture[opening], .3)
        self.assertTrue(np.all(s.parcel_collision_suture[closing] > .1))
        expected_convergence = -rows[0]['normal_speed_km_myr'] # half closing area, two Myr
        self.assertAlmostEqual(s.collision_contacts[0]['cumulative_convergence_km'], expected_convergence, places=10)
        old = s.parcel_collision_suture.copy(); contacts.refresh(s, 2.)
        np.testing.assert_array_equal(s.parcel_collision_suture, old)
        for key in ('mass', 'parcel_patch', 'parcel_collision_sheet'):
            np.testing.assert_array_equal(getattr(s, key), getattr(before, key))
        np.testing.assert_array_equal(s.structure['thickness_km'], before.structure['thickness_km'])
        np.testing.assert_array_equal(s.material_surface['area_km2'], before.material_surface['area_km2'])

    def test_actual_spherical_point_touch_is_not_an_arc_connection(self):
        a, b, c, d = unit([[1., 0., 0.], [1., .1, 0.], [1., .2, 0.], [1., .05, 0.]])
        tol = collision_fronts.CONNECT_TOLERANCE_KM/R
        self.assertEqual(collision_fronts.shared_arc(a, b, b, c, tol), 0.)
        self.assertGreater(collision_fronts.shared_arc(a, b, d, c, tol), .04)
        first = unit([[1., 0., 0.], [1., .1, 0.], [1., 0., .1]])
        second = unit([[1., .1, 0.], [1., .2, 0.], [1., .1, -.1]])
        self.assertFalse(collision_fronts._touches(first, second, tol))

    def test_deformation_admission_uses_current_omega_after_forces(self):
        s = fixture(); historical = deepcopy(s.collision_contacts)
        initial = contacts.deformation_boundaries(s, empty_boundaries())
        self.assertEqual(len(initial['bl']), 1)
        self.assertLess(initial['bmid'][0, 1], -.1)
        s.omega *= -1.
        reversed_motion = contacts.deformation_boundaries(s, empty_boundaries())
        self.assertEqual(len(reversed_motion['bl']), 1)
        self.assertGreater(reversed_motion['bmid'][0, 1], .1)
        self.assertEqual(s.collision_contacts, historical)
        s.omega[:] = 0.
        self.assertEqual(len(contacts.deformation_boundaries(s, empty_boundaries())['bl']), 0)

    def test_coincident_footprints_do_not_invent_velocity_normals(self):
        from tests.test_collision_polarity import fixture as coincident
        s = coincident(); s.omega[:] = [[0., .01, -.02], [.02, -.01, .03]]
        contacts.refresh(s)
        rows = fronts(s); self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]['direction_resolved'])
        np.testing.assert_array_equal(rows[0]['normal'], 0.)
        self.assertEqual(rows[0]['normal_speed_km_myr'], 0.)
        self.assertEqual(len(contacts.deformation_boundaries(s, empty_boundaries())['bl']), 0)


if __name__ == '__main__':
    unittest.main()
