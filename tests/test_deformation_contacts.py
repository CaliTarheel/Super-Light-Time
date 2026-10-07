"""Detached bodies need a contact; anchored continental margins retain loading."""
import copy
import unittest
import numpy as np

from deforming_regions import deform
from material_surface import initialize_surface, spherical_face_areas, sample_surface
from ridge_geometry import rotate

R = 6371.


def unit(value):
    return value/np.linalg.norm(value, axis=-1, keepdims=True)


def chart(u, v):
    u, v = np.broadcast_arrays(u, v)
    return unit(np.column_stack((np.ones(u.size), u.ravel()/R, v.ravel()/R)))


def patch(u, v, owner=0, kind=1):
    x, y = np.meshgrid(u, v)
    faces = []
    for row in range(len(v)-1):
        for col in range(len(u)-1):
            a = row*len(u)+col
            faces.extend(((a, a+1, a+1+len(u)), (a, a+1+len(u), a+len(u))))
    return initialize_surface(chart(x, y), np.array(faces), np.full(len(faces), owner), np.full(len(faces), kind))


def combine(*surfaces):
    starts = np.r_[0, np.cumsum([len(s['vertices']) for s in surfaces])]
    return initialize_surface(np.concatenate([s['vertices'] for s in surfaces]),
        np.concatenate([s['faces']+starts[i] for i, s in enumerate(surfaces)]),
        np.concatenate([s['face_owner'] for s in surfaces]), np.concatenate([s['face_kind'] for s in surfaces]))


def loading():
    return np.array([[0., 0., 20./R], [0., 0., -20./R], [0., 0., -30./R]]), dict(
        bmid=np.array([[1., 0., 0.]]), bn=np.array([[0., 1., 0.]]),
        bl=np.array([4000.]), bp=np.array([0]), bq=np.array([1]))


def subdivide(s):
    vertices, faces = s['vertices'], s['faces']
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    unique, inv = np.unique(edges, axis=0, return_inverse=True)
    ab, bc, ca = (inv.reshape(3, len(faces))+len(vertices))
    a, b, c = faces.T
    refined = np.concatenate((np.column_stack((a, ab, ca)), np.column_stack((ab, b, bc)),
                              np.column_stack((ca, bc, c)), np.column_stack((ab, bc, ca))))
    return initialize_surface(np.vstack((vertices, unit(vertices[unique].sum(axis=1)))), refined,
                              np.tile(s['face_owner'], 4), np.tile(s['face_kind'], 4))


class MaterialContactDeformationTests(unittest.TestCase):
    def test_detached_arc_near_trench_stays_rigid_even_if_coarse_front_crosses_it(self):
        omega, boundary = loading()
        for centre in (-180., 0., -390.):
            s = patch(np.linspace(centre-40, centre+40, 5), np.linspace(-60, 60, 5), kind=3)
            old = deform(s, omega, boundary, 2.)
            self.assertGreater(np.max(np.abs(old['areal_strain'])), 1e-5)
            result = deform(s, omega, boundary, 2., require_material_contact=True)
            np.testing.assert_array_equal(result['vertices'], result['rigid_vertices'])
            np.testing.assert_array_equal(result['areal_strain'], 0.)
            self.assertEqual(result['diagnostics']['material_contact']['rejected_detached_components'], 1)

    def test_continental_interior_anchor_keeps_subduction_margin_shortening(self):
        omega, boundary = loading()
        s = patch(np.linspace(-1600, 0, 17), np.linspace(-1000, 1000, 21))
        result = deform(s, omega, boundary, 2., require_material_contact=True)
        self.assertGreater(result['diagnostics']['material_contact']['anchored_front_pairs'], 0)
        self.assertLess(result['area_ratio'].min(), .98)
        self.assertGreater(result['diagnostics']['deforming_vertices'], 0)
        old_area = spherical_face_areas(s['vertices'], s['faces'])
        np.testing.assert_allclose(result['face_area_km2']*35/result['area_ratio'], old_area*35, rtol=2e-12)

    def test_same_owner_detached_island_does_not_inherit_mainland_contact(self):
        omega, boundary = loading()
        mainland = patch(np.linspace(-1600, 0, 17), np.linspace(-700, 700, 15))
        island = patch(np.linspace(-240, -120, 5), np.linspace(1000, 1120, 5), kind=3)
        s = combine(mainland, island)
        result = deform(s, omega, boundary, 2., require_material_contact=True)
        np.testing.assert_array_equal(result['areal_strain'][len(mainland['faces']):], 0.)
        self.assertLess(result['area_ratio'][:len(mainland['faces'])].min(), .99)
        self.assertEqual(result['diagnostics']['material_contact']['rejected_detached_components'], 1)

    def test_actual_arc_continent_and_arc_arc_contacts_can_deform(self):
        omega, boundary = loading()
        for other_kind in (1, 3):
            s = combine(patch(np.linspace(-140, 10, 4), np.linspace(-80, 80, 4), kind=3),
                        patch(np.linspace(-10, 140, 4), np.linspace(-80, 80, 4), owner=1, kind=other_kind))
            result = deform(s, omega, boundary, 2., require_material_contact=True)
            self.assertGreater(result['diagnostics']['material_contact']['resolved_triangle_contacts'], 0)
            self.assertEqual(result['diagnostics']['material_contact']['buoyant_contact_pairs'], 2)
            self.assertLess(result['area_ratio'].min(), .995)
            self.assertTrue(np.all(result['face_area_km2'] > 0))

    def test_admission_is_rotation_and_refinement_invariant(self):
        omega, boundary = loading()
        for s in (patch(np.linspace(-1600, 0, 17), np.linspace(-1000, 1000, 21)),
                  patch(np.linspace(-240, -120, 5), np.linspace(-80, 80, 5), kind=3)):
            result = deform(s, omega, boundary, 2., require_material_contact=True)
            refined = deform(subdivide(s), omega, boundary, 2., require_material_contact=True)
            for key in ('anchored_components', 'anchored_front_pairs', 'rejected_detached_components'):
                self.assertEqual(result['diagnostics']['material_contact'][key], refined['diagnostics']['material_contact'][key])
            axis = np.array([0., -np.pi/2, 0.])
            turned = copy.deepcopy(s); turned['vertices'] = rotate(s['vertices'], axis)
            rotated_boundary = copy.deepcopy(boundary)
            for key in ('bmid', 'bn'): rotated_boundary[key] = rotate(boundary[key], axis)
            rotated = deform(turned, rotate(omega, axis), rotated_boundary, 2., require_material_contact=True)
            np.testing.assert_allclose(rotated['vertices'], rotate(result['vertices'], axis), atol=2e-13)

    def test_contact_is_specific_to_the_opposite_owner_and_front(self):
        omega, boundary = loading()
        s = combine(patch(np.linspace(-260, -130, 4), np.linspace(-70, 70, 4), kind=3),
                    patch(np.linspace(-150, -20, 4), np.linspace(-70, 70, 4), owner=1, kind=3))
        second = chart(-180., 0.)[0]
        boundary = dict(bmid=np.vstack((boundary['bmid'], second)),
                        bn=np.vstack((boundary['bn'], unit(np.array([0., 1., 0.])-second*second[1]))),
                        bl=np.array([4000., 4000.]), bp=np.array([0, 0]), bq=np.array([1, 2]))
        result = deform(s, omega, boundary, 2., require_material_contact=True)
        source = result['source_boundary'][result['source_boundary'] >= 0]
        self.assertGreater(len(source), 0)
        np.testing.assert_array_equal(source, 0)

    def test_crossing_thin_bodies_collide_even_when_no_vertex_hits_the_other(self):
        omega, boundary = loading()
        s = combine(patch([-200., 200.], [-15., 15.], kind=3),
                    patch([-15., 15.], [-200., 200.], owner=1, kind=1))
        hits = sample_surface(s, s['vertices'])
        self.assertFalse(np.any(s['vertex_owner'][hits['query_index']] != hits['owner']))
        result = deform(s, omega, boundary, 2., require_material_contact=True)
        self.assertGreater(result['diagnostics']['material_contact']['resolved_triangle_contacts'], 0)
        self.assertGreater(np.max(np.abs(result['areal_strain'])), 1e-5)

    def test_point_contact_alone_does_not_transmit_a_finite_traction(self):
        omega, boundary = loading()
        s = combine(patch([-200., -100.], [-100., 0.], kind=3),
                    patch([-100., 0.], [0., 100.], owner=1, kind=1))
        result = deform(s, omega, boundary, 2., require_material_contact=True)
        self.assertEqual(result['diagnostics']['material_contact']['resolved_triangle_contacts'], 0)
        np.testing.assert_array_equal(result['areal_strain'], 0.)


if __name__ == '__main__': unittest.main()
