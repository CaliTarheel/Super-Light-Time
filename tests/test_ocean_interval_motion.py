"""Independent point, area and differential checks for a finite spherical map."""
from copy import deepcopy
import io
import json
import unittest

import numpy as np

import checkpoint
import mesh_geometry
import ocean_interval_motion as motion


def triangle(colatitude):
    azimuth = np.arange(3)*2.*np.pi/3.
    return np.column_stack((np.sin(colatitude)*np.cos(azimuth),
                            np.sin(colatitude)*np.sin(azimuth), np.full(3, np.cos(colatitude))))


def fixture():
    return motion.prepare(triangle(.18)[None], triangle(.30)[None], [7], [15],
                          source_epoch_myr=168., duration_myr=2., radius_m=6371e3)


def unit(points):
    return points/np.linalg.norm(points, axis=-1, keepdims=True)


class OceanIntervalMotionTests(unittest.TestCase):
    def test_rigid_rotation_maps_points_differentials_and_unit_area(self):
        angle = .43
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                             [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
        source = triangle(.24)
        context = motion.prepare(source[None], (source@rotation.T)[None], [7], [15],
            source_epoch_myr=168., duration_myr=2., radius_m=6371e3)
        points = np.vstack((source, unit(np.array([[.01, .02, 1.]]))))
        donor = np.full(len(points), 15); owner = np.full(len(points), 7)
        mapped = motion.map_points(context, donor, owner, points)
        np.testing.assert_allclose(mapped, points@rotation.T, atol=3e-16)
        np.testing.assert_allclose(motion.area_jacobian(context, donor, owner, points), 1., atol=6e-15)
        vectors = np.cross(points, np.array([1., 2., 3.]))
        np.testing.assert_allclose(motion.push_tangent(context, donor, owner, points, vectors),
                                   vectors@rotation.T, atol=2e-15)

    def test_expansion_matches_analytic_polar_jacobian_and_roundtrip(self):
        context = fixture(); point = np.array([[0., 0., 1.]])
        expected = (np.tan(.30)/np.tan(.18))**2
        measured = motion.area_jacobian(context, [15], [7], point)[0]
        self.assertAlmostEqual(measured, expected, places=13)
        self.assertGreater(measured, 1.)
        points = unit(np.array([[.02, .03, 1.], [-.04, .01, 1.]]))
        ids, owners = [15, 15], [7, 7]
        mapped = motion.map_points(context, ids, owners, points)
        back = motion.map_points(context, ids, owners, mapped, inverse=True)
        np.testing.assert_allclose(back, points, atol=3e-16)
        jac = motion.area_jacobian(context, ids, owners, points)
        reverse = motion.area_jacobian(context, ids, owners, mapped, inverse=True)
        np.testing.assert_allclose(jac*reverse, 1., atol=2e-15)

    def test_local_jacobian_and_differential_match_independent_finite_difference(self):
        context = fixture(); point = unit(np.array([[.03, -.02, 1.]]))
        first = unit(np.cross(point, np.array([[1., 0., 0.]])))
        second = np.cross(point, first)
        mapped = motion.map_points(context, [15], [7], point)
        epsilon = 1e-6
        def difference(vector):
            plus = motion.map_points(context, [15], [7], unit(point+epsilon*vector))
            minus = motion.map_points(context, [15], [7], unit(point-epsilon*vector))
            return (plus-minus)/(2.*epsilon)
        a, b = difference(first), difference(second)
        np.testing.assert_allclose(motion.push_tangent(context, [15], [7], point, first), a,
                                   rtol=1e-8, atol=2e-10)
        numerical = float(np.einsum('ni,ni->n', np.cross(a, b), mapped)[0])
        self.assertAlmostEqual(motion.area_jacobian(context, [15], [7], point)[0], numerical, places=8)
        returned = motion.push_tangent(context, [15], [7], mapped,
            motion.push_tangent(context, [15], [7], point, first), inverse=True)
        np.testing.assert_allclose(returned, first, atol=1e-15)

    def test_shared_edge_attachments_agree_and_order_does_not_change_identity(self):
        mesh = mesh_geometry.icosphere(0)
        face_indices = mesh['edge_faces'][0]
        source = mesh['vertices'][mesh['faces'][face_indices]]
        mapped = unit(source@np.array([[1.1, .03, 0.], [0., .9, .02], [0., 0., 1.]]))
        context = motion.prepare(source, mapped, [7, 7], [40, 3],
            source_epoch_myr=168., duration_myr=2., radius_m=6371e3)
        a, b = mesh['vertices'][mesh['edge_vertices'][0]]
        point = unit(np.array([a*.37+b*.63]))
        left = motion.map_points(context, [40], [7], point)
        right = motion.map_points(context, [3], [7], point)
        np.testing.assert_allclose(left, right, atol=3e-16)
        reordered = motion.prepare(source[::-1], mapped[::-1], [7, 7], [3, 40],
            source_epoch_myr=168., duration_myr=2., radius_m=6371e3)
        np.testing.assert_array_equal(left, motion.map_points(reordered, [40], [7], point))

    def test_rotation_of_nonrigid_source_and_endpoint_preserves_map(self):
        context = fixture()
        q, _ = np.linalg.qr(np.random.default_rng(34).normal(size=(3, 3)))
        q[:, 0] *= np.linalg.det(q)
        rotated = motion.prepare(context['source_triangles_xyz']@q, context['mapped_triangles_xyz']@q,
            [7], [15], source_epoch_myr=168., duration_myr=2., radius_m=6371e3)
        points = unit(np.array([[.03, .01, 1.]]))
        np.testing.assert_allclose(motion.map_points(rotated, [15], [7], points@q),
                                   motion.map_points(context, [15], [7], points)@q, atol=5e-16)
        np.testing.assert_allclose(motion.area_jacobian(rotated, [15], [7], points@q),
                                   motion.area_jacobian(context, [15], [7], points), atol=8e-15)

    def test_typed_checkpoint_roundtrip_preserves_interval_and_results(self):
        context = fixture(); arrays = {}
        encoded = checkpoint._encode(context, arrays)
        buffer = io.BytesIO(); np.savez(buffer, **arrays); buffer.seek(0)
        with np.load(buffer, allow_pickle=False) as archive:
            restored = checkpoint._decode(json.loads(json.dumps(encoded)), archive)
        point = np.array([[0., 0., 1.]])
        self.assertEqual(restored['fingerprint'], context['fingerprint'])
        np.testing.assert_array_equal(motion.map_points(restored, [15], [7], point),
                                      motion.map_points(context, [15], [7], point))
        np.testing.assert_array_equal(motion.area_jacobian(restored, [15], [7], point),
                                      motion.area_jacobian(context, [15], [7], point))

    def test_owner_identity_disambiguates_coincident_donor_ids_and_positions(self):
        source = np.repeat(triangle(.18)[None], 2, axis=0)
        mapped = np.array([triangle(.3), triangle(.1)])
        context = motion.prepare(source, mapped, [7, 9], [15, 15],
            source_epoch_myr=168., duration_myr=2., radius_m=6371e3)
        point = unit(np.array([[.02, .03, 1.]]))
        first = motion.map_points(context, [15], [7], point)
        second = motion.map_points(context, [15], [9], point)
        self.assertGreater(np.linalg.norm(first-second), .01)
        np.testing.assert_allclose(motion.map_points(context, [15], [7], first, inverse=True), point, atol=3e-16)
        np.testing.assert_allclose(motion.map_points(context, [15], [9], second, inverse=True), point, atol=3e-16)
        with self.assertRaises(ValueError):
            motion.prepare(source, mapped, [7, 7], [15, 15],
                source_epoch_myr=168., duration_myr=2., radius_m=6371e3)

    def test_resolved_skinny_triangle_does_not_reject_its_own_vertices(self):
        source = unit(np.array([[-.1, 0., 1.], [.1, 0., 1.], [0., 1e-7, 1.]]))
        mapped = triangle(.3)
        q, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(3, 3)))
        q[:, 0] *= np.linalg.det(q)
        source, mapped = source@q, mapped@q
        context = motion.prepare(source[None], mapped[None], [7], [15],
            source_epoch_myr=168., duration_myr=2., radius_m=6371e3)
        result = motion.map_points(context, [15]*3, [7]*3, source)
        # The inverse is condition-limited; containment must still accept
        # represented endpoints rather than misclassify them as other donors.
        np.testing.assert_allclose(result, mapped, rtol=0., atol=3e-9)

    def test_extreme_vector_scale_cannot_hide_radial_part_or_return_infinity(self):
        context = fixture(); point = np.array([[0., 0., 1.]])
        with self.assertRaisesRegex(ValueError, 'tangent'):
            motion.push_tangent(context, [15], [7], point, [[1e308, 0., 1e308]])
        with self.assertRaisesRegex(ValueError, 'finite numerical range'):
            motion.push_tangent(context, [15], [7], point, [[1.1e308, 0., 0.]])
        tiny = motion.push_tangent(context, [15], [7], point, [[1e-300, 0., 0.]])
        self.assertGreater(tiny[0, 0], 1e-300)
        self.assertTrue(np.isfinite(tiny).all())

    def test_forward_boundary_points_remain_valid_for_inverse_attachment(self):
        for height in (.01, .001, 1e-5, 1e-7):
            base = unit(np.array([[-.1, 0., 1.], [.1, 0., 1.], [0., height, 1.]]))
            for seed in range(10):
                q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
                q[:, 0] *= np.linalg.det(q)
                source, mapped = base@q, triangle(.3)@q
                context = motion.prepare(source[None], mapped[None], [7], [15],
                    source_epoch_myr=168., duration_myr=2., radius_m=6371e3)
                points = np.vstack((source, unit(source+np.roll(source, -1, axis=0))))
                endpoint = motion.map_points(context, [15]*6, [7]*6, points)
                restored = motion.map_points(context, [15]*6, [7]*6, endpoint, inverse=True)
                with self.subTest(height=height, seed=seed):
                    np.testing.assert_allclose(restored, points, rtol=0., atol=2e-9)

    def test_rejects_wrong_attachment_mutation_fold_and_radial_differential(self):
        context = fixture(); point = np.array([[0., 0., 1.]])
        for ids, owners, points in (([99], [7], point), ([15], [8], point),
                                     ([15], [7], -point), ([15.], [7], point)):
            with self.subTest(ids=ids, owners=owners), self.assertRaises(ValueError):
                motion.map_points(context, ids, owners, points)
        for name in ('source_epoch_myr', 'duration_myr', 'radius_m'):
            changed = deepcopy(context); changed[name] += 1.
            with self.assertRaises(ValueError): motion.map_points(changed, [15], [7], point)
        changed = deepcopy(context); changed['mapped_triangles_xyz'][0] *= -1.
        with self.assertRaises(ValueError): motion.map_points(changed, [15], [7], point)
        with self.assertRaises(ValueError):
            motion.push_tangent(context, [15], [7], point, point)
        with self.assertRaises(ValueError):
            motion.prepare(triangle(.18)[None], triangle(.3)[None, ::-1], [7], [15],
                source_epoch_myr=168., duration_myr=2., radius_m=6371e3)
        with self.assertRaises(ValueError):
            motion.prepare(triangle(.18)[None], triangle(.3)[None], [7], [15],
                source_epoch_myr=168., duration_myr=True, radius_m=6371e3)
        empty = motion.map_points(context, np.array([], dtype=int), np.array([], dtype=int), np.empty((0, 3)))
        self.assertEqual(empty.shape, (0, 3))


if __name__ == '__main__':
    unittest.main()
