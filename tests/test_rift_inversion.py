"""Independent compression, geometry and finite-budget inversion fixtures."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from rift_inversion import inversion_increment, RADIUS_KM


def boundary(closing=30., length=900., code=4):
    return SimpleNamespace(
        bmid=np.array([[1., 0., 0.]]), bn=np.array([[0., 1., 0.]]),
        bp=np.array([0]), bq=np.array([1]), down=np.array([0]),
        bcode=np.array([code]), bl=np.array([length]),
        omega=np.array([[0., 0., closing / RADIUS_KM], [0., 0., 0.]]))


def samples(distance_km=0., owner=0, extension=200., inverted=0., strike='north'):
    angle = distance_km / RADIUS_KM
    point = np.array([[np.cos(angle), np.sin(angle), 0.]])
    tangent = np.array([[0., 0., 1.]]) if strike == 'north' else np.array([[-np.sin(angle), np.cos(angle), 0.]])
    return point, tangent, np.array([owner]), np.array([extension]), np.array([inverted])


class RiftInversionTests(unittest.TestCase):
    def test_actual_cross_rift_compression_builds_from_recorded_extension(self):
        gain = inversion_increment(boundary(), *samples(), 2.)[0]
        #900km coherent contact / (900+300) at30km/Myr gives22.5m/Myr
        # initial construction rate and a600m inherited-rift capacity.
        expected = 600. * (1. - np.exp(-22.5 * 2. / 600.))
        self.assertAlmostEqual(gain, expected, places=11)
        self.assertGreater(gain, 40.)

    def test_unextended_or_exhausted_material_cannot_invert(self):
        self.assertEqual(inversion_increment(boundary(), *samples(extension=0.), 2.)[0], 0.)
        self.assertEqual(inversion_increment(boundary(), *samples(inverted=600.), 2.)[0], 0.)
        self.assertEqual(inversion_increment(boundary(), *samples(inverted=900.), 2.)[0], 0.)

    def test_common_rotation_zero_load_and_small_motion_noise_do_not_uplift(self):
        s = boundary()
        s.omega[:] = [0.03, -.01, .005]
        self.assertEqual(inversion_increment(s, *samples(), 2.)[0], 0.)
        s.omega[1] += [1e-10, -1e-10, 1e-10]
        self.assertEqual(inversion_increment(s, *samples(), 2.)[0], 0.)
        s.omega[:] = 0.
        self.assertEqual(inversion_increment(s, *samples(), 2.)[0], 0.)

    def test_extension_and_shear_alone_cannot_reactivate_a_scar(self):
        self.assertEqual(inversion_increment(boundary(closing=-30.), *samples(), 2.)[0], 0.)
        s = boundary()
        s.omega[:] = 0.
        s.omega[0, 1] = .01  # Pure strike-slip at this boundary geometry.
        self.assertEqual(inversion_increment(s, *samples(), 2.)[0], 0.)
        s = boundary(closing=3.)
        s.omega[0, 1] = .02  # Compression is too small relative to shear.
        self.assertEqual(inversion_increment(s, *samples(), 2.)[0], 0.)

    def test_subduction_loads_overrider_only_and_collision_loads_both(self):
        s = boundary(code=2)
        self.assertEqual(inversion_increment(s, *samples(owner=0), 2.)[0], 0.)
        self.assertGreater(inversion_increment(s, *samples(owner=1), 2.)[0], 0.)
        s.down[:] = 1
        self.assertGreater(inversion_increment(s, *samples(owner=0), 2.)[0], 0.)
        self.assertEqual(inversion_increment(s, *samples(owner=1), 2.)[0], 0.)
        for owner in (0, 1):
            self.assertGreater(inversion_increment(boundary(), *samples(owner=owner), 2.)[0], 0.)

    def test_unattached_owner_and_nonconvergent_boundary_types_do_not_load(self):
        self.assertEqual(inversion_increment(boundary(), *samples(owner=2), 2.)[0], 0.)
        for code in (1, 3, 5):
            self.assertEqual(inversion_increment(boundary(code=code), *samples(), 2.)[0], 0.)

    def test_locality_tapers_even_for_a_very_long_distant_boundary(self):
        s = boundary(length=1e9)
        near = inversion_increment(s, *samples(), 2.)[0]
        middle = inversion_increment(s, *samples(distance_km=900.), 2.)[0]
        far = inversion_increment(s, *samples(distance_km=1700.), 2.)[0]
        outside = inversion_increment(s, *samples(distance_km=1801.), 2.)[0]
        self.assertGreater(near, middle)
        self.assertGreater(middle, far)
        self.assertLess(far, near * .02)
        self.assertEqual(outside, 0.)

    def test_compression_along_rift_strike_has_no_inversion_effect(self):
        cross = inversion_increment(boundary(), *samples(strike='north'), 2.)[0]
        along = inversion_increment(boundary(), *samples(strike='east'), 2.)[0]
        self.assertGreater(cross, 0.)
        self.assertLess(along, 1e-12)
        data = list(samples())
        data[1] = np.array([[0., 1., 1.]]) / np.sqrt(2.)
        half = inversion_increment(boundary(), *data, 2.)[0]
        self.assertAlmostEqual(half, 600. * (1 - np.exp(-11.25 * 2 / 600.)), places=10)

    def test_short_contact_penalty_and_segment_refinement_do_not_amplify_uplift(self):
        short = inversion_increment(boundary(length=1.), *samples(), 2.)[0]
        long = inversion_increment(boundary(length=900.), *samples(), 2.)[0]
        self.assertLess(short, long * .01)
        s = boundary()
        for name in ('bmid', 'bn', 'bp', 'bq', 'down', 'bcode'):
            setattr(s, name, np.repeat(getattr(s, name), 12, axis=0))
        s.bl = np.full(12, 75.)
        self.assertAlmostEqual(inversion_increment(s, *samples(), 2.)[0], long, places=11)

    def test_analytic_integration_is_timestep_consistent_and_capacity_limited(self):
        s = boundary(closing=100., length=1e6)
        for extension, capacity in ((50., 150.), (1000., 2000.)):
            data = list(samples(extension=extension))
            once = inversion_increment(s, *data, 40.)[0]
            accumulated = 0.
            for _ in range(20):
                data[4][:] = accumulated
                accumulated += inversion_increment(s, *data, 2.)[0]
            self.assertAlmostEqual(accumulated, once, places=10)
            data[4][:] = 0.
            self.assertLessEqual(inversion_increment(s, *data, 2.)[0], 60.)
            self.assertLessEqual(inversion_increment(s, *data, 10000.)[0], capacity)

    def test_rotation_across_poles_and_seam_preserves_transport_and_loading(self):
        source = boundary()
        data = samples(distance_km=850.)
        expected = inversion_increment(source, *data, 2.)
        for angle in (np.pi/2, np.pi-.002, -np.pi+.002):
            for axis in ('y', 'z'):
                c, sn = np.cos(angle), np.sin(angle)
                rotation = (np.array([[c, 0, sn], [0, 1, 0], [-sn, 0, c]]) if axis == 'y'
                            else np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]]))
                s = deepcopy(source)
                for name in ('bmid', 'bn', 'omega'):
                    setattr(s, name, getattr(s, name) @ rotation.T)
                moved = list(data)
                moved[0], moved[1] = data[0] @ rotation.T, data[1] @ rotation.T
                np.testing.assert_allclose(inversion_increment(s, *moved, 2.), expected, rtol=1e-12, atol=1e-12)

    def test_budget_and_geometry_inputs_are_not_mutated(self):
        s, data = boundary(), samples()
        original = deepcopy((s, data))
        inversion_increment(s, *data, 2.)
        for name, values in vars(s).items():
            np.testing.assert_array_equal(values, getattr(original[0], name))
        for actual, expected in zip(data, original[1]):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(inversion_increment(s, *data, 0.)[0], 0.)

    def test_spatial_bins_and_chunks_match_an_independent_spherical_reference(self):
        rng = np.random.default_rng(71)
        count, edge_count = 610, 41
        points = rng.normal(size=(count, 3))
        points /= np.linalg.norm(points, axis=1)[:, None]
        mid = rng.normal(size=(edge_count, 3))
        mid /= np.linalg.norm(mid, axis=1)[:, None]
        normal = np.cross(np.array([0., 0., 1.]), mid)
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        tangent = np.cross(points, np.cross(np.array([0., 0., 1.]), points))
        tangent /= np.linalg.norm(tangent, axis=1)[:, None]
        owners = rng.integers(0, 3, count)
        extension = rng.uniform(20, 800, count)
        inverted = rng.uniform(0, 40, count)
        s = SimpleNamespace(bmid=mid, bn=normal, bp=np.zeros(edge_count, int),
                            bq=np.ones(edge_count, int), down=np.full(edge_count, -1),
                            bcode=np.full(edge_count, 4), bl=rng.uniform(10, 400, edge_count),
                            omega=np.array([[0., 0., .006], [0., 0., 0.]]))
        # Slow reference uses explicit transported3-vectors and the equivalent
        # weighted-mean/coherence formulation, with no spatial bin selection.
        velocity = np.cross(s.omega[1]-s.omega[0], mid) * RADIUS_KM
        closing = -np.sum(velocity * normal, axis=1)
        expected = np.zeros(count)
        for i in range(count):
            if owners[i] not in (0, 1):
                continue
            cosine = np.clip(mid @ points[i], -1., 1.)
            distance = np.arccos(cosine) * RADIUS_KM
            use = (distance < 1800.) & (closing > 2.)
            if not np.any(use):
                continue
            transported = normal[use] - ((normal[use] @ points[i]) / (1+cosine[use]))[:, None] * (mid[use]+points[i])
            alignment = transported @ np.cross(points[i], tangent[i])
            weight = (1-(distance[use]/1800.)**2)**2
            length_weight = s.bl[use] * weight
            total = length_weight.sum()
            mean = np.average(closing[use] * weight * alignment**2, weights=length_weight)
            effective = mean * total/(total+300.)
            capacity = min(2000., 3*extension[i])
            rate = min(effective, 30.)
            expected[i] = max(capacity-inverted[i], 0.) * (1-np.exp(-rate*2/capacity))
        data = (points, tangent, owners, extension, inverted)
        for chunk in (7, 256):
            with patch('rift_inversion.PAIR_CHUNK', chunk):
                actual = inversion_increment(s, *data, 2.)
            np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)


if __name__ == '__main__':
    unittest.main()
