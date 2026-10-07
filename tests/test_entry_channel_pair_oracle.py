"""Geometry and work checks for the read-only face-pair channel oracle."""

import unittest

import numpy as np

import continental_entry
import gravitational_relaxation as gravity
import material_surface
from experiments.entry_channel_pair_oracle import pair_energy, pair_handoff
from ridge_geometry import rotate
from tests.test_entry_stack_work_oracle import FACES, NORMAL, SHEETS, geometry


class EntryChannelPairOracleTests(unittest.TestCase):
    def setUp(self):
        self.points = geometry()
        self.area = float(material_surface.spherical_face_areas(
            self.points, FACES)[0])
        self.volume = self.area * 35.
        self.mass = self.volume * 1e9 * 2800.

    def energy(self, points=None, normal=None, upper=None, **kwargs):
        return pair_energy(
            self.points if points is None else points, FACES[0],
            FACES[1] if upper is None else upper,
            NORMAL[0] if normal is None else normal, 50.,
            self.volume, self.mass, kwargs.pop('base', 35000.),
            1000., 3300. * 9.81, **kwargs)

    def test_clipped_area_matches_independent_overlap_ledger(self):
        result = self.energy()
        fraction = continental_entry.reference_entry_fractions(
            self.points[FACES[:1]], NORMAL, radius_km=6371.)
        ledger = continental_entry.entry_stack_overlap_ledger(
            self.points, FACES, SHEETS, np.array([0]), NORMAL,
            fraction, radius_km=6371.)
        self.assertGreater(result['area_km2'], 0.)
        self.assertLess(result['area_km2'], self.area)
        np.testing.assert_allclose(result['area_km2'],
                                   sum(row['area_km2'] for row in ledger),
                                   rtol=3e-13)
        self.assertGreater(result['area_mean_reaction_pa'], 0.)
        self.assertGreater(result['area_mean_upper_uplift_m'], 0.)
        self.assertGreater(result['quadrature_samples'], 35)

    def test_upper_subdivision_preserves_area_and_converged_energy(self):
        whole = self.energy()
        p = self.points
        midpoints = np.array([p[3] + p[4], p[4] + p[5], p[5] + p[3]])
        midpoints /= np.linalg.norm(midpoints, axis=1)[:, None]
        vertices = np.vstack((p, midpoints))
        children = ([3, 6, 8], [6, 4, 7], [8, 7, 5], [6, 7, 8])
        results = [self.energy(vertices, upper=np.array(face))
                   for face in children]
        np.testing.assert_allclose(sum(r['area_km2'] for r in results),
                                   whole['area_km2'], rtol=3e-13)
        np.testing.assert_allclose(sum(r['energy_j'] for r in results),
                                   whole['energy_j'], rtol=2e-8)
        np.testing.assert_allclose(
            sum(r['area_mean_reaction_pa'] * r['area_km2'] for r in results),
            whole['area_mean_reaction_pa'] * whole['area_km2'], rtol=2e-8)

    def test_base_derivative_equals_integrated_contact_reaction(self):
        result = self.energy()
        step = 1.
        measured = (self.energy(base=35000. + step)['energy_j']
                    - self.energy(base=35000. - step)['energy_j']) / (2 * step)
        predicted = result['area_mean_reaction_pa'] * result['area_km2'] * 1e6
        np.testing.assert_allclose(measured, predicted, rtol=2e-7)

    def test_finite_front_excludes_contact_beyond_its_endpoints(self):
        midpoint = np.array([1., 0., 0.])
        half_length = 50.
        finite = self.energy(finite_midpoint=midpoint,
                             finite_half_length_km=half_length)
        infinite = self.energy()
        fraction = continental_entry.reference_entry_fractions(
            self.points[FACES[:1]], NORMAL, radius_km=6371.,
            finite_midpoints=midpoint[None],
            finite_half_lengths_km=np.array([half_length]))
        ledger = continental_entry.entry_stack_overlap_ledger(
            self.points, FACES, SHEETS, np.array([0]), NORMAL, fraction,
            radius_km=6371., finite_midpoints=midpoint[None],
            finite_half_lengths_km=np.array([half_length]))
        self.assertGreater(finite['area_km2'], 0.)
        self.assertLess(finite['area_km2'], infinite['area_km2'])
        np.testing.assert_allclose(
            finite['area_km2'], sum(row['area_km2'] for row in ledger),
            rtol=3e-13)

    def test_common_rotation_and_two_plate_work_are_reciprocal(self):
        initial = self.energy()
        spin = np.array([.2, -.1, .3])
        moved = self.energy(rotate(self.points, spin), rotate(NORMAL[0], spin))
        np.testing.assert_allclose(moved['energy_j'], initial['energy_j'],
                                   rtol=2e-12)
        np.testing.assert_allclose(moved['area_km2'], initial['area_km2'],
                                   rtol=2e-12)
        step = 1e-6
        for axis in np.eye(3):
            def energy(owner, direction):
                points = self.points.copy()
                selected = slice(3 * owner, 3 * owner + 3)
                points[selected] = rotate(points[selected], axis * direction * step)
                normal = (rotate(NORMAL[0], axis * direction * step)
                          if owner == 1 else NORMAL[0])
                return self.energy(points, normal)['energy_j']

            lower = (energy(0, 1) - energy(0, -1)) / (2 * step)
            upper_and_hinge = (energy(1, 1) - energy(1, -1)) / (2 * step)
            self.assertLess(abs(lower + upper_and_hinge)
                            / max(abs(lower), abs(upper_and_hinge)), 1e-7)

    def test_disjoint_upper_face_has_no_candidate_contact(self):
        points = self.points.copy()
        points[3:] = rotate(points[3:], [0., 0., .4])
        result = self.energy(points)
        self.assertEqual(result['area_km2'], 0.)
        self.assertEqual(result['energy_j'], 0.)
        self.assertEqual(result['quadrature_samples'], 0)

    def test_handoff_removes_exact_old_entry_and_stack_terms(self):
        points = self.points.copy()
        points[3:] = points[:3]
        points = rotate(points, [0., 0., .03])
        areas = material_surface.spherical_face_areas(points, FACES)
        volumes = areas * 35.
        for lower_density in (2800., 3450.):
            with self.subTest(lower_density=lower_density):
                masses = volumes * 1e9 * np.array([lower_density, 2800.])
                candidate = pair_energy(
                    points, FACES[0], FACES[1], NORMAL[0], 50.,
                    volumes[0], masses[0], 35000., 1000., 3300. * 9.81)
                handoff = pair_handoff(
                    points, FACES[0], FACES[1], NORMAL[0], 50.,
                    volumes[0], masses[0], volumes[1], masses[1],
                    35000., 1000., 3300. * 9.81,
                    exclusive_upper_coverage=True)
                old_entry = continental_entry.evaluate(
                    points, FACES[:1], volumes[:1], masses[:1], NORMAL,
                    [50.])['energy_j']
                profile = dict(
                    dense_fraction=np.array(
                        [1. if lower_density == 3450. else 0., 0.]),
                    sheet_order={2: {1}})
                old_stack = gravity.reference_energy(
                    points, FACES, volumes, [35., 35.], SHEETS,
                    density_profile=profile) * gravity.ENERGY_UNIT_J
                np.testing.assert_allclose(
                    candidate['entry_reference_area_km2'], areas[0],
                    rtol=2e-13)
                np.testing.assert_allclose(
                    candidate['previous_entry_energy_j'], old_entry,
                    rtol=2e-13)
                np.testing.assert_allclose(
                    handoff['removed_stack_energy_j'], old_stack,
                    rtol=2e-13)
                np.testing.assert_allclose(
                    old_entry + old_stack + handoff['correction_j'],
                    candidate['energy_j'], rtol=2e-13)

    def test_reference_handoff_matches_finite_native_entry_measure(self):
        points = self.points.copy()
        points[3:] = points[:3]
        midpoint = np.array([1., 0., 0.])
        result = self.energy(
            points, finite_midpoint=midpoint, finite_half_length_km=50.)
        old = continental_entry.evaluate(
            points, FACES[:1], [self.volume], [self.mass], NORMAL,
            [50.], finite_midpoints=midpoint[None],
            finite_half_lengths_km=np.array([50.]))
        np.testing.assert_allclose(
            result['previous_entry_energy_j'], old['energy_j'],
            rtol=2e-13)
        np.testing.assert_allclose(
            result['entry_reference_area_km2'],
            self.area * old['entered_reference_fraction'][0], rtol=2e-13)

    def test_partial_pair_handoff_survives_upper_subdivision(self):
        whole = pair_handoff(
            self.points, FACES[0], FACES[1], NORMAL[0], 50.,
            self.volume, self.mass, self.area * 35.,
            self.area * 35e9 * 2800., 35000., 1000., 3300. * 9.81,
            exclusive_upper_coverage=True)
        p = self.points
        midpoints = np.array([p[3] + p[4], p[4] + p[5], p[5] + p[3]])
        midpoints /= np.linalg.norm(midpoints, axis=1)[:, None]
        points = np.vstack((p, midpoints))
        children = ([3, 6, 8], [6, 4, 7], [8, 7, 5], [6, 7, 8])
        pieces = []
        for child in children:
            area = float(material_surface.spherical_face_areas(
                points, np.array([child]))[0])
            pieces.append(pair_handoff(
                points, FACES[0], child, NORMAL[0], 50., self.volume,
                self.mass, area * 35., area * 35e9 * 2800.,
                35000., 1000., 3300. * 9.81,
                exclusive_upper_coverage=True))
        for field, tolerance in (
                ('area_km2', 3e-13),
                ('entry_reference_area_km2', 3e-13),
                ('removed_entry_energy_j', 3e-13),
                ('removed_stack_energy_j', 3e-13),
                ('channel_energy_j', 2e-8),
                ('correction_j', 2e-7)):
            np.testing.assert_allclose(
                sum(row[field] for row in pieces), whole[field],
                rtol=tolerance)

    def test_handoff_correction_has_reciprocal_two_plate_work(self):
        axis = np.array([.2, -.3, .5])
        axis /= np.linalg.norm(axis)
        step = 1e-6

        def correction(owner, direction):
            points = self.points.copy()
            selected = slice(3 * owner, 3 * owner + 3)
            points[selected] = rotate(points[selected], axis * direction * step)
            normal = (rotate(NORMAL[0], axis * direction * step)
                      if owner == 1 else NORMAL[0])
            return pair_handoff(
                points, FACES[0], FACES[1], normal, 50.,
                self.volume, self.mass, self.volume, self.mass,
                35000., 1000., 3300. * 9.81,
                exclusive_upper_coverage=True)['correction_j']

        lower = (correction(0, 1) - correction(0, -1)) / (2 * step)
        upper = (correction(1, 1) - correction(1, -1)) / (2 * step)
        self.assertLess(abs(lower + upper) / max(abs(lower), abs(upper)), 1e-7)

    def test_mixed_measure_energy_matches_high_order_full_face_oracle(self):
        points = self.points.copy()
        points[3:] = points[:3]
        points = rotate(points, [0., 0., .03])
        lower = points[FACES[0]]
        area = float(material_surface.spherical_face_areas(
            points, FACES[:1])[0])
        volume = area * 35.
        nodes, weights = np.polynomial.legendre.leggauss(50)
        nodes = (nodes + 1.) / 2.
        weights = weights / 2.
        u, v = np.meshgrid(nodes, nodes, indexing='ij')
        bary = np.stack([1. - u - (1. - u) * v, u, (1. - u) * v],
                        axis=-1)
        direction = bary @ lower
        determinant = np.dot(
            lower[0], np.cross(lower[1] - lower[0],
                               lower[2] - lower[0]))
        jacobian = (6371.**2 * determinant
                    / (2. * area * np.linalg.norm(direction, axis=-1)**3))
        preferred = (bary @ (6371e3 * np.arcsin(lower @ NORMAL[0]))
                     * np.sin(np.deg2rad(50.)))
        quadrature = 2. * weights[:, None] * weights[None, :] * (1. - u)
        stiffness = 1000.
        upper_stiffness = 3300. * 9.81
        for density, base in ((2800., 200000.), (3450., 1.)):
            with self.subTest(density=density):
                buoyancy = 9.81 * (3300. - density) * 35000.
                effective = 1. / (1. / stiffness
                                  + 1. / (jacobian * upper_stiffness))
                reaction = effective * np.maximum(
                    base - preferred + buoyancy / stiffness, 0.)
                depth = preferred - buoyancy / stiffness + reaction / stiffness
                uplift = reaction / (jacobian * upper_stiffness)
                density_energy = (
                    .5 * stiffness * (depth - preferred)**2
                    + buoyancy * depth
                    + .5 * jacobian * upper_stiffness * uplift**2)
                result = pair_energy(
                    points, FACES[0], FACES[1], NORMAL[0], 50.,
                    volume, volume * 1e9 * density, base,
                    stiffness, upper_stiffness)
                independent = area * 1e6 * np.sum(
                    quadrature * density_energy)
                np.testing.assert_allclose(
                    result['energy_j'], independent, rtol=3e-11)
                np.testing.assert_allclose(
                    result['area_mean_upper_uplift_m'],
                    np.sum(quadrature * jacobian * uplift), rtol=3e-11,
                    atol=1e-12)
                np.testing.assert_allclose(
                    result['area_mean_reaction_pa'],
                    np.sum(quadrature * reaction), rtol=3e-11, atol=1e-5)
                if density == 3450.:
                    self.assertEqual(result['area_mean_reaction_pa'], 0.)
                    old_entry = continental_entry.evaluate(
                        points, FACES[:1], [volume],
                        [volume * 1e9 * density], NORMAL, [50.])['energy_j']
                    np.testing.assert_allclose(
                        result['energy_j'],
                        old_entry - area * 1e6
                        * buoyancy**2 / (2. * stiffness), rtol=3e-11)

    def test_rejects_invalid_mechanics_even_for_a_disjoint_pair(self):
        points = self.points.copy()
        points[3:] = rotate(points[3:], [0., 0., .4])
        with self.assertRaisesRegex(ValueError, 'valid ordered faces'):
            self.energy(points, base=-1.)
        with self.assertRaisesRegex(ValueError, 'valid ordered faces'):
            self.energy(points, quadrature_tolerance=0.)
        with self.assertRaisesRegex(ValueError, 'exclusive upper coverage'):
            pair_handoff(
                self.points, FACES[0], FACES[1], NORMAL[0], 50.,
                self.volume, self.mass, self.volume, self.mass,
                35000., 1000., 3300. * 9.81,
                exclusive_upper_coverage=False)


if __name__ == '__main__':
    unittest.main()
