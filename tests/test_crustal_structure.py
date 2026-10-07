"""Column physics, history accounting and spherical foreland locality."""
from copy import deepcopy
import io
import unittest

import numpy as np

from crustal_structure import (AIRY_M_PER_KM, COOLING_MYR, MAX_FORELAND_M,
                               MAX_RIFT_HEAT_M, MAX_THICKNESS_KM, MIN_THICKNESS_KM,
                               RADIUS_KM, STATE_FIELDS, WATER_FACTOR,
                               coalesce_structure, elevation, evolve_structure,
                               foreland_profile, foreland_target, initialize_structure)
from ridge_geometry import rotate


def point(distance_km, along_km=0.):
    p = np.array([np.cos(along_km/RADIUS_KM), 0., np.sin(along_km/RADIUS_KM)])
    return np.cos(distance_km/RADIUS_KM)*p+np.sin(distance_km/RADIUS_KM)*np.array([0., 1., 0.])


class CrustalStructureTests(unittest.TestCase):
    def test_initial_columns_preserve_given_land_and_submerged_heights(self):
        initial = np.array([220., 440., -1500., 970.])
        s = initialize_structure([1, 2, 1, 3], initial)
        np.testing.assert_allclose(elevation(s), initial, atol=1e-12)
        np.testing.assert_array_equal(s['thickness_km'], [35., 42., 35., 25.])
        np.testing.assert_array_equal(s['area_factor'], np.ones(4))
        np.testing.assert_array_equal(s['rift_age_myr'], np.full(4, -1.))
        self.assertEqual(set(s), set(STATE_FIELDS))

    def test_airy_height_uses_total_thickness_and_water_loading(self):
        s = initialize_structure([1, 1], [0., 100.])
        s['thickness_km'] += [10., -10.]
        expected = [10*1000*(3300-2800)/3300,
                    (100.-10*1000*(3300-2800)/3300)*3300/(3300-1030)]
        np.testing.assert_allclose(elevation(s), expected, atol=1e-12)
        self.assertAlmostEqual(float(elevation(s)[0]), 1515.1515151515152)

    def test_mechanical_strain_conserves_effective_column_volume(self):
        s = initialize_structure([1, 2, 3], [220., 440., 970.])
        new, budget = evolve_structure(s, 20., shortening_km_myr=[20., 20., 0.],
                                      extension_km_myr=[0., 0., 20.], strength=[1., .35, 1.])
        np.testing.assert_allclose(new['thickness_km']*new['area_factor'],
                                   s['thickness_km']*s['area_factor'], atol=1e-13)
        self.assertAlmostEqual(new['thickness_km'][0], 35*np.exp(.25*20/400*20))
        self.assertGreater(new['thickness_km'][0]-35, new['thickness_km'][1]-42)
        self.assertLess(new['thickness_km'][2], 25.)
        self.assertGreater(budget['shortening_gain_m'][0], 0.)
        self.assertGreater(budget['extension_loss_m'][2], 0.)
        np.testing.assert_array_equal(budget['removed_volume_km_per_reference_km2'], np.zeros(3))

    def test_rift_thinning_persists_while_support_cools_and_subsidence_continues(self):
        original = initialize_structure([1], [220.])
        rifted, stretching = evolve_structure(original, 20., extension_km_myr=40.)
        self.assertLess(rifted['thickness_km'][0], 35.)
        self.assertGreater(rifted['rift_heat_m'][0], 0.)
        self.assertEqual(rifted['rift_age_myr'][0], 0.)
        self.assertLess(elevation(rifted)[0], elevation(original)[0])
        cooled, cooling = evolve_structure(rifted, 2*COOLING_MYR)
        np.testing.assert_allclose(cooled['rift_heat_m'], rifted['rift_heat_m']*np.exp(-2), atol=1e-12)
        np.testing.assert_array_equal(cooled['thickness_km'], rifted['thickness_km'])
        self.assertLess(cooling['thermal_delta_m'][0], 0.)
        self.assertEqual(cooled['rift_age_myr'][0], 2*COOLING_MYR)
        ancient, _ = evolve_structure(cooled, 1000.)
        self.assertLess(elevation(ancient)[0], -1000.)
        self.assertGreater(stretching['extension_loss_m'][0], stretching['thermal_delta_m'][0])

    def test_no_extension_cannot_invent_rift_heat_or_age(self):
        s = initialize_structure([1, 2, 3], [220., 440., 970.])
        new, budget = evolve_structure(s, 1000.)
        for name in STATE_FIELDS:
            np.testing.assert_array_equal(new[name], s[name], err_msg=name)
        np.testing.assert_array_equal(budget['total_delta_m'], np.zeros(3))

    def test_constant_rift_heating_and_cooling_are_time_partition_consistent(self):
        s = initialize_structure([1], [220.])
        one, _ = evolve_structure(s, 20., extension_km_myr=20.)
        many = s
        for _ in range(10):
            many, _ = evolve_structure(many, 2., extension_km_myr=20.)
        for name in STATE_FIELDS:
            np.testing.assert_allclose(one[name], many[name], atol=2e-11, rtol=1e-13, err_msg=name)
        np.testing.assert_allclose(elevation(one), elevation(many), atol=2e-10)

    def test_erosion_unloads_column_and_reports_rebound_without_double_counting(self):
        s = initialize_structure([1, 1], [2000., -1000.])
        new, budget = evolve_structure(s, 2., denudation_m=1000.)
        np.testing.assert_allclose(new['thickness_km'], s['thickness_km']-1., atol=1e-14)
        expected_loss = np.array([AIRY_M_PER_KM, AIRY_M_PER_KM*WATER_FACTOR])
        np.testing.assert_allclose(budget['net_erosion_loss_m'], expected_loss, atol=1e-12)
        np.testing.assert_allclose(budget['rebound_m'], 1000.-expected_loss, atol=1e-12)
        np.testing.assert_allclose(budget['total_delta_m'], -expected_loss, atol=1e-12)
        np.testing.assert_array_equal(budget['shortening_gain_m'], np.zeros(2))
        np.testing.assert_array_equal(budget['removed_volume_km_per_reference_km2'], np.ones(2))

    def test_all_height_terms_close_and_inputs_remain_unchanged(self):
        s = initialize_structure([1, 2, 3], [50., 440., -800.])
        original = deepcopy(s)
        new, b = evolve_structure(s, 2., shortening_km_myr=[0., 30., 0.],
                                 extension_km_myr=[25., 0., 35.], denudation_m=[0., 200., 100.],
                                 foreland_target_m=[1000., 0., 5000.])
        accounted = b['mechanical_delta_m']+b['thermal_delta_m']+b['foreland_delta_m']-b['net_erosion_loss_m']
        np.testing.assert_allclose(b['total_delta_m'], accounted, atol=1e-12)
        np.testing.assert_allclose(elevation(new)-elevation(s), accounted, atol=1e-12)
        np.testing.assert_allclose(b['denudation_m']-b['rebound_m'], b['net_erosion_loss_m'], atol=1e-12)
        for name in STATE_FIELDS:
            np.testing.assert_array_equal(s[name], original[name])
        self.assertTrue(np.all(new['foreland_m'] <= MAX_FORELAND_M))

    def test_finite_limits_and_actual_removed_volume_are_accounted(self):
        s = initialize_structure([1], [220.])
        thick, _ = evolve_structure(s, 1000., shortening_km_myr=1e5)
        self.assertAlmostEqual(thick['thickness_km'][0], MAX_THICKNESS_KM)
        eroded, b = evolve_structure(thick, 2., denudation_m=1e9)
        self.assertAlmostEqual(eroded['thickness_km'][0], MIN_THICKNESS_KM)
        np.testing.assert_allclose(thick['thickness_km']*thick['area_factor']
                                   -eroded['thickness_km']*eroded['area_factor'],
                                   b['removed_volume_km_per_reference_km2'], atol=1e-13)
        limited, _ = evolve_structure(eroded, 1000., extension_km_myr=1e5)
        self.assertLessEqual(limited['rift_heat_m'][0], MAX_RIFT_HEAT_M)
        self.assertAlmostEqual(limited['rift_heat_m'][0], 0.)
        self.assertTrue(all(np.isfinite(v).all() for v in limited.values()))

    def test_foreland_is_finite_delayed_and_persists_after_loading_stops(self):
        values = foreland_profile([0., 60., 250., 440., 1000.], 4500.)
        np.testing.assert_allclose(values, [0., 0., 1000., 0., 0.], atol=1e-12)
        self.assertEqual(float(foreland_profile(250., 1e9)), MAX_FORELAND_M)
        self.assertEqual(float(foreland_profile(250., 300.)), 0.)
        s = initialize_structure([1], [220.])
        loaded, _ = evolve_structure(s, 12., foreland_target_m=1000.)
        self.assertAlmostEqual(loaded['foreland_m'][0], 1000*(1-np.exp(-1)))
        quiet, b = evolve_structure(loaded, 100.)
        self.assertAlmostEqual(quiet['foreland_m'][0], loaded['foreland_m'][0]*np.exp(-1))
        self.assertGreater(b['foreland_delta_m'][0], 0.)
        self.assertGreater(quiet['foreland_m'][0], 0.)

    def test_foreland_targets_only_nearby_matching_owners_and_finite_segments(self):
        points = np.array([point(0), point(250), point(-250), point(250), point(1000), point(250, 2000)])
        result = foreland_target(points, [1, 1, 2, 3, 1, 1], [[1., 0., 0.]],
                                 [[0., 1., 0.]], [1000.], [1], [2], [4500.], chunk_size=2)
        np.testing.assert_allclose(result, [0., 1000., 1000., 0., 0., 0.], atol=1e-9)

    def test_foreland_geometry_is_equivariant_at_seam_and_pole(self):
        points = np.array([point(d, along) for along in (-400., 0., 400., 800.)
                           for d in (-400., -250., 0., 100., 250., 400.)])
        mid, normal = np.array([[1., 0., 0.]]), np.array([[0., 1., 0.]])
        owners = np.ones(len(points), int)
        reference = foreland_target(points, owners, mid, normal, [1000.], [1], [2], [4500.])
        for rotation in ([0., 0., np.pi], [0., -np.pi/2, 0.], [.4, -.7, 1.1]):
            moved = foreland_target(rotate(points, rotation), owners, rotate(mid, rotation),
                                    rotate(normal, rotation), [1000.], [1], [2], [4500.], chunk_size=3)
            np.testing.assert_allclose(moved, reference, atol=2e-8)

    def test_nearest_belt_does_not_place_foredeep_under_adjacent_mountain_peaks(self):
        mids = np.array([point(0., -300.), point(0., 300.)])
        targets = foreland_target(mids, [1, 1], mids, [[0., 1., 0.], [0., 1., 0.]],
                                  [600., 600.], [1, 1], [2, 2], [4500., 7000.])
        np.testing.assert_array_equal(targets, np.zeros(2))

    def test_juvenile_coalescence_preserves_mean_height_effective_area_and_volume(self):
        s = initialize_structure([3, 3, 3], [850., -1200., 1000.])
        s['thickness_km'][:] = [15., 30., 25.]
        s['area_factor'][:] = [2., .5, 1.]
        s['rift_heat_m'][:] = [1000., 0., 100.]
        s['rift_age_myr'][:] = [10., -1., 0.]
        s['foreland_m'][:] = [500., 0., 100.]
        weights, groups = np.array([2., 1., 4.]), np.array([0, 0, 1])
        merged = coalesce_structure(s, weights, groups)
        totals = np.bincount(groups, weights=weights)
        for source, target in ((elevation(s), elevation(merged)),
                               (s['area_factor'], merged['area_factor']),
                               (s['thickness_km']*s['area_factor'], merged['thickness_km']*merged['area_factor'])):
            np.testing.assert_allclose(np.bincount(groups, weights=weights*source), target*totals, atol=3e-12)
        self.assertEqual(merged['rift_age_myr'][0], 10.)

    def test_plain_array_state_roundtrip_continues_exactly(self):
        s, _ = evolve_structure(initialize_structure([1, 2, 3], [220., 440., 970.]),
                                 2., extension_km_myr=[20., 0., 30.], foreland_target_m=[0., 1000., 0.])
        buffer = io.BytesIO()
        np.savez(buffer, **s)
        buffer.seek(0)
        with np.load(buffer, allow_pickle=False) as archive:
            restored = {name: archive[name].copy() for name in archive.files}
        first, b1 = evolve_structure(s, 2., shortening_km_myr=12., denudation_m=40.)
        second, b2 = evolve_structure(restored, 2., shortening_km_myr=12., denudation_m=40.)
        for left, right in ((first, second), (b1, b2)):
            for name in left:
                np.testing.assert_array_equal(left[name], right[name], err_msg=name)

    def test_invalid_inputs_and_empty_material_are_explicit(self):
        s = initialize_structure(np.array([], np.uint8), np.array([]))
        new, budget = evolve_structure(s, 2.)
        self.assertEqual(elevation(new).shape, (0,))
        self.assertEqual(budget['total_delta_m'].shape, (0,))
        with self.assertRaises(ValueError):
            initialize_structure([0], [0.])
        with self.assertRaises(ValueError):
            evolve_structure(initialize_structure([1]), 0.)
        with self.assertRaises(ValueError):
            evolve_structure(initialize_structure([1]), 2., denudation_m=-1.)
        with self.assertRaises(ValueError):
            foreland_target([[1., 0., 0.]], [1], [[1., 0., 0.]], [[1., 0., 0.]], [100.], [1], [2], [4500.])


if __name__ == '__main__':
    unittest.main()
