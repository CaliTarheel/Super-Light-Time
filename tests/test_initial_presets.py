import unittest
import hashlib

import numpy as np

from fracture import component_labels
from tectonics import make_initial


def spherical_area(width, height):
    edges = np.pi/2-np.arange(height+1)*np.pi/height
    return np.repeat((np.sin(edges[:-1])-np.sin(edges[1:]))/(2*width), width)


class InitialPresetTests(unittest.TestCase):
    def test_land65_uses_spherical_area_and_many_separate_cratons_at_each_grid(self):
        for width in (64, 128, 512):
            height = width//2
            with self.subTest(width=width):
                result = make_initial(dict(width=width, height=height), preset='land65')
                crust = result['crust']
                area = spherical_area(width, height)
                fraction = area[crust > 0].sum()
                self.assertLessEqual(abs(fraction-.65), area.max())
                self.assertAlmostEqual(result['land_fraction'], fraction)
                self.assertEqual(result['preset'], 'land65')
                _, continents = component_labels(crust > 0, width, height)
                _, cratons = component_labels(crust == 2, width, height)
                self.assertEqual(continents, 1)
                self.assertEqual(cratons, 18)
                self.assertEqual(result['craton_count'], cratons)
                self.assertTrue(np.all(np.isin(crust, [0, 1, 2])))
                classic = make_initial(dict(width=width, height=height))
                _, classic_cratons = component_labels(classic['crust'] == 2, width, height)
                self.assertGreater(cratons, classic_cratons)

    def test_highland65_has_six_continents_one_ocean_and_eighteen_cratons(self):
        for width in (48, 64, 128, 512):
            height = width//2
            with self.subTest(width=width):
                result = make_initial(dict(width=width, height=height, seed=41), preset='highland65')
                crust = result['crust']
                area = spherical_area(width, height)
                fraction = area[crust > 0].sum()
                continents, continent_count = component_labels(crust > 0, width, height)
                ocean, ocean_count = component_labels(crust == 0, width, height)
                _, craton_count = component_labels(crust == 2, width, height)
                continent_area = np.bincount(continents[continents >= 0],
                                             weights=area[continents >= 0], minlength=continent_count)
                ocean_area = np.bincount(ocean[ocean >= 0],
                                         weights=area[ocean >= 0], minlength=ocean_count)
                self.assertLessEqual(abs(fraction-.65), area.max())
                self.assertEqual(result['preset'], 'highland65')
                self.assertEqual(continent_count, 6)
                self.assertEqual(ocean_count, 1)
                self.assertEqual(craton_count, 18)
                self.assertEqual(result['continent_count'], 6)
                self.assertEqual(result['ocean_component_count'], 1)
                self.assertEqual(result['craton_count'], 18)
                self.assertAlmostEqual(result['land_fraction'], fraction)
                self.assertAlmostEqual(result['continental_fraction'], fraction)
                self.assertAlmostEqual(result['largest_continent_fraction'], continent_area.max())
                self.assertAlmostEqual(result['largest_ocean_fraction'], ocean_area.max())
                self.assertLess(result['largest_continent_fraction'], .13)
                self.assertGreater(result['largest_ocean_fraction'], .34)
                self.assertEqual(result['target_continental_fraction'], .65)

    def test_highland65_topology_survives_extreme_supported_grid_shapes(self):
        for width, height in ((48, 1024), (2048, 24)):
            with self.subTest(width=width, height=height):
                result = make_initial(dict(width=width, height=height, seed=93), preset='highland65')
                _, continents = component_labels(result['crust'] > 0, width, height)
                _, oceans = component_labels(result['crust'] == 0, width, height)
                _, cratons = component_labels(result['crust'] == 2, width, height)
                self.assertEqual((continents, oceans, cratons), (6, 1, 18))

    def test_poles_and_longitude_seam_are_one_connected_continent(self):
        width, height = 128, 64
        for seed in (0, 12, 93):
            result = make_initial(dict(width=width, height=height, seed=seed), preset='land65')
            crust = result['crust'].reshape(height, width)
            self.assertTrue(np.all(crust[0] > 0))
            self.assertTrue(np.all(crust[-1] > 0))
            # The seam runs through continental polar caps and the far ocean;
            # it never appears as a numerical crack or an isolated polar island.
            labels, count = component_labels(crust > 0, width, height)
            self.assertEqual(count, 1)
            self.assertEqual(labels[0], labels[width-1])
            self.assertEqual(labels[-width], labels[-1])
            self.assertLessEqual(abs(result['land_fraction']-.65), spherical_area(width, height).max())
            self.assertEqual(result['craton_count'], 18)

    def test_seed_is_repeatable_and_alternate_seeds_change_the_world(self):
        config = dict(width=128, height=64, seed=81)
        for preset in ('land65', 'highland65'):
            with self.subTest(preset=preset):
                first = make_initial(config, preset=preset)
                repeated = make_initial(config, preset=preset)
                np.testing.assert_array_equal(first['crust'], repeated['crust'])
                other = make_initial(dict(config, seed=82), preset=preset)
                self.assertFalse(np.array_equal(first['crust'], other['crust']))

    def test_classic_default_is_unchanged_and_unknown_preset_rejects(self):
        # Frozen pre-preset output guards the classic generator without copying
        # its implementation into the test.
        width, height, seed = 128, 64, 12
        config = dict(width=width, height=height, seed=seed)
        default = make_initial(config)
        explicit = make_initial(config, preset='pangaea')
        self.assertEqual(hashlib.sha256(default['crust'].tobytes()).hexdigest(),
                         '74b49456fb357ca25ed9920017a7223674b52d59d72c200153422a917665fddb')
        np.testing.assert_array_equal(default['crust'], explicit['crust'])
        with self.assertRaises(ValueError):
            make_initial(config, preset='unknown')


if __name__ == '__main__':
    unittest.main()
