"""Density and physical-area checks for upper contact rock loads."""

from copy import deepcopy
import unittest

import numpy as np

import channel_upper_rock_loads
import column_density
import crust_inventory
import crustal_structure
import dense_crust


class UpperRockLoadTests(unittest.TestCase):
    def test_ordinary_and_retained_dense_mass_use_current_physical_area(self):
        column = crustal_structure.initialize_structure(np.array([1]))
        crust_inventory.initialize(column)
        dense_crust.initialize(column, 1000.)
        ordinary = channel_upper_rock_loads.physical_rock_column(column)
        self.assertEqual(ordinary['dense_thickness_km'], 0.)
        self.assertAlmostEqual(ordinary['rock_mass_kg_m2'],
                               35000. * column_density.RHO_CRUST)
        self.assertAlmostEqual(ordinary['current_surface_elevation_m'],
                               crustal_structure.elevation(column)[0])
        dense_crust.advance(column, [1.], [1000.], [10.], 100.,
                            minimum_thickness_km=8.)
        result = channel_upper_rock_loads.physical_rock_column(column)
        self.assertGreater(result['dense_thickness_km'], 0.)
        self.assertAlmostEqual(result['rock_mass_kg_m2'], 1000. * (
            column_density.RHO_CRUST * result['ordinary_thickness_km']
            + column_density.RHO_DENSE * result['dense_thickness_km']))
        strained = deepcopy(column)
        strained['area_factor'] *= 2.
        strained['thickness_km'] /= 2.
        spread = channel_upper_rock_loads.physical_rock_column(strained)
        np.testing.assert_allclose(spread['rock_mass_kg_m2'],
                                   result['rock_mass_kg_m2'] / 2., rtol=2e-12)


if __name__ == '__main__':
    unittest.main()
