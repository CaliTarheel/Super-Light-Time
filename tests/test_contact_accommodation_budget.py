"""Contact accommodation must not run out of active-set budget on a large region.

Captured from run SEP21T (20260921-011257-b2f85b) at 60 Myr: contact region 3,
931 faces over 6.2e6 km2. Its preferred 2 Myr motion pushes 39 faces past
their unchanged area bounds by 7,283 km2 in total. The constrained solve needs
38 simultaneous bounds; a flat budget of 32 refused it, and the coupled step
then retried down to 0.0625 Myr.
"""
import json
import unittest
from pathlib import Path

import numpy as np

import bounded_gravity
import contact_response
from material_surface import spherical_face_areas

FIXTURE = Path(__file__).resolve().parent/'fixtures'/'contact_accommodation_budget_sep21t_60myr.npz'


def captured():
    with np.load(FIXTURE) as data:
        kwargs = json.loads(str(data['kwargs']))
        weights = data['viscosity_weights']
        kwargs['viscosity_weights'] = None if weights.ndim == 0 and np.isnan(weights) else weights
        args = [data[name] for name in ('points', 'faces', 'area', 'preferred', 'rigid', 'minimum', 'maximum')]
        return args, float(data['dt']), kwargs


class ContactAccommodationBudgetTests(unittest.TestCase):
    def test_captured_region_is_accommodated_within_unchanged_bounds(self):
        (points, faces, area, preferred, rigid, minimum, maximum), dt, kwargs = captured()
        self.assertEqual(len(faces), 931)
        endpoint, velocity, solver, detail = contact_response.redistribute(
            points, faces, area, preferred, rigid, minimum, maximum, dt, **kwargs)
        self.assertTrue(solver['converged'])
        self.assertGreater(detail['maximum_active_bounds'], 32)
        self.assertLessEqual(detail['maximum_active_bounds'], detail['active_set_budget'])
        self.assertEqual(detail['active_set_budget'],
                         max(bounded_gravity.MINIMUM_ACTIVE_BUDGET,
                             int(bounded_gravity.ACTIVE_SET_FACE_FRACTION*len(faces))))
        actual = spherical_face_areas(endpoint, faces, kwargs.get('radius', 6371.))
        self.assertTrue(np.all(actual >= minimum*(1-1e-12)))
        self.assertTrue(np.all(actual <= maximum*(1+1e-12)))

    def test_budget_is_not_a_flat_cap(self):
        self.assertIsNone(contact_response.ACTIVE_SET_BUDGET)


if __name__ == '__main__':
    unittest.main()
