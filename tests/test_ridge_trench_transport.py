"""Ridge ownership guards preserve subcell retreat and real arriving plates."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

import ridge_spreading
import trench_history
from raster_engine import Simulation, _rotate


def ocean_world(width=128, velocity=(.005, 0.)):
    height = width//2
    s = Simulation(dict(width=width, height=height, duration_myr=100, dt_myr=2),
                   dict(width=width, height=height, crust=np.zeros(width*height, np.uint8)))
    s.plate = (s.xyz[:, 1] >= 0).astype(np.int16)
    s.support[:] = 0
    s.support[s.plate, np.arange(s.n)] = 1
    s.active[:] = False
    s.active[:2] = True
    s.count = 2
    s.plate_uid[:] = 0
    s.plate_uid[:2] = [1, 2]
    s.next_plate_uid = 3
    s.omega[:] = 0
    s.omega[:2, 2] = velocity
    s.age[:] = np.where(s.plate == 0, 80., 20.)
    s.polarity[:] = -1
    s._boundaries()
    # The prescribed velocity experiment starts with a developed slab. Discard
    # random-world records after replacing its geometry, and retain the real
    # local polarity/transport gates rather than bypassing the lifecycle.
    del s.trench_systems
    trench_history.initialize(s)
    for row in s.trench_systems:
        row.update(phase='mature', maturity=1., active_myr=10., shortening_km=300.)
    trench_history.prepare(s)
    return s


def equatorial_trench(s):
    values = s.support[1].reshape(s.h, s.w)[s.h//2]
    longitude = (np.arange(s.w)+.5)*2*np.pi/s.w-np.pi
    cross = np.flatnonzero((values[:-1] <= .5) & (values[1:] > .5))
    k = cross[np.argmin(np.abs(longitude[cross]))]
    return float(longitude[k]+(.5-values[k])/(values[k+1]-values[k])*(longitude[k+1]-longitude[k]))


class RidgeTrenchTransportTests(unittest.TestCase):
    def test_stationary_advection_preserves_real_fractional_forearc_claims(self):
        s = ocean_world()
        self.assertTrue(np.any((s.bcode == 2) & (s.down == 0)))
        s._migrate_trenches(2.)
        near = (s.xyz[:, 0] > .8) & (np.abs(s.xyz[:, 2]) < .5)
        partial = near & (s.plate == 0) & (s.support[1] > 1e-5) & (s.support[1] < .5)
        self.assertTrue(np.any(partial), 'Actual retreat must deposit unresolved overriding membership')
        before = s.support.copy()
        # Zero motion is a strict transport identity. A categorical-source-only
        # guard used to erase these valid claims before they could accumulate.
        s.omega[:] = 0
        s._advect(2.)
        np.testing.assert_allclose(s.support[:, near], before[:, near], atol=2e-7, rtol=0)
        self.assertTrue(np.all(s.support[1, partial] > 0))
        self.assertEqual(s.spreading_diagnostics['active_segments'], 0)

    def test_stationary_overrider_accumulates_retreat_through_repeated_advection(self):
        s = ocean_world()
        no_retreat = deepcopy(s)
        effect = []
        for _ in range(30):
            for world in (s, no_retreat):
                world._advect(2.)
                world.t += 2.
                world._rasterize()
                world._boundaries()
                if world is s:
                    world._migrate_trenches(2.)
                trench_history.update(world, 2.)
                ridge_spreading.refresh(world)
            effect.append(equatorial_trench(no_retreat)-equatorial_trench(s))
        # The existing convergence transport also moves this trench. Compare
        # matched worlds to isolate accumulated retreat rather than demanding
        # a particular absolute hinge velocity from the simplified transport.
        self.assertGreater(effect[-1], .3*2*np.pi/s.w)
        self.assertGreater(effect[-1]-effect[4], .2*2*np.pi/s.w)
        near = (s.xyz[:, 0] > .8) & (np.abs(s.xyz[:, 2]) < .5)
        self.assertTrue(np.any(near & (no_retreat.plate == 0) & (s.plate == 1)))
        np.testing.assert_array_equal(s.omega[1], np.zeros(3))
        np.testing.assert_allclose(s.support.sum(axis=0), 1., atol=2e-7)
        self.assertGreater(s.process_totals['trench_swept_km2'], 0.)

    def test_real_third_plate_arriving_across_multiple_cells_survives_ridge_update(self):
        s = ocean_world(256, (-.003, .003))
        longitude = np.arctan2(s.xyz[:, 1], s.xyz[:, 0])
        latitude = np.arcsin(s.xyz[:, 2])
        third = ((longitude > np.deg2rad(4.)) & (longitude < np.deg2rad(7.))
                 & (np.abs(latitude) < np.deg2rad(8.)))
        s.active[2] = True
        s.plate_uid[2], s.next_plate_uid = 3, 4
        s.count = 3
        s.plate[third] = 2
        s.support[:, third] = 0
        s.support[2, third] = 1
        s.omega[2] = [0., 0., -.014]
        s._boundaries()
        dt = 5.
        lookup = s._sample_coordinates(_rotate(s.xyz, -s.omega[2]*dt))
        arriving = s._sample((s.plate == 2).astype(float), lookup)
        target = ((arriving > .4) & (s.plate != 2) & (np.abs(longitude) < np.deg2rad(1.5))
                  & (np.abs(latitude) < np.deg2rad(4.)))
        for neighbor in (s.east, s.west, s.north, s.south):
            target &= s.plate[neighbor] != 2
        self.assertTrue(np.any(target), 'The incoming plate must outrun the old one-cell neighborhood guard')
        unprotected = deepcopy(s)
        s._advect(dt)
        self.assertTrue(np.all(s.support[2, target] > .1))
        self.assertTrue(np.any(s.plate[target] == 2))
        # Counterfactual isolates the original guard defect: the very same
        # transported third plate is erased if only the old territory is checked.
        original = ridge_spreading.advance
        with patch.object(ridge_spreading, 'advance',
                          side_effect=lambda world, field, step, arrivals=None:
                          original(world, field, step, arrivals=None)):
            unprotected._advect(dt)
        np.testing.assert_array_equal(unprotected.support[2, target], np.zeros(np.count_nonzero(target)))
        np.testing.assert_allclose(s.support.sum(axis=0), 1., atol=2e-7)


if __name__ == '__main__':
    unittest.main()
