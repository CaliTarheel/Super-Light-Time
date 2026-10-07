"""Spreading is measured against prescribed rotations and material budgets."""
import unittest

import numpy as np

from raster_engine import Simulation, _rotate
import ridge_spreading


def globe(width=128, velocity=(-.003, .003), rotation=None):
    h = width//2
    s = Simulation(dict(width=width, height=h, duration_myr=2),
                   dict(width=width, height=h, crust=np.zeros(width*h, np.uint8)))
    s.plate = (s.xyz[:, 1] >= 0).astype(np.int16)
    s.support[:] = 0
    s.support[s.plate, np.arange(s.n)] = 1
    s.active[:] = False
    s.active[:2] = True
    s.omega[:] = 0
    s.omega[:2, 2] = velocity
    s.age[:] = 80.
    if rotation is not None:
        back = _rotate(s.xyz, -np.asarray(rotation))
        s.plate = (back[:, 1] >= 0).astype(np.int16)
        s.support[:] = 0
        s.support[s.plate, np.arange(s.n)] = 1
        s.omega[:2] = _rotate(s.omega[:2], np.asarray(rotation))
    s._boundaries()
    return s


def advance(s, steps):
    for _ in range(steps):
        s._advect(2.)
        s.t += 2
        s._rasterize()
        s._boundaries()
        ridge_spreading.refresh(s)


def equatorial_front(s):
    fronts = s.spreading_fronts['mid']
    near = np.flatnonzero(fronts[:, 0] > 0)
    i = near[np.argmin(np.abs(fronts[near, 2]))]
    return float(np.arctan2(fronts[i, 1], fronts[i, 0]))


class RidgeSpreadingTests(unittest.TestCase):
    def test_spherical_opening_area_and_wide_strips_are_accounted_once(self):
        for width, speed, dt in ((128, .003, 2.), (256, .014, 5.)):
            with self.subTest(width=width, dt=dt):
                s = globe(width, (-speed, speed))
                s._advect(dt)
                expected = 2*6371.**2*(2*speed*dt)
                d = s.spreading_diagnostics
                self.assertAlmostEqual(d['generated_area_km2']/expected, 1., delta=.04)
                self.assertAlmostEqual(d['side_p_area_km2']+d['side_q_area_km2'],
                                       d['generated_area_km2'], delta=1e-6)
                self.assertAlmostEqual(d['side_p_area_km2']/d['side_q_area_km2'], 1., delta=1e-12)
                self.assertTrue(np.all(s.age >= dt*.5-1e-9))

    def test_closing_or_stopped_front_retires_without_new_production(self):
        s = globe()
        advance(s, 2)
        before = s.process_totals['ocean_created_km2']
        s.omega[:] = 0.
        advance(s, 2)
        self.assertEqual(s.process_totals['ocean_created_km2'], before)
        self.assertEqual(len(s.spreading_fronts['key']), 0)

    def test_stationary_axis_keeps_spreading_and_creating_paired_crust(self):
        s = globe()
        advance(s, 30)
        self.assertAlmostEqual(equatorial_front(s), 0., delta=.003)
        self.assertGreater(s.process_totals['ocean_created_km2'], 1e6)
        d = s.spreading_diagnostics
        self.assertAlmostEqual(d['side_p_area_km2']/d['side_q_area_km2'], 1., delta=.01)
        self.assertAlmostEqual(d['mean_axis_speed_cm_yr'], 0., delta=1e-5)
        self.assertGreater(d['mean_full_rate_cm_yr'], 1.)
        np.testing.assert_allclose(s.support.sum(axis=0), 1., atol=2e-7)

    def test_asymmetric_velocities_move_axis_at_half_stage_across_cells(self):
        errors = []
        for width in (64, 128, 256):
            with self.subTest(width=width):
                s = globe(width, (.001, .005))
                advance(s, 40)
                # The shared curved field is remapped on the raster. Its
                # measured zero crossing must converge under refinement and
                # stay within half a cell of the analytic finite-stage axis.
                # Exact per-face plane pinning passed a tighter local check
                # but generated hundreds of false pieces in a real world.
                error = abs(equatorial_front(s)-.003*80)
                self.assertLess(error, 2*np.pi/width*.5)
                errors.append(error)
                self.assertGreater(s.process_totals['ocean_created_km2'], 1e6)
        self.assertTrue(all(b < a for a, b in zip(errors, errors[1:])))

    def test_curved_spreading_does_not_create_disconnected_plate_pieces(self):
        from fracture import component_labels
        for width in (64, 128):
            with self.subTest(width=width):
                s = globe(width)
                contour = np.sin(s.lon.ravel()+.22*np.sin(5*s.lat.ravel()))
                s.plate = (contour >= 0).astype(np.int16)
                s.support[:] = 0
                s.support[s.plate, np.arange(s.n)] = 1
                s._boundaries()
                advance(s, 40)
                pieces = sum(component_labels(s.plate == p, s.w, s.h)[1]
                             for p in np.unique(s.plate))
                self.assertEqual(pieces, 2)

    def test_old_ocean_ages_normally_away_from_ridge(self):
        s = globe()
        advance(s, 5)
        away = (np.abs(s.xyz[:, 1]) > .8) & (np.abs(s.xyz[:, 2]) < .3)
        np.testing.assert_allclose(s.age[away], 90., atol=1e-6)
        near = (s.xyz[:, 0] > .99) & (np.abs(s.xyz[:, 2]) < .05)
        self.assertLess(float(s.age[near].mean()), 90.)
        self.assertTrue(np.all(s.age >= 0))

    def test_remote_minority_claim_cannot_nucleate_in_spreading_gap(self):
        s = globe()
        s.active[2] = True
        # A real third plate exists on the opposite side of the globe, while
        # its old diffuse numerical tail reaches the opening ridge.
        remote = s.xyz[:, 0] < -.95
        s.plate[remote] = 2
        s.support *= .99
        s.support[2] += .01
        s.support[:, remote] = 0
        s.support[2, remote] = 1
        s._boundaries()
        advance(s, 30)
        self.assertFalse(np.any(s.plate[s.xyz[:, 0] > .5] == 2))
        self.assertTrue(np.any(s.plate[remote] == 2))

    def test_real_third_plate_at_junction_is_not_painted_over(self):
        s = globe()
        third = (np.abs(s.xyz[:, 1]) < .08) & (s.xyz[:, 2] > .2) & (s.xyz[:, 2] < .45) & (s.xyz[:, 0] > 0)
        s.active[2] = True
        s.plate[third] = 2
        s.support[:, third] = 0
        s.support[2, third] = 1
        s._boundaries()
        before = s.plate.copy()
        transported = s.support.copy()
        ridge_spreading.advance(s, transported, 2.)
        self.assertTrue(np.all(np.argmax(transported[:, third], axis=0) == 2))
        np.testing.assert_array_equal(s.plate, before)

    def test_common_motion_produces_no_ocean(self):
        s = globe(64, (.004, .004))
        advance(s, 5)
        self.assertEqual(s.process_totals['ocean_created_km2'], 0.)
        np.testing.assert_allclose(s.age, 90., atol=1e-6)

    def test_production_does_not_depend_on_axis_orientation(self):
        results = []
        for rotation in (None, np.array([0., 0., np.pi]), np.array([0., np.pi/2, 0.])):
            s = globe(128, rotation=rotation)
            advance(s, 1)
            results.append(s.process_totals['ocean_created_km2'])
            self.assertTrue(np.isfinite(s.support).all())
            self.assertGreater(s.spreading_diagnostics['active_segments'], 0)
        self.assertLess(max(results)/min(results), 1.12)


if __name__ == '__main__':
    unittest.main()
