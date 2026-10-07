"""Initial relief stays coherent and material-attached through rigid motion."""
from copy import deepcopy
from pathlib import Path
import unittest

import numpy as np

from basement_relief import initial_basement_relief
from raster_engine import Simulation, _xyz


def sphere_points(count):
    index = np.arange(count, dtype=float)
    z = 1.-2.*(index+.5)/count
    longitude = index*np.pi*(3.-np.sqrt(5.))
    radius = np.sqrt(1.-z*z)
    return np.column_stack((radius*np.cos(longitude), radius*np.sin(longitude), z))


class BasementFieldTests(unittest.TestCase):
    def test_seeded_field_is_bounded_with_resolution_independent_spherical_mean(self):
        xyz = sphere_points(32_768)
        first = initial_basement_relief(xyz, 12)
        np.testing.assert_array_equal(first, initial_basement_relief(xyz, 12))
        self.assertTrue(np.all((first >= 0.) & (first <= 220.)))
        self.assertAlmostEqual(float(first.mean()), 110., delta=.01)
        self.assertGreater(float(np.ptp(first)), 100.)
        changed = initial_basement_relief(xyz, 93)
        self.assertGreater(float(np.std(first-changed)), 20.)
        self.assertAlmostEqual(float(changed.mean()), 110., delta=.01)

    def test_same_coordinates_have_same_height_in_any_sampling_grid_or_batch(self):
        xyz = sphere_points(70_003)  # Cross the bounded-memory block boundary.
        whole = initial_basement_relief(xyz, 15)
        indices = np.array([0, 33, 2000, 65_535, 65_536, 70_002])
        np.testing.assert_allclose(initial_basement_relief(xyz[indices], 15),
                                   whole[indices], atol=1e-12, rtol=0)
        np.testing.assert_allclose(initial_basement_relief(xyz[::-1], 15)[::-1],
                                   whole, atol=1e-12, rtol=0)

    def test_seam_and_polar_neighbours_are_continuous(self):
        eps = 1e-7
        latitude = np.linspace(-1.5, 1.5, 100)
        east = _xyz(np.full(100, np.pi-eps), latitude)
        west = _xyz(np.full(100, -np.pi+eps), latitude)
        self.assertLess(float(np.max(np.abs(initial_basement_relief(east, 8)
                                             -initial_basement_relief(west, 8)))), .001)
        longitude = np.linspace(-np.pi, np.pi, 100)
        for pole in (-1., 1.):
            cap = _xyz(longitude, np.full(100, pole*(np.pi/2-eps)))
            self.assertLess(float(np.ptp(initial_basement_relief(cap, 8))), .001)

    def test_short_displacement_has_small_relief_change_everywhere(self):
        xyz = sphere_points(1024)
        tangent = np.cross(xyz, np.array([.3, .8, .5]))
        tangent /= np.linalg.norm(tangent, axis=1)[:, None]
        # Ten km of travel must not reproduce the old independent parcel jumps.
        angle = 10./6371.
        moved = np.cos(angle)*xyz+np.sin(angle)*tangent
        delta = initial_basement_relief(moved, 12)-initial_basement_relief(xyz, 12)
        self.assertLess(float(np.max(np.abs(delta))), 1.)
        self.assertEqual(initial_basement_relief(np.empty((0, 3)), 12).shape, (0,))


class BasementMaterialTests(unittest.TestCase):
    def simulation(self, code=None, observed_class=Simulation):
        config = dict(width=48, height=24, mechanics_nodes=128,
                      duration_myr=2, plate_count=4, seed=12)
        initial = None if code is None else dict(width=48, height=24,
                                                crust=np.full(48*24, code).tolist())
        return observed_class(config, initial)

    def test_initialization_consumes_exact_legacy_draw_without_perturbing_rng(self):
        class ObservedSimulation(Simulation):
            def _make_parcels(self):
                expected_rng = np.random.default_rng()
                expected_rng.bit_generator.state = deepcopy(self.rng.bit_generator.state)
                super()._make_parcels()
                expected_rng.uniform(0, 220, len(self.mass))
                self.expected_parcel_rng_state = expected_rng.bit_generator.state
                self.actual_parcel_rng_state = deepcopy(self.rng.bit_generator.state)

        simulation = self.simulation(observed_class=ObservedSimulation)
        self.assertEqual(simulation.actual_parcel_rng_state,
                         simulation.expected_parcel_rng_state)
        np.testing.assert_allclose(simulation.relief,
                                   initial_basement_relief(simulation.pos, 12), atol=1e-12)
        np.testing.assert_allclose(simulation.structure['reference_elevation_m'],
                                   220.+simulation.relief+180.*(simulation.kind == 2))

    def test_rigid_polar_motion_advects_birth_relief_without_resampling_it(self):
        simulation = self.simulation()
        pos = simulation.pos.copy()
        relief = simulation.relief.copy()
        reference = simulation.structure['reference_elevation_m'].copy()
        trace_reference = simulation.trace_structure['reference_elevation_m'].copy()
        simulation.omega[:] = [0., np.pi/4, 0.]
        simulation._advect(2.)
        self.assertGreater(float(np.max(np.linalg.norm(simulation.pos-pos, axis=1))), 1.)
        self.assertTrue(np.any(pos[:, 2]*simulation.pos[:, 2] < 0.))
        np.testing.assert_array_equal(simulation.relief, relief)
        np.testing.assert_array_equal(simulation.structure['reference_elevation_m'], reference)
        np.testing.assert_array_equal(simulation.trace_structure['reference_elevation_m'],
                                      trace_reference)
        self.assertGreater(float(np.std(initial_basement_relief(simulation.pos, 12)-relief)), 10.)

    def test_flat_material_remains_flat_in_snapshots_before_and_after_polar_motion(self):
        simulation = self.simulation(code=1)
        simulation.relief[:] = 280.
        simulation.structure['reference_elevation_m'][:] = 500.
        simulation.omega[:] = [0., np.pi/4, 0.]
        for moved in (False, True):
            if moved:
                simulation._advect(2.)
            simulation._rasterize()
            frame = simulation.snapshot()
            self.assertTrue(np.all(frame['crust'] > 0))
            # A flat, continuous continent cannot acquire world-fixed hills,
            # holes or shifting source texture merely by crossing the pole.
            np.testing.assert_allclose(frame['elevation'], 500., atol=.001, rtol=0)

    def test_new_initial_condition_helper_is_preserved_in_run_source_archive(self):
        from server import capture_auxiliary_sources
        root = Path(__file__).parents[1]
        sources = capture_auxiliary_sources((root/'tectonics.py').read_bytes())
        self.assertEqual(sources['basement_relief.py'], (root/'basement_relief.py').read_bytes())


if __name__ == '__main__':
    unittest.main()
