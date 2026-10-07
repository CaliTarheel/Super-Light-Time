"""Independent contract checks for the forward model.

These tests check repeatability, initial-condition fidelity, usable saved history,
and numerical survival for a billion-year run. They are not calibration against
Earth or a claim of geological predictive accuracy.

Run from the project directory: python -m unittest discover -s tests -v
"""
import math
import unittest

import numpy as np

from raster_engine import (DEFAULT_CONFIG, RADIUS_KM, Simulation, _rotate,
                       make_initial, validate_config)


def config(**updates):
    result = dict(DEFAULT_CONFIG)
    result.update(width=96, height=48, duration_myr=24, dt_myr=2,
                  snapshot_myr=10, plate_count=8, seed=12)
    result.update(updates)
    return validate_config(result)


def initial_map(cfg, codes):
    return {"width": cfg["width"], "height": cfg["height"],
            "crust": np.asarray(codes, dtype=np.uint8).reshape(-1).tolist()}


class InitialConditionTests(unittest.TestCase):
    def test_generated_world_reproducible_and_contains_cratons(self):
        cfg = config()
        first, second = make_initial(cfg), make_initial(cfg)
        np.testing.assert_array_equal(first["crust"], second["crust"])
        self.assertEqual((first["width"], first["height"]), (96, 48))
        codes = np.asarray(first["crust"])
        self.assertEqual(codes.size, 96 * 48)
        self.assertEqual(set(np.unique(codes)), {0, 1, 2})
        changed = make_initial(config(seed=93))
        self.assertFalse(np.array_equal(codes, changed["crust"]))

    def test_arbitrary_painted_map_preserved_at_time_zero(self):
        cfg = config()
        codes = np.zeros((48, 96), dtype=np.uint8)
        # Include the map seam and near-polar cells, not just a central continent.
        codes[3:19, :9] = 1
        codes[3:19, -7:] = 1
        codes[7:12, :3] = 2
        codes[31:43, 46:71] = 1
        codes[33:37, 53:58] = 2
        first = next(Simulation(cfg, initial_map(cfg, codes)).snapshots())
        self.assertEqual(first["time_myr"], 0)
        np.testing.assert_array_equal(np.asarray(first["crust"]).reshape(48, 96), codes)

    def test_all_ocean_and_all_continent_inputs_remain_valid(self):
        cfg = config(duration_myr=10)
        for code in (0, 1, 2):
            with self.subTest(code=code):
                codes = np.full(96 * 48, code, dtype=np.uint8)
                frames = list(Simulation(cfg, initial_map(cfg, codes)).snapshots())
                np.testing.assert_array_equal(frames[0]["crust"], codes)
                self.assertEqual(frames[-1]["time_myr"], 10)
                self.assertTrue(np.isfinite(frames[-1]["elevation"]).all())


class SphericalGeometryTests(unittest.TestCase):
    def test_euler_rotation_has_correct_direction_and_preserves_geometry(self):
        points = np.eye(3)
        quarter_turn = np.array([0., 0., np.pi / 2])
        rotated = _rotate(points, quarter_turn)
        expected = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
        np.testing.assert_allclose(rotated, expected, atol=1e-14)
        np.testing.assert_allclose(rotated @ rotated.T, points @ points.T, atol=1e-14)
        np.testing.assert_allclose(_rotate(rotated, -quarter_turn), points, atol=1e-14)
        np.testing.assert_array_equal(_rotate(points, np.zeros(3)), points)

    def test_surface_area_and_material_weights_include_polar_geometry(self):
        cfg = config()
        codes = np.full(96 * 48, 2, dtype=np.uint8)
        simulation = Simulation(cfg, initial_map(cfg, codes))
        sphere_area = 4 * np.pi * RADIUS_KM ** 2
        self.assertAlmostEqual(simulation.earth_area / sphere_area, 1., places=13)
        self.assertAlmostEqual(simulation.original_mass / sphere_area, 1., places=13)
        self.assertLess(simulation.cell_area[0], simulation.cell_area[24 * 96] / 10)

    def test_subcell_ocean_motion_accumulates_instead_of_freezing(self):
        cfg = config()
        simulation = Simulation(cfg, initial_map(cfg, np.zeros(96 * 48)))
        # The requested initial ocean is now one plate. Build a deliberate
        # two-domain transport fixture to test advection of ocean membership.
        simulation.plate = (simulation.xyz[:, 0] > 0).astype(np.int16)
        simulation.active[:2] = True
        simulation.support[:] = 0
        simulation.support[simulation.plate, np.arange(simulation.n)] = 1
        original = simulation.plate.copy()
        original_age = float(np.mean(simulation.age))
        # Each 2 Myr step moves only 0.229 degrees, much less than one cell.
        # After 100 Myr the exact rigid displacement is 0.2 radians.
        simulation.omega[:] = [0., 0., .002]
        expected = original[simulation._indices(
            _rotate(simulation.xyz, np.array([0., 0., -.2])))]
        for _ in range(50):
            simulation._advect(2)
        self.assertGreater(float(np.mean(simulation.plate != original)), .05)
        self.assertGreater(float(np.mean(simulation.plate == expected)), .97)
        # No convergence/divergence is possible during a common rigid rotation.
        self.assertAlmostEqual(float(np.mean(simulation.age)) - original_age,
                               100., delta=.01)


class HistoryTests(unittest.TestCase):
    def assert_valid_frame(self, frame, cfg):
        self.assertEqual((frame["width"], frame["height"]),
                         (cfg["width"], cfg["height"]))
        size = cfg["width"] * cfg["height"]
        for key in ("elevation", "plate", "crust", "age", "boundary"):
            values = np.asarray(frame[key])
            self.assertEqual(values.size, size, key)
            self.assertEqual(values.ndim, 1, key)
            self.assertTrue(np.isfinite(values).all(), key)
        self.assertTrue((np.asarray(frame["age"]) >= 0).all())
        self.assertTrue(np.isin(frame["crust"], [0, 1, 2, 3]).all())
        self.assertTrue(np.isin(frame["boundary"], [0, 1, 2, 3, 4, 5]).all())
        self.assertTrue(np.issubdtype(np.asarray(frame["plate"]).dtype, np.integer))
        self.assertIsInstance(frame["stats"], dict)
        self.assertIsInstance(frame["events"], list)
        self.assertIsInstance(frame["plates"], list)
        for value in frame["stats"].values():
            if isinstance(value, (int, float, np.number)):
                self.assertTrue(math.isfinite(float(value)))
        plate_area = sum(plate["area_km2"] for plate in frame["plates"])
        self.assertAlmostEqual(plate_area / (4 * np.pi * RADIUS_KM ** 2), 1., places=11)

    def test_final_time_saved_when_not_a_snapshot_multiple(self):
        cfg = config()
        frames = list(Simulation(cfg).snapshots())
        times = [frame["time_myr"] for frame in frames]
        self.assertEqual(times[0], 0)
        self.assertEqual(times[-1], 24)
        self.assertTrue(all(a < b for a, b in zip(times, times[1:])), times)
        for frame in frames:
            self.assert_valid_frame(frame, cfg)

    def test_repeatable_motion_and_snapshots_do_not_alias_live_state(self):
        cfg = config()
        simulation = Simulation(cfg)
        stream = simulation.snapshots()
        first = next(stream)
        frozen = {key: np.asarray(first[key]).copy()
                  for key in ("elevation", "plate", "crust", "age", "boundary")}
        rest = list(stream)
        for key, value in frozen.items():
            np.testing.assert_array_equal(first[key], value, err_msg=key)
        repeated = list(Simulation(cfg).snapshots())
        self.assertEqual(len(rest) + 1, len(repeated))
        for a, b in zip([first] + rest, repeated):
            for key in frozen:
                np.testing.assert_array_equal(a[key], b[key], err_msg=key)
        self.assertFalse(np.array_equal(first["age"], rest[-1]["age"]))

    def test_cancel_before_evolution(self):
        frames = list(Simulation(config()).snapshots(cancel=lambda: True))
        self.assertLessEqual(len(frames), 1)
        if frames:
            self.assertEqual(frames[0]["time_myr"], 0)

    def test_default_billion_year_world_keeps_boundaries_and_produces_arcs(self):
        # This particular generated scenario is a regression fixture for a
        # previous failure that absorbed almost all ocean plates and stalled.
        # Three plates is not a proposed universal law for arbitrary planets.
        cfg = config(width=128, height=64, plate_count=12,
                     duration_myr=1000, snapshot_myr=100)
        simulation = Simulation(cfg)
        previous = -1
        count = 0
        saw_boundary = False
        saw_resolved_arc = False
        for frame in simulation.snapshots():
            self.assert_valid_frame(frame, cfg)
            self.assertGreater(frame["time_myr"], previous)
            previous = frame["time_myr"]
            count += 1
            saw_boundary |= bool(np.any(frame["boundary"]))
            saw_resolved_arc |= bool(np.any(np.asarray(frame["crust"]) == 3))
            self.assertTrue(np.all(simulation.active[simulation.parcel_plate]),
                            f"Orphaned buoyant material at {frame['time_myr']} Myr")
            if frame["time_myr"] >= 200:
                self.assertGreaterEqual(frame["stats"]["active_plates"], 3,
                                        f"Default world stalled at {frame['time_myr']} Myr")
                self.assertTrue(np.any(frame["boundary"]),
                                f"Default world lost its boundaries at {frame['time_myr']} Myr")
            self.assertAlmostEqual(frame["stats"]["continental_mass_retained_fraction"],
                                   1., places=11)
            self.assertAlmostEqual(frame["stats"]["craton_mass_retained_fraction"],
                                   1., places=11)
        self.assertEqual(previous, 1000)
        self.assertGreaterEqual(count, 11)
        self.assertTrue(saw_boundary)
        self.assertTrue(saw_resolved_arc, "The default history should contain mapped island-arc crust.")
        self.assertGreater(frame["stats"]["arc_added_km2"], 0)


class ConfigurationTests(unittest.TestCase):
    def test_impossible_or_nonfinite_parameters_are_rejected(self):
        cases = ({"width": -2}, {"height": 0}, {"dt_myr": 0},
                 {"duration_myr": float("nan")}, {"duration_myr": float("inf")})
        for invalid in cases:
            with self.subTest(invalid=invalid):
                candidate = dict(DEFAULT_CONFIG)
                candidate.update(invalid)
                with self.assertRaises((ValueError, TypeError)):
                    validate_config(candidate)


if __name__ == "__main__":
    unittest.main()
