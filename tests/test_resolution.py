"""Check genuine higher-resolution evolution and allocation-only chunking."""

import unittest
from unittest.mock import patch

import numpy as np

from raster_engine import DEFAULT_CONFIG, Simulation, make_initial, validate_config


class ResolutionTests(unittest.TestCase):
    def test_resolution_limits_and_default(self):
        self.assertEqual((DEFAULT_CONFIG['width'], DEFAULT_CONFIG['height']), (512, 256))
        result = validate_config(dict(width=2048, height=1024))
        self.assertEqual((result['width'], result['height']), (2048, 1024))
        # Arbitrary aspect ratios remain supported for existing inputs/tests.
        self.assertEqual(validate_config(dict(width=1024, height=300))['height'], 300)
        for updates in (dict(width=2049), dict(height=1025), dict(width=1024.5)):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                validate_config(updates)

    def test_chunking_preserves_cell_and_material_evolution_exactly(self):
        cfg = dict(width=96, height=48, duration_myr=6)
        complete, chunked = Simulation(cfg), Simulation(cfg)
        for _ in range(3):
            complete.step(2)
            with patch('raster_engine.ADVECTION_CHUNK_CELLS', 257):
                chunked.step(2)
            a, b = complete.snapshot(), chunked.snapshot()
            for key, value in a.items():
                if isinstance(value, np.ndarray):
                    np.testing.assert_array_equal(value, b[key], err_msg=key)
                else:
                    self.assertEqual(value, b[key], key)
            for name in ('pos', 'mass', 'relief', 'support', 'omega', 'suture'):
                np.testing.assert_array_equal(getattr(complete, name), getattr(chunked, name),
                                              err_msg=name)

    def test_1024_grid_evolves_real_fields_and_retains_dense_history(self):
        cfg = dict(width=1024, height=512, duration_myr=2)
        painting = make_initial(cfg)
        simulation = Simulation(cfg, painting)
        first = simulation.snapshot()
        np.testing.assert_array_equal(first['crust'], painting['crust'])
        simulation.step(2)
        final = simulation.snapshot()
        self.assertEqual(final['time_myr'], 2)
        self.assertEqual(final['stats']['integration_steps'], 1)
        for name in ('elevation', 'plate', 'crust', 'age', 'boundary'):
            self.assertEqual(final[name].shape, (1024 * 512,))
            self.assertTrue(np.all(np.isfinite(final[name])), name)
        self.assertTrue(np.any(final['boundary']))
        self.assertTrue(np.any(final['plate'] != first['plate']))
        self.assertTrue(np.all(simulation.active[simulation.parcel_plate]))
        self.assertTrue(np.all(simulation.active[simulation.trace_plate]))
        self.assertEqual(len(first['trace_id']), 16384)
        self.assertEqual(final['history']['initial_marker_limit'], 16384)
        self.assertEqual(final['history']['arc_marker_limit'], 4096)
        np.testing.assert_array_equal(final['trace_id'][:16384], first['trace_id'])
        self.assertFalse(np.array_equal(final['trace_xyz'][:16384], first['trace_xyz']))
        self.assertAlmostEqual(final['stats']['continental_mass_retained_fraction'], 1., places=10)
        self.assertAlmostEqual(final['stats']['craton_mass_retained_fraction'], 1., places=10)


if __name__ == '__main__':
    unittest.main()
