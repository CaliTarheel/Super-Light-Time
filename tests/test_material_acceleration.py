"""Scientific solver equality, freeze provenance and native input safety."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
import material_acceleration as native
import rift_mechanics


@unittest.skipUnless(native.active(), 'Native Windows build unavailable')
class MaterialAccelerationTests(unittest.TestCase):
    def test_complete_loading_is_exact(self):
        rng = np.random.default_rng(837)
        for size in (12, 37, 128):
            xyz = rng.normal(size=(size, 3))
            xyz /= np.linalg.norm(xyz, axis=1)[:, None]
            edges = np.column_stack((np.arange(size-1), np.arange(1, size)))
            loads = rng.normal(size=(size, 3))
            strength = np.exp(rng.normal(size=size))
            weights = rng.random(size)
            arguments = dict(xyz=xyz, edges=edges, strength=strength,
                             loads=loads, load_weights=weights, iterations=1000)
            with native.reference_backend():
                reference = rift_mechanics.solve_loading(**arguments)
            accelerated = rift_mechanics.solve_loading(**arguments)
            self.assertEqual(reference.keys(), accelerated.keys())
            for key in reference:
                if isinstance(reference[key], np.ndarray):
                    self.assertEqual(reference[key].dtype, accelerated[key].dtype)
                    self.assertEqual(reference[key].tobytes(), accelerated[key].tobytes(), key)
                else:
                    self.assertEqual(reference[key], accelerated[key], key)

    def test_invalid_inputs_are_rejected_before_native_call(self):
        values = [np.array([0]), np.array([1]), np.ones((1, 3)), np.ones((1, 3)),
                  np.ones((1, 3)), np.ones(1), np.ones(2)]
        for slot, replacement in ((0, np.array([.5])), (1, np.array([2])),
                                  (2, np.ones((1, 2))), (5, np.array([np.nan]))):
            bad = list(values)
            bad[slot] = replacement
            with self.assertRaises(ValueError):
                native.prepare_operator(*bad)
        operator = native.prepare_operator(*values)
        with self.assertRaises(ValueError):
            operator(np.full((2, 3), np.inf))
        with self.assertRaises(ValueError):
            operator(np.zeros((2, 2)))

    def test_frozen_source_and_backend_record(self):
        import server
        import checkpoint
        self.assertIn('material_acceleration.py', server.AUXILIARY_SOURCES)
        self.assertIn('_material_native_build.py', server.AUXILIARY_SOURCES)
        expected = server.SimulationManager.compatibility()
        changed = copy.deepcopy(expected)
        changed['material_backend']['binary_sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'material-loading backend'):
            checkpoint.verify_compatibility(changed, expected)
        with patch.object(native, '_LIBRARY', None), patch.object(native.build, 'SOURCE_SHA256', 'bad'):
            with self.assertRaisesRegex(ValueError, 'source.*fingerprint'):
                native._library()


if __name__ == '__main__':
    unittest.main()
