"""Budget extensions retain certified responses and reject unresolved motion."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import viscous_sheet as sheet


_spec = importlib.util.spec_from_file_location('iteration_extension_dense_oracle',
    Path(__file__).with_name('test_viscous_sheet_graded_modes.py'))
_oracle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_oracle)


def binding(value):
    return value.dtype.str, value.shape, value.tobytes()


class IterationExtensionTests(unittest.TestCase):
    def problem(self):
        points, faces, area, basis, matrix, modes, body, length = _oracle.fixture(n=4)
        zero = np.zeros_like(points)
        mask = np.zeros(len(points), bool)
        return (points, faces, area, zero, mask, mask, length), body, basis, matrix, modes

    def test_certified_response_at_original_boundary_is_byte_exact(self):
        args, body, basis, matrix, modes = self.problem()
        baseline, first = sheet.solve(*args, body_force=body, iterations=512, tolerance=1e-8)
        self.assertTrue(first['converged'], first)
        boundary = first['iterations']
        original, old = sheet.solve(*args, body_force=body, iterations=boundary, tolerance=1e-8)
        extended, new = sheet.solve(*args, body_force=body, iterations=boundary,
            extension_iterations=512, tolerance=1e-8)
        self.assertTrue(old['converged'], old)
        self.assertTrue(new['converged'], new)
        self.assertEqual(binding(baseline), binding(original))
        self.assertEqual(binding(original), binding(extended))
        self.assertEqual(old['iterations'], new['iterations'])
        self.assertEqual(old['true_stationarity_norm'], new['true_stationarity_norm'])
        actual = np.einsum('nia,ni->na', basis, extended).ravel()
        load = matrix @ modes
        self.assertLessEqual(np.linalg.norm(load - matrix @ actual) / np.linalg.norm(load), 1e-8)

    def test_inadequate_extended_budget_emits_zero(self):
        args, body, *_ = self.problem()
        before = [binding(value) for value in args if isinstance(value, np.ndarray)]
        velocity, info = sheet.solve(*args, body_force=body, iterations=1,
            extension_iterations=1, tolerance=1e-13)
        self.assertFalse(info['converged'], info)
        self.assertEqual(info['iterations'], 2)
        self.assertEqual(info['maximum_iterations'], 2)
        self.assertTrue(info['emitted_zero_on_failure'])
        self.assertEqual(binding(velocity), binding(np.zeros_like(velocity)))
        self.assertEqual(before, [binding(value) for value in args if isinstance(value, np.ndarray)])

    def test_sufficient_extension_passes_independent_variational_gate(self):
        args, body, basis, matrix, modes = self.problem()
        velocity, info = sheet.solve(*args, body_force=body, iterations=1,
            extension_iterations=511, tolerance=1e-8)
        self.assertTrue(info['converged'], info)
        self.assertLessEqual(info['relative_residual'], 1e-8)
        self.assertGreaterEqual(info['residual_restarts'], 1)
        actual = np.einsum('nia,ni->na', basis, velocity).ravel()
        load = matrix @ modes
        self.assertLessEqual(np.linalg.norm(load - matrix @ actual) / np.linalg.norm(load), 1e-8)
        self.assertGreater(float(actual @ matrix @ actual), 0)

    def test_zero_extension_preserves_default_response(self):
        args, body, *_ = self.problem()
        default, old = sheet.solve(*args, body_force=body, iterations=512, tolerance=1e-8)
        explicit, new = sheet.solve(*args, body_force=body, iterations=512,
            extension_iterations=0, tolerance=1e-8)
        self.assertEqual(binding(default), binding(explicit))
        self.assertEqual(old, new)

    def test_extension_budget_requires_nonnegative_integer(self):
        args, body, *_ = self.problem()
        for invalid in (-1, True, np.bool_(False), 1.5, None):
            with self.subTest(value=invalid):
                with self.assertRaisesRegex(ValueError, 'extension'):
                    sheet.solve(*args, body_force=body, extension_iterations=invalid)

    def test_numpy_integer_budget_is_reported_without_integer_overflow(self):
        args, body, *_ = self.problem()
        # An equilibrium solve exits before iterating, so even this large finite
        # input exercises only scalar bookkeeping and cannot spend that budget.
        velocity, info = sheet.solve(*args, iterations=np.int64(2**63-1),
            extension_iterations=np.int64(1))
        self.assertTrue(info['converged'], info)
        self.assertEqual(info['iterations'], 0)
        self.assertEqual(info['maximum_iterations'], 2**63)
        self.assertEqual(binding(velocity), binding(np.zeros_like(velocity)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
