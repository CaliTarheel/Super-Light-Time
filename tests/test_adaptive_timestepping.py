"""Coupled rollback properties, including RNG and partially completed halves."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np

import adaptive_timestepping as stepping
from deforming_regions import IncompleteContactStepError


def world():
    cells = np.array([1., 2., 3.])
    return SimpleNamespace(t=0., steps=0, mass=cells, alias=cells,
                           rng=np.random.default_rng(12), history=[], nested={'volume': 6.})


def accept(s, dt):
    s.mass += dt
    s.nested['volume'] += dt * 3
    s.history.append((dt, float(s.rng.random())))
    s.steps += 1
    s.t += dt


class CoupledTimesteppingTests(unittest.TestCase):
    def test_retry_matches_uninterrupted_small_steps_including_rng(self):
        actual, expected = world(), world()
        def trial(dt):
            accept(actual, dt)
            if dt > .5:
                raise IncompleteContactStepError('Contact interval not completed')
        stepping.advance(actual, 2., trial)
        for _ in range(4):
            accept(expected, .5)
        np.testing.assert_array_equal(actual.mass, expected.mass)
        self.assertIs(actual.mass, actual.alias)
        self.assertEqual(actual.history, expected.history)
        self.assertEqual(actual.rng.bit_generator.state, expected.rng.bit_generator.state)
        self.assertEqual(actual.t, 2.)
        self.assertEqual(actual.steps, 4)
        self.assertEqual(actual.timestep_diagnostics['rejected_trials'], 3)
        self.assertEqual(actual.timestep_diagnostics['minimum_dt_myr'], .5)
        self.assertFalse(hasattr(actual, '_adaptive_step_active'))

    def test_unrecoverable_later_half_rolls_back_earlier_accepted_half(self):
        s = world()
        baseline = deepcopy(vars(s))
        def trial(dt):
            old_time = s.t
            accept(s, dt)
            if dt > .5 or old_time >= .5:
                raise IncompleteContactStepError('Still violates contact bounds')
        with self.assertRaises(IncompleteContactStepError):
            stepping.advance(s, 1., trial)
        self.assertEqual(s.t, baseline['t'])
        self.assertEqual(s.steps, 0)
        self.assertEqual(s.history, [])
        self.assertEqual(s.rng.bit_generator.state, baseline['rng'].bit_generator.state)
        self.assertEqual(s.nested, baseline['nested'])
        np.testing.assert_array_equal(s.mass, baseline['mass'])
        self.assertIs(s.mass, s.alias)
        self.assertFalse(hasattr(s, '_adaptive_step_active'))

    def test_unrelated_error_is_not_hidden_by_retries(self):
        s = world()
        calls = []
        def trial(dt):
            calls.append(dt)
            accept(s, dt)
            raise ValueError('Broken conservation law')
        with self.assertRaisesRegex(ValueError, 'conservation'):
            stepping.advance(s, 1., trial)
        self.assertEqual(calls, [1.])
        self.assertEqual(s.t, 0.)
        self.assertEqual(s.history, [])

    def test_incomplete_time_is_rejected_without_advancing_world(self):
        s = world()
        with self.assertRaisesRegex(ValueError, 'physical interval'):
            stepping.advance(s, 1., lambda dt: accept(s, dt / 2))
        self.assertEqual(s.t, 0.)

    def test_event_boundary_recursion_is_included_in_outer_transaction(self):
        s = world()
        def trial(dt):
            if s.t == 0 and dt > .25:
                stepping.advance(s, .25, trial)
                stepping.advance(s, dt - .25, trial)
            else:
                accept(s, dt)
        stepping.advance(s, 1., trial)
        self.assertEqual(s.t, 1.)
        self.assertEqual([row[0] for row in s.history], [.25, .75])
        self.assertEqual(s.timestep_diagnostics['completed_dt_myr'], 1.)


if __name__ == '__main__':
    unittest.main()
