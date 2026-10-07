"""Measured work timing stays live, excludes pauses, and adapts to throughput."""
import math
import unittest

from progress_timing import ProgressTiming


class Clock:
    def __init__(self, now=0.):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class ProgressTimingTests(unittest.TestCase):
    def test_cold_start_requires_distinct_advances_and_measured_duration(self):
        clock = Clock()
        timer = ProgressTiming(total_units=100, clock=clock)
        self.assertEqual(timer.snapshot(), {'elapsed_seconds': 0., 'eta_seconds': None})
        for units in (1, 2, 3, 4):
            timer.update(units)
        self.assertIsNone(timer.snapshot()['eta_seconds'])
        for units in (5, 6, 7):
            clock.advance(.25)
            self.assertIsNone(timer.update(units)['eta_seconds'])
        clock.advance(1.25)
        measured = timer.snapshot()
        self.assertEqual(measured['elapsed_seconds'], 2.)
        self.assertEqual(measured['eta_seconds'], 62.)  # 3 measured units / 2 s

    def test_elapsed_and_eta_respond_to_stalled_operation_without_callbacks(self):
        clock = Clock()
        timer = ProgressTiming(total_units=100, clock=clock)
        for units in (10, 20, 30):
            clock.advance(2)
            status = timer.update(units)
        self.assertEqual(status, {'elapsed_seconds': 6., 'eta_seconds': 14.})
        clock.advance(4)
        self.assertEqual(timer.snapshot(), {'elapsed_seconds': 10., 'eta_seconds': 23.33})
        self.assertEqual(timer.update(30), timer.snapshot())
        clock.advance(100)
        self.assertEqual(timer.snapshot()['eta_seconds'], 256.67)  # stall stays in the measured rate

    def test_sparse_multi_minute_steps_produce_eta_and_keep_stalls_in_the_rate(self):
        clock = Clock()
        timer = ProgressTiming(total_units=20, clock=clock)
        for units in (1, 2):
            clock.advance(150)
            self.assertIsNone(timer.update(units)['eta_seconds'])
        clock.advance(150)
        self.assertEqual(timer.update(3), {'elapsed_seconds': 450., 'eta_seconds': 2550.})
        clock.advance(150)
        self.assertEqual(timer.snapshot(), {'elapsed_seconds': 600., 'eta_seconds': 3400.})
        # Faster new observations reduce the estimate while retaining the long
        # operation that still crosses the recent window's edge.
        for units in (4, 5, 6, 7):
            clock.advance(1)
            timer.update(units)
        self.assertEqual(timer.snapshot(), {'elapsed_seconds': 604., 'eta_seconds': 500.5})

    def test_pause_resume_preserves_elapsed_but_excludes_pause_and_old_rate(self):
        clock = Clock()
        timer = ProgressTiming(total_units=100, clock=clock)
        for units in (10, 20, 30):
            clock.advance(2)
            timer.update(units)
        clock.advance(1)
        self.assertEqual(timer.pause(), {'elapsed_seconds': 7., 'eta_seconds': None})
        clock.advance(3600)
        self.assertEqual(timer.pause(), {'elapsed_seconds': 7., 'eta_seconds': None})
        self.assertEqual(timer.resume(), {'elapsed_seconds': 7., 'eta_seconds': None})
        for units in (40, 50, 60):
            clock.advance(4)
            status = timer.update(units)
        self.assertEqual(status, {'elapsed_seconds': 19., 'eta_seconds': 16.})
        timer.resume()  # repeated resume while running must not reset the rate
        self.assertEqual(timer.snapshot(), status)

    def test_recreated_timer_uses_saved_offset_not_old_process_clock(self):
        clock = Clock(-9000)
        timer = ProgressTiming(total_units=1000, completed_units=600,
                               elapsed_seconds=25.25, running=False, clock=clock)
        clock.advance(1e6)
        self.assertEqual(timer.snapshot(), {'elapsed_seconds': 25.25, 'eta_seconds': None})
        timer.resume()
        for units in (620, 640, 660):
            clock.advance(5)
            timer.update(units)
        self.assertEqual(timer.snapshot(), {'elapsed_seconds': 40.25, 'eta_seconds': 85.})

    def test_recent_window_adapts_after_real_rate_change(self):
        clock = Clock()
        timer = ProgressTiming(total_units=1000, clock=clock, window_seconds=10)
        for units in range(10, 101, 10):
            clock.advance(1)
            timer.update(units)
        self.assertEqual(timer.snapshot()['eta_seconds'], 90.)
        for units in range(102, 123, 2):
            clock.advance(1)
            timer.update(units)
        self.assertEqual(timer.snapshot()['eta_seconds'], 439.)  # now 2 units / s

    def test_same_progress_callbacks_cannot_manufacture_a_warm_estimate(self):
        clock = Clock()
        timer = ProgressTiming(clock=clock)
        for _ in range(20):
            clock.advance(1)
            self.assertIsNone(timer.update(0)['eta_seconds'])
        clock.advance(1)
        self.assertIsNone(timer.update(.1)['eta_seconds'])
        for _ in range(20):
            clock.advance(1)
            self.assertIsNone(timer.update(.1)['eta_seconds'])

    def test_finalization_time_counts_until_finish_then_freezes(self):
        clock = Clock()
        timer = ProgressTiming(clock=clock)
        clock.advance(10)
        self.assertEqual(timer.update(1), {'elapsed_seconds': 10., 'eta_seconds': None})
        clock.advance(2)
        self.assertEqual(timer.finish(), {'elapsed_seconds': 12., 'eta_seconds': 0.})
        clock.advance(500)
        self.assertEqual(timer.snapshot(), {'elapsed_seconds': 12., 'eta_seconds': 0.})
        self.assertEqual(timer.resume(), timer.snapshot())

    def test_zero_work_and_cancellation_have_defined_timings(self):
        clock = Clock()
        empty = ProgressTiming(total_units=0, clock=clock)
        self.assertEqual(empty.finish()['eta_seconds'], 0.)
        clock.advance(100)
        self.assertEqual(empty.snapshot()['elapsed_seconds'], 0.)
        cancelled = ProgressTiming(clock=clock)
        clock.advance(5)
        self.assertEqual(cancelled.pause(), {'elapsed_seconds': 5., 'eta_seconds': None})
        with self.assertRaises(ValueError):
            cancelled.update(.1)

    def test_invalid_work_and_clock_values_fail_without_poisoning_state(self):
        for field, value in [('total_units', -1), ('completed_units', math.nan),
                             ('elapsed_seconds', math.inf), ('window_seconds', 0),
                             ('min_seconds', 0), ('min_samples', 0), ('max_samples', 3)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                ProgressTiming(**{field: value})
        clock = Clock(100)
        timer = ProgressTiming(clock=clock)
        clock.advance(1)
        timer.update(.2)
        for value in (-.1, .1, 1.1, math.nan, math.inf):
            with self.subTest(work=value), self.assertRaises(ValueError):
                timer.update(value)
        for value in (100, math.nan, math.inf, 'bad'):
            clock.now = value
            with self.subTest(clock=value), self.assertRaises(ValueError):
                timer.snapshot()
        clock.now = 102
        self.assertEqual(timer.snapshot()['elapsed_seconds'], 2.)


if __name__ == '__main__':
    unittest.main()
